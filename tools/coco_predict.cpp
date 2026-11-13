#include "ort_backend.h"
#include "yolo_processing.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>


// 模块：精度数据导出程序（yolo_coco_predict）
// 作用：加载一次 ONNX 模型，对数据集里每张图做推理，把检测框按 COCO 格式写入 predictions.json，
//       供前面的 Python 脚本计算 AP。预处理和后处理直接复用项目主程序的代码，保证测的就是实际部署的流程。
// 参数：标注 JSON、图片根目录、ONNX 模型、类别名文件、输出 JSON。
// 置信度门槛故意设得很低（0.001），目的是保留尽可能多的候选框。
// 通俗讲：算 AP 需要看"不同置信度下"的表现，如果一开始就把低分框丢了，曲线就不完整，AP 会算低。


int main(int argc, char** argv) {
    try {
        // 参数个数必须正好 6 个（程序名 + 5 个参数），否则打印用法并报错
        if (argc != 6)
            throw std::invalid_argument(
                "usage: yolo_coco_predict annotations.json images_root model.onnx classes.txt predictions.json");
        // 用 OpenCV 的 FileStorage 读取标注 JSON
        cv::FileStorage dataset(argv[1], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!dataset.isOpened())
            throw std::runtime_error("cannot read annotations");
        // 取出标注文件里所有类别的 ID，排序后用于"模型类别序号 → COCO 类别 ID"的映射
        std::vector<int> category_ids;
        for (const auto& row : dataset["categories"])
            category_ids.push_back(int(row["id"]));
        std::sort(category_ids.begin(), category_ids.end());
        // 必须正好 80 个，且没有重复（adjacent_find 在已排序数组里找相邻相等元素）
        if (category_ids.size() != 80 ||
            std::adjacent_find(category_ids.begin(), category_ids.end()) != category_ids.end())
            throw std::runtime_error(
                "requires 80 unique COCO category IDs in canonical class order");
        // 类别名文件必须与主程序一致，避免导出器悄悄使用另一套类别约定。
        std::ifstream classes(argv[4]);
        std::string line;
        size_t count = 0;
        std::vector<std::string> class_names;
        // 逐行读类别名，跳过空行；兼容 Windows 换行（行尾多一个 '\r'，要去掉）
        while (std::getline(classes, line))
            if (!line.empty() && line != "\r") {
                if (line.back() == '\r')
                    line.pop_back();
                class_names.push_back(line);
                ++count;
            }
        if (count != 80)
            throw std::runtime_error("requires 80 class names");
        // 逐类核对：标注里第 i 小的类别 ID，其名字必须等于类别名文件第 i 行。
        // 通俗讲：模型输出的是"第几类"，这里确保"第几类"和标注里的类别名对得上，防止张冠李戴
        for (const auto& category : dataset["categories"]) {
            const auto position =
                std::lower_bound(category_ids.begin(), category_ids.end(), int(category["id"]));
            if (std::string(category["name"]) !=
                class_names.at(size_t(position - category_ids.begin())))
                throw std::runtime_error(
                    "dataset category order/names disagree with model classes");
        }
        // 加载模型，只加载一次，后面所有图片复用；第二个参数 false 的具体含义取决于 OrtBackend 的定义
        OrtBackend backend(argv[3], false);
        // 打开输出文件，开始写 JSON：先写阈值参数，再开一个 predictions 数组
        cv::FileStorage output(argv[5], cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!output.isOpened())
            throw std::runtime_error("cannot write predictions");
        output << "confidence_threshold" << 0.001 << "nms_iou" << 0.45 << "predictions" << "[";
        // canonical 会解析成绝对路径并消除 ".."、符号链接，后面用来做越界检查
        const auto root = std::filesystem::canonical(argv[2]);


        // 模块：逐图推理
        // 作用：对每张图执行 读取 → 预处理 → 推理 → 解码 → 写出结果。
        //       禁止图片路径越出根目录；坐标还原和按类别 NMS 与主程序完全相同。


        for (const auto& row : dataset["images"]) {
            const std::string name = row["file_name"];
            const auto path = std::filesystem::canonical(root / name);
            // 路径安全检查：算出相对根目录的路径，如果以 ".." 开头说明跑到根目录外面去了，
            // 比如标注里写了 "../../secret.jpg"，这里会拒绝
            const auto relative = path.lexically_relative(root);
            if (relative.empty() || *relative.begin() == "..")
                throw std::runtime_error("image path escapes dataset root");
            const auto image = cv::imread(path.string());
            if (image.empty())
                throw std::runtime_error("cannot decode dataset image");
            // 标注里记录的宽高必须与实际图片一致，否则框的坐标换算会错
            if (int(row["width"]) != image.cols || int(row["height"]) != image.rows)
                throw std::runtime_error("annotation image size mismatch");
            // info 记录缩放比例和填充量，后处理时要靠它把框还原到原图坐标
            LetterboxInfo info;
            PreprocessTimes times;  // 预处理各步骤耗时，这里用不到，但接口要求传入
            // 预处理：等比缩放 + 灰边填充 + 转张量，与 Python 里的 prepare 对应
            const auto input = prepare_yolo(image, info, times);
            // 推理，得到原始输出 [1,84,8400]
            const auto raw = backend.infer(input);
            // 后处理：按 0.001 的分数门槛筛选，按类别做 NMS（IoU 阈值 0.45），
            // 并把框从 640×640 空间还原到原图坐标
            const auto detections = decode_yolo(raw, image.size(), info, 0.001f, 0.45f);
            // 把每个检测框写成 COCO 预测格式
            for (const auto& detection : detections) {
                output << "{" << "image_id" << int(row["id"])
                       // 模型输出的是 0~79 的类别序号，这里映射回 COCO 的真实类别 ID
                       << "category_id" << category_ids.at(size_t(detection.class_id))
                       // bbox 格式：左上角 x、y，宽，高（像素），与 COCO 一致
                       << "bbox" << "["
                       << detection.box.x << detection.box.y << detection.box.width
                       << detection.box.height << "]"
                       << "score" << detection.score << "}";
            }
        }
        // 关闭 predictions 数组
        output << "]";
        return 0;
    } catch (const std::exception& error) {
        // 任何错误都打印到标准错误并返回 1，Python 端 subprocess 的 check=True 会据此报错终止
        std::cerr << error.what() << '\n';
        return 1;
    }
}
