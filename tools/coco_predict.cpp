// YOLO 检测部署 | tools/coco_predict.cpp
// validate_coco8.py 调用本文件编出的 yolo_coco_predict，按标注清单逐张读取原图。
// 复用 yolo_processing.h 的预处理/框解码和 ort_backend.h 的 OrtBackend，把预测写成 JSON。
// 每条记录的图像 ID、类别 ID 和原图坐标，是评估器与真值配对的依据。
//
// 数据怎么走：
//   输入：annotations.json（validate_coco8.py 的 convert 生成）、图片目录、ONNX 模型、类别名文件
//   输出：predictions.json，形如 {阈值参数, predictions:[...]}。
//         只有 predictions 数组是 COCO 预测格式，Python 端读的是 json 里的 ['predictions']。
//   看实现：src/yolo_processing.cpp（准备张量、还原框）、src/ort_backend.cpp（运行模型）。看评分：回到 validate_coco8.py 的 main。
//
// 阅读地图（按源码顺序）：
//   模块 1 main 前半：检查参数、读标注、核对 80 个类别、加载模型、打开输出文件
//   模块 2 逐图推理：对每张图 读取 → 预处理 → 推理 → 解码 → 写出


#include "ort_backend.h"
#include "yolo_processing.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>


// 模块 1：精度数据导出程序（yolo_coco_predict）
// 功能：加载一次 ONNX 模型，对数据集里每张图做推理，把检测框按 COCO 预测格式写入 predictions.json。
// 目的：给 Python 脚本计算 AP 提供数据。预处理和框解码共用项目里的同一份代码，
//       Detector::detect 调用 decode_yolo 的默认分数门槛 0.5，这里评估用 0.001；两处的 NMS IoU 阈值都是 0.45。
// 参数：标注 JSON、图片根目录、ONNX 模型、类别名文件、输出 JSON。
// 置信度门槛故意设得很低（0.001），目的是保留尽可能多的候选框。
// 通俗讲：算 AP 需要看"不同置信度下"的表现，如果一开始就把低分框丢了，可能漏掉低分但正确的检测，限制这次能测到的召回率。


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
        // 为什么会有两种类别编号：
        //   模型输出的是"第几类"（下标 0~79，连续）；
        //   标注里的 category_id 是数据集自己的编号，两套不一定相等。
        // 本流程的标注由 validate_coco8.py 生成，编号是 1~80（下标 + 1）；
        // 官方 COCO 标注则是 1~90 且中间有空缺。
        // 所以先取出标注里所有类别 ID 并排序，约定"模型第 i 类 ↔ 第 i 小的 ID"。
        std::vector<int> category_ids;
        for (const auto& row : dataset["categories"])
            category_ids.push_back(int(row["id"]));
        std::sort(category_ids.begin(), category_ids.end());
        // 必须正好 80 个，且没有重复（adjacent_find 在已排序数组里找相邻相等元素）
        if (category_ids.size() != 80 ||
            std::adjacent_find(category_ids.begin(), category_ids.end()) != category_ids.end())
            throw std::runtime_error(
                "requires 80 unique COCO category IDs in canonical class order");
        // 读取类别名文件。本程序只检查"数量为 80"以及"与标注里的类别名一一对应"，
        // 并不会与主程序比对；两边是否一致，靠调用方传入同一份 data/coco.txt 保证。
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
        // 逐类核对：标注里第 i 小的类别 ID，其名字必须等于类别名数组下标 i 对应的名字（文件的第 i+1 个非空行，i 从 0 起）。
        // 通俗讲：模型输出的是"第几类"，这里确保"第几类"和标注里的类别名对得上，防止张冠李戴
        for (const auto& category : dataset["categories"]) {
            const auto position =
                std::lower_bound(category_ids.begin(), category_ids.end(), int(category["id"]));
            if (std::string(category["name"]) !=
                class_names.at(size_t(position - category_ids.begin())))
                throw std::runtime_error(
                    "dataset category order/names disagree with model classes");
        }
        // 加载模型，只加载一次，后面所有图片复用。
        // 第二个参数叫 use_cuda。传 false 时 makeSessionOptions 不注册 CUDA 执行后端，这批图片都用同一个 CPU 会话推理。
        OrtBackend backend(argv[3], false);
        // 打开输出文件，开始写 JSON：先写阈值参数，再开一个 predictions 数组。
        // output << 的写法：键、值交替写入；"{" "[" 表示开对象/数组，"}" "]" 表示收尾。
        // 注意：0.001 和 0.45 在下面调用 decode_yolo 时又写了一遍，改的时候两处要同步
        cv::FileStorage output(argv[5], cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!output.isOpened())
            throw std::runtime_error("cannot write predictions");
        output << "confidence_threshold" << 0.001 << "nms_iou" << 0.45 << "predictions" << "[";
        // canonical 会解析成绝对路径并消除 ".."、符号链接，后面用来做越界检查
        const auto root = std::filesystem::canonical(argv[2]);


        // 模块 2：逐图推理
        // 功能：对每张图执行 读取 → 预处理 → 推理 → 解码 → 写出结果。
        // 目的：产出每张图的预测框。路径不允许越出图片根目录；
        //       预处理、坐标还原和按类别 NMS 调用的是 yolo_processing.h 里的同一套函数。


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
            // 预处理：等比缩放 + 灰边填充 + 转张量。
            // tools/cpu_quantize.py 的 prepare 同样做等比缩放、灰边填充、BGR 转 RGB 和除以 255，产出 NCHW 张量。
            // validate_coco8.py 用 Python 张量的哈希检查重复；这条评估链没有把每张图的 C++ 张量与它逐字节比对。
            const auto input = prepare_yolo(image, info, times);
            // 推理。本项目的 YOLO11n ONNX 导出在 640×640 输入下约定输出 [1,84,8400]：
            // 1 张图；84 = 4 个框参数 + 80 个类别分数；
            // 8400 = 80×80 + 40×40 + 20×20（三层特征图的格子数）。换模型这些数字会变
            const auto raw = backend.infer(input);
            // 后处理：丢掉分数低于 0.001 的框，按类别做 NMS（IoU 阈值 0.45），
            // 并把框从 640×640 空间还原到原图坐标
            const auto detections = decode_yolo(raw, image.size(), info, 0.001f, 0.45f);
            // 把每个检测框写成 COCO 预测格式
            for (const auto& detection : detections) {
                output << "{" << "image_id" << int(row["id"])
                       // class_id 是模型输出的类别下标（0~79）；
                       // 例如本脚本生成的类别表以 [1,2,3,…] 开头，class_id=2 就写 ID 3。评估器按这套 ID 找同类真值。
                       // at 越界会抛异常，不会悄悄读错
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
