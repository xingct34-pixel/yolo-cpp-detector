#include "ort_backend.h"
#include "yolo_processing.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>


// 模块：精度数据导出：加载一次ORT模型，复用项目预后处理，按COCO类别ID输出原图框
// 参数：标注JSON、图片根目录、ONNX、类别名文件、输出JSON。评估使用低分数门槛以保留召回。


int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: yolo_coco_predict annotations.json images_root model.onnx classes.txt predictions.json");
        cv::FileStorage dataset(argv[1], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!dataset.isOpened())
            throw std::runtime_error("cannot read annotations");
        std::vector<int> category_ids;
        for (const auto& row : dataset["categories"])
            category_ids.push_back(int(row["id"]));
        std::sort(category_ids.begin(), category_ids.end());
        if (category_ids.size() != 80 ||
            std::adjacent_find(category_ids.begin(), category_ids.end()) != category_ids.end())
            throw std::runtime_error(
                "requires 80 unique COCO category IDs in canonical class order");
        // 类别名文件必须与主程序一致，避免导出器悄悄使用另一套类别约定。
        std::ifstream classes(argv[4]);
        std::string line;
        size_t count = 0;
        std::vector<std::string> class_names;
        while (std::getline(classes, line))
            if (!line.empty() && line != "\r") {
                if (line.back() == '\r')
                    line.pop_back();
                class_names.push_back(line);
                ++count;
            }
        if (count != 80)
            throw std::runtime_error("requires 80 class names");
        for (const auto& category : dataset["categories"]) {
            const auto position =
                std::lower_bound(category_ids.begin(), category_ids.end(), int(category["id"]));
            if (std::string(category["name"]) !=
                class_names.at(size_t(position - category_ids.begin())))
                throw std::runtime_error(
                    "dataset category order/names disagree with model classes");
        }
        OrtBackend backend(argv[3], false);
        cv::FileStorage output(argv[5], cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!output.isOpened())
            throw std::runtime_error("cannot write predictions");
        output << "confidence_threshold" << 0.001 << "nms_iou" << 0.45 << "predictions" << "[";
        const auto root = std::filesystem::canonical(argv[2]);


        // 模块：逐图执行：禁止越出图片根目录；输出使用与主程序相同的坐标还原和按类NMS


        for (const auto& row : dataset["images"]) {
            const std::string name = row["file_name"];
            const auto path = std::filesystem::canonical(root / name);
            const auto relative = path.lexically_relative(root);
            if (relative.empty() || *relative.begin() == "..")
                throw std::runtime_error("image path escapes dataset root");
            const auto image = cv::imread(path.string());
            if (image.empty())
                throw std::runtime_error("cannot decode dataset image");
            if (int(row["width"]) != image.cols || int(row["height"]) != image.rows)
                throw std::runtime_error("annotation image size mismatch");
            LetterboxInfo info;
            PreprocessTimes times;
            const auto input = prepare_yolo(image, info, times);
            const auto raw = backend.infer(input);
            const auto detections = decode_yolo(raw, image.size(), info, 0.001f, 0.45f);
            for (const auto& detection : detections) {
                output << "{" << "image_id" << int(row["id"]) << "category_id"
                       << category_ids.at(size_t(detection.class_id)) << "bbox" << "["
                       << detection.box.x << detection.box.y << detection.box.width
                       << detection.box.height << "]"
                       << "score" << detection.score << "}";
            }
        }
        output << "]";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
