#include "detector.h"
#include <fstream>
#include <chrono>
using namespace std;
using namespace cv;

Detector::Detector(const string& model_path, const string& classes_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session_(env_, model_path.c_str(), session_options_) {
    
    // 读取类别名称
    ifstream f(classes_path);
    string line;
    while (getline(f, line)) {
        class_names_.push_back(line);
    }
    cout << "加载类别数量：" << class_names_.size() << endl;
    cout << "模型加载成功" << endl;
}

vector<float> Detector::preprocess(Mat& img, int& img_w, int& img_h) {
    img_w = img.cols;
    img_h = img.rows;
    
    Mat blob;                            // blob：存放预处理中间结果的图像矩阵
    resize(img, blob, Size(640, 640));    //拉伸缩放原图，模型固定输入尺寸是640*640
    blob.convertTo(blob, CV_32F, 1.0 / 255.0);     //归一化，像素值转化为32位浮点数，取值范围由0-255转化为0-1
    cvtColor(blob, blob, COLOR_BGR2RGB);             //通道用BGR转化为RGB
    
    Mat channels[3];
    split(blob, channels);  //HWC转CHW（split+循环插入），把三个通道分开，按R全部→G全部→B全部排列，ONNX要求的内存格式

    vector<float> input_data;
    for (int c = 0; c < 3; c++) {
        input_data.insert(input_data.end(),
            (float*)channels[c].data,             //起始指针
            (float*)channels[c].data + 640 * 640);  //结尾指针   后移640*640的位置
    }
    return input_data;
}

//后处理
void Detector::postprocess(Mat& img, float* data, int img_w, int img_h) {  
    float conf_threshold = 0.5;           //置信阈值，大于0.5被保存
    vector<Rect> boxes;                   //保存筛选出来的框
    vector<float> scores;
    vector<int> class_ids;

    for (int i = 0; i < 8400; i++) {
        float max_score = 0;
        int class_id = 0;
        for (int c = 0; c < 80; c++) {
            float score = data[c * 8400 + 4 * 8400 + i];  //前四行是xywh，后面c（80）是coco数据集类别，i是在这一行里面取第 i 个预测框的置信度值
            if (score > max_score) {
                max_score = score;
                class_id = c;                 //更新最大置信值和他对应的id
            }
        }
        if (max_score > conf_threshold) {               //判断是否大于阈值
            float cx = data[0 * 8400 + i] * img_w / 640;
            float cy = data[1 * 8400 + i] * img_h / 640;
            float w  = data[2 * 8400 + i] * img_w / 640;
            float h  = data[3 * 8400 + i] * img_h / 640;
            int x = (int)(cx - w / 2);                                   //x轴往右变大，y轴往下变大，各减一半xy为图片左上角
            int y = (int)(cy - h / 2);
            boxes.push_back(Rect(x, y, (int)w, (int)h));
            scores.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, conf_threshold, 0.45, indices);       // 执行非极大值抑制(NMS)，过滤重叠的重复检测框，IOU 阈>0.45就会失效
    
    for (int idx : indices) {  
        rectangle(img, boxes[idx], Scalar(0, 255, 0), 2);  //OpenCV绘图函数，在图像上画矩形
        string label = class_names_[class_ids[idx]] + 
                       " " + to_string((int)(scores[idx] * 100)) + "%";          //转化为百分比
        putText(img, label, Point(boxes[idx].x, boxes[idx].y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }
}

Mat Detector::detect(Mat& img) {
    int img_w, img_h;
    auto input_data = preprocess(img, img_w, img_h);

    array<int64_t, 4> input_shape{1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    auto start = chrono::high_resolution_clock::now();

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    auto end = chrono::high_resolution_clock::now();
    float fps = 1000.0 / chrono::duration<float, milli>(end - start).count();
    putText(img, "FPS: " + to_string((int)fps),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    float* data = outputs[0].GetTensorMutableData<float>();
    postprocess(img, data, img_w, img_h);

    return img;
}
