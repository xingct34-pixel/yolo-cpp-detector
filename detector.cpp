#include "detector.h"
#include <fstream>
#include <chrono>
using namespace std;
using namespace cv;

// 新增：创建一份限制线程数的SessionOptions
Ort::SessionOptions Detector::make_session_options() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);   // 限制ONNX Runtime内部只用1个线程做单次推理
    return options;
}



Detector::Detector(const string& model_path, const string& classes_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session_(env_, model_path.c_str(), make_session_options()) {
    

    // ifstream：input-file-stream，文件输入流，专门用于从磁盘文件读取数据
    ifstream f(classes_path);
    string line;                  // 用于临时存储从类别文件中读取的每一行文字

    // 逐行读取类别名称，并保存到class_names_容器中
    while (getline(f, line)) {
        class_names_.push_back(line);
    }

    // 输出类别数量
    cout << "加载类别数量：" << class_names_.size() << endl;

    // 构造Detector对象时创建ONNX Runtime推理会话，
    cout << "模型加载成功" << endl;
}

//预处理
vector<float> Detector::preprocess(Mat& img, int& img_w, int& img_h) {

    // 保存原始图像尺寸，后面需要将640×640模型输出的坐标映射回原图尺寸
    img_w = img.cols;
    img_h = img.rows;
    
    Mat blob;                            // blob：存放预处理中间结果的图像矩阵

    // 将原始图像缩放到模型要求的固定输入尺寸640×640
    resize(img, blob, Size(640, 640));

    // 将图像数据转换成32位浮点数，并进行归一化
    // 原始像素值范围为0~255，这里乘以1/255后变成0~1
    blob.convertTo(blob, CV_32F, 1.0 / 255.0);

    // 将OpenCV默认的BGR通道顺序转换为RGB
    cvtColor(blob, blob, COLOR_BGR2RGB);

    // 创建3个单通道矩阵，分别保存R、G、B三个通道
    Mat channels[3];

    // 原始图像的数据排列可以理解为HWC（ Height, Width, Channel）：HWC，opencv常用格式，CHW是Pytorch常用格式，加个N事批量处理数据（batch size）
    // 像素1的R、G、B → 像素2的R、G、B → ...
    split(blob, channels);

    // ONNX模型要求的输入通常是CHW格式：
    // 所有R通道数据 → 所有G通道数据 → 所有B通道数据
    // 因此后面的循环会按照R、G、B的顺序把三个通道依次放入一维数组
    vector<float> input_data;

    for (int c = 0; c < 3; c++) {
        input_data.insert(input_data.end(),

            // 当前通道数据的起始内存地址
            (float*)channels[c].data,

            // 当前通道数据的结束位置
            // 每个通道有640×640=409600个float元素
            // 指针向后偏移640×640个float元素后，
            // 就得到当前通道数据末尾的下一个位置
            (float*)channels[c].data + 640 * 640);
    }

    // 返回按照CHW格式排列的一维浮点数组，
    // 后续会将它包装成ONNX Runtime可以识别的输入Tensor
    return input_data;
}

//后处理
void Detector::postprocess(Mat& img, float* data, int img_w, int img_h) {  

    // 设置置信度阈值，只有最大类别置信度大于0.5的候选框才会被保留
    float conf_threshold = 0.5;

    // boxes：保存通过置信度筛选的检测框
    vector<Rect> boxes;

    // scores：保存每个检测框对应的最大置信度
    vector<float> scores;

    // class_ids：保存每个检测框对应的类别ID
    vector<int> class_ids;

    // YOLO输出中一共有8400个候选检测位置
    // 8400 = 80×80 + 40×40 + 20×20
    // 对应三个不同尺度的检测特征图上的所有位置
    for (int i = 0; i < 8400; i++) {

        // 用于记录当前候选框在80个类别中的最高置信度
        float max_score = 0;

        // 保存最高置信度对应的类别ID
        int class_id = 0;

        // 遍历COCO数据集的80个类别
        for (int c = 0; c < 80; c++) {

            // YOLO输出采用通道优先的内存排列方式
            // 前4个通道对应x、y、w、h
            // 后面的80个通道对应80个类别的置信度
            // (4+c)*8400+i可以表示为：
            // 第c个类别对应的通道 × 8400个候选位置 + 当前候选位置i
            float score = data[c * 8400 + 4 * 8400 + i];

            // 找到当前候选框80个类别中的最大置信度
            if (score > max_score) {
                max_score = score;

                // 保存最大置信度对应的类别ID
                class_id = c;
            }
        }

        // 判断当前候选框的最大置信度是否达到设定阈值
        // 低于阈值的候选框直接舍弃
        if (max_score > conf_threshold) {

            // 从模型输出中读取目标中心点x坐标
            // 模型坐标基于640×640输入图像，
            // 因此乘以img_w/640映射回原始图像尺寸
            float cx = data[0 * 8400 + i] * img_w / 640;

            // 读取目标中心点y坐标，并映射回原始图像尺寸
            float cy = data[1 * 8400 + i] * img_h / 640;

            // 读取检测框宽度，并映射回原始图像尺寸
            float w  = data[2 * 8400 + i] * img_w / 640;

            // 读取检测框高度，并映射回原始图像尺寸
            float h  = data[3 * 8400 + i] * img_h / 640;

            // 模型输出的是检测框中心点(cx, cy)和宽高(w, h)
            // OpenCV Rect需要左上角坐标(x, y)
            // 因此中心点分别减去宽度和高度的一半
            int x = (int)(cx - w / 2);
            int y = (int)(cy - h / 2);

            // 将当前有效检测框保存到boxes中
            // Rect参数依次为：左上角x、左上角y、框宽w、框高h
            boxes.push_back(Rect(x, y, (int)w, (int)h));

            // 保存当前检测框的最大置信度
            // 后续NMS需要根据置信度决定保留哪个框
            scores.push_back(max_score);

            // 保存当前检测框对应的类别ID
            class_ids.push_back(class_id);
        }
    }

    // 保存NMS执行后最终保留下来的检测框索引
    vector<int> indices;

    // 执行非极大值抑制（NMS，Non-Maximum Suppression）
    // boxes：所有通过置信度筛选的检测框
    // scores：每个检测框对应的置信度
    // conf_threshold：置信度筛选阈值
    // 0.45：NMS的IoU阈值
    // indices：保存NMS最终保留下来的框在boxes中的索引
    //
    // 当检测框之间的重叠程度（IoU）达到NMS阈值时，
    // 会抑制置信度较低的重复检测框，减少同一个目标出现多个框的情况
    dnn::NMSBoxes(boxes, scores, conf_threshold, 0.45, indices);

    // 遍历NMS最终保留下来的检测框
    for (int idx : indices) {

        // 在原始图像上绘制检测框
        // boxes[idx]：取出当前保留的检测框
        // Scalar(0,255,0)：绿色
        // 2：矩形框线条粗细
        rectangle(img, boxes[idx], Scalar(0, 255, 0), 2);

        // 构造检测结果标签：
        // 类别名称 + 空格 + 置信度百分比
        string label = class_names_[class_ids[idx]] + 
                       " " + to_string((int)(scores[idx] * 100)) + "%";

        // 在检测框左上角附近绘制类别和置信度文字
        // boxes[idx].x：检测框左上角x坐标
        // boxes[idx].y - 5：在检测框上方5个像素的位置
        putText(img, label, Point(boxes[idx].x, boxes[idx].y - 5),

                // FONT_HERSHEY_SIMPLEX：OpenCV自带字体
                // 0.5：字体缩放比例
                // Scalar(0,255,0)：绿色文字
                // 1：文字线条粗细
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }
}

// 推理主流程：预处理 → 创建Tensor → 推理 → 计算FPS → 取输出 → 后处理
cv::Mat Detector::detect(cv::Mat& img) {

    // 第一步：调用preprocess，把原始图像转换成模型输入需要的一维浮点数组
    int img_w, img_h;
    vector<float> input_data = preprocess(img, img_w, img_h);

    // 第二步：创建输入Tensor
    // MemoryInfo：告诉ONNX Runtime数据存放在CPU内存里（区别于GPU显存）
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 输入形状：batch=1（一次一张图）, channel=3（RGB三通道）, height=640, width=640
    vector<int64_t> input_shape = {1, 3, 640, 640};

    // 把input_data这块内存包装成ONNX Runtime认识的Tensor
    // 传入内存信息、数据指针、数据长度、形状指针、形状维度数
    // Tensor不会重新拷贝数据，而是直接指向input_data底层内存
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 第三步：准备输入输出节点名字
    // 这两个名字是模型导出ONNX时确定的，需要和模型实际的输入输出节点名一致
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    // 第四步：计时开始，记录推理前的时间点
    auto start = chrono::high_resolution_clock::now();

    // 第五步：session.Run()推理
    // 把输入Tensor送进模型，指定输入输出节点名字及个数，得到输出结果outputs
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,              // 输入Tensor个数
        output_names,
        1               // 输出Tensor个数
    );

    // 第六步：计时结束，计算FPS
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();  // 单位：秒
    double fps = 1.0 / elapsed;
    cout << "推理耗时：" << elapsed * 1000 << " ms, FPS: " << fps << endl;

    // 第七步：取出输出数据
    // GetTensorMutableData：拿到输出Tensor底层的float指针，方便postprocess用下标直接访问
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // 第八步：调用postprocess，解析结果+NMS+画框
    postprocess(img, output_data, img_w, img_h);

    // 返回画好检测框的图片
    return img;
}

/*
第一步：调用preprocess，把图片预处理成input_data

第二步：创建输入Tensor，告诉ONNX数据形状[1,3,640,640]，把input_data包装成Tensor

第三步：计时开始， 记录推理前时间

第四步：session.Run()推理， 把Tensor送进模型， 得到输出outputs

第五步：计时结束， 计算FPS

第六步：取出输出数据，GetTensorMutableData

第七步：调用postprocess， 解析结果+NMS+画框

第八步：返回画好框的图片
*/
