#pragma once

#include <string>
#include <vector>

// ==================== 推理后端统一接口 ====================
//
// InferenceBackend：推理后端的统一“插座”
//
// ORT / TensorRT 都只需要实现下面几个函数，
// Detector 不需要关心底层到底使用哪个推理框架。

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    // ==================== 推理模块 ====================
    //
    // 输入：
    //   已经完成预处理的 CHW float 数据
    //
    // 输出：
    //   模型原始输出
    virtual std::vector<float> infer(
        const std::vector<float>& input_data) = 0;

    // ==================== 性能统计模块 ====================
    //
    // 返回最近一次推理耗时
    virtual double last_latency_ms() const = 0;

    // ==================== 后端信息模块 ====================
    //
    // 例如：
    //   "ONNX Runtime"
    //   "TensorRT"
    virtual std::string name() const = 0;
};


/*不同推理框架的 API 完全不同
InferenceBackend 就是把这个共性提炼出来的契约，只暴露三件事：

infer()：输入 CHW float，输出原始张量
last_latency_ms()：最近一次推理耗时
name()：后端名称

所有框架特有的细节（会话、显存、绑定、流）都藏在各自的实现类里，不会泄漏到接口上。
*/


/*
2. 为什么 Detector 不直接依赖 OrtBackend？

如果 Detector 持有的是 OrtBackend，会出现几个问题：

依赖扩散：detector.h 要包含 ort_backend.h，进而包含 onnxruntime_cxx_api.h。所有用到 Detector 的文件都被迫依赖 ORT，没装 TensorRT 或 ORT 的环境就编不过。
扩展要改 Detector：加一个 TensorRT 就得在 Detector 里写分支，每多一个后端就多一轮修改。
违反依赖倒置：高层逻辑（检测流程）不应该依赖低层细节（具体框架），而是两者都依赖抽象。

现在的依赖方向是：

Detector ──依赖──▶ InferenceBackend ◀──实现── OrtBackend
                                    ◀──实现── TrtBackend

Detector 只认识接口。只有 createBackend() 这一个地方知道具体类，具体类的头文件只在它的 .cpp 里包含即可。
*/

/*
3. virtual 是干什么的？

virtual 开启运行时多态：通过基类指针调用函数时，实际执行哪个版本，由对象的真实类型在运行时决定，而不是由指针的静态类型决定。

cpp
std::unique_ptr<InferenceBackend> backend_ = std::make_unique<OrtBackend>(...);
backend_->infer(input);   // 实际调用的是 OrtBackend::infer

换成 TrtBackend 之后，同一行 backend_->infer(input) 就会走 TensorRT 的实现，Detector 的代码一个字都不用改。

实现层面是靠每个对象里的虚表指针（vptr）找到虚函数表，多一次间接调用，开销很小，相比一次模型推理（毫秒级）可以忽略。
*/

/*
为什么 infer() 是纯虚函数？

= 0 表示这个函数在基类里没有实现，且子类必须实现。这样做有两层作用：

强制契约：任何后端想被 Detector 使用，就必须提供 infer()，忘了实现，编译器会直接报错，而不是运行时才出问题。
基类变成抽象类：InferenceBackend 不能被直接实例化，这是合理的，因为"一个什么都不会的推理后端"没有意义。

last_latency_ms() 和 name() 也是纯虚的，道理相同，它们同样是每个后端都必须回答的问题。

补充一点：这几个函数后面带 const 的，是承诺不修改对象状态，所以通过 const 引用也能调用。
*/

/*
5. 为什么需要 virtual ~InferenceBackend()？

这是多态基类里最容易踩的坑。Detector 持有的是 unique_ptr<InferenceBackend>，销毁时通过基类指针删除对象：

如果析构函数不是 virtual：只会调用基类的析构函数，OrtBackend 的析构函数不会执行。它持有的 Ort::Session、TensorRT 的 engine、CUDA 显存就泄漏了，严格来说这还是未定义行为。
如果析构函数是 virtual：先调用子类析构，再调用基类析构，资源被正确释放。

= default 表示不需要额外逻辑，只是为了把它声明成 virtual。经验法则：只要一个类打算被继承并通过基类指针使用，就必须有虚析构函数。
*/

/*
6. 这种设计有什么好处？
好处	说明
解耦	Detector 不依赖任何具体推理框架，编译依赖干净
开闭原则	加新后端只需新增一个类并在工厂里注册，不改 Detector
可替换	运行时通过 BackendType 切换后端，方便 ORT 和 TensorRT 做性能对比
可测试	可以写一个 MockBackend，返回固定输出，单独测试预处理、后处理和 NMS，不需要真实模型
职责清晰	框架相关的复杂度（显存管理、会话配置）被限制在各自的实现类中
统一计时	last_latency_ms() 由各后端自己测，口径统一，便于公平对比

接入新后端（CPU、CUDA、OpenVINO 等）的步骤就很固定：

新建一个类继承 InferenceBackend，实现三个函数。
在 BackendType 里加一个枚举值。
在 createBackend() 里加一个 case。

有两点设计上的提醒：

ORT 的 CPU 和 CUDA 没必要做成两个类，它们只是同一个后端的不同 execution provider，通过 use_cuda 参数区分就够了。只有框架不同（ORT、TensorRT、OpenVINO）才值得单独成类。
当前接口只传 vector<float>，输入输出形状由 Detector 的常量约定。如果以后要支持动态输入尺寸或多输出模型，可以考虑让接口额外传形状，或者返回一个带形状信息的结构体，这样 infer() 会更自描述，也不用在后处理里硬编码 8400。
*/
