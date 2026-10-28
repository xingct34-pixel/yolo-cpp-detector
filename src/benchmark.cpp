#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

void Benchmark::record(
    double latency_ms)
{
    if (latency_ms < 0.0 || !std::isfinite(latency_ms))
    {
        return;
    }

    if(total_count_==0)minimum_=maximum_=latency_ms;
    minimum_=std::min(minimum_,latency_ms);maximum_=std::max(maximum_,latency_ms);
    sum_+=latency_ms;++total_count_;
    if(latencies_.size()<10000)latencies_.push_back(latency_ms);
    else latencies_[next_++%10000]=latency_ms;
}

double Benchmark::average() const
{
    return total_count_?sum_/total_count_:0;
}

double Benchmark::percentile(
    double p) const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument(
            "percentile must be between 0 and 1.");
    }

    std::vector<double> values =
        latencies_;

    std::sort(
        values.begin(),
        values.end());

    const double index =
        p *
        static_cast<double>(
            values.size() - 1);

    const size_t lower =
        static_cast<size_t>(index);

    const size_t upper =
        lower + 1;

    if (upper >= values.size())
    {
        return values[lower];
    }

    const double weight =
        index -
        static_cast<double>(lower);

    return values[lower] *
               (1.0 - weight) +
           values[upper] *
               weight;
}

double Benchmark::p50() const
{
    return percentile(0.50);
}

double Benchmark::p95() const
{
    return percentile(0.95);
}

double Benchmark::min() const
{
    return minimum_;
}

double Benchmark::max() const
{
    return maximum_;
}

size_t Benchmark::count() const
{
    return total_count_;
}

void Benchmark::print(
    const char* name) const
{
    std::cout
        << "\n========== Benchmark =========="
        << std::endl;

    std::cout
        << "Backend: "
        << name
        << std::endl;

    std::cout
        << "Samples: "
        << count()
        << std::endl;

    if (latencies_.empty())
    {
        std::cout
            << "没有可用的推理延迟数据。"
            << std::endl;

        std::cout
            << "=============================="
            << std::endl;

        return;
    }

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Average: "
        << average()
        << " ms"
        << std::endl;

    std::cout
        << "P50:     "
        << p50()
        << " ms"
        << std::endl;

    std::cout
        << "P95:     "
        << p95()
        << " ms"
        << std::endl;

    std::cout
        << "Min:     "
        << min()
        << " ms"
        << std::endl;

    std::cout
        << "Max:     "
        << max()
        << " ms"
        << std::endl;

    if (average() > 0.0)
    {
        std::cout
            << "Model FPS: "
            << 1000.0 / average()
            << std::endl;
    }

    std::cout
        << "=============================="
        << std::endl;
}
/*
⑫ 核心问题：为什么不能只看平均 FPS？

一句话：平均值只描述"中心"，不描述"分布"，而用户感受到的是最慢的那些帧。

平均值会被离群值拉偏或掩盖。 100 次推理里 95 次 10ms、5 次 200ms，平均 = 19.5ms，对应约 51 FPS，看起来很流畅。但 P95 已经接近 200ms，每 20 帧就卡一次。
你的 Model FPS = 1000 / average 只是"平均延迟的倒数"，不是任何一帧的真实表现。
首次推理的冷启动会污染平均值和 max，所以要先 warmup。
FPS 是吞吐视角，延迟是体验视角。 实时场景（视频流、自动驾驶、在线服务）关心的是每一帧能否在截止时间内完成，必须看分布。

面试常见指标对照：

指标	含义	注意点
平均延迟	总耗时 ÷ 次数	易被离群值扭曲
P50	中位数，典型体验	看不到长尾
P95 / P99	95% / 99% 的请求不超过该值	用于 SLA，反映长尾
吞吐量	单位时间处理的样本数	批处理会提高吞吐但可能增加单次延迟，不等于 1/延迟
模型纯推理延迟	只计 forward	GPU 需同步后再停表
端到端延迟	输入到最终结果的全流程	含预处理、拷贝、后处理
⑬ 重点问题
P50 是什么？ 延迟排序后位于 50% 位置的值（中位数）。一半请求比它快，一半比它慢，代表典型延迟。
P95 是什么？ 排序后位于 95% 位置的值。95% 的请求不超过它，剩下 5% 更慢。你的代码里是 percentile(0.95)，按 0.95 × (n-1) 定位并对相邻两个值线性插值。
为什么 P95 有意义？ 它度量长尾。平均值和 P50 都看不到偶发卡顿，P95 能。优化是否真正提升了稳定性、是否满足 SLA，都要看 P95/P99。
为什么第一次推理可能特别慢？
CUDA context 初始化
显存分配、权重上传
cuDNN / TensorRT 的 kernel 选择与 autotune
延迟加载和 JIT 编译
缓存冷启动、GPU 频率未升起
解决办法：先空跑数十次（warmup），不计入 record。
模型推理 FPS 和 Pipeline FPS 有什么区别？
模型推理 FPS = 1000 / 纯 forward 延迟，只衡量模型本身。
Pipeline FPS 包含读图、预处理、H2D 拷贝、推理、D2H 拷贝、后处理（如 NMS）、绘制等全部环节，通常明显更低。
多线程流水线下，吞吐取决于最慢的一级，但单帧延迟仍是各阶段之和。
*/
