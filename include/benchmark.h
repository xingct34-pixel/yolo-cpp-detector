#pragma once

#include <cstddef>
#include <vector>

// 模块：Benchmark 性能统计模块
//
// Benchmark：专门记录推理耗时，统计平均值、P50、P95、FPS 等指标。
//
// P50：50% 的请求低于这个耗时
// P95：95% 的请求低于这个耗时
//
// 这些指标比单纯看一次 FPS 更适合性能分析。

class Benchmark {
public:
    // 模块：样本记录：累计耗时并维护有界统计缓冲

    void record(double latency_ms);

    // 模块：指标计算：提供均值、分位、极值及样本数量

    double average() const;

    double p50() const;

    double p95() const;

    double min() const;

    double max() const;

    size_t count() const;

    // 模块：指标展示：打印已收集的性能统计

    void print(const char* name) const;

private:
    double percentile(double p) const;

private:
    std::vector<double> latencies_;
    size_t total_count_ = 0, next_ = 0;
    double sum_ = 0, minimum_ = 0, maximum_ = 0;
};
