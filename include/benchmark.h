#pragma once

#include <cstddef>
#include <vector>

// ==================== Benchmark 性能统计模块 ====================
//
// Benchmark：专门记录推理耗时，统计平均值、P50、P95、FPS 等指标。
//
// P50：50% 的请求低于这个耗时
// P95：95% 的请求低于这个耗时
//
// 这些指标比单纯看一次 FPS 更适合性能分析。

class Benchmark
{
public:
    // ==================== 数据记录模块 ====================

    void record(double latency_ms);

    // ==================== 统计模块 ====================

    double average() const;

    double p50() const;

    double p95() const;

    double min() const;

    double max() const;

    size_t count() const;

    // ==================== 输出模块 ====================

    void print(const char* name) const;

private:
    double percentile(double p) const;

private:
    std::vector<double> latencies_;
};
