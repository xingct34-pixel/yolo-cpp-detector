#pragma once

#include <cstddef>
#include <vector>

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
