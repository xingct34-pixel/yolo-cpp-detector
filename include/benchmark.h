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

// 性能基准统计类：记录每次耗时，并提供均值、分位数、极值等指标


class Benchmark {
public:
    // 模块：样本记录：累计耗时并维护有界统计缓冲

    // 记录一次耗时样本（单位：毫秒），同时更新总和、极值和计数
    void record(double latency_ms);

    // 模块：指标计算：提供均值、分位、极值及样本数量

    // 返回平均耗时（总耗时 / 总样本数）
    double average() const;

    // 返回中位数耗时（P50）
    double p50() const;

    // 返回 95 分位耗时（P95），常用于衡量尾部延迟
    double p95() const;

    // 返回历史最小耗时
    double min() const;

    // 返回历史最大耗时
    double max() const;

    // 返回累计记录的样本总数
    size_t count() const;

    // 模块：指标展示：打印已收集的性能统计

    // 以 name 为标签打印所有统计指标
    void print(const char* name) const;

private:
    // 计算指定分位数，p 为分位比例（如 0.5 表示 P50）
    double percentile(double p) const;

private:
    // 有界耗时缓冲区，仅保留最近的样本用于分位数计算
    std::vector<double> latencies_;
    // total_count_：累计样本总数；next_：缓冲区下一个写入位置（环形覆盖）
    size_t total_count_ = 0, next_ = 0;
    // sum_：耗时总和；minimum_/maximum_：历史最小/最大耗时
    double sum_ = 0, minimum_ = 0, maximum_ = 0;
};
