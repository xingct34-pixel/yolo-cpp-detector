#include "benchmark.h"

#include <algorithm>
#include <iostream>
#include <numeric>

// ==================== 数据记录模块 ====================

void Benchmark::record(double latency_ms)
{
    if (latency_ms >= 0.0)
    {
        latencies_.push_back(latency_ms);
    }
}

// ==================== 基础统计模块 ====================

double Benchmark::average() const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    const double sum =
        std::accumulate(
            latencies_.begin(),
            latencies_.end(),
            0.0);

    return sum / latencies_.size();
}

double Benchmark::min() const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    return *std::min_element(
        latencies_.begin(),
        latencies_.end());
}

double Benchmark::max() const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    return *std::max_element(
        latencies_.begin(),
        latencies_.end());
}

// ==================== Percentile 统计模块 ====================

double Benchmark::percentile(double p) const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    std::vector<double> sorted =
        latencies_;

    std::sort(
        sorted.begin(),
        sorted.end());

    const double index =
        p * (sorted.size() - 1);

    const size_t lower =
        static_cast<size_t>(index);

    const size_t upper =
        std::min(
            lower + 1,
            sorted.size() - 1);

    const double fraction =
        index - lower;

    return sorted[lower] * (1.0 - fraction)
         + sorted[upper] * fraction;
}

double Benchmark::p50() const
{
    return percentile(0.50);
}

double Benchmark::p95() const
{
    return percentile(0.95);
}

// ==================== 信息查询模块 ====================

size_t Benchmark::count() const
{
    return latencies_.size();
}

// ==================== 结果输出模块 ====================

void Benchmark::print(const char* name) const
{
    std::cout << "\n========== Benchmark ==========\n";

    std::cout << "Backend: " << name << '\n';
    std::cout << "Samples: " << count() << '\n';
    std::cout << "Average: " << average() << " ms\n";
    std::cout << "P50:     " << p50() << " ms\n";
    std::cout << "P95:     " << p95() << " ms\n";
    std::cout << "Min:     " << min() << " ms\n";
    std::cout << "Max:     " << max() << " ms\n";

    if (average() > 0.0)
    {
        std::cout
            << "FPS:     "
            << 1000.0 / average()
            << '\n';
    }

    std::cout
        << "===============================\n";
}
