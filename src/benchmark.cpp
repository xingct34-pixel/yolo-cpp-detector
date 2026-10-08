#include "benchmark.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

void Benchmark::record(
    double latency_ms)
{
    if (latency_ms < 0.0)
    {
        return;
    }

    latencies_.push_back(
        latency_ms);
}

double Benchmark::average() const
{
    if (latencies_.empty())
    {
        return 0.0;
    }

    double sum = 0.0;

    for (double latency :
         latencies_)
    {
        sum += latency;
    }

    return sum /
           static_cast<double>(
               latencies_.size());
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

size_t Benchmark::count() const
{
    return latencies_.size();
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
