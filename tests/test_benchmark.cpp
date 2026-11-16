// YOLO 检测部署 | tests/test_benchmark.cpp
// 功能：用可手算的递增耗时检验 Benchmark。输入 0 到 10999 后样本数超过保存窗口，
// 全程均值仍为 5499.5，近期样本的中位数变为 5999.5，可据此查出统计范围混用。
// 目的：让有限样本窗口与全程累计量的混用能被一个确定的期望值发现。


#include "benchmark.h"  // 把多次计时样本汇总成均值和分位数
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
int main(){Benchmark b;assert(b.count()==0&&b.p95()==0);b.record(-1);b.record(std::numeric_limits<double>::quiet_NaN());assert(b.count()==0);


// 模块：超过窗口后的统计
// 保留最近 10000 个数时，P50 取 5999 与 6000 的中点；总次数仍是 11000。
// 输入 0…10999，最近一万个数从 1000 开始。中间两个是 5999 和 6000，所以窗口中位数要是 5999.5；全程均值则还包含开头的 0…999。
// 输入超过 10000 个样本后，分位数读保留窗口。检查次数时却应看到包括旧样本的累计值。
for(int n=0;n<11000;++n)b.record(n);assert(b.count()==11000);assert(b.average()==5499.5);assert(b.min()==0&&b.max()==10999);assert(b.p50()==5999.5);}
