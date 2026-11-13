// YOLO 检测部署 | tests/test_benchmark.cpp
// 用可手算的递增耗时检验 Benchmark。输入 0 到 10999 后样本数超过保存窗口，
// 全程均值仍为 5499.5，近期样本的中位数变为 5999.5，可据此查出统计范围混用。


#include "benchmark.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
int main(){Benchmark b;assert(b.count()==0&&b.p95()==0);b.record(-1);b.record(std::numeric_limits<double>::quiet_NaN());assert(b.count()==0);


// 模块：超过窗口后的统计
// 保留最近 10000 个数时，P50 取 5999 与 6000 的中点；总次数仍是 11000。


for(int n=0;n<11000;++n)b.record(n);assert(b.count()==11000);assert(b.average()==5499.5);assert(b.min()==0&&b.max()==10999);assert(b.p50()==5999.5);}
