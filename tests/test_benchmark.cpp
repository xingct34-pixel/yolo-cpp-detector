#include "benchmark.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
int main(){Benchmark b;assert(b.count()==0&&b.p95()==0);b.record(-1);b.record(std::numeric_limits<double>::quiet_NaN());assert(b.count()==0);
for(int n=0;n<11000;++n)b.record(n);assert(b.count()==11000);assert(b.average()==5499.5);assert(b.min()==0&&b.max()==10999);assert(b.p50()==5999.5);}
