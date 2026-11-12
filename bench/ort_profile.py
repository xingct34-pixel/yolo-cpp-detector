#!/usr/bin/env python3
"""Summarize an existing ORT JSON profile; node sums are not end-to-end latency."""
import argparse
import collections
import json
from pathlib import Path


# 模块：算子归档：只计kernel_time事件，按实际算子和执行后端汇总


def summarize(events):
    groups = collections.defaultdict(lambda: {'calls': 0, 'duration_us': 0.0})
    for event in events:
        if event.get('cat') != 'Node' or not event.get('name', '').endswith('_kernel_time'):
            continue
        args = event.get('args', {})
        key = (args.get('op_name', 'unknown'), args.get('provider', 'unknown'))
        groups[key]['calls'] += 1
        groups[key]['duration_us'] += float(event.get('dur', 0))
    return sorted(
        [dict(op=op, provider=provider, **row) for (op, provider), row in groups.items()],
        key=lambda row: row['duration_us'],
        reverse=True,
    )


# 模块：报告入口：只读分析Profile文件，保留原始Chrome Trace以便查看时间线


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {
        'scope': 'sum of ORT Node kernel_time events; includes warmup if profiled; not wall time',
        'operators': summarize(json.loads(args.profile.read_text())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
