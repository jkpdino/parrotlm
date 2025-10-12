#!/usr/bin/env python3
"""
Analyze PyTorch profiling traces and display key metrics.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys


def analyze_trace(trace_path: Path):
    """Analyze a PyTorch profiler trace file."""
    print(f"\nAnalyzing trace: {trace_path}")
    print(f"File size: {trace_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    with open(trace_path, 'r') as f:
        data = json.load(f)

    # Extract events
    events = data.get('traceEvents', [])
    print(f"Total events: {len(events):,}\n")

    # Analyze by category
    categories = defaultdict(lambda: {'count': 0, 'total_dur': 0, 'self_time': 0})
    operations = defaultdict(lambda: {'count': 0, 'total_dur': 0, 'avg_dur': 0})

    cuda_events = []
    cpu_events = []

    for event in events:
        if event.get('ph') == 'X':  # Complete events (have duration)
            name = event.get('name', 'Unknown')
            dur = event.get('dur', 0)
            cat = event.get('cat', 'Unknown')

            # Track by category
            categories[cat]['count'] += 1
            categories[cat]['total_dur'] += dur

            # Track operations
            if dur > 0:
                operations[name]['count'] += 1
                operations[name]['total_dur'] += dur

                # Separate CUDA and CPU
                if 'cuda' in cat.lower() or 'gpu' in cat.lower():
                    cuda_events.append((name, dur))
                else:
                    cpu_events.append((name, dur))

    # Calculate averages
    for op in operations:
        if operations[op]['count'] > 0:
            operations[op]['avg_dur'] = operations[op]['total_dur'] / operations[op]['count']

    # Print category summary
    print("="*80)
    print("CATEGORY SUMMARY")
    print("="*80)
    print(f"{'Category':<30} {'Count':>10} {'Total Time (ms)':>20} {'Avg Time (μs)':>20}")
    print("-"*80)

    sorted_cats = sorted(categories.items(), key=lambda x: x[1]['total_dur'], reverse=True)
    for cat, stats in sorted_cats[:20]:
        avg_us = (stats['total_dur'] / stats['count']) if stats['count'] > 0 else 0
        print(f"{cat:<30} {stats['count']:>10,} {stats['total_dur']/1000:>20,.2f} {avg_us:>20,.2f}")

    # Print top operations by total time
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY TOTAL TIME")
    print("="*80)
    print(f"{'Operation':<50} {'Count':>10} {'Total (ms)':>15} {'Avg (μs)':>15}")
    print("-"*80)

    sorted_ops = sorted(operations.items(), key=lambda x: x[1]['total_dur'], reverse=True)
    for op_name, stats in sorted_ops[:20]:
        op_display = op_name[:47] + "..." if len(op_name) > 50 else op_name
        print(f"{op_display:<50} {stats['count']:>10,} {stats['total_dur']/1000:>15,.2f} {stats['avg_dur']:>15,.2f}")

    # Print top operations by average time
    print("\n" + "="*80)
    print("TOP 20 SLOWEST OPERATIONS (BY AVERAGE TIME)")
    print("="*80)
    print(f"{'Operation':<50} {'Count':>10} {'Avg (μs)':>15} {'Total (ms)':>15}")
    print("-"*80)

    sorted_ops_avg = sorted(operations.items(), key=lambda x: x[1]['avg_dur'], reverse=True)
    for op_name, stats in sorted_ops_avg[:20]:
        op_display = op_name[:47] + "..." if len(op_name) > 50 else op_name
        print(f"{op_display:<50} {stats['count']:>10,} {stats['avg_dur']:>15,.2f} {stats['total_dur']/1000:>15,.2f}")

    # Look for our labeled sections
    print("\n" + "="*80)
    print("TRAINING STEP BREAKDOWN (record_function labels)")
    print("="*80)

    labeled_sections = {}
    for op_name, stats in operations.items():
        # Look for our specific labels
        if op_name in ['data_loading', 'forward', 'loss_computation', 'backward',
                       'gradient_clipping', 'optimizer_step']:
            labeled_sections[op_name] = stats

    if labeled_sections:
        print(f"{'Section':<30} {'Count':>10} {'Total (ms)':>15} {'Avg (ms)':>15}")
        print("-"*80)
        for section, stats in labeled_sections.items():
            print(f"{section:<30} {stats['count']:>10,} {stats['total_dur']/1000:>15,.2f} {stats['avg_dur']/1000:>15,.2f}")
    else:
        print("No labeled sections found. Make sure profiling captured the training steps.")

    # Memory usage (if available)
    print("\n" + "="*80)
    print("MEMORY ANALYSIS")
    print("="*80)

    # Look for memory allocation events
    memory_allocs = []
    memory_frees = []
    memory_snapshots = []

    for event in events:
        name = event.get('name', '')
        cat = event.get('cat', '')

        # CUDA memory events
        if '[memory]' in name or 'cudaMalloc' in name or 'Allocated' in name:
            args = event.get('args', {})
            if 'Addr' in args or 'Device Id' in args:
                memory_allocs.append(event)

        if 'cudaFree' in name or 'Freed' in name:
            memory_frees.append(event)

        # Memory snapshot counters
        if event.get('ph') == 'C' and 'memory' in name.lower():
            memory_snapshots.append(event)

    print(f"Memory allocations: {len(memory_allocs):,}")
    print(f"Memory frees: {len(memory_frees):,}")
    print(f"Memory snapshots: {len(memory_snapshots):,}")

    # Analyze allocation sizes
    if memory_allocs:
        print("\n" + "-"*80)
        print("TOP 20 LARGEST MEMORY ALLOCATIONS")
        print("-"*80)
        print(f"{'Operation':<50} {'Size (MB)':>15} {'Device':>10}")
        print("-"*80)

        alloc_sizes = []
        for event in memory_allocs:
            args = event.get('args', {})
            size = args.get('Bytes', args.get('bytes', 0))
            if size > 0:
                name = event.get('name', 'Unknown')
                device = args.get('Device Id', args.get('device', 'N/A'))
                alloc_sizes.append((name, size, device))

        alloc_sizes.sort(key=lambda x: x[1], reverse=True)
        for name, size, device in alloc_sizes[:20]:
            name_display = name[:47] + "..." if len(name) > 50 else name
            print(f"{name_display:<50} {size/1024/1024:>15.2f} {str(device):>10}")

        if alloc_sizes:
            total_allocated = sum(size for _, size, _ in alloc_sizes)
            print("-"*80)
            print(f"{'Total memory allocated:':<50} {total_allocated/1024/1024:>15.2f} MB")

    # Look for memory usage over time (counter events)
    if memory_snapshots:
        print("\n" + "-"*80)
        print("MEMORY USAGE OVER TIME (Counter Events)")
        print("-"*80)

        # Group by counter name
        counter_data = defaultdict(list)
        for event in memory_snapshots:
            name = event.get('name', '')
            args = event.get('args', {})
            ts = event.get('ts', 0)

            # Extract the actual memory value
            for key, value in args.items():
                if isinstance(value, (int, float)):
                    counter_data[name].append((ts, value))

        for counter_name, values in sorted(counter_data.items()):
            if values:
                values.sort(key=lambda x: x[0])
                max_val = max(v for _, v in values)
                min_val = min(v for _, v in values)
                avg_val = sum(v for _, v in values) / len(values)

                # Convert bytes to MB if the values are large
                if max_val > 1024 * 1024:
                    unit = "MB"
                    max_val /= (1024 * 1024)
                    min_val /= (1024 * 1024)
                    avg_val /= (1024 * 1024)
                else:
                    unit = "bytes"

                print(f"\n{counter_name}:")
                print(f"  Min: {min_val:,.2f} {unit}")
                print(f"  Avg: {avg_val:,.2f} {unit}")
                print(f"  Max: {max_val:,.2f} {unit}")
                print(f"  Samples: {len(values):,}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. Open chrome://tracing in Chrome browser")
    print("2. Click 'Load' and select the trace file")
    print("3. Use WASD keys to navigate the timeline")
    print("4. Click on events to see detailed information")
    print(f"\nTrace file: {trace_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler traces")
    parser.add_argument(
        "trace_path",
        nargs='?',
        type=str,
        help="Path to the trace file (.pt.trace.json)"
    )

    args = parser.parse_args()

    # Find trace file
    if args.trace_path:
        trace_path = Path(args.trace_path)
    else:
        # Look in profiling_logs directory (check parent dir if run from analyze/)
        profiling_dir = Path("profiling_logs")
        if not profiling_dir.exists():
            profiling_dir = Path("../profiling_logs")

        if profiling_dir.exists():
            traces = list(profiling_dir.glob("*.pt.trace.json"))
            if traces:
                # Use the most recent one
                trace_path = max(traces, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent trace: {trace_path}")
            else:
                print("Error: No trace files found in profiling_logs/")
                sys.exit(1)
        else:
            print("Error: profiling_logs directory not found")
            print("Usage: python analyze_profile.py [trace_file.pt.trace.json]")
            sys.exit(1)

    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}")
        sys.exit(1)

    analyze_trace(trace_path)


if __name__ == "__main__":
    main()
