#!/usr/bin/env python3
"""
Analyze memory timeline from PyTorch profiler traces.
Creates CSV of memory usage over time and identifies peak memory causes.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import sys


def analyze_memory_timeline(trace_path: Path, output_csv: Path):
    """Analyze memory timeline and export to CSV."""

    print(f"Loading trace: {trace_path}")
    with open(trace_path, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])
    print(f"Total events: {len(events):,}\n")

    # Collect all events with timestamps
    print("Processing events...")

    # Memory allocations
    memory_events = []  # (timestamp, delta_bytes, device, addr, operation_context)
    allocations_by_addr = {}  # addr -> (size, device)

    # Track what operations are happening at each time
    operation_timeline = []  # (start_ts, end_ts, operation_name, category)

    for event in events:
        name = event.get('name', '')
        args = event.get('args', {})
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        cat = event.get('cat', '')
        ph = event.get('ph', '')

        # Track memory allocations
        if '[memory]' in name:
            addr = args.get('Addr')
            size = args.get('Bytes', 0)
            device = args.get('Device Id', -1)

            if addr and size:
                memory_events.append((ts, size, device, addr, name))
                allocations_by_addr[addr] = (size, device)

        # Track memory frees
        if 'free' in name.lower() or 'dealloc' in name.lower():
            addr = args.get('Addr')
            if addr and addr in allocations_by_addr:
                size, device = allocations_by_addr[addr]
                memory_events.append((ts, -size, device, addr, name))
                del allocations_by_addr[addr]

        # Track operations (complete events with duration)
        if ph == 'X' and dur > 0:
            operation_timeline.append((ts, ts + dur, name, cat))

    print(f"Found {len(memory_events):,} memory events")
    print(f"Found {len(operation_timeline):,} operations\n")

    # Sort by timestamp
    memory_events.sort(key=lambda x: x[0])
    operation_timeline.sort(key=lambda x: x[0])

    print("Building memory timeline...")

    # Build timeline with cumulative memory
    timeline = []
    current_memory_by_device = defaultdict(int)
    peak_memory = 0
    peak_time = 0
    peak_context = None

    # Create index of operations by start time for faster lookup
    print("Indexing operations for faster lookup...")
    ops_by_time = {}
    for op_start, op_end, op_name, op_cat in operation_timeline:
        # Convert to int and only index important operations
        op_start = int(op_start)
        op_end = int(op_end)
        if op_end - op_start > 1000:  # Only operations > 1ms
            for t in range(op_start, op_end, 100000):  # Sample every 100ms
                if t not in ops_by_time:
                    ops_by_time[t] = []
                ops_by_time[t].append((op_name, op_cat))

    print(f"Processing {len(memory_events):,} memory events...")
    for idx, (ts, delta, device, addr, mem_op) in enumerate(memory_events):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(memory_events):,} events...")

        # Update memory
        current_memory_by_device[device] += delta
        total_memory = sum(current_memory_by_device.values())

        # Find active operations (use indexed lookup)
        active_ops = []
        # Round to nearest 100ms for lookup
        lookup_time = (ts // 100000) * 100000
        if lookup_time in ops_by_time:
            active_ops = ops_by_time[lookup_time][:10]  # Max 10 ops

        # Record this point in timeline
        device_str = "CPU" if device == -1 else f"GPU_{device}"

        timeline.append({
            'timestamp_us': ts,
            'timestamp_sec': ts / 1_000_000,
            'memory_bytes': total_memory,
            'memory_mb': total_memory / (1024 * 1024),
            'memory_gb': total_memory / (1024 * 1024 * 1024),
            'device': device_str,
            'device_memory_mb': current_memory_by_device[device] / (1024 * 1024),
            'delta_mb': delta / (1024 * 1024),
            'operation': mem_op[:100],  # Truncate long operation names
            'active_operations_count': len(active_ops),
            'top_active_op': active_ops[0][0][:100] if active_ops else '',
            'top_active_cat': active_ops[0][1] if active_ops else ''
        })

        # Track peak
        if total_memory > peak_memory:
            peak_memory = total_memory
            peak_time = ts
            # For peak, do detailed operation lookup
            detailed_active_ops = []
            for op_start, op_end, op_name, op_cat in operation_timeline:
                if op_start <= ts <= op_end and op_end - op_start > 10000:  # > 10ms
                    detailed_active_ops.append((op_name, op_cat))
                    if len(detailed_active_ops) >= 20:
                        break

            peak_context = {
                'timestamp': ts,
                'memory': total_memory,
                'device': device_str,
                'memory_op': mem_op,
                'active_ops': detailed_active_ops
            }

    print(f"Timeline points: {len(timeline):,}")

    # Write CSV
    print(f"\nWriting CSV to: {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp_us', 'timestamp_sec', 'memory_bytes', 'memory_mb', 'memory_gb',
            'device', 'device_memory_mb', 'delta_mb', 'operation',
            'active_operations_count', 'top_active_op', 'top_active_cat'
        ])
        writer.writeheader()
        writer.writerows(timeline)

    csv_size = output_csv.stat().st_size / (1024 * 1024)
    print(f"CSV size: {csv_size:.2f} MB")

    # Analyze peak memory
    print("\n" + "="*80)
    print("PEAK MEMORY ANALYSIS")
    print("="*80)

    if peak_context:
        peak_mb = peak_context['memory'] / (1024 * 1024)
        peak_gb = peak_context['memory'] / (1024 * 1024 * 1024)
        peak_sec = peak_context['timestamp'] / 1_000_000

        print(f"\nPeak Memory: {peak_mb:,.2f} MB ({peak_gb:.2f} GB)")
        print(f"Timestamp: {peak_sec:.2f} seconds")
        print(f"Device: {peak_context['device']}")
        print(f"Memory operation: {peak_context['memory_op']}")

        print(f"\nActive operations at peak ({len(peak_context['active_ops'])} total):")
        print("-"*80)

        # Group by category
        ops_by_cat = defaultdict(list)
        for op_name, op_cat in peak_context['active_ops']:
            ops_by_cat[op_cat].append(op_name)

        for cat, ops in sorted(ops_by_cat.items()):
            print(f"\n[{cat}] ({len(ops)} operations)")
            for op in ops[:5]:  # Show top 5 per category
                op_display = op[:70] + "..." if len(op) > 73 else op
                print(f"  - {op_display}")
            if len(ops) > 5:
                print(f"  ... and {len(ops) - 5} more")

    # Find memory growth periods
    print("\n" + "="*80)
    print("MEMORY GROWTH ANALYSIS")
    print("="*80)

    # Sample timeline at intervals to find rapid growth
    if len(timeline) > 100:
        sample_interval = len(timeline) // 100
        sampled = timeline[::sample_interval]

        max_growth_rate = 0
        max_growth_period = None

        for i in range(1, len(sampled)):
            prev = sampled[i-1]
            curr = sampled[i]

            time_diff = curr['timestamp_sec'] - prev['timestamp_sec']
            if time_diff > 0:
                mem_diff = curr['memory_mb'] - prev['memory_mb']
                growth_rate = mem_diff / time_diff  # MB per second

                if growth_rate > max_growth_rate:
                    max_growth_rate = growth_rate
                    max_growth_period = (prev, curr)

        if max_growth_period:
            prev, curr = max_growth_period
            print(f"\nFastest memory growth:")
            print(f"  Rate: {max_growth_rate:.2f} MB/second")
            print(f"  Period: {prev['timestamp_sec']:.2f}s to {curr['timestamp_sec']:.2f}s")
            print(f"  Memory change: {prev['memory_mb']:.2f} MB → {curr['memory_mb']:.2f} MB")
            print(f"  Active operation: {curr['top_active_op']}")

    # Memory distribution by device
    print("\n" + "="*80)
    print("MEMORY DISTRIBUTION BY DEVICE")
    print("="*80)

    device_stats = defaultdict(lambda: {'max': 0, 'avg': 0, 'count': 0, 'total': 0})

    for point in timeline:
        device = point['device']
        mem = point['device_memory_mb']
        device_stats[device]['max'] = max(device_stats[device]['max'], mem)
        device_stats[device]['total'] += mem
        device_stats[device]['count'] += 1

    for device, stats in sorted(device_stats.items()):
        stats['avg'] = stats['total'] / stats['count'] if stats['count'] > 0 else 0
        print(f"\n{device}:")
        print(f"  Peak: {stats['max']:,.2f} MB")
        print(f"  Average: {stats['avg']:,.2f} MB")
        print(f"  Samples: {stats['count']:,}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"CSV file created: {output_csv}")
    print(f"Timeline points: {len(timeline):,}")
    print(f"Peak memory: {peak_mb:,.2f} MB ({peak_gb:.2f} GB)")
    print("\nYou can now:")
    print("  1. Open the CSV in Excel/Numbers/Google Sheets")
    print("  2. Plot memory_mb vs timestamp_sec to visualize memory usage")
    print("  3. Filter by device to see per-device memory")
    print("  4. Identify memory spikes by looking at delta_mb column")
    print("="*80)


def main():
    if len(sys.argv) > 1:
        trace_path = Path(sys.argv[1])
    else:
        # Find most recent trace (check parent dir if run from analyze/)
        profiling_dir = Path("profiling_logs")
        if not profiling_dir.exists():
            profiling_dir = Path("../profiling_logs")

        if profiling_dir.exists():
            traces = list(profiling_dir.glob("*.pt.trace.json"))
            # Exclude _with_memory traces
            traces = [t for t in traces if '_with_memory' not in t.name]
            if traces:
                trace_path = max(traces, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent trace: {trace_path}\n")
            else:
                print("No trace files found!")
                sys.exit(1)
        else:
            print("profiling_logs directory not found!")
            sys.exit(1)

    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}")
        sys.exit(1)

    # Create output CSV path
    output_csv = trace_path.parent / f"{trace_path.stem}_memory_timeline.csv"

    analyze_memory_timeline(trace_path, output_csv)


if __name__ == "__main__":
    main()
