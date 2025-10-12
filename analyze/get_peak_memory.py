#!/usr/bin/env python3
"""
Calculate peak memory usage from PyTorch profiler trace.
"""

import json
from pathlib import Path
from collections import defaultdict

def get_peak_memory(trace_path: Path):
    """Calculate peak memory usage from trace events."""
    print(f"Loading trace: {trace_path}")

    with open(trace_path, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])
    print(f"Total events: {len(events):,}\n")

    # Track memory allocations and frees over time
    memory_timeline = []  # (timestamp, delta_bytes, operation)
    allocations_by_addr = {}  # addr -> size for tracking frees

    for event in events:
        name = event.get('name', '')
        args = event.get('args', {})
        ts = event.get('ts', 0)  # microseconds

        # Look for memory allocation events
        if '[memory]' in name:
            addr = args.get('Addr')
            size = args.get('Bytes', 0)
            device = args.get('Device Id', args.get('Device Type', 'unknown'))

            if addr and size:
                # This is an allocation
                memory_timeline.append((ts, size, f'alloc_{addr}', device))
                allocations_by_addr[addr] = size

        # Look for free events (they might have different naming)
        if 'free' in name.lower() or 'dealloc' in name.lower():
            addr = args.get('Addr')
            if addr and addr in allocations_by_addr:
                size = allocations_by_addr[addr]
                memory_timeline.append((ts, -size, f'free_{addr}', device))
                del allocations_by_addr[addr]

    print(f"Found {len(memory_timeline):,} memory events")

    if not memory_timeline:
        print("\nNo memory timeline events found!")
        print("The trace may not have detailed memory tracking enabled.")
        return

    # Sort by timestamp
    memory_timeline.sort(key=lambda x: x[0])

    # Calculate cumulative memory usage over time
    current_memory = defaultdict(int)  # by device
    peak_memory = defaultdict(int)
    peak_timestamp = defaultdict(int)
    total_current = 0
    total_peak = 0
    peak_total_ts = 0

    print("\nCalculating memory usage over time...")

    for ts, delta, op, device in memory_timeline:
        current_memory[device] += delta
        total_current = sum(current_memory.values())

        # Track peak for each device
        if current_memory[device] > peak_memory[device]:
            peak_memory[device] = current_memory[device]
            peak_timestamp[device] = ts

        # Track overall peak
        if total_current > total_peak:
            total_peak = total_current
            peak_total_ts = ts

    # Print results
    print("\n" + "="*80)
    print("PEAK MEMORY USAGE")
    print("="*80)

    if peak_memory:
        print("\nBy Device:")
        print("-"*80)
        for device, peak in sorted(peak_memory.items()):
            peak_mb = peak / (1024 * 1024)
            peak_gb = peak / (1024 * 1024 * 1024)
            ts_sec = peak_timestamp[device] / 1_000_000

            device_str = f"Device {device}" if device != 'unknown' else "Unknown Device"
            if device == -1:
                device_str = "CPU"

            print(f"{device_str}:")
            print(f"  Peak: {peak_mb:,.2f} MB ({peak_gb:.2f} GB)")
            print(f"  At timestamp: {ts_sec:.2f} seconds")
            print()

    if total_peak > 0:
        print("-"*80)
        print(f"TOTAL PEAK MEMORY USAGE: {total_peak / (1024*1024):,.2f} MB ({total_peak / (1024*1024*1024):.2f} GB)")
        print(f"At timestamp: {peak_total_ts / 1_000_000:.2f} seconds")
        print("="*80)

    # Show current memory at end
    print(f"\nMemory at end of trace:")
    for device, mem in sorted(current_memory.items()):
        device_str = f"Device {device}" if device != 'unknown' else "Unknown"
        if device == -1:
            device_str = "CPU"
        print(f"  {device_str}: {mem / (1024*1024):,.2f} MB ({mem / (1024*1024*1024):.2f} GB)")

    print(f"  Total: {sum(current_memory.values()) / (1024*1024):,.2f} MB ({sum(current_memory.values()) / (1024*1024*1024):.2f} GB)")

    if sum(current_memory.values()) > 0:
        print(f"\nWARNING: Memory not freed at end of trace! Potential memory leak.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        trace_path = Path(sys.argv[1])
    else:
        # Find most recent trace (check parent dir if run from analyze/)
        profiling_dir = Path("profiling_logs")
        if not profiling_dir.exists():
            profiling_dir = Path("../profiling_logs")

        if profiling_dir.exists():
            traces = list(profiling_dir.glob("*.pt.trace.json"))
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

    get_peak_memory(trace_path)
