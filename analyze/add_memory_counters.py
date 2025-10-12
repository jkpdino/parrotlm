#!/usr/bin/env python3
"""
Add memory usage counters to PyTorch trace for better Perfetto visualization.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys


def add_memory_counters(input_trace: Path, output_trace: Path):
    """Add memory counter events to trace for Perfetto visualization."""

    print(f"Loading trace: {input_trace}")
    with open(input_trace, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])
    print(f"Original events: {len(events):,}")

    # Track memory allocations over time
    memory_timeline = []
    allocations_by_addr = {}

    print("Processing memory events...")

    for event in events:
        name = event.get('name', '')
        args = event.get('args', {})
        ts = event.get('ts', 0)
        pid = event.get('pid', 0)
        tid = event.get('tid', 0)

        if '[memory]' in name:
            addr = args.get('Addr')
            size = args.get('Bytes', 0)
            device = args.get('Device Id', -1)

            if addr and size:
                memory_timeline.append((ts, size, device, pid, tid))
                allocations_by_addr[addr] = size

    print(f"Found {len(memory_timeline):,} memory allocations")

    if not memory_timeline:
        print("No memory events found! Copying original trace...")
        import shutil
        shutil.copy(input_trace, output_trace)
        return

    # Sort by timestamp
    memory_timeline.sort(key=lambda x: x[0])

    # Calculate cumulative memory and create counter events
    current_memory_by_device = defaultdict(int)
    counter_events = []

    print("Creating counter events...")

    for ts, delta, device, pid, tid in memory_timeline:
        current_memory_by_device[device] += delta
        total_memory = sum(current_memory_by_device.values())

        device_str = "CPU" if device == -1 else f"GPU_{device}"

        # Add counter event for this device
        counter_events.append({
            "name": f"Memory Usage ({device_str})",
            "ph": "C",  # Counter event
            "ts": ts,
            "pid": pid,
            "tid": tid,
            "args": {
                "Allocated (MB)": current_memory_by_device[device] / (1024 * 1024)
            }
        })

        # Add counter for total memory
        counter_events.append({
            "name": "Total Memory Usage",
            "ph": "C",
            "ts": ts,
            "pid": pid,
            "tid": tid,
            "args": {
                "Total (MB)": total_memory / (1024 * 1024)
            }
        })

    print(f"Created {len(counter_events):,} counter events")

    # Add counter events to trace
    data['traceEvents'].extend(counter_events)

    print(f"Total events after adding counters: {len(data['traceEvents']):,}")

    # Write output
    print(f"\nWriting enhanced trace to: {output_trace}")
    with open(output_trace, 'w') as f:
        json.dump(data, f)

    output_size = output_trace.stat().st_size / (1024 * 1024)
    print(f"Output size: {output_size:.2f} MB")
    print("\n" + "="*80)
    print("SUCCESS! Open this file in Perfetto:")
    print(f"  https://ui.perfetto.dev")
    print(f"\nOr in Chrome Tracing:")
    print(f"  chrome://tracing")
    print(f"\nLook for tracks named:")
    print(f"  - 'Memory Usage (CPU)'")
    print(f"  - 'Total Memory Usage'")
    print("="*80)


def main():
    if len(sys.argv) > 1:
        input_trace = Path(sys.argv[1])
    else:
        # Find most recent trace (check parent dir if run from analyze/)
        profiling_dir = Path("profiling_logs")
        if not profiling_dir.exists():
            profiling_dir = Path("../profiling_logs")

        if profiling_dir.exists():
            traces = list(profiling_dir.glob("*.pt.trace.json"))
            if traces:
                input_trace = max(traces, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent trace: {input_trace}\n")
            else:
                print("No trace files found!")
                sys.exit(1)
        else:
            print("profiling_logs directory not found!")
            sys.exit(1)

    if not input_trace.exists():
        print(f"Trace file not found: {input_trace}")
        sys.exit(1)

    # Create output filename
    output_trace = input_trace.parent / f"{input_trace.stem}_with_memory.json"

    add_memory_counters(input_trace, output_trace)


if __name__ == "__main__":
    main()
