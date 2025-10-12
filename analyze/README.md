# Profiling Analysis Scripts

This directory contains scripts for analyzing PyTorch profiler traces.

## Scripts

### 1. `analyze_profile.py`

Provides a comprehensive text-based analysis of profiling traces.

**Usage:**
```bash
python analyze/analyze_profile.py [trace_file.pt.trace.json]
```

If no file is specified, it uses the most recent trace in `profiling_logs/`.

**Output:**
- Category summary (CPU ops, CUDA ops, Python functions)
- Top 20 operations by total time
- Top 20 slowest operations by average time
- Training step breakdown (data_loading, forward, backward, etc.)
- Memory allocation summary

**Example:**
```bash
# Analyze most recent trace
python analyze/analyze_profile.py

# Analyze specific trace
python analyze/analyze_profile.py profiling_logs/my_trace.pt.trace.json
```

---

### 2. `get_peak_memory.py`

Calculates peak memory usage from profiling traces.

**Usage:**
```bash
python analyze/get_peak_memory.py [trace_file.pt.trace.json]
```

**Output:**
- Peak memory usage by device (CPU/GPU)
- Total peak memory usage
- Timestamp when peak occurred
- Memory remaining at end of trace

**Example:**
```bash
# Get peak memory from most recent trace
python analyze/get_peak_memory.py

# Get peak memory from specific trace
python analyze/get_peak_memory.py profiling_logs/my_trace.pt.trace.json
```

---

### 3. `add_memory_counters.py`

Enhances profiling traces with explicit memory counter events for better visualization in Perfetto/Chrome Tracing.

**Usage:**
```bash
python analyze/add_memory_counters.py [trace_file.pt.trace.json]
```

**Output:**
- Creates a new trace file with `_with_memory.json` suffix
- Adds memory counter tracks that show memory usage over time as line graphs

**Example:**
```bash
# Enhance most recent trace
python analyze/add_memory_counters.py

# Enhance specific trace
python analyze/add_memory_counters.py profiling_logs/my_trace.pt.trace.json
```

**Viewing Enhanced Traces:**

1. **Perfetto UI (Recommended):**
   - Go to https://ui.perfetto.dev
   - Drag and drop the `*_with_memory.json` file
   - Look for "Memory Usage (CPU)" and "Total Memory Usage" tracks

2. **Chrome Tracing:**
   - Open `chrome://tracing` in Chrome
   - Click "Load" and select the `*_with_memory.json` file
   - Scroll down to see memory counter tracks

---

## Quick Start

After running training with profiling enabled:

```bash
# 1. Get overall analysis
python analyze/analyze_profile.py

# 2. Get peak memory
python analyze/get_peak_memory.py

# 3. Create enhanced trace for visualization
python analyze/add_memory_counters.py

# 4. Open in Perfetto
open "https://ui.perfetto.dev"
# Then drag and drop the *_with_memory.json file
```

---

## Training with Profiling

To generate profiling traces, run training with the `--profile` flag:

```bash
python train.py --config configs/training/my_config.yml --profile
```

**Profiling Options:**
- `--profile`: Enable profiling
- `--profile-dir`: Output directory (default: `./profiling_logs`)
- `--profile-wait`: Steps to skip before profiling (default: 5)
- `--profile-warmup`: Warmup steps (default: 2)
- `--profile-active`: Steps to actively profile (default: 3)
- `--profile-repeat`: Number of profiling cycles (default: 1)

**Example:**
```bash
python train.py --config configs/training/tinystories_tiny.yml \
    --profile \
    --profile-dir ./my_profiling_logs \
    --profile-wait 10 \
    --profile-active 5
```

---

## Notes

- All scripts automatically find the most recent trace if no file is specified
- Traces can be very large (500+ MB), so processing may take a minute
- The scripts work with both CPU and GPU profiling traces
- Enhanced traces (`*_with_memory.json`) are slightly larger than originals
