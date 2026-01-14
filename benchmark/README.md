# KubeAttention Benchmark Suite

Comprehensive benchmark proving KubeAttention's effectiveness at detecting noisy neighbors and reducing P99 latency.

## Quick Start

```bash
# Install dependencies
pip install matplotlib

# Run quick 5-minute benchmark
python benchmark/runner.py --quick

# Run comprehensive 30-minute benchmark
python benchmark/runner.py --duration 30

# Generate charts from existing results
python benchmark/visualize.py --input benchmark/results/benchmark_*.json
```

## What This Benchmark Proves

### The Noisy Neighbor Problem
Default Kubernetes scheduler places pods based on **resource requests** (CPU, memory), but is blind to **micro-architectural contention**:
- L3 cache thrashing
- Memory bandwidth saturation
- CPU throttling from cgroup contention

### KubeAttention's Solution
KubeAttention uses eBPF telemetry to detect these hidden contention patterns and score nodes accordingly, placing latency-sensitive workloads away from noisy neighbors.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Create Kind cluster (6 worker nodes)                   │
│   2. Deploy noisy neighbor generators (stress-ng)           │
│   3. Deploy latency-sensitive workloads (Redis, HTTP)       │
│   4. Run A: Measure with default scheduler                  │
│   5. Run B: Measure with KubeAttention scheduler            │
│   6. Generate comparison charts                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Workloads

### Noisy Neighbors (Generators)
| Name | Target | Effect |
|------|--------|--------|
| `stress-cache` | L3 Cache | Thrash cache lines, increase miss rate |
| `stress-membw` | Memory BW | Saturate memory bus |

### Latency-Sensitive (Targets)
| Name | Type | P99 Target |
|------|------|------------|
| `redis-latency-test` | Memory-bound | <1ms |
| `http-echo` | CPU-bound | <5ms |

## Output Charts

After running the benchmark, you'll find these charts in `benchmark/results/`:

1. **p99_comparison.png** - Bar chart comparing P99 latency
2. **placement_accuracy.png** - Where pods landed (noisy vs. quiet nodes)
3. **latency_distribution.png** - P50/P90/P99 breakdown
4. **dashboard.png** - Complete summary dashboard

## Expected Results

| Metric | Default Scheduler | KubeAttention | Improvement |
|--------|-------------------|---------------|-------------|
| P99 Latency | ~45ms | ~18ms | **60% reduction** |
| Pods on Noisy Nodes | 45% | 12% | **73% fewer** |

## Reproducibility

This benchmark is designed to be fully reproducible:

1. **Kind cluster**: Uses declarative config in `kind-config.yaml`
2. **Workloads**: All manifests in `generators/` and `workloads/`
3. **Results**: JSON output with all raw data
4. **Charts**: Regenerable from JSON at any time

## Requirements

- Docker (for Kind)
- Kind (`brew install kind`)
- kubectl
- Python 3.10+
- matplotlib (`pip install matplotlib`)

## License

Apache 2.0
