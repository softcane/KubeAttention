# KubeAttention 🧠

**Transformer-Based Kubernetes Scheduling for Noisy Neighbor Avoidance**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go)](https://go.dev)
[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)

KubeAttention is a residency-aware scheduler plugin that uses a **Transformer neural network** to detect and avoid **noisy neighbor** interference. By analyzing real-time eBPF telemetry (L3 cache misses, memory bandwidth, etc.), it places latency-sensitive workloads on nodes where they are least likely to suffer from resource contention.

---

## 🎯 The Problem

Standard Kubernetes schedulers are "blind" to micro-architectural contention. They only see CPU/Memory allocations. When a latency-critical pod is placed on a node with a **hidden noisy neighbor** (cache-thrashing or memory-bandwidth-heavy workload):

- **P99 Latency spikes** by up to 65%.
- **Tail latency** becomes unpredictable.
- **Hardware resources** (L3 Cache, Memory BW) are saturated despite low CPU utilization.

## 💡 The Solution: KubeAttention

KubeAttention adds a deep-learning "Brain" to the Kubernetes Scheduling Framework:

1.  **eBPF Telemetry**: Ingests 11 low-level metrics from **Tetragon** (L3 cache miss rate, memory bandwidth, etc.).
2.  **Context-Aware Scoring**: Uses **Multi-Head Attention** to evaluate nodes not just by themselves, but in the context of the entire cluster and the specific pod requirements.
3.  **High-Performance Architecture**: 
    - **Background TelemetryStore**: Polling happens out-of-band to ensure zero-latency scheduling.
    - **Batch Inference**: Amortizes gRPC overhead by scoring all candidate nodes in a single call.
    - **Circuit Breaker**: Falls back to default scheduling if inference exceeds 45ms.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    gRPC/UDS    ┌─────────────────────┐ │
│  │   Gopher    │◄──────────────►│       Brain         │ │
│  │ (Go Plugin) │   (BatchScore) │  (PyTorch Xformer)  │ │
│  └──────┬──────┘                └──────────┬──────────┘ │
│         │                                  │            │
│  ┌──────▼──────┐                ┌──────────▼──────────┐ │
│  │ Telemetry   │                │     Tetragon        │ │
│  │   Store     │                │  (eBPF Observer)    │ │
│  └─────────────┘                └─────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deep dive into the Transformer layers and eBPF feature set.

---

## 🚀 Getting Started

### Prerequisites
- **Kubernetes 1.29+**
- **Go 1.22+** (for the Gopher plugin)
- **Python 3.11+** (for the Brain server)
- **Tetragon** installed in the cluster for eBPF metrics.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/softcane/KubeAttention.git
cd KubeAttention

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r brain/requirements.txt
```

### 2. Training Strategy

KubeAttention emphasizes **real-world data** captured from your production environment.

#### A. Production: Real Data Collection (Recommended)
The system uses the `Collector` component to gather high-fidelity training data from the cluster without impacting production performance.

1.  **Enable Shadow Mode**: Set `shadowMode: true` in the scheduler arguments. KubeAttention will generate recommendations and log them to annotations without binding pods.
2.  **Run the Collector**: The collector watches for these decisions and records the outcome (e.g., if the pod was evicted or OOM-killed after placement).
3.  **Train on Real Data**:
    ```bash
    # Train using the JSONL output from the collector
    PYTHONPATH=. python brain/training/train.py --data_path /path/to/collector_output.jsonl
    ```

#### B. Bootstrapping: Synthetic Data (Sandbox/CI only)
For quick verification or CI/CD pipelines, you can bootstrap a model using synthetic data that simulates common noisy neighbor patterns:

```bash
# Generate synthetic bootstrapping data
PYTHONPATH=. python brain/training/dataset.py

# Run initial training
PYTHONPATH=. python brain/training/train.py --epochs 20
```

### 3. Running the Brain

The Brain runs as a gRPC server communicating over a Unix Domain Socket for maximum speed:

```bash
# Start the Brain server
PYTHONPATH=. python -m brain.server --socket /tmp/brain.sock
```

### 4. Building the Scheduler

```bash
# Build the Go scheduler plugin
cd pkg/scheduler
go build -o kube-attention-scheduler
```

---

## 🛡️ Reliability & Safety

- **Shadow Mode**: Run KubeAttention in parallel with the default scheduler to gather metrics without affecting placement.
- **Fail-Safe**: If the Brain is unreachable or too slow, KubeAttention automatically falls back to a neutral score (50/100).
- **Hardened Inference**: Validates inputs for NaN/Inf and clamps outputs to [0, 100].

## 📜 License
Apache 2.0 - see [LICENSE](LICENSE)
