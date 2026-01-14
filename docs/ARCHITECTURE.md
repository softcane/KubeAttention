# KubeAttention Architecture

## Overview

KubeAttention uses lightweight machine learning models (MLP or XGBoost) to score Kubernetes nodes based on real-time eBPF telemetry. The system detects noisy neighbor patterns that traditional schedulers miss, combined with a high-performance Go middleware that ensures minimal impact on scheduling throughput.

The previous Transformer-based architecture has been replaced with simpler, faster models that achieve the same accuracy with sub-millisecond inference latency.

---

## System Components

```
                           KubeAttention
+----------------------------------------------------------------+
|                                                                 |
|  +-------------+         gRPC/UDS          +------------------+ |
|  |   Gopher    |<------------------------->|      Brain       | |
|  | Score Plugin|       (BatchScore)        | (MLP / XGBoost)  | |
|  +------+------+                           +--------+---------+ |
|         |                                           |           |
|  +------v------+                                    | PyTorch / |
|  | Telemetry   |                                    | XGBoost   |
|  |   Store     |                                    v           |
|  +------+------+                           +------------------+ |
|         |                                  |    Tetragon      | |
|         +--------------------------------->| (eBPF Metrics)   | |
|                                            +------------------+ |
+----------------------------------------------------------------+
```

---

## The Brain: Model Architecture

### Model Options

KubeAttention supports two scoring models, selectable via `brain/config.py`:

| Model | Parameters | Inference | Training | Model Size |
|-------|------------|-----------|----------|------------|
| MLP (2-layer) | ~3,500 | 0.05ms | 2.5s | 16KB |
| XGBoost | ~6,400 trees | 0.34ms | 0.1s | 100KB |

**Default**: MLP is recommended for production due to lower inference latency.

### MLP Architecture

```
Input (20 features)
    |
    v
+-------------------+
| Linear(20 -> 64)  |
| ReLU              |
+-------------------+
    |
    v
+-------------------+
| Linear(64 -> 32)  |
| ReLU              |
+-------------------+
    |
    v
+-------------------+     +-------------------+
| Score Head        |     | Confidence Head   |
| Linear(32 -> 1)   |     | Linear(32 -> 1)   |
| Sigmoid * 100     |     | Sigmoid           |
+-------------------+     +-------------------+
    |                         |
    v                         v
 Score [0-100]          Confidence [0-1]
```

**Code**: `brain/models/mlp_scorer.py`

### XGBoost Architecture

- Gradient boosted decision trees (100 estimators)
- Max depth of 6 per tree
- Trained with squared error objective
- Min-max score normalization at inference

**Code**: `brain/models/xgboost_scorer.py`

---

## Feature Set

The Brain receives 15 eBPF-derived features plus 5 pod context features:

### Node Features (15)

| Feature | Source | Purpose |
|---------|--------|---------|
| cpu_utilization | sched:sched_stat_runtime | Basic CPU load |
| cpu_throttle_rate | cgroup:cgroup_throttle | CPU contention indicator |
| memory_utilization | kprobe:__alloc_pages | Memory pressure |
| memory_bandwidth_gbps | perf:mem_load_retired | Noisy neighbor signal |
| l3_cache_miss_rate | perf:cache_misses | Critical: LLC contention |
| l3_cache_occupancy_mb | perf:llc_occupancy | Cache pressure |
| disk_io_wait_ms | block:block_rq_complete | I/O bottleneck |
| disk_iops | block:block_rq_issue | I/O load |
| network_rx_packets_sec | net:netif_receive_skb | Network load |
| network_tx_packets_sec | net:net_dev_xmit | Network load |
| network_drop_rate | skb:kfree_skb | Network saturation |
| node_cost_index | metadata | Cost optimization |
| is_spot_instance | metadata | Resilience scoring |
| spot_interruption_risk | metadata | Risk assessment |
| zone_diversity_score | computed | Zone spread incentive |

### Pod Context Features (5)

| Feature | Description |
|---------|-------------|
| cpu_normalized | Requested CPU / 4000m |
| memory_normalized | Requested memory / 16GB |
| workload_type | One-hot encoded (cpu/mem/io/balanced) |
| criticality | Priority level (low/medium/high) |
| priority | Scheduling priority weight |

---

## Scoring Pipeline

### Input Processing

```python
# Node features: (N, 15) matrix for N candidate nodes
node_features = extract_from_telemetry_cache(nodes)

# Pod features: (5,) vector for the pod being scheduled
pod_features = encode_pod_context(pod)

# Concatenate for model input: (N, 20)
X = concatenate(node_features, broadcast(pod_features, N))
```

### Forward Pass

```python
# MLP inference
scores, confidences = model(X)  # Returns (N,) scores in [0, 100]

# Generate reasoning for each node
for i, node in enumerate(nodes):
    reasoning = generate_reasoning(node.name, scores[i], node_features[i])
```

### Output

```
ScoringResult {
    node_name: str       # "node-1"
    score: int           # 0-100 (higher is better)
    confidence: float    # 0-1 (model's confidence)
    reasoning: str       # "Node node-1: low CPU load, ample memory (CPU=30%, Mem=40%)"
}
```

---

## High-Performance Gopher Plugin

To meet the strict latency requirements of the Kubernetes scheduler, the Go plugin implements several critical optimizations:

### TelemetryStore (Background Polling)

The TelemetryStore runs as a singleton background process. It periodically fetches metrics from Tetragon and caches node states. This ensures that the Score function never makes a synchronous network call to fetch metrics.

### PreScore Batching

Instead of calling the Brain gRPC endpoint for every node sequentially (which would scale O(N) where N is the number of nodes), KubeAttention uses the PreScore phase to send one batch request for all candidate nodes. This reduces the total scheduling overhead to O(1) gRPC roundtrips.

### Circuit Breaker and Safety

The BrainClient monitors latency and error rates. If the Brain takes over 45ms or returns errors, the plugin automatically trips the circuit breaker and falls back to neutral scores (50/100), ensuring cluster stability even if the ML components fail.

---

## Why Lightweight Models?

The original Transformer architecture was replaced for several reasons:

| Concern | Transformer | MLP/XGBoost |
|---------|-------------|-------------|
| Inference latency | 5-10ms | 0.05-0.34ms |
| Model complexity | ~500K params | ~3.5K params |
| Memory footprint | 2GB+ | 16-100KB |
| Training time | Hours | Seconds |
| Cold start | Slow | Instant |

The key insight is that node scoring is primarily a tabular regression problem. The input features are well-structured eBPF metrics, not sequences or images. Simple models perform equally well with dramatically better latency.

---

## Training the Model

### Data Collection

Training data is collected via the Collector component watching scheduling events:

```python
# Each training sample contains:
{
    "node_telemetry": {...},    # 15 features per node
    "pod_context": {...},       # Pod requirements
    "chosen_node": "node-1",    # Where scheduler placed the pod
    "outcome": "running",       # running / oom_killed / evicted
}
```

### Label Construction

| Outcome | Label | Weight |
|---------|-------|--------|
| running | 1.0 | 1.0 |
| restarted | 0.5 | 1.5 |
| terminated | 0.3 | 2.0 |
| oom_killed | 0.0 | 3.0 |
| evicted | 0.0 | 3.0 |

### Training Loop

```python
# Load data with pod context features
X, y, weights = prepare_training_data("events.jsonl")

# Initialize model
model = get_model("mlp", input_dim=X.shape[1])

# Train
model.train(X, y, weights=weights, epochs=50, lr=1e-3)

# Save
model.save("checkpoints/best_model.pt")
```

---

## Performance Constraints

| Constraint | Target | Implementation |
|------------|--------|----------------|
| Inference Latency | < 1ms | Lightweight MLP |
| Fallback Behavior | 50/100 score | Circuit breaker |
| Telemetry Staleness | < 10s | Staleness guard |
| Memory Footprint | < 50MB | No GPU required |

---

## Proactive Rebalancer (Phase 4)

The Rebalancer runs as a background audit loop that identifies pods on sub-optimal nodes:

1. Scans all running pods every 60 seconds
2. Scores current node vs all alternatives
3. If current node scores below 40 AND an alternative is 20+ points better:
   - Annotates the pod with `kubeattention.io/rebalance-target`
   - External controller can use this annotation to trigger eviction

---

## Further Reading

- [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Tetragon eBPF](https://tetragon.io/) - Runtime security observability
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
