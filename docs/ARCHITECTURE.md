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
Input (26 features: 15 node + 11 pod)
    |
    v
+-------------------+
| Linear(26 -> 64)  |
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

The Brain receives 15 eBPF-derived node features plus 11 pod context features:

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

### Pod Context Features (11)

| Feature | Description |
|---------|-------------|
| cpu_normalized | Requested CPU / 8000m |
| memory_normalized | Requested memory / 32GB |
| workload_cpu_bound | One-hot: CPU-intensive workload |
| workload_memory_bound | One-hot: Memory-intensive workload |
| workload_io_bound | One-hot: I/O-intensive workload |
| workload_balanced | One-hot: Balanced workload |
| workload_unknown | One-hot: Unknown workload type |
| criticality_unknown | One-hot: Unknown criticality |
| criticality_low | One-hot: Low priority |
| criticality_medium | One-hot: Medium priority |
| criticality_high | One-hot: High/critical priority |

---

## Scoring Pipeline

### Input Processing

```python
# Node features: (N, 15) matrix for N candidate nodes
node_features = extract_from_telemetry_cache(nodes)

# Pod features: (11,) vector for the pod being scheduled
pod_features = encode_pod_context(pod)

# Concatenate for model input: (N, 26)
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

Training data is collected via the Collector component watching scheduling events. The Collector fetches real-time CPU/memory metrics from the Kubernetes **metrics-server API** (`metrics.k8s.io/v1beta1`).

```python
# Each training sample contains:
{
    "node_telemetry": {
        "node-1": {
            "cpu_utilization": 0.092,    # From metrics-server
            "memory_utilization": 0.111, # From metrics-server
            "l3_cache_miss_rate": 0.0,   # From Tetragon (when available)
            ...
        }
    },
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

## Cost Function

The training objective uses a **weighted Mean Squared Error (MSE)** loss that penalizes critical failures more heavily than successful placements.

### Label Scoring Formula

The target label for each node is a quantitative "goodness" score. These formulas are central to the KubeAttention policy and are **fully configurable** in `brain/config.py` under `TrainingConfig`.

For each node, the baseline score is computed from telemetry:
```
label(node) = w_cpu × (1 - cpu_util) + w_mem × (1 - mem_util) + w_cache × (1 - l3_cache_miss)
```
*Default Weights: w_cpu=0.4, w_mem=0.4, w_cache=0.2*

For the **chosen node** (where the pod was actually placed), the outcome is factored in using a "Trust Factor" blend:

```
label(chosen) = (1 - trust_outcome) × telemetry_score + trust_outcome × outcome_score
```
*Default Outcome Trust: 0.7*

Where `outcome_score` maps real-world results to target values:

| Outcome | Score | Weight | Rationale |
|---------|-------|--------|-----------|
| running | 1.0 | 1.0× | Successful placement |
| restarted | 0.5 | 1.5× | Minor issue |
| terminated | 0.3 | 2.0× | Failure |
| oom_killed | 0.0 | 3.0× | Critical: noisy neighbor OOM |
| evicted | 0.0 | 3.0× | Critical: resource contention |
| failed | 0.0 | 2.0× | General failure |

### Training Loss

The weighted MSE loss is:

```
L = (1/N) × Σᵢ wᵢ × (ŷᵢ - yᵢ)²
```

Where:
- `ŷᵢ` = predicted score for node i (0-1 range during training)
- `yᵢ` = target label for node i
- `wᵢ` = outcome weight (1.0-3.0×, from table above)

This weighting scheme ensures the model learns aggressively from failures (OOM, eviction) while treating successful placements as baseline.

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

### Verified E2E Results

In our Kind cluster testing (January 2026), the Rebalancer successfully identified 4 pods for migration with score deltas of 33-34 points:

```
benchmark/http-echo-bcvr4          -> worker (delta: 33)
benchmark/redis-latency-test-cc58n -> worker (delta: 34)
benchmark/stress-membw-v2q2p       -> worker (delta: 33)
kubeattention/collector-xsgqm      -> worker (delta: 33)
```

---

## Model Loading

The Brain server loads a pre-trained model on startup from:

```
/app/brain/models/trained_model.pt
```

If no model is found, the server uses random initialization and logs a warning. To bake a model into the Docker image, place it in `brain/models/trained_model.pt` before building.

---

## Further Reading

- [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Tetragon eBPF](https://tetragon.io/) - Runtime security observability
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
