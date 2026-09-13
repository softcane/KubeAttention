# KubeAttention Architecture

## Overview

KubeAttention runs as an out-of-tree Kubernetes scheduler. The Go plugin obtains node utilization from Metrics Server, sends all feasible nodes to the Python Brain in one protobuf request, and contributes the validated model scores to placement. Shadow mode records recommendations without changing scores.

```text
Metrics Server -> TelemetryStore -> Scheduler plugin -> Brain
                                      |                 |
                                      |                 +-> compatible MLP/XGBoost checkpoint
                                      v
                                Kubernetes binding
                                      |
                                      v
                              persistent Collector JSONL
```

CPU and memory utilization are the current active-scoring measurements. The schema also reserves hardware contention fields. They remain explicitly unavailable until a supported per-node source populates them.

---

## System components

- **Scheduler:** Kubernetes 1.35 Scheduling Framework executable with `PreEnqueue`, `PreScore`, `Score`, normalization, and `PostBind`.
- **TelemetryStore:** continuously discovers nodes and caches timestamped Metrics API samples.
- **Brain:** validates request shape, feature-schema version, freshness, required measurements, and checkpoint compatibility before inference.
- **Collector:** records measured candidate telemetry, placement, and delayed outcomes to a persistent JSONL file.
- **Trainer:** fits MLP or XGBoost models and promotes a candidate only when measured held-out error beats the resource-pressure baseline.

---

## The Brain: Model Architecture

### Model Options

KubeAttention supports MLP and XGBoost scorers. `brain.modelType` selects the checkpoint format. Both backends emit candidate-independent quality values in `[0, 1]`, which the serving boundary converts to absolute integer scores in `[0, 100]`.

MLP is the default. Model promotion, rather than a hard-coded backend recommendation, decides whether a trained artifact is eligible for deployment.

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
+-------------------+
| Score Head        |
| Linear(32 -> 1)   |
| Sigmoid           |
+-------------------+
          |
          v
   Quality [0-1]
```

**Code**: `brain/models/mlp_scorer.py`

### XGBoost Architecture

- Gradient boosted decision trees
- Squared-error training objective
- Sigmoid conversion of each raw prediction to an absolute quality value
- No candidate-relative min/max normalization

**Code**: `brain/models/xgboost_scorer.py`

---

## Feature Set

The Brain receives 15 node features plus 11 pod-context features. Every `NodeTelemetry` message carries an availability mask and a schema version, so zero and missing have different meanings.

### Node features (15)

| Feature | Current runtime source |
|---------|------------------------|
| `cpu_utilization` | Metrics Server usage / allocatable CPU |
| `memory_utilization` | Metrics Server working set / allocatable memory |
| `cpu_throttle_rate` | unavailable |
| `memory_bandwidth_gbps` | unavailable |
| `l3_cache_miss_rate` | unavailable |
| `l3_cache_occupancy_mb` | unavailable |
| `disk_io_wait_ms` | unavailable |
| `disk_iops` | unavailable |
| `network_rx_packets_sec` | unavailable |
| `network_tx_packets_sec` | unavailable |
| `network_drop_rate` | unavailable |
| `node_cost_index` | node-label/default metadata |
| `is_spot_instance` | node-label metadata |
| `spot_interruption_risk` | node-label/default metadata |
| `zone_diversity_score` | node topology metadata |

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
qualities = model.predict_quality(X)  # Candidate-independent values in [0, 1]
scores = clip(qualities * 100, 0, 100)

for i, node in enumerate(nodes):
    reasoning = generate_reasoning(node.name, scores[i], node_features[i])
```

### Output

```text
NodeScore {
    node_name: string
    score: int64
    reasoning: string
}
```

The API does not expose a confidence value. Neither backend trains or calibrates one.

---

## Go scheduler plugin

### TelemetryStore

One background store follows the shared node informer and refreshes Metrics Server samples once per second. `PreScore` reads only the cache; it does not perform one telemetry request per candidate.

### PreScore batching

`PreScore` sends one `BatchScore` RPC containing all feasible nodes. `Score` reads the corresponding result from typed cycle state. This keeps network round trips constant while local encoding and model work still scale with the number of candidates.

### Connection and failure safety

The Brain client uses generated protobuf stubs, reconnects after startup or service loss, enforces the scheduler's RPC deadline, and opens one circuit breaker after repeated failures. Disconnection, timeout, malformed response, stale data, or missing measurements produce bounded neutral scores.

---

## Why lightweight models?

The feature tensor is small tabular data rather than a sequence. MLP and gradient-boosted trees fit this contract without a transformer. A candidate still must beat the held-out non-ML baseline before promotion; model type alone is not evidence of accuracy or latency protection.

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
            "l3_cache_miss_rate": 0.0,   # Unavailable in the current source
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

## Runtime constraints

| Constraint | Behavior |
|------------|----------|
| Scheduler RPC deadline | 50 ms by default |
| Brain inference ceiling | bounded by both the caller deadline and Brain safety limit |
| Fallback | neutral score `50` |
| Telemetry staleness | samples older than 30 seconds are degraded |
| Missing required metrics | neutral batch; no inferred zero load |
| Missing/incompatible checkpoint | Brain readiness remains false |

---

## Proactive rebalancer

The rebalancer is disabled by default. When explicitly enabled, it scans running pods and writes recommendation annotations only. It does not evict, migrate, or reschedule workloads. Operators must treat its annotations as advisory.

---

## Model loading

The Brain reads `MODEL_PATH`, which the Helm chart sets to:

```text
/models/best_model.pt
```

Startup validates the checkpoint's input width, feature-schema version, and feature names. A missing or incompatible artifact keeps the Brain unready; the server never advertises a random model as healthy. A newly promoted artifact takes effect after the Brain pod restarts.

---

## Further Reading

- [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Kubernetes Metrics Server](https://github.com/kubernetes-sigs/metrics-server)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
