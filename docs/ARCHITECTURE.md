# KubeAttention Architecture

## Overview

KubeAttention uses a **Transformer neural network** to score Kubernetes nodes based on real-time telemetry. The key innovation is using **attention mechanisms** to detect noisy neighbor patterns that traditional schedulers miss, combined with a **high-performance Go middleware** that ensures zero impact on scheduling throughput.

---

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         KubeAttention                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         gRPC/UDS          ┌────────────────┐  │
│  │    Gopher    │◄────────────────────────►│     Brain       │  │
│  │  Score Plugin│        (BatchScore)       │   Transformer   │  │
│  └──────┬───────┘                           └───────┬────────┘  │
│         │                                           │           │
│  ┌──────▼──────┐                                    │ PyTorch   │
│  │ Telemetry   │                                    │           │
│  │   Store     │                                    ▼           │
│  └──────┬──────┘                            ┌────────────────┐  │
│         │ eBPF/K8s Metrics                  │   Tetragon     │  │
│         └──────────────────────────────────►│  eBPF Metrics  │  │
│                                             └────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Brain: Transformer Architecture

### Input: ClusterTensor

The Brain receives cluster state encoded as a **ClusterTensor**:

```
ClusterTensor {
    node_features: Tensor[N, T, F]    # N nodes × T timesteps × F features
    pod_context:   Tensor[P]          # Pod requirements vector
    attention_mask: Tensor[N]         # Valid node mask
}
```

Where:
- **N** = Number of candidate nodes (typically 3-100)
- **T** = Temporal window (default: 10 timesteps)
- **F** = 11 eBPF features from Tetragon
- **P** = 7 pod context features

### Feature Set (F=11)

| # | Feature | Source | Why It Matters |
|---|---------|--------|----------------|
| 1 | cpu_utilization | sched:sched_stat_runtime | Basic load |
| 2 | cpu_throttle_rate | cgroup:cgroup_throttle | CPU contention |
| 3 | memory_utilization | kprobe:__alloc_pages | Memory pressure |
| 4 | memory_bandwidth_gbps | perf:mem_load_retired | **Noisy neighbor signal** |
| 5 | l3_cache_miss_rate | perf:cache_misses | **Critical: LLC contention** |
| 6 | l3_cache_occupancy_mb | perf:llc_occupancy | Cache pressure |
| 7 | disk_io_wait_ms | block:block_rq_complete | I/O bottleneck |
| 8 | disk_iops | block:block_rq_issue | I/O load |
| 9 | network_rx_packets_sec | net:netif_receive_skb | Network load |
| 10 | network_tx_packets_sec | net:net_dev_xmit | Network load |
| 11 | network_drop_rate | skb:kfree_skb | Network saturation |

---

## Attention Mechanisms

The Brain uses **two types of attention** to understand cluster state:

### 1. Node Self-Attention

```
                    ┌─────────────────────────────────┐
                    │     Node Self-Attention         │
                    │                                 │
   Node 1 ──────────┤  Q₁ ─────────┐                 │
   Node 2 ──────────┤  Q₂ ─────────┼──► Attention ───┼──► Updated Nodes
   Node 3 ──────────┤  Q₃ ─────────┘    Weights      │
                    │                                 │
                    │  "Which nodes affect each      │
                    │   other's performance?"        │
                    └─────────────────────────────────┘
```

**Purpose**: Detect inter-node relationships.

**How it works**:
1. Each node generates Query (Q), Key (K), Value (V) vectors
2. Attention scores: `softmax(Q · K^T / √d)`
3. Nodes "attend" to other nodes with correlated telemetry

**What it learns**:
- Nodes on the same rack often have correlated network patterns
- Nodes with high L3 cache miss rates often share noisy neighbors
- Memory bandwidth saturation propagates across NUMA nodes

**Code**: [`brain/model.py:NodeAttentionBlock`](file:///Users/pradeepsingh/code/tools/KubeAttention/brain/model.py#L35-L90)

```python
class NodeAttentionBlock(nn.Module):
    def forward(self, x, mask):
        # x: (B, N, D) - batch × nodes × features
        
        q = self.q_proj(x)  # What am I looking for?
        k = self.k_proj(x)  # What do I have?
        v = self.v_proj(x)  # What information to share?
        
        # Scaled dot-product attention
        attn = softmax(q @ k.T / sqrt(d))  # (N, N) attention matrix
        
        # Each node aggregates info from related nodes
        out = attn @ v
        return out
```

---

### 2. Pod Cross-Attention (The Key Innovation)

```
                    ┌─────────────────────────────────┐
                    │     Pod Cross-Attention         │
                    │                                 │
   Node 1 ──────────┤  Q₁ ─────┐                     │
   Node 2 ──────────┤  Q₂ ─────┼──► Attention ───────┼──► Pod-Aware Scores
   Node 3 ──────────┤  Q₃ ─────┘    to Pod          │
                    │              ▲                  │
                    │              │                  │
   Pod Context ─────┤──────────────┘                 │
   (cpu, mem,       │  K, V                          │
    workload_type)  │  "How well does this node     │
                    │   match the pod's needs?"      │
                    └─────────────────────────────────┘
```

**Purpose**: Context-aware scoring based on pod requirements.

**How it works**:
1. Pod requirements become Key (K) and Value (V)
2. Each node generates Query (Q) asking "Am I suitable?"
3. Cross-attention computes relevance: `softmax(Q_node · K_pod^T)`

**What it learns**:
- CPU-bound pods should avoid nodes with high throttle rates
- Memory-bound pods need nodes with available bandwidth
- I/O-bound pods need low disk wait times

**Code**: [`brain/model.py:PodCrossAttention`](file:///Users/pradeepsingh/code/tools/KubeAttention/brain/model.py#L93-L140)

```python
class PodCrossAttention(nn.Module):
    def forward(self, nodes, pod_ctx):
        # nodes: (B, N, D)
        # pod_ctx: (B, P) - pod requirements
        
        q = self.q_proj(nodes)      # Node asks: "Am I suitable?"
        k = self.k_proj(pod_ctx)    # Pod says: "I need these resources"
        v = self.v_proj(pod_ctx)    # Pod provides: context for scoring
        
        # Each node attends to pod requirements
        attn = softmax(q @ k.T / sqrt(d))
        
        # Nodes update their representations with pod context
        out = attn @ v
        return nodes + out  # Residual connection
```

---

## Full Forward Pass

```
Input: ClusterTensor
         │
         ▼
┌─────────────────────────────────┐
│  1. Feature Projection          │
│     (F=11) → (D=128)            │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. Temporal Pooling            │
│     Mean over T timesteps       │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. Node Self-Attention ×3      │  ← "Which nodes are related?"
│     Multi-head (4 heads)        │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  4. Pod Cross-Attention         │  ← "Which node fits the pod?"
│     Nodes attend to pod reqs    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  5. Score Head                  │
│     Linear → Sigmoid × 100      │
└─────────────────────────────────┘
         │
         ▼
Output: scores[N], confidence[N]
        (0-100 per node)
```

## High-Performance Gopher Plugin

To meet the strict latency requirements of the Kubernetes scheduler, the Go plugin (`Gopher`) implements several critical optimizations:

### 1. TelemetryStore (Background Polling)
The `TelemetryStore` runs as a singleton background process. It periodically fetches metrics from Tetragon and Prometheus, keeping a hot cache of node states. This ensures that the `Score` function **never** makes a synchronous network call to fetch metrics.

### 2. PreScore Batching
Instead of calling the Brain gRPC endpoint for every node sequentially (which would scale O(N) where N is the number of nodes), KubeAttention uses the `PreScore` phase to send **one batch request** for all candidate nodes. This reduces the total scheduling overhead to O(1) gRPC roundtrips.

### 3. Circuit Breaker & Safety
The `BrainClient` monitors latency and error rates. If the Brain takes >45ms or returns errors, the plugin automatically trips the circuit breaker and falls back to neutral scores, ensuring cluster stability even if the AI components fail.

### Traditional Schedulers

**Problem**: Score each node independently.

```
score(node) = f(node_metrics)  # No context about other nodes or pod
```

### KubeAttention

**Solution**: Score nodes in context of:
1. **Other nodes** (self-attention) - detect cluster-wide patterns
2. **Pod requirements** (cross-attention) - match workload needs

```
score(node) = Attention(node, all_nodes, pod_requirements)
```

### Example: Detecting L3 Cache Contention

1. **Input**: Node A has high `l3_cache_miss_rate` (0.8)
2. **Self-Attention**: Node A attends to Node B (same rack, similar pattern)
3. **Cross-Attention**: Incoming pod is "memory-bound" workload type
4. **Output**: Node A gets low score (15/100) - avoid cache contention
5. **Result**: Pod scheduled to Node C (clean node, score 85/100)

---

## Training the Model

### Data Collection

```python
# Collect (cluster_state, pod, outcome) tuples
training_data = []
for scheduling_event in shadow_mode_events:
    cluster_tensor = collect_telemetry(event.timestamp)
    pod_context = encode_pod(event.pod)
    
    # Label: Did the actual placement result in good latency?
    label = measure_pod_performance(event.pod, event.node)
    
    training_data.append((cluster_tensor, pod_context, label))
```

### Loss Function

```python
# Binary cross-entropy on "good placement" prediction
loss = BCE(predicted_score, actual_performance_label)
```

### Training Loop

```python
for epoch in range(100):
    for cluster, pod, label in dataloader:
        scores, confidence = model(cluster, pod)
        loss = BCE(scores, label) + confidence_regularization
        loss.backward()
        optimizer.step()
```

---

## Performance Constraints

| Constraint | Target | Implementation |
|------------|--------|----------------|
| Inference Latency | <50ms | Circuit breaker fallback |
| Memory Footprint | <2GB | Int8 quantization |
| Model Size | ~500K params | 3 attention layers, D=128 |

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Tetragon eBPF](https://tetragon.io/) - Runtime security observability
