# KubeAttention

**Kubernetes scheduling with measured node pressure and validated ML scoring**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go)](https://go.dev)
[![Python Version](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python)](https://python.org)

KubeAttention is an out-of-tree Kubernetes scheduler. It scores all feasible nodes in one Brain RPC, fails safely when telemetry or inference is unavailable, and records measured scheduling outcomes for later training.

The supported runtime telemetry source is Kubernetes Metrics Server. CPU and memory utilization participate in active scoring. Hardware-counter fields remain in the versioned feature schema, but the scheduler marks them unavailable unless a supported source supplies them. Missing values never masquerade as idle nodes.

---

## How it works

1. The Go scheduler continuously caches node metrics.
2. `PreScore` sends every feasible node and the complete pod context to the Python Brain.
3. The Brain validates schema, timestamps, required measurements, and checkpoint compatibility before inference.
4. Active mode contributes the returned scores to Kubernetes placement. Shadow mode returns neutral scores and writes the Brain recommendation to pod annotations after binding.
5. The Collector persists candidate nodes, measured telemetry, the chosen node, and the observed pod outcome as JSONL.

```text
Kubernetes Metrics API
          |
          v
  TelemetryStore -----> KubeAttention scheduler ----gRPC----> Brain
                              |                                 |
                              | Score / bind                    | validated checkpoint
                              v                                 |
                           Pod event ----------------------------+
                              |
                              v
                      persistent Collector JSONL
```

The model input contract contains 15 node features and 11 pod features. Every checkpoint records the feature-schema version and input names. MLP and XGBoost return candidate-independent scores on the same absolute 0–100 scale.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the feature and RPC contracts.

---

## Getting started

### Prerequisites

- Kubernetes 1.35
- Go 1.25
- Python 3.12 or Docker
- Helm 3
- Kubernetes Metrics Server

### Build

```bash
docker build -t kubeattention/scheduler:latest -f deploy/scheduler.Dockerfile .
docker build -t kubeattention/brain:latest -f deploy/brain.Dockerfile .
docker build -t kubeattention/collector:latest -f deploy/collector.Dockerfile .
```

### Run the Kind acceptance path

```bash
bash scripts/e2e-kind-full.sh
```

Set `KEEP_CLUSTER=1` to retain the cluster for inspection.

### Deploy the Helm chart

```bash
helm upgrade --install kubeattention helm/kubeattention \
  --set scheduler.mode=shadow \
  --set training.enabled=false
```

Pods opt in with:

```yaml
spec:
  schedulerName: kubeattention-scheduler
```

After validating shadow annotations and the measured comparison, switch to active scoring:

```bash
helm upgrade kubeattention helm/kubeattention \
  --reuse-values \
  --set scheduler.mode=active
```

The scheduler deployment checksum restarts the process when its profile changes.

### Train and promote a model

Use Collector JSONL from a persistent volume. Training rejects synthetic, incomplete, or unmeasured records when the measured-data gate is enabled.

```bash
PYTHONPATH=. python brain/training/train.py \
  --train-data /path/to/train-events.jsonl \
  --val-data /path/to/held-out-events.jsonl \
  --model mlp
```

Promotion requires the candidate to improve held-out mean squared error over the non-ML CPU/memory/cache-pressure baseline. Copy only a promoted, schema-compatible artifact to `/models/best_model.pt`, then restart the Brain deployment so it loads that artifact.

### Run a measured comparison

With the stack running in active mode:

```bash
python benchmark/runner.py \
  --requests-per-pod 10000 \
  --warmup-seconds 30 \
  --output benchmark-results.json
```

The runner deploys the same Redis workload under controlled pressure with `default-scheduler` and `kubeattention-scheduler`. It records every pod's node, profile, request count, and Redis P50/P95/P99/maximum latency. Add `--require-improvement` when a non-positive worst-pod P99 result must fail CI.

---

## Reliability and safety

- **Shadow by default:** recommendations are annotations; returned scheduler scores remain neutral.
- **Bounded fallback:** unavailable Brain RPCs and incomplete telemetry return a score of 50 without panicking.
- **Freshness guard:** node samples older than 30 seconds cannot drive active scoring.
- **Strict readiness:** Brain health is false until a compatible checkpoint loads.
- **Measured metrics:** the Prometheus endpoint reports observed request count, latency, readiness, and returned node scores.
- **No automatic migration:** the rebalancer is disabled by default and only writes recommendations when explicitly enabled.

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `scheduler.mode` | `shadow` | Neutral annotation mode or active scoring |
| `scheduler.timeoutMs` | `50` | Scheduler-to-Brain RPC deadline |
| `brain.modelType` | `mlp` | MLP or XGBoost checkpoint format |
| `MAX_STALENESS_MS` | `30000` | Maximum Brain telemetry age |
| `rebalancer.enabled` | `false` | Enables annotation-only recommendations |

---

## Project Structure

```
KubeAttention/
├── brain/                  # Python ML components
│   ├── models/             # MLP and XGBoost scorers
│   ├── training/           # Training pipeline
│   ├── server.py           # gRPC Brain server
│   └── config.py           # Centralized configuration
├── pkg/                    # Go scheduler plugin
│   ├── scheduler/          # Kubernetes scheduling framework plugin
│   └── collector/          # Telemetry collection
├── proto/                  # Protocol buffer definitions
├── deploy/                 # Kubernetes manifests
└── test/                   # E2E and unit tests
```

---

## License

Apache 2.0 - see [LICENSE](LICENSE)
