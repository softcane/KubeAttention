#!/usr/bin/env bash

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-kubeattention-e2e}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to a promoted, schema-compatible MLP checkpoint}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE="kubeattention"
NAMESPACE="kubeattention"

cleanup() {
    if [[ "$KEEP_CLUSTER" != "1" ]]; then
        kind delete cluster --name "$CLUSTER_NAME"
    fi
}
trap cleanup EXIT

for binary in docker kind kubectl helm jq; do
    command -v "$binary" >/dev/null 2>&1 || {
        echo "required command not found: $binary" >&2
        exit 1
    }
done
[[ -f "$MODEL_CHECKPOINT" ]] || {
    echo "model checkpoint not found: $MODEL_CHECKPOINT" >&2
    exit 1
}

kind create cluster --name "$CLUSTER_NAME" --config "$PROJECT_ROOT/benchmark/kind-config.yaml"
kubectl wait --for=condition=Ready nodes --all --timeout=180s

docker build -t kubeattention/scheduler:acceptance -f "$PROJECT_ROOT/deploy/scheduler.Dockerfile" "$PROJECT_ROOT"
docker build -t kubeattention/brain:acceptance -f "$PROJECT_ROOT/deploy/brain.Dockerfile" "$PROJECT_ROOT"
docker build -t kubeattention/collector:acceptance -f "$PROJECT_ROOT/deploy/collector.Dockerfile" "$PROJECT_ROOT"
docker run --rm \
    -v "$MODEL_CHECKPOINT:/model/best_model.pt:ro" \
    kubeattention/brain:acceptance \
    python -c "from brain.models.mlp_scorer import MLPScorer; model=MLPScorer(); model.load('/model/best_model.pt'); assert model.ready"
kind load docker-image kubeattention/scheduler:acceptance --name "$CLUSTER_NAME"
kind load docker-image kubeattention/brain:acceptance --name "$CLUSTER_NAME"
kind load docker-image kubeattention/collector:acceptance --name "$CLUSTER_NAME"

helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/ --force-update
helm repo update metrics-server
helm upgrade --install metrics-server metrics-server/metrics-server \
    --namespace kube-system \
    --set 'args[0]=--kubelet-insecure-tls' \
    --set 'args[1]=--kubelet-preferred-address-types=InternalIP'
kubectl rollout status deployment/metrics-server -n kube-system --timeout=180s

helm upgrade --install "$RELEASE" "$PROJECT_ROOT/helm/kubeattention" \
    --set brain.enabled=false \
    --set collector.enabled=false \
    --set scheduler.enabled=false \
    --set training.enabled=false \
    --set grafana.enabled=false

cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: kubeattention-model-loader
  namespace: kubeattention
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
  containers:
    - name: loader
      image: busybox:1.36
      command: ["sh", "-c", "sleep 600"]
      volumeMounts:
        - name: models
          mountPath: /models
  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: kubeattention-models
EOF
kubectl wait --for=condition=Ready pod/kubeattention-model-loader -n "$NAMESPACE" --timeout=180s
kubectl cp "$MODEL_CHECKPOINT" "$NAMESPACE/kubeattention-model-loader:/models/best_model.pt"
kubectl delete pod kubeattention-model-loader -n "$NAMESPACE" --wait=true

helm upgrade "$RELEASE" "$PROJECT_ROOT/helm/kubeattention" \
    --set brain.image.tag=acceptance \
    --set brain.image.pullPolicy=Never \
    --set collector.image.tag=acceptance \
    --set collector.image.pullPolicy=Never \
    --set collector.outcomeWaitMinutes=0.1 \
    --set scheduler.image.tag=acceptance \
    --set scheduler.image.pullPolicy=Never \
    --set scheduler.mode=shadow \
    --set training.enabled=false \
    --set grafana.enabled=false

kubectl rollout status deployment/kubeattention-brain -n "$NAMESPACE" --timeout=300s
kubectl rollout status deployment/kubeattention-collector -n "$NAMESPACE" --timeout=300s
kubectl rollout status deployment/kubeattention-scheduler -n "$NAMESPACE" --timeout=300s

for attempt in $(seq 1 36); do
    if kubectl top nodes >/dev/null 2>&1; then
        break
    fi
    if [[ "$attempt" == "36" ]]; then
        echo "metrics.k8s.io did not become available" >&2
        exit 1
    fi
    sleep 5
done

kubectl create namespace kubeattention-acceptance
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: default-proof
  namespace: kubeattention-acceptance
spec:
  containers:
    - name: sleeper
      image: busybox:1.36
      command: ["sh", "-c", "sleep 600"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
---
apiVersion: v1
kind: Pod
metadata:
  name: kubeattention-proof
  namespace: kubeattention-acceptance
spec:
  schedulerName: kubeattention-scheduler
  containers:
    - name: sleeper
      image: busybox:1.36
      command: ["sh", "-c", "sleep 600"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
kubectl wait --for=condition=Ready pod/default-proof pod/kubeattention-proof \
    -n kubeattention-acceptance --timeout=180s

[[ "$(kubectl get pod default-proof -n kubeattention-acceptance -o jsonpath='{.spec.schedulerName}')" == "default-scheduler" ]]
[[ "$(kubectl get pod kubeattention-proof -n kubeattention-acceptance -o jsonpath='{.spec.schedulerName}')" == "kubeattention-scheduler" ]]
[[ "$(kubectl get pod kubeattention-proof -n kubeattention-acceptance -o jsonpath='{.metadata.annotations.kubeattention\.io/mode}')" == "shadow" ]]

sleep 10
kubectl exec -n "$NAMESPACE" deployment/kubeattention-collector -- cat /data/events.jsonl \
    | jq -e 'select(.pod_name == "kubeattention-proof" and .scheduler_name == "kubeattention-scheduler" and .outcome != "pending")' >/dev/null
kubectl exec -n "$NAMESPACE" deployment/kubeattention-brain -- \
    python -c "import urllib.request; body=urllib.request.urlopen('http://127.0.0.1:8080/metrics').read().decode(); assert 'kubeattention_brain_scoring_requests_total' in body"

if kubectl get pods -A -o json | jq -e '.items[] | select(.metadata.annotations["kubeattention.io/rebalance-target"] != null)' >/dev/null; then
    echo "rebalancer annotation found while rebalancer is disabled" >&2
    exit 1
fi

echo "PASS: scheduler, Brain RPC, measured CPU/memory telemetry, Collector persistence, probes, and shadow annotations were exercised"
echo "NOTE: this control-path acceptance does not claim a latency improvement; use a measured A/B workload before active rollout"
