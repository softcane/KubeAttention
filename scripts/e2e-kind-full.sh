#!/bin/bash
# KubeAttention Full E2E Test Script
# 
# This script deploys the complete KubeAttention stack in a Kind cluster:
# 1. Creates Kind cluster with 6 worker nodes
# 2. Installs Tetragon for eBPF telemetry
# 3. Deploys the Brain server
# 4. Deploys the Collector
# 5. Runs stress workloads to generate training data
# 6. Tests Rebalancer annotations

set -e

CLUSTER_NAME="kubeattention-e2e"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "KubeAttention Full E2E Test"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    log_step "Cleaning up..."
    kind delete cluster --name $CLUSTER_NAME 2>/dev/null || true
}

# Check prerequisites
check_prereqs() {
    log_step "Checking prerequisites..."
    
    command -v kind >/dev/null 2>&1 || { log_error "kind not found. Install: brew install kind"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl not found"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm not found. Install: brew install helm"; exit 1; }
    
    echo "  kind: $(kind version)"
    echo "  kubectl: $(kubectl version --client -o json | jq -r '.clientVersion.gitVersion')"
    echo "  helm: $(helm version --short)"
}

# Create Kind cluster
create_cluster() {
    log_step "Creating Kind cluster: $CLUSTER_NAME"
    
    # Check if cluster exists
    if kind get clusters | grep -q $CLUSTER_NAME; then
        log_warn "Cluster $CLUSTER_NAME already exists"
        kubectl config use-context kind-$CLUSTER_NAME
        return
    fi
    
    kind create cluster --name $CLUSTER_NAME --config $PROJECT_ROOT/benchmark/kind-config.yaml
    kubectl config use-context kind-$CLUSTER_NAME
    
    log_step "Waiting for nodes to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=120s
    
    echo ""
    kubectl get nodes
}

# Build and load local images
build_images() {
    log_step "Building local Docker images..."
    
    # Build Brain
    log_step "Building Brain image..."
    docker build -t kubeattention/brain:latest -f $PROJECT_ROOT/deploy/brain.Dockerfile $PROJECT_ROOT
    
    # Build Collector
    log_step "Building Collector image..."
    docker build -t kubeattention/collector:latest -f $PROJECT_ROOT/deploy/collector.Dockerfile $PROJECT_ROOT
    
    log_step "Loading images into Kind cluster..."
    kind load docker-image kubeattention/brain:latest --name $CLUSTER_NAME
    kind load docker-image kubeattention/collector:latest --name $CLUSTER_NAME
}

# Install Tetragon
install_tetragon() {
    log_step "Installing Tetragon for eBPF telemetry..."
    
    # Add Tetragon Helm repo
    helm repo add cilium https://helm.cilium.io 2>/dev/null || true
    helm repo update
    
    # Check if already installed
    if helm list -n kube-system | grep -q tetragon; then
        log_warn "Tetragon already installed"
        return
    fi
    
    # Install Tetragon
    helm install tetragon cilium/tetragon \
        --namespace kube-system \
        --set tetragon.exportFilename=/var/log/tetragon/tetragon.log \
        --set tetragon.enableProcessCred=true \
        --set tetragon.enableProcessNs=true
    
    log_step "Waiting for Tetragon pods to be ready..."
    kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=tetragon -n kube-system --timeout=120s
    
    echo ""
    kubectl get pods -n kube-system -l app.kubernetes.io/name=tetragon
}

# Deploy Brain server
deploy_brain() {
    log_step "Deploying Brain server..."
    
    kubectl apply -f $PROJECT_ROOT/deploy/brain-deployment.yaml
    
    # Restart to pick up baked-in model if image was just loaded
    kubectl rollout restart deployment/kubeattention-brain -n kubeattention-system
    
    log_step "Waiting for Brain to be ready..."
    kubectl rollout status deployment/kubeattention-brain -n kubeattention-system
    
    echo ""
    kubectl get pods -n kubeattention-system -l app=kubeattention-brain
}

# Deploy Collector
deploy_collector() {
    log_step "Deploying event Collector..."
    
    kubectl apply -f $PROJECT_ROOT/deploy/collector-deployment.yaml
    
    log_step "Waiting for Collector to be ready..."
    sleep 10  # Collector doesn't have readiness probe
    
    echo ""
    kubectl get pods -n kubeattention -l app=collector
}

# Deploy stress workloads
deploy_stress_workloads() {
    log_step "Deploying stress workloads..."
    
    # Create benchmark namespace
    kubectl create namespace benchmark 2>/dev/null || true
    
    # Deploy noisy neighbors
    kubectl apply -f $PROJECT_ROOT/benchmark/generators/
    
    # Deploy latency-sensitive workloads
    kubectl apply -f $PROJECT_ROOT/benchmark/workloads/
    
    log_step "Waiting for workloads to be scheduled..."
    kubectl wait --for=condition=Available deployment --all -n benchmark --timeout=120s || true
    
    echo ""
    echo "Stress workloads:"
    kubectl get pods -n benchmark -o wide
}

# Collect training data
collect_training_data() {
    log_step "Collecting training data for 5 minutes..."
    
    DURATION=300  # 5 minutes
    echo "Waiting $DURATION seconds for scheduling events..."
    
    # Show progress
    for i in $(seq 1 10); do
        sleep 30
        EVENTS=$(kubectl exec -n kubeattention deployment/collector -- wc -l /data/events.jsonl 2>/dev/null || echo "0")
        echo "  [$i/10] Events collected: $EVENTS"
    done
    
    log_step "Exporting training data..."
    kubectl exec -n kubeattention deployment/collector -- cat /data/events.jsonl > /tmp/kubeattention_events.jsonl 2>/dev/null || true
    
    TOTAL=$(wc -l < /tmp/kubeattention_events.jsonl 2>/dev/null || echo "0")
    echo "Total events exported: $TOTAL"
}

# Test Rebalancer
test_rebalancer() {
    log_step "Testing Rebalancer annotations..."
    
    echo "Waiting for Rebalancer loop (65s)..."
    sleep 65
    
    # Check for rebalance annotations
    ANNOTATED=$(kubectl get pods -A -o json | jq '[.items[] | select(.metadata.annotations["kubeattention.io/rebalance-target"] != null)] | length')
    
    echo "Pods with rebalance annotations: $ANNOTATED"
    
    if [ "$ANNOTATED" -gt 0 ]; then
        echo ""
        echo "Rebalance recommendations:"
        kubectl get pods -A -o json | jq -r '.items[] | select(.metadata.annotations["kubeattention.io/rebalance-target"] != null) | "\(.metadata.namespace)/\(.metadata.name) -> \(.metadata.annotations["kubeattention.io/rebalance-target"])"'
    fi
}

# Summary
print_summary() {
    echo ""
    echo "============================================================"
    echo "E2E Test Complete"
    echo "============================================================"
    echo ""
    
    log_step "Cluster Status:"
    kubectl get nodes
    echo ""
    
    log_step "KubeAttention Components:"
    kubectl get pods -n kubeattention
    echo ""
    
    log_step "Tetragon Status:"
    kubectl get pods -n kube-system -l app.kubernetes.io/name=tetragon
    echo ""
    
    log_step "Benchmark Workloads:"
    kubectl get pods -n benchmark
    echo ""
    
    log_step "Next Steps:"
    echo "  1. View training data: cat /tmp/kubeattention_events.jsonl | jq"
    echo "  2. Train model: PYTHONPATH=. python brain/training/train.py --train-data /tmp/kubeattention_events.jsonl"
    echo "  3. Delete cluster: kind delete cluster --name $CLUSTER_NAME"
}

# Train model before building images
train_model() {
    log_step "Pre-training model for deployment..."
    
    # Generate fresh synthetic data
    PYTHONPATH=. ./.venv/bin/python3 brain/training/dataset.py
    
    # Train
    PYTHONPATH=. ./.venv/bin/python3 brain/training/train.py --train-data training_data.jsonl --model mlp --epochs 50
    
    # Copy to brain package to be baked into image
    mkdir -p $PROJECT_ROOT/brain/models
    cp checkpoints/best_model.pt $PROJECT_ROOT/brain/models/trained_model.pt
    log_step "Model baked into brain/models/trained_model.pt"
}

# Main
main() {
    # trap cleanup EXIT
    
    check_prereqs
    create_cluster
    
    # NEW: Train model before building
    train_model
    
    build_images
    install_tetragon
    deploy_brain
    deploy_collector
    deploy_stress_workloads
    
    log_step "Verifying architecture..."
    BRAIN_POD=$(kubectl get pods -n kubeattention-system -l app=kubeattention-brain -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n kubeattention-system $BRAIN_POD -- ls -l /app/brain/models/trained_model.pt
    
    # validate_inference
    test_rebalancer
    print_summary
    
    # Don't cleanup on success so user can inspect
    trap - EXIT
    
    echo ""
    log_step "SUCCESS! Cluster $CLUSTER_NAME is running with a baked-in ML model."
}

# Run with optional args
case "${1:-}" in
    --cleanup)
        cleanup
        ;;
    --skip-tetragon)
        log_warn "Skipping Tetragon installation"
        install_tetragon() { true; }
        main
        ;;
    *)
        main
        ;;
esac
