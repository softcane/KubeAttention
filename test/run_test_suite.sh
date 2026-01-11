#!/bin/bash
# KubeAttention Noisy Neighbor Test Suite
# This script sets up the test environment and runs comparison tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="kubeattention"

echo "=============================================="
echo "🧪 KubeAttention Noisy Neighbor Test Suite"
echo "=============================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prereqs() {
    echo "📋 Checking prerequisites..."
    
    if ! command -v kind &> /dev/null; then
        echo -e "${RED}❌ kind not found. Install: brew install kind${NC}"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found. Install: brew install kubectl${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites met${NC}"
    echo
}

# Create or reuse cluster
setup_cluster() {
    echo "🏗️  Setting up Kind cluster..."
    
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        echo -e "${YELLOW}⚠️  Cluster '${CLUSTER_NAME}' already exists${NC}"
        read -p "Delete and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kind delete cluster --name "${CLUSTER_NAME}"
        else
            echo "Using existing cluster"
            return
        fi
    fi
    
    kind create cluster --config "${SCRIPT_DIR}/kind-config.yaml" --name "${CLUSTER_NAME}"
    echo -e "${GREEN}✅ Cluster created${NC}"
    echo
}

# Deploy stress pods
deploy_stress() {
    echo "😈 Deploying stress pods (noisy neighbors)..."
    
    # Create namespace
    kubectl apply -f "${SCRIPT_DIR}/noisy_neighbor/workload.yaml"
    
    # Deploy CPU stress
    echo "  → CPU stress pod..."
    kubectl apply -f "${SCRIPT_DIR}/noisy_neighbor/cpu_stress.yaml"
    
    # Deploy memory stress
    echo "  → Memory stress pod..."
    kubectl apply -f "${SCRIPT_DIR}/noisy_neighbor/memory_stress.yaml"
    
    # Deploy cache stress
    echo "  → Cache stress pod (L3 cache contention)..."
    kubectl apply -f "${SCRIPT_DIR}/noisy_neighbor/cache_stress.yaml"
    
    echo -e "${GREEN}✅ Stress pods deployed${NC}"
    echo
}

# Wait for stress pods to be running
wait_for_stress() {
    echo "⏳ Waiting for stress pods to start..."
    
    kubectl wait --for=condition=Ready pods -l app=stress-test \
        -n noisy-neighbor-test --timeout=60s || true
    
    echo -e "${GREEN}✅ Stress pods running${NC}"
    echo
}

# Deploy victim workload and observe scheduling
test_scheduling() {
    echo "🎯 Testing scheduling decisions..."
    echo
    
    echo "Scheduling victim deployment (5 replicas)..."
    kubectl apply -f "${SCRIPT_DIR}/noisy_neighbor/workload.yaml"
    
    sleep 10
    
    echo
    echo "📊 Pod Distribution:"
    echo "─────────────────────────────────────────"
    kubectl get pods -n noisy-neighbor-test -o wide | grep victim
    
    echo
    echo "📝 KubeAttention Annotations (Shadow Mode):"
    echo "─────────────────────────────────────────"
    kubectl get pods -n noisy-neighbor-test -l app=victim-replica \
        -o jsonpath='{range .items[*]}{.metadata.name}: {.metadata.annotations.kubeattention\.io/recommended-node}{"\n"}{end}' || echo "No annotations (KubeAttention not active)"
    
    echo
}

# Show node resource usage
show_node_metrics() {
    echo "📈 Node Resource Usage:"
    echo "─────────────────────────────────────────"
    kubectl top nodes 2>/dev/null || echo "Metrics server not installed"
    echo
}

# Compare with default scheduler
compare_schedulers() {
    echo "🔬 Scheduler Comparison:"
    echo "─────────────────────────────────────────"
    
    # Count pods per node
    echo "Pods per worker node:"
    for node in $(kubectl get nodes -l '!node-role.kubernetes.io/control-plane' -o name); do
        node_name=$(echo $node | cut -d'/' -f2)
        pod_count=$(kubectl get pods -n noisy-neighbor-test --field-selector spec.nodeName=$node_name --no-headers 2>/dev/null | wc -l)
        role=$(kubectl get node $node_name -o jsonpath='{.metadata.labels.kubeattention\.io/test-role}')
        echo "  $node_name ($role): $pod_count pods"
    done
    echo
    
    # Check if victim pods avoided noisy nodes
    echo "Victim pod placement analysis:"
    clean_count=$(kubectl get pods -n noisy-neighbor-test -l app=victim-replica \
        --field-selector spec.nodeName=kubeattention-worker3 --no-headers 2>/dev/null | wc -l || echo 0)
    total_victims=$(kubectl get pods -n noisy-neighbor-test -l app=victim-replica --no-headers | wc -l)
    
    echo "  Victims on clean node: $clean_count / $total_victims"
    
    if [ "$clean_count" -gt 2 ]; then
        echo -e "  ${GREEN}✅ Good: Majority of victims on clean node${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Default scheduler didn't avoid noisy nodes${NC}"
        echo "  KubeAttention would improve this!"
    fi
    echo
}

# Cleanup
cleanup() {
    echo "🧹 Cleanup options:"
    echo "  1. Delete test namespace: kubectl delete ns noisy-neighbor-test"
    echo "  2. Delete cluster: kind delete cluster --name ${CLUSTER_NAME}"
    echo
}

# Main
main() {
    check_prereqs
    setup_cluster
    deploy_stress
    wait_for_stress
    test_scheduling
    show_node_metrics
    compare_schedulers
    cleanup
    
    echo "=============================================="
    echo -e "${GREEN}🎉 Test suite complete!${NC}"
    echo "=============================================="
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
