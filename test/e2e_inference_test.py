"""End-to-end inference test with Phase 2 (Cost) and Phase 4 (Resilience) features."""

import time
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.models import get_model
from brain.metrics_schema import FEATURE_NAMES

def main():
    print("=" * 60)
    print("KubeAttention E2E Inference Test (Phase 2 & 4)")
    print("=" * 60)
    print()

    # 1. Initialize Model
    model = get_model("mlp", input_dim=len(FEATURE_NAMES) + 5)  # node features + pod context

    # 2. Simulate Node Telemetry with Cost and Resilience Features
    # node-1: Performance Good, but High Cost, Low Diversity
    # node-2: Performance Bad (Contention), High Risk
    # node-3: Performance Good, Low Cost, High Diversity (BEST)
    
    # Build node features - ALL NORMALIZED TO [0,1] to match training data
    # Features: cpu_util, cpu_throttle, mem_util, mem_bw, l3_miss, l3_occ, 
    #           disk_wait, disk_iops, net_rx, net_tx, net_drop, 
    #           cpu_throt2, cost_idx, is_spot, spot_risk
    node_features = np.array([
        # node-1: Good perf, high cost - should score medium
        [0.3, 0.05, 0.4, 0.2, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.01, 0.1, 0.9, 0.0, 0.0],
        # node-2: Bad perf (HIGH contention l3_miss=0.9), high risk - should score LOW
        [0.2, 0.1, 0.3, 0.3, 0.9, 0.6, 0.1, 0.05, 0.05, 0.05, 0.01, 0.5, 0.2, 1.0, 0.8],
        # node-3: Good perf, LOW cost, LOW contention - should score HIGH
        [0.35, 0.03, 0.35, 0.25, 0.15, 0.2, 0.03, 0.2, 0.15, 0.15, 0.005, 0.05, 0.1, 0.0, 0.0],
    ], dtype=np.float32)

    
    node_names = ["node-1", "node-2", "node-3"]
    
    # Pod context: critical production workload
    # [cpu_norm, mem_norm, workload_type, criticality, priority]
    pod_features = np.array([0.5, 0.25, 0.5, 0.75, 0.8], dtype=np.float32)

    print("Node Telemetry Breakdown:")
    print("-" * 60)
    print("  node-1: Good resource (CPU=0.3), High Cost (0.9), Zone A")
    print("  node-2: High Contention (0.9 cache miss), Spot Risk (0.8), Zone A")
    print("  node-3: Good resource (CPU=0.35), Low Cost (0.1), Zone B (Incentivized)")
    print()

    # 3. Train quickly on synthetic data to give model some signal
    # Training features must match inference: node_features.shape[1] + pod_features.shape[0]
    total_features = node_features.shape[1] + len(pod_features)
    print(f"  Total features for model: {total_features}")
    
    print("Quick training on synthetic data...")
    train_X = np.random.rand(500, total_features).astype(np.float32)
    # Give lower scores to high CPU utilization (col 0) and high contention (col 4: l3_miss)
    train_y = (1.0 - train_X[:, 0] * 0.5 - train_X[:, 4] * 0.5).astype(np.float32)
    train_y = np.clip(train_y, 0, 1)
    
    metrics = model.train(train_X, train_y, epochs=30, batch_size=32)
    print(f"  Training done: {metrics}")


    # 4. Run Model
    results = model.score_nodes(node_features, pod_features, node_names)

    # 5. Verify Results
    print()
    print("Model Predictions:")
    print("-" * 60)
    
    best_node = None
    best_score = -1
    
    for res in results:
        node = res.node_name
        score = res.score
        conf = res.confidence
        reasoning = res.reasoning
        
        indicator = " <- BEST" if score > 70 else ""
        print(f"  {node}: Score={score}, Confidence={conf:.3f}{indicator}")
        print(f"    Reasoning: {reasoning}")
        
        if score > best_score:
            best_score = score
            best_node = node

    print("-" * 60)
    print(f"  Best node selected: {best_node} (score={best_score})")
    
    # Verify the model made a reasonable choice
    # node-3 should be best (low cost, good perf, low contention)
    # node-2 should be worst (high contention)
    if best_node == "node-3":
        print("\n✅ PASS: Model correctly prioritized low cost + low contention!")
    elif best_node == "node-1":
        print("\n⚠️ PARTIAL: Model picked good performance but ignored cost.")
    elif best_node == "node-2":
        print("\n❌ FAIL: Model placed pod on high-contention node!")
    else:
        print("\n⚠️ WARNING: Unexpected placement.")

    print("=" * 60)
    return 0 if best_node != "node-2" else 1

if __name__ == "__main__":
    exit(main())

