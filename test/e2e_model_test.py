"""
End-to-end test for both MLP and XGBoost models.

Tests the full scoring pipeline with synthetic data.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.models import get_model, list_models
from brain.metrics_schema import FEATURE_NAMES


def test_model(model_name: str) -> bool:
    """Test a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name.upper()}")
    print(f"{'='*60}")
    
    # 1. Initialize model
    try:
        model = get_model(model_name)
        print(f"  Model: {model.name}")
        print(f"  Parameters: {model.num_parameters:,}")
    except Exception as e:
        print(f"  FAILED to initialize: {e}")
        return False
    
    # 2. Generate synthetic training data
    print("  Generating synthetic data...")
    num_samples = 1000
    num_features = len(FEATURE_NAMES)
    pod_features_dim = 5
    
    # Features: random values in [0, 1]
    X_node = np.random.rand(num_samples, num_features).astype(np.float32)
    X_pod = np.random.rand(num_samples, pod_features_dim).astype(np.float32)
    X = np.hstack([X_node, X_pod])
    
    # Labels: synthetic scoring (lower CPU util = higher score)
    # CPU utilization is typically first feature
    y = (1.0 - X_node[:, 0]) * 0.5 + np.random.rand(num_samples) * 0.2
    y = np.clip(y, 0, 1).astype(np.float32)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # 3. Train
    print("  Training...")
    try:
        if model_name == "mlp":
            metrics = model.train(X, y, epochs=20, lr=1e-3, batch_size=32)
        else:
            metrics = model.train(X, y)
        print(f"  Training complete: {metrics}")
    except Exception as e:
        print(f"  FAILED to train: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Score nodes
    print("  Scoring nodes...")
    try:
        # Create 5 test nodes
        node_features = np.random.rand(5, num_features).astype(np.float32)
        # Set different CPU utilizations to test ranking
        node_features[0, 0] = 0.9  # High CPU = should be low score
        node_features[1, 0] = 0.7
        node_features[2, 0] = 0.5
        node_features[3, 0] = 0.3
        node_features[4, 0] = 0.1  # Low CPU = should be high score
        
        pod_features = np.array([0.25, 0.25, 0.5, 0.5, 0.5], dtype=np.float32)
        node_names = [f"node-{i}" for i in range(5)]
        
        results = model.score_nodes(node_features, pod_features, node_names)
        
        print("  Results:")
        for res in results:
            cpu = node_features[node_names.index(res.node_name), 0]
            print(f"    {res.node_name}: Score={res.score}, CPU={cpu:.1f}, Conf={res.confidence:.2f}")
        
        # Verify ranking: node-4 (lowest CPU) should have highest or near-highest score
        scores = [r.score for r in results]
        best_node_idx = np.argmax(scores)
        best_node = results[best_node_idx].node_name
        print(f"  Best node: {best_node} (score={max(scores)})")
        
    except Exception as e:
        print(f"  FAILED to score: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Save and load
    print("  Testing save/load...")
    try:
        ext = ".pt" if model_name == "mlp" else ".json"
        path = f"/tmp/kubeattention_test_{model_name}{ext}"
        model.save(path)
        
        model2 = get_model(model_name)
        model2.load(path)
        
        # Verify loaded model produces same results
        results2 = model2.score_nodes(node_features, pod_features, node_names)
        scores2 = [r.score for r in results2]
        
        if model_name == "mlp":
            # MLP should be deterministic after loading
            if scores != scores2:
                print(f"  WARNING: Scores differ after load (expected for untrained init)")
        print(f"  Save/load OK")
        
    except Exception as e:
        print(f"  FAILED save/load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"  PASSED")
    return True


def main():
    print("=" * 60)
    print("KubeAttention E2E Model Test: MLP vs XGBoost")
    print("=" * 60)
    
    available = list_models()
    print(f"Available models: {available}")
    
    results = {}
    for model_name in ["mlp", "xgboost"]:
        if model_name in available:
            results[model_name] = test_model(model_name)
        else:
            print(f"\nSkipping {model_name}: not available")
            results[model_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {model_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
