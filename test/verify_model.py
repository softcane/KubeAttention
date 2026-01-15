
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.models import get_model
from brain.metrics_schema import NodeMetricsSnapshot
from brain.tensor_encoder import PodContext

def verify():
    print("Verifying model scoring logic...")
    
    # Initialize model and load trained state
    model = get_model("mlp", input_dim=26)
    model_path = "checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model.load(model_path)
    print(f"Successfully loaded model from {model_path}")

    # Case 1: Great Node
    good_snap = NodeMetricsSnapshot(
        node_name="good-node",
        cpu_utilization=0.1,
        memory_utilization=0.1,
        l3_cache_miss_rate=0.01,
        memory_bandwidth_gbps=1.0
    )
    
    # Case 2: Terrible Node (High CPU, High Cache Miss)
    bad_snap = NodeMetricsSnapshot(
        node_name="bad-node",
        cpu_utilization=0.9,
        memory_utilization=0.9,
        l3_cache_miss_rate=0.8,
        memory_bandwidth_gbps=150.0
    )
    
    # Pod Context
    pod = PodContext(pod_name="test", pod_namespace="default", cpu_milli=1000, memory_bytes=1024**3)
    pod_features = np.array(pod.to_feature_vector(), dtype=np.float32)
    
    # Inference
    node_features = np.array([
        good_snap.to_feature_vector(),
        bad_snap.to_feature_vector()
    ], dtype=np.float32)
    
    results = model.score_nodes(node_features, pod_features, ["good-node", "bad-node"])
    
    for res in results:
        print(f"Node: {res.node_name}, Score: {res.score}, Reasoning: {res.reasoning}")
        
    # Check if good node > bad node
    good_score = [r.score for r in results if r.node_name == "good-node"][0]
    bad_score = [r.score for r in results if r.node_name == "bad-node"][0]
    
    if good_score > bad_score:
        print("\n✅ SUCCESS: Model correctly prefers the better node!")
    else:
        print("\n❌ FAILURE: Model still has inverted logic or hasn't learned properly.")

if __name__ == "__main__":
    verify()
