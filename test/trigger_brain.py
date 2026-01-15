
import grpc
import sys
import os
import time

# Add generated stubs to path
sys.path.insert(0, '/app/gen/python')
import scheduler_pb2
import scheduler_pb2_grpc

def trigger():
    channel = grpc.insecure_channel('localhost:50051')
    stub = scheduler_pb2_grpc.BrainStub(channel)
    
    # Simulate telemetry for 3 nodes
    nodes = []
    for i in range(1, 4):
        node_name = f"kubeattention-e2e-worker{'' if i==1 else i if i<3 else '3'}"
        # Node names are: kubeattention-e2e-worker, kubeattention-e2e-worker2, kubeattention-e2e-worker3
        if i == 1: node_name = "kubeattention-e2e-worker"
        elif i == 2: node_name = "kubeattention-e2e-worker2"
        else: node_name = "kubeattention-e2e-worker3"
        
        # Make worker3 look catastrophically bad
        # Make others look perfect
        cpu = 0.99 if i == 3 else 0.01
        l3 = 0.99 if i == 3 else 0.01
        
        node = scheduler_pb2.NodeTelemetry(
            node_name=node_name,
            timestamp_unix_ms=int(time.time() * 1000),
            cpu_utilization=cpu,
            memory_utilization=0.99 if i == 3 else 0.01,
            l3_cache_miss_rate=l3,
            memory_bandwidth_gbps=100.0 if i == 3 else 0.1,
            network_drop_rate=0.99 if i == 3 else 0.0,
            cpu_throttle_rate=500.0 if i == 3 else 0.0
        )
        nodes.append(node)
    
    req = scheduler_pb2.BatchScoreRequest(
        nodes=nodes,
        pod_requirements=scheduler_pb2.PodRequirements(
            cpu_milli=1000,
            memory_bytes=1024*1024*1024
        )
    )
    
    print(f"Sending BatchScore request for {len(nodes)} nodes...")
    resp = stub.BatchScore(req)
    for score in resp.scores:
        print(f"Node: {score.node_name}, Score: {score.score}, Reasoning: {score.reasoning}")

if __name__ == "__main__":
    trigger()
