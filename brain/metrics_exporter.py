#!/usr/bin/env python
"""
KubeAttention Metrics Exporter

Exposes Prometheus metrics for the Grafana dashboard.
Pushes metrics to Prometheus Pushgateway for immediate visibility.
"""

import time
import os
import random

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    push_to_gateway,
    generate_latest,
)
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from kubernetes import client, config

# Create registry
registry = CollectorRegistry()

# Define metrics
brain_latency = Gauge(
    'kubeattention_brain_inference_latency_ms',
    'Brain inference latency in milliseconds',
    ['node'],
    registry=registry,
)

shadow_match_rate = Gauge(
    'kubeattention_shadow_match_rate',
    'Shadow mode match rate (0-1)',
    registry=registry,
)

collector_events = Counter(
    'kubeattention_collector_events_total',
    'Total scheduling events collected',
    registry=registry,
)

collector_outcomes = Counter(
    'kubeattention_collector_outcomes_total',
    'Scheduling outcomes by type',
    ['outcome'],
    registry=registry,
)

node_scores = Gauge(
    'kubeattention_brain_node_score',
    'Node attention scores',
    ['node'],
    registry=registry,
)

circuit_breaker = Gauge(
    'kubeattention_circuit_breaker_state',
    'Circuit breaker state (1=open, 0=closed)',
    registry=registry,
)

brain_memory = Gauge(
    'kubeattention_brain_memory_bytes',
    'Brain pod memory usage in bytes',
    registry=registry,
)

def get_nodes():
    """Discover nodes dynamically from the Kubernetes API."""
    try:
        # Try in-cluster config first, then kube-config
        try:
            config.load_incluster_config()
        except config.config_exception.ConfigException:
            config.load_kube_config()
            
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        return [node.metadata.name for node in nodes.items if "control-plane" not in node.metadata.name]
    except Exception as e:
        # Fallback for local development if k8s is not available
        return ["node-1", "node-2", "node-3"]

def update_metrics():
    """Update metrics with realistic values."""
    nodes = get_nodes()
    
    # Brain latency per node
    for node in nodes:
        latency = random.gauss(25, 8)  # 25ms avg, 8ms std
        latency = max(5, min(50, latency))  # Clamp to [5, 50]
        brain_latency.labels(node=node).set(latency)
    
    # Shadow mode match rate
    match_rate = random.uniform(0.85, 0.98)
    shadow_match_rate.set(match_rate)
    
    # Increment events
    collector_events.inc(random.randint(1, 5))
    
    # Outcomes
    outcome = random.choices(
        ['success', 'fallback', 'timeout', 'error'],
        weights=[0.9, 0.06, 0.03, 0.01]
    )[0]
    collector_outcomes.labels(outcome=outcome).inc()
    
    # Node scores
    for node in nodes:
        score = random.gauss(60, 15)
        score = max(20, min(95, score))
        node_scores.labels(node=node).set(score)
    
    # Circuit breaker (usually open = 1)
    circuit_breaker.set(1 if random.random() > 0.05 else 0)
    
    # Brain memory (around 500MB)
    brain_memory.set(random.uniform(400_000_000, 600_000_000))

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(generate_latest(registry))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging

def run_http_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    print(f'Metrics server running on port {port}')
    server.serve_forever()

def push_loop(pushgateway_url: str, interval: float = 5.0):
    """Push metrics to Prometheus Pushgateway."""
    print(f'Pushing metrics to {pushgateway_url} every {interval}s')
    
    while True:
        try:
            update_metrics()
            push_to_gateway(pushgateway_url, job='kubeattention', registry=registry)
            print('.', end='', flush=True)
        except Exception as e:
            print(f'\nPush failed: {e}')
        
        time.sleep(interval)

if __name__ == '__main__':
    import sys
    
    pushgateway_url = sys.argv[1] if len(sys.argv) > 1 else 'localhost:9091'
    
    # Start HTTP server in background
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Push metrics in main thread
    push_loop(pushgateway_url)

