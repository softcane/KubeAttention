#!/usr/bin/env python3
"""Verify that a compatible checkpoint prefers lower measured node pressure."""

import argparse
import sys
from pathlib import Path



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", choices=("mlp", "xgboost"), default="mlp")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    import numpy as np

    from brain.metrics_schema import NodeMetricsSnapshot
    from brain.models import get_model
    from brain.tensor_encoder import PodContext


    model = get_model(args.model)
    model.load(str(args.checkpoint))
    quiet = NodeMetricsSnapshot(
        node_name="quiet-node",
        cpu_utilization=0.1,
        memory_utilization=0.1,
    )
    pressured = NodeMetricsSnapshot(
        node_name="pressured-node",
        cpu_utilization=0.9,
        memory_utilization=0.9,
    )
    pod = PodContext(
        pod_name="verification",
        pod_namespace="default",
        cpu_milli=1000,
        memory_bytes=1024**3,
    )
    results = model.score_nodes(
        np.asarray(
            [quiet.to_feature_vector(), pressured.to_feature_vector()],
            dtype=np.float32,
        ),
        np.asarray(pod.to_feature_vector(), dtype=np.float32),
        [quiet.node_name, pressured.node_name],
    )
    scores = {result.node_name: result.score for result in results}
    print(scores)
    if scores[quiet.node_name] <= scores[pressured.node_name]:
        print("checkpoint did not prefer the lower-pressure node", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
