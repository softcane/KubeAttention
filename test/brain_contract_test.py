import asyncio
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import grpc
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gen" / "python"))

import scheduler_pb2

from brain.metrics_schema import (
    FEATURE_SCHEMA_VERSION,
    MODEL_INPUT_DIM,
    POD_FEATURE_NAMES,
)
from brain.models.base import BaseScorer, ScoringResult
from brain.models.mlp_scorer import MLPScorer
from brain.models.xgboost_scorer import XGBoostScorer
from brain.training.train import evaluate_promotion, prepare_training_data
from brain.server import BrainServicer


class FakeContext:
    def __init__(self, remaining=None):
        self.code = grpc.StatusCode.OK
        self.details = ""
        self.remaining = remaining

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def time_remaining(self):
        return self.remaining


class RecordingModel(BaseScorer):
    def __init__(self, delay=0.0):
        self.delay = delay
        self.last_node_features = None
        self.last_pod_features = None

    @property
    def name(self):
        return "recording"

    @property
    def num_parameters(self):
        return 0

    @property
    def ready(self):
        return True

    def score_nodes(self, node_features, pod_features, node_names):
        if self.delay:
            time.sleep(self.delay)
        self.last_node_features = node_features.copy()
        self.last_pod_features = pod_features.copy()
        return [ScoringResult(name, 73, "recorded") for name in node_names]

    def train(self, X, y, weights=None):
        return {}

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


def valid_node(name="node-a", **overrides):
    values = {
        "node_name": name,
        "cpu_utilization": 0.2,
        "memory_utilization": 0.3,
        "memory_bandwidth_gbps": 12.0,
        "l3_cache_miss_rate": 0.1,
        "timestamp_unix_ms": int(time.time() * 1000) - 100,
        "available_metrics": [
            scheduler_pb2.NODE_METRIC_CPU_UTILIZATION,
            scheduler_pb2.NODE_METRIC_MEMORY_UTILIZATION,
            scheduler_pb2.NODE_METRIC_MEMORY_BANDWIDTH,
            scheduler_pb2.NODE_METRIC_L3_CACHE_MISS_RATE,
        ],
        "telemetry_source": "contract-test",
        "observation_window_ms": 1_000,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
    values.update(overrides)
    return scheduler_pb2.NodeTelemetry(**values)


def valid_request(nodes=None, **pod_overrides):
    pod_values = {
        "pod_name": "workload",
        "pod_namespace": "default",
        "cpu_milli": 1_000,
        "memory_bytes": 1024**3,
        "priority": 900_000_000,
        "workload_type": "cpu-bound",
        "criticality": scheduler_pb2.CRITICALITY_HIGH,
        "labels": {"kubeattention.io/workload-type": "cpu-bound"},
    }
    pod_values.update(pod_overrides)
    return scheduler_pb2.BatchScoreRequest(
        pod_requirements=scheduler_pb2.PodRequirements(**pod_values),
        nodes=nodes if nodes is not None else [valid_node()],
    )


class FixedPredictor:
    def predict(self, features):
        return np.asarray(features[:, 0], dtype=np.float32)

class PredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_quality(self, features):
        return self.predictions

class BrainContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_valid_request_preserves_versioned_node_and_pod_features(self):
        model = RecordingModel()
        servicer = BrainServicer(model=model)
        context = FakeContext(remaining=1.0)

        response = await servicer.BatchScore(valid_request(), context)

        self.assertEqual(context.code, grpc.StatusCode.OK)
        self.assertEqual(response.scores[0].score, 73)
        self.assertEqual(model.last_node_features.shape, (1, 15))
        self.assertEqual(len(model.last_pod_features), len(POD_FEATURE_NAMES))
        self.assertGreater(model.last_pod_features[2], 0.9)
        self.assertEqual(model.last_pod_features[-1], 1.0)

    async def test_malformed_requests_never_reach_model(self):
        cases = [
            valid_request(nodes=[]),
            valid_request(nodes=[valid_node(timestamp_unix_ms=0)]),
            valid_request(nodes=[valid_node(timestamp_unix_ms=int(time.time() * 1000) + 60_000)]),
            valid_request(nodes=[valid_node(cpu_utilization=float("nan"))]),
            valid_request(cpu_milli=-1),
            valid_request(nodes=[valid_node("node-a"), valid_node("node-a")]),
        ]
        for request in cases:
            with self.subTest(request=request):
                model = RecordingModel()
                context = FakeContext(remaining=1.0)
                response = await BrainServicer(model=model).BatchScore(request, context)
                self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)
                self.assertEqual(len(response.scores), 0)
                self.assertIsNone(model.last_node_features)

    async def test_incompatible_schema_is_failed_precondition(self):
        context = FakeContext(remaining=1.0)
        response = await BrainServicer(model=RecordingModel()).BatchScore(
            valid_request(nodes=[valid_node(schema_version="old")]), context
        )
        self.assertEqual(context.code, grpc.StatusCode.FAILED_PRECONDITION)
        self.assertEqual(len(response.scores), 0)

    async def test_missing_required_measurement_is_neutral(self):
        context = FakeContext(remaining=1.0)
        node = valid_node(available_metrics=[scheduler_pb2.NODE_METRIC_CPU_UTILIZATION])
        response = await BrainServicer(model=RecordingModel()).BatchScore(
            valid_request(nodes=[node]), context
        )
        self.assertEqual(context.code, grpc.StatusCode.OK)
        self.assertEqual(response.scores[0].score, 50)
        self.assertIn("missing required metrics", response.scores[0].reasoning)

    async def test_caller_deadline_bounds_inference(self):
        context = FakeContext(remaining=0.01)
        started = time.perf_counter()
        response = await BrainServicer(model=RecordingModel(delay=0.2)).BatchScore(
            valid_request(), context
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.1)
        self.assertEqual(response.scores[0].score, 50)
        self.assertIn("deadline exceeded", response.scores[0].reasoning)

    async def test_health_requires_compatible_checkpoint(self):
        servicer = BrainServicer(model=MLPScorer())
        response = await servicer.HealthCheck(scheduler_pb2.HealthCheckRequest(), FakeContext())
        self.assertFalse(response.healthy)
        self.assertEqual(response.model_schema_version, "")

    def test_mlp_checkpoint_records_and_validates_schema(self):
        model = MLPScorer()
        X = np.zeros((2, MODEL_INPUT_DIM), dtype=np.float32)
        y = np.array([0.25, 0.75], dtype=np.float32)
        model.train(X, y, epochs=1, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            model.save(path)
            loaded = MLPScorer()
            loaded.load(path)
            self.assertTrue(loaded.ready)

    def test_xgboost_scores_have_candidate_independent_absolute_meaning(self):
        model = XGBoostScorer()
        model.model = FixedPredictor()
        model._is_trained = True
        pod = np.zeros(len(POD_FEATURE_NAMES), dtype=np.float32)
        node_a = np.zeros((1, 15), dtype=np.float32)
        node_a[0, 0] = 0.2
        alone = model.score_nodes(node_a, pod, ["a"])[0].score

        candidates = np.zeros((2, 15), dtype=np.float32)
        candidates[:, 0] = [0.2, 0.8]
        together = model.score_nodes(candidates, pod, ["a", "b"])[0].score

        self.assertEqual(alone, 20)
        self.assertEqual(together, 20)

    def test_promotion_requires_measured_validation_and_baseline_win(self):
        features = np.zeros((2, MODEL_INPUT_DIM), dtype=np.float32)
        features[:, 0] = [0.1, 0.9]
        features[:, 2] = [0.1, 0.9]
        features[:, 4] = [0.1, 0.9]
        labels = np.array([0.9, 0.1], dtype=np.float32)
        accepted = evaluate_promotion(
            PredictionModel(labels), features, labels, minimum_improvement=0.01
        )
        self.assertTrue(accepted["promoted"])

        rejected = evaluate_promotion(
            PredictionModel(np.array([0.5, 0.5], dtype=np.float32)),
            features,
            labels,
            minimum_improvement=0.01,
        )
        self.assertFalse(rejected["promoted"])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unmeasured.jsonl")
            with open(path, "w") as events_file:
                events_file.write('{"event_id":"one","outcome":"running"}\n')
            with self.assertRaisesRegex(ValueError, "evidence_source='measured'"):
                prepare_training_data(path, require_measured=True)

            path = os.path.join(directory, "incomplete.jsonl")
            with open(path, "w") as events_file:
                events_file.write(
                    '{"event_id":"two","outcome":"running",'
                    '"evidence_source":"measured","candidate_nodes":["node-a"],'
                    '"node_telemetry":{"node-a":{"available_metrics":'
                    '["cpu_utilization"]}}}\n'
                )
            with self.assertRaisesRegex(
                ValueError, "complete required interference measurements"
            ):
                prepare_training_data(path, require_measured=True)


if __name__ == "__main__":
    unittest.main()
