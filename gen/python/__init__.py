"""
Generated protobuf __init__.py for scheduler module.
This file enables Python imports from the generated proto directory.
"""

from .scheduler_pb2 import (
    ScoreRequest,
    ScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    NodeTelemetry,
    PodRequirements,
    NodeScore,
    HealthCheckRequest,
    HealthCheckResponse,
)

from .scheduler_pb2_grpc import (
    BrainServicer,
    BrainStub,
    add_BrainServicer_to_server,
)

__all__ = [
    'ScoreRequest',
    'ScoreResponse',
    'BatchScoreRequest',
    'BatchScoreResponse',
    'NodeTelemetry',
    'PodRequirements',
    'NodeScore',
    'HealthCheckRequest',
    'HealthCheckResponse',
    'BrainServicer',
    'BrainStub',
    'add_BrainServicer_to_server',
]
