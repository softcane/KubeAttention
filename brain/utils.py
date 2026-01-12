"""
KubeAttention Utility Functions

Shared helper functions to ensure consistent behavior across the codebase.
"""

from typing import Dict, Any

from .config import INFERENCE


def create_neutral_result(node_name: str, reason: str) -> Dict[str, Any]:
    """
    Create a neutral fallback result with standard values.
    
    Used when:
    - Telemetry data is stale
    - Inference timeout exceeded
    - Input contains NaN/Inf values
    
    Args:
        node_name: Name of the node
        reason: Human-readable explanation for the fallback
        
    Returns:
        Standard result dict with neutral score
    """
    return {
        "node_name": node_name,
        "score": INFERENCE.FALLBACK_SCORE,
        "confidence": INFERENCE.FALLBACK_CONFIDENCE,
        "reasoning": reason,
    }
