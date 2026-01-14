"""
Data exporter for converting collected events to Parquet format.

Provides utilities for:
- Converting JSONL to Parquet
- Loading training data
- Data preprocessing and filtering
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("PyArrow not installed. Run: pip install pyarrow")


def export_to_parquet(
    jsonl_path: str,
    parquet_path: str,
    min_outcome_wait_seconds: float = 300,  # 5 minutes
) -> int:
    """
    Convert JSONL scheduling events to Parquet format.
    
    Args:
        jsonl_path: Path to input JSONL file
        parquet_path: Path to output Parquet file
        min_outcome_wait_seconds: Minimum time before outcome (filter incomplete)
        
    Returns:
        Number of events exported
    """
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow required for Parquet export. Run: pip install pyarrow")
    
    # Load events
    events = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                # Filter out pending outcomes
                if event.get("outcome") != "pending":
                    events.append(event)
    
    if not events:
        print("No completed events to export")
        return 0
    
    # Flatten nested structures for Parquet
    flattened = []
    for event in events:
        flat = {
            "event_id": event.get("event_id", ""),
            "timestamp": event.get("timestamp", ""),
            "pod_uid": event.get("pod_uid", ""),
            "pod_name": event.get("pod_name", ""),
            "pod_namespace": event.get("pod_namespace", ""),
            "cpu_request_milli": event.get("cpu_request_milli", 0),
            "memory_request_bytes": event.get("memory_request_bytes", 0),
            "chosen_node": event.get("chosen_node", ""),
            "scheduler_name": event.get("scheduler_name", ""),
            "outcome": event.get("outcome", "unknown"),
            "outcome_timestamp": event.get("outcome_timestamp", ""),
            "p99_latency_ms": event.get("p99_latency_ms", 0.0),
        }
        
        # Flatten node telemetry (store as JSON string for now)
        flat["node_telemetry_json"] = json.dumps(event.get("node_telemetry", {}))
        flat["pod_labels_json"] = json.dumps(event.get("pod_labels", {}))
        flat["candidate_nodes_json"] = json.dumps(event.get("candidate_nodes", []))
        
        flattened.append(flat)
    
    # Create PyArrow table
    table = pa.Table.from_pylist(flattened)
    
    # Write to Parquet
    pq.write_table(table, parquet_path, compression='snappy')
    
    print(f"Exported {len(flattened)} events to {parquet_path}")
    return len(flattened)


def load_training_data(
    parquet_path: str,
    outcome_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load training data from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
        outcome_filter: Optional list of outcomes to include
        
    Returns:
        List of event dictionaries
    """
    if not HAS_PYARROW:
        raise RuntimeError("PyArrow required for Parquet loading. Run: pip install pyarrow")
    
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    events = []
    for _, row in df.iterrows():
        event = row.to_dict()
        
        # Parse JSON fields
        event["node_telemetry"] = json.loads(event.pop("node_telemetry_json", "{}"))
        event["pod_labels"] = json.loads(event.pop("pod_labels_json", "{}"))
        event["candidate_nodes"] = json.loads(event.pop("candidate_nodes_json", "[]"))
        
        # Apply outcome filter
        if outcome_filter and event.get("outcome") not in outcome_filter:
            continue
        
        events.append(event)
    
    print(f"Loaded {len(events)} events from {parquet_path}")
    return events


def get_training_stats(data_path: str) -> Dict[str, Any]:
    """
    Get statistics about the training data.
    
    Returns:
        Dictionary with counts by outcome, namespace, etc.
    """
    if data_path.endswith(".parquet"):
        events = load_training_data(data_path)
    else:
        with open(data_path) as f:
            events = [json.loads(line) for line in f if line.strip()]
    
    # Count outcomes
    outcomes = {}
    namespaces = {}
    schedulers = {}
    
    for event in events:
        outcome = event.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        ns = event.get("pod_namespace", "unknown")
        namespaces[ns] = namespaces.get(ns, 0) + 1
        
        sched = event.get("scheduler_name", "unknown")
        schedulers[sched] = schedulers.get(sched, 0) + 1
    
    return {
        "total_events": len(events),
        "outcomes": outcomes,
        "namespaces": namespaces,
        "schedulers": schedulers,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export training data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_training_stats(args.input)
        print("\nTraining Data Statistics:")
        print(f"   Total events: {stats['total_events']}")
        print(f"   Outcomes: {stats['outcomes']}")
        print(f"   Namespaces: {stats['namespaces']}")
    else:
        export_to_parquet(args.input, args.output)
