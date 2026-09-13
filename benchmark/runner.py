#!/usr/bin/env python3
"""Compare measured Redis tail latency under two Kubernetes schedulers."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time


@dataclass
class PodLatency:
    pod_name: str
    node_name: str
    node_profile: str
    requests: int
    average_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float


@dataclass
class RunResult:
    scheduler_name: str
    start_time: str
    end_time: str
    requests: int
    worst_pod_p99_ms: float
    mean_pod_p99_ms: float
    pods: list[dict]


def run(command: list[str], *, input_text: str | None = None) -> str:
    print("  $", " ".join(command))
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def kubectl(*args: str, input_text: str | None = None) -> str:
    return run(["kubectl", *args], input_text=input_text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_json(path: Path, namespace: str) -> dict:
    raw = kubectl(
        "create", "--dry-run=client", "-f", str(path), "-o", "json"
    )
    manifest = json.loads(raw)
    manifest.setdefault("metadata", {})["namespace"] = namespace
    return manifest


def apply_manifest(manifest: dict) -> None:
    kubectl("apply", "-f", "-", input_text=json.dumps(manifest))


def deploy_pressure(project_root: Path, namespace: str) -> None:
    generator_dir = project_root / "benchmark" / "generators"
    for path in sorted(generator_dir.glob("*.yaml")):
        apply_manifest(manifest_json(path, namespace))
    kubectl(
        "wait",
        "--for=condition=Available",
        "deployment",
        "-l",
        "role=noisy-neighbor",
        "-n",
        namespace,
        "--timeout=300s",
    )


def workload_manifest(
    project_root: Path, namespace: str, scheduler_name: str, run_name: str
) -> dict:
    manifest = manifest_json(
        project_root / "benchmark" / "workloads" / "redis-latency-test.yaml",
        namespace,
    )
    manifest["metadata"]["name"] = run_name
    manifest["metadata"].setdefault("labels", {})["benchmark-run"] = run_name
    template = manifest["spec"]["template"]
    template["metadata"]["labels"]["app"] = run_name
    template["metadata"]["labels"]["benchmark-run"] = run_name
    manifest["spec"]["selector"]["matchLabels"] = {"app": run_name}
    template["spec"]["schedulerName"] = scheduler_name
    return manifest


def parse_redis_benchmark(output: str, pod_name: str) -> tuple[float, ...]:
    for row in csv.reader(io.StringIO(output)):
        if row and row[0] == "PING_INLINE" and len(row) >= 8:
            return tuple(float(value) for value in (row[2], row[4], row[5], row[6], row[7]))
    raise ValueError(f"redis-benchmark returned no PING_INLINE percentile row for {pod_name}: {output!r}")


def measure_pods(namespace: str, run_name: str, requests_per_pod: int) -> list[PodLatency]:
    pods = json.loads(
        kubectl(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"benchmark-run={run_name}",
            "-o",
            "json",
        )
    )["items"]
    measurements = []
    for pod in pods:
        pod_name = pod["metadata"]["name"]
        node_name = pod["spec"]["nodeName"]
        node = json.loads(kubectl("get", "node", node_name, "-o", "json"))
        node_profile = node["metadata"].get("labels", {}).get(
            "kubeattention.io/profile", "unlabelled"
        )
        output = kubectl(
            "exec",
            pod_name,
            "-n",
            namespace,
            "-c",
            "latency-probe",
            "--",
            "redis-benchmark",
            "-h",
            "localhost",
            "-t",
            "ping_inline",
            "-n",
            str(requests_per_pod),
            "-c",
            "1",
            "--csv",
        )
        average, p50, p95, p99, maximum = parse_redis_benchmark(output, pod_name)
        measurements.append(
            PodLatency(
                pod_name=pod_name,
                node_name=node_name,
                node_profile=node_profile,
                requests=requests_per_pod,
                average_ms=average,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                maximum_ms=maximum,
            )
        )
    if not measurements:
        raise ValueError(f"no pods found for benchmark run {run_name}")
    return measurements


def run_profile(
    project_root: Path,
    namespace: str,
    scheduler_name: str,
    requests_per_pod: int,
    warmup_seconds: int,
) -> RunResult:
    run_name = "redis-default" if scheduler_name == "default-scheduler" else "redis-kubeattention"
    start_time = utc_now()
    apply_manifest(workload_manifest(project_root, namespace, scheduler_name, run_name))
    kubectl(
        "wait",
        "--for=condition=Available",
        f"deployment/{run_name}",
        "-n",
        namespace,
        "--timeout=300s",
    )
    time.sleep(warmup_seconds)
    pod_metrics = measure_pods(namespace, run_name, requests_per_pod)
    p99_values = [metric.p99_ms for metric in pod_metrics]
    result = RunResult(
        scheduler_name=scheduler_name,
        start_time=start_time,
        end_time=utc_now(),
        requests=sum(metric.requests for metric in pod_metrics),
        worst_pod_p99_ms=max(p99_values),
        mean_pod_p99_ms=sum(p99_values) / len(p99_values),
        pods=[asdict(metric) for metric in pod_metrics],
    )
    kubectl("delete", f"deployment/{run_name}", "-n", namespace, "--wait=true")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-per-pod", type=int, default=10_000)
    parser.add_argument("--warmup-seconds", type=int, default=30)
    parser.add_argument("--namespace", default="kubeattention-benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args()
    if args.requests_per_pod <= 0 or args.warmup_seconds < 0:
        parser.error("request count must be positive and warmup must be non-negative")

    project_root = Path(__file__).resolve().parent.parent
    kubectl("get", "deployment/kubeattention-brain", "-n", "kubeattention")
    kubectl("get", "deployment/kubeattention-scheduler", "-n", "kubeattention")
    scheduler_config = kubectl(
        "get",
        "configmap/kubeattention-scheduler-config",
        "-n",
        "kubeattention",
        "-o",
        "jsonpath={.data.scheduler-config\\.yaml}",
    )
    if "mode: active" not in scheduler_config:
        raise RuntimeError("KubeAttention scheduler must be in active mode for an A/B comparison")
    kubectl("get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes")
    kubectl("create", "namespace", args.namespace)
    try:
        deploy_pressure(project_root, args.namespace)
        default = run_profile(
            project_root,
            args.namespace,
            "default-scheduler",
            args.requests_per_pod,
            args.warmup_seconds,
        )
        kubeattention = run_profile(
            project_root,
            args.namespace,
            "kubeattention-scheduler",
            args.requests_per_pod,
            args.warmup_seconds,
        )
        delta = default.worst_pod_p99_ms - kubeattention.worst_pod_p99_ms
        percent = (
            delta / default.worst_pod_p99_ms * 100.0
            if default.worst_pod_p99_ms
            else 0.0
        )
        report = {
            "evidence_source": "measured",
            "metric": "worst per-pod P99 from redis-benchmark PING_INLINE",
            "default": asdict(default),
            "kubeattention": asdict(kubeattention),
            "p99_improvement_ms": delta,
            "p99_improvement_percent": percent,
            "improved": delta > 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.require_improvement and delta <= 0:
            print("KubeAttention did not improve measured worst-pod P99 latency")
            return 1
        return 0
    finally:
        if not args.keep:
            kubectl("delete", "namespace", args.namespace, "--wait=true")


if __name__ == "__main__":
    raise SystemExit(main())
