package scheduler

import (
	"context"
	"math"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	corelisters "k8s.io/client-go/listers/core/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

func TestKubernetesMetricsSourceNormalizesNodeUsage(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-a"},
		Status: v1.NodeStatus{Allocatable: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("4"),
			v1.ResourceMemory: resource.MustParse("8Gi"),
		}},
	}
	if err := indexer.Add(node); err != nil {
		t.Fatalf("add node: %v", err)
	}
	observedAt := time.Now().UTC().Truncate(time.Second)
	metric := &metricsv1beta1.NodeMetrics{
		ObjectMeta: metav1.ObjectMeta{Name: "node-a"},
		Timestamp:  metav1.NewTime(observedAt),
		Window:     metav1.Duration{Duration: 30 * time.Second},
		Usage: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1"),
			v1.ResourceMemory: resource.MustParse("2Gi"),
		},
	}
	client := metricsfake.NewSimpleClientset()
	client.Fake.PrependReactor("get", "nodes", func(k8stesting.Action) (bool, runtime.Object, error) {
		return true, metric, nil
	})
	source := newKubernetesMetricsSource(corelisters.NewNodeLister(indexer), client)

	got, err := source.GetNodeMetrics(context.Background(), "node-a")
	if err != nil {
		t.Fatalf("get node metrics: %v", err)
	}
	if math.Abs(got.CPUUtilization-0.25) > 0.0001 || math.Abs(got.MemoryUtilization-0.25) > 0.0001 {
		t.Fatalf("normalized utilization = cpu %.4f memory %.4f", got.CPUUtilization, got.MemoryUtilization)
	}
	if !got.Available.Has(MetricCPUUtilization) || !got.Available.Has(MetricMemoryUtilization) {
		t.Fatalf("CPU and memory availability not recorded: %b", got.Available)
	}
	if got.Available.Has(MetricL3CacheMissRate) {
		t.Fatal("unobserved cache metric marked available")
	}
	if !got.Timestamp.Equal(observedAt) || got.Window != 30*time.Second {
		t.Fatalf("freshness metadata = %s/%s", got.Timestamp, got.Window)
	}
}

func TestActiveScoringRequiresFreshResourcePressureSignals(t *testing.T) {
	now := time.Now()
	required := MetricCPUUtilization | MetricMemoryUtilization
	sample := &NodeMetrics{Timestamp: now.Add(-time.Second), Available: required}
	if !sample.ActiveScoringReady(now, 10*time.Second) {
		t.Fatal("fresh CPU and memory telemetry rejected")
	}

	sample.Available &^= MetricMemoryUtilization
	if sample.ActiveScoringReady(now, 10*time.Second) {
		t.Fatal("telemetry without memory pressure accepted")
	}

	sample.Available = required
	sample.Timestamp = now.Add(-11 * time.Second)
	if sample.ActiveScoringReady(now, 10*time.Second) {
		t.Fatal("stale telemetry accepted")
	}
}

func TestSchedulerDefaultsToShadowMode(t *testing.T) {
	args := &KubeAttentionArgs{}
	args.SetDefaults()
	if !args.shadowMode() {
		t.Fatalf("default mode = %q, want shadow", args.Mode)
	}
}

func TestNodeTelemetryReflectsMeasuredContention(t *testing.T) {
	node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}
	available := MetricCPUUtilization | MetricMemoryUtilization
	quiet := nodeTelemetry(node, &NodeMetrics{
		Timestamp:         time.UnixMilli(1_000),
		Available:         available,
		Source:            "metrics.k8s.io/v1beta1",
		CPUUtilization:    0.1,
		MemoryUtilization: 0.2,
	})
	busy := nodeTelemetry(node, &NodeMetrics{
		Timestamp:         time.UnixMilli(2_000),
		Available:         available,
		Source:            "metrics.k8s.io/v1beta1",
		CPUUtilization:    0.9,
		MemoryUtilization: 0.7,
	})

	if quiet.GetCpuUtilization() == busy.GetCpuUtilization() ||
		quiet.GetMemoryUtilization() == busy.GetMemoryUtilization() {
		t.Fatal("measured contention did not change features sent to Brain")
	}
	if len(busy.GetAvailableMetrics()) != 2 || busy.GetSchemaVersion() != FeatureSchemaVersion {
		t.Fatalf("measurement metadata not preserved: %+v", busy)
	}
}
