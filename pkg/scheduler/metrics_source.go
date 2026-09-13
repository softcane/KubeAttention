package scheduler

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	metricsclient "k8s.io/metrics/pkg/client/clientset/versioned"
)

// MetricSet records which numeric values came from a real measurement.
type MetricSet uint32

const (
	MetricCPUUtilization MetricSet = 1 << iota
	MetricCPUThrottleRate
	MetricMemoryUtilization
	MetricMemoryBandwidth
	MetricL3CacheMissRate
	MetricL3CacheOccupancy
	MetricDiskIOWait
	MetricDiskIOPS
	MetricNetworkRXPackets
	MetricNetworkTXPackets
	MetricNetworkDropRate
)

// Has reports whether a measurement is available.
func (set MetricSet) Has(metric MetricSet) bool {
	return set&metric != 0
}

// NodeMetrics is one measured node sample with provenance and freshness.
type NodeMetrics struct {
	NodeName            string
	Timestamp           time.Time
	Window              time.Duration
	Source              string
	Available           MetricSet
	DegradationReason   string
	CPUUtilization      float64
	CPUThrottleRate     float64
	MemoryUtilization   float64
	MemoryBandwidthGbps float64
	L3CacheMissRate     float64
	L3CacheOccupancyMB  float64
	DiskIOWaitMs        float64
	DiskIOPS            float64
	NetworkRxPacketsSec float64
	NetworkTxPacketsSec float64
	NetworkDropRate     float64
}

// Fresh reports whether the sample can be used at the given time.
func (m *NodeMetrics) Fresh(now time.Time, maxAge time.Duration) bool {
	return m != nil && !m.Timestamp.IsZero() && !m.Timestamp.After(now) && now.Sub(m.Timestamp) <= maxAge
}

// ActiveScoringReady reports whether the sample has fresh node-level CPU and
// memory pressure. Optional hardware counters improve the model when present.
func (m *NodeMetrics) ActiveScoringReady(now time.Time, maxAge time.Duration) bool {
	const required = MetricCPUUtilization | MetricMemoryUtilization
	return m.Fresh(now, maxAge) && m.Available&required == required
}

// KubernetesMetricsSource reads CPU and memory use from metrics.k8s.io.
type KubernetesMetricsSource struct {
	nodes   corelisters.NodeLister
	metrics metricsclient.Interface
}

// NewKubernetesMetricsSource creates a source using the scheduler's API configuration.
func NewKubernetesMetricsSource(config *rest.Config, nodes corelisters.NodeLister) (*KubernetesMetricsSource, error) {
	client, err := metricsclient.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("create metrics client: %w", err)
	}
	return &KubernetesMetricsSource{nodes: nodes, metrics: client}, nil
}

func newKubernetesMetricsSource(nodes corelisters.NodeLister, client metricsclient.Interface) *KubernetesMetricsSource {
	return &KubernetesMetricsSource{nodes: nodes, metrics: client}
}

// GetNodeMetrics returns normalized CPU and memory utilization for one node.
func (s *KubernetesMetricsSource) GetNodeMetrics(ctx context.Context, nodeName string) (*NodeMetrics, error) {
	node, err := s.nodes.Get(nodeName)
	if err != nil {
		return nil, fmt.Errorf("get node %s: %w", nodeName, err)
	}
	sample, err := s.metrics.MetricsV1beta1().NodeMetricses().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get metrics for node %s: %w", nodeName, err)
	}

	cpuCapacity := node.Status.Allocatable.Cpu().MilliValue()
	memoryCapacity := node.Status.Allocatable.Memory().Value()
	if cpuCapacity <= 0 || memoryCapacity <= 0 {
		return nil, fmt.Errorf("node %s has invalid allocatable capacity", nodeName)
	}
	cpuUsage := sample.Usage.Cpu().MilliValue()
	memoryUsage := sample.Usage.Memory().Value()

	return &NodeMetrics{
		NodeName:          nodeName,
		Timestamp:         sample.Timestamp.Time,
		Window:            sample.Window.Duration,
		Source:            "metrics.k8s.io/v1beta1",
		Available:         MetricCPUUtilization | MetricMemoryUtilization,
		DegradationReason: "optional hardware contention measurements unavailable",
		CPUUtilization:    clampRatio(float64(cpuUsage) / float64(cpuCapacity)),
		MemoryUtilization: clampRatio(float64(memoryUsage) / float64(memoryCapacity)),
	}, nil
}

func clampRatio(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
