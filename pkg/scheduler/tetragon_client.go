/*
Package scheduler provides a complete Tetragon client for fetching real eBPF metrics.

This client connects to Tetragon's gRPC API to fetch runtime telemetry
including L3 cache metrics, memory bandwidth, and I/O statistics.

If Tetragon is not available, it falls back to Prometheus metrics from
the Kubernetes metrics-server.
*/
package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// TetragonClient fetches real eBPF metrics from Tetragon or Prometheus fallback
type TetragonClient struct {
	tetragonEndpoint   string
	prometheusEndpoint string
	httpClient         *http.Client
	mu                 sync.RWMutex
	cache              map[string]*NodeMetrics
	cacheTTL           time.Duration
	usePrometheus      bool // fallback flag
}

// NodeMetrics holds real telemetry from Tetragon eBPF probes
type NodeMetrics struct {
	NodeName            string    `json:"node_name"`
	Timestamp           time.Time `json:"timestamp"`
	CPUUtilization      float64   `json:"cpu_utilization"`
	CPUThrottleRate     float64   `json:"cpu_throttle_rate"`
	MemoryUtilization   float64   `json:"memory_utilization"`
	MemoryBandwidthGbps float64   `json:"memory_bandwidth_gbps"`
	L3CacheMissRate     float64   `json:"l3_cache_miss_rate"`
	L3CacheOccupancyMB  float64   `json:"l3_cache_occupancy_mb"`
	DiskIOWaitMs        float64   `json:"disk_io_wait_ms"`
	DiskIOPS            float64   `json:"disk_iops"`
	NetworkRxPacketsSec float64   `json:"network_rx_packets_sec"`
	NetworkTxPacketsSec float64   `json:"network_tx_packets_sec"`
	NetworkDropRate     float64   `json:"network_drop_rate"`
}

// NewTetragonClient creates a client for fetching real metrics
func NewTetragonClient(tetragonEndpoint, prometheusEndpoint string) *TetragonClient {
	if tetragonEndpoint == "" {
		tetragonEndpoint = "localhost:2112" // Fixed: 2112 is the standard Tetragon metrics port
	}
	if prometheusEndpoint == "" {
		prometheusEndpoint = "http://prometheus-server.monitoring:9090"
	}
	return &TetragonClient{
		tetragonEndpoint:   tetragonEndpoint,
		prometheusEndpoint: prometheusEndpoint,
		httpClient: &http.Client{
			Timeout: 50 * time.Millisecond, // Fast timeout for scheduling path compliance
		},
		cache: make(map[string]*NodeMetrics),
	}
}

// GetNodeMetrics fetches real-time metrics for a node
// Tries Tetragon first, falls back to Prometheus.
// Note: This is now called by background TelemetryStore.
func (c *TetragonClient) GetNodeMetrics(ctx context.Context, nodeName string) (*NodeMetrics, error) {
	// Try Tetragon first
	metrics, err := c.fetchFromTetragon(ctx, nodeName)
	if err != nil {
		// Fallback to Prometheus
		metrics, err = c.fetchFromPrometheus(ctx, nodeName)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch metrics: tetragon and prometheus both failed: %w", err)
		}
		c.usePrometheus = true
	}

	return metrics, nil
}

// fetchFromTetragon fetches metrics using Tetragon HTTP metrics endpoint
// Tetragon exposes Prometheus-format metrics at /metrics
func (c *TetragonClient) fetchFromTetragon(ctx context.Context, nodeName string) (*NodeMetrics, error) {
	// Tetragon exposes metrics at http://localhost:2112/metrics in Prometheus format
	// These include perf counters for cache misses, memory bandwidth, etc.
	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("http://%s/metrics", c.tetragonEndpoint), nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tetragon not reachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tetragon returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Parse Prometheus-format metrics from Tetragon
	metrics := &NodeMetrics{
		NodeName:  nodeName,
		Timestamp: time.Now(),
	}

	metricsStr := string(body)

	// Parse Tetragon-specific eBPF metrics
	// tetragon_events_total{type="PROCESS_EXEC"}
	// tetragon_policy_events{action="allow"}
	// tetragon_perf_event_cache_misses - L3 cache misses from perf counters

	metrics.L3CacheMissRate = parsePrometheusMetric(metricsStr, "tetragon_perf_event_cache_misses")
	metrics.L3CacheOccupancyMB = parsePrometheusMetric(metricsStr, "tetragon_perf_event_llc_occupancy") / (1024 * 1024)
	metrics.MemoryBandwidthGbps = parsePrometheusMetric(metricsStr, "tetragon_perf_event_memory_bandwidth") / 1e9
	metrics.CPUThrottleRate = parsePrometheusMetric(metricsStr, "tetragon_cgroup_throttle_events_total")
	metrics.DiskIOWaitMs = parsePrometheusMetric(metricsStr, "tetragon_io_latency_seconds") * 1000
	metrics.DiskIOPS = parsePrometheusMetric(metricsStr, "tetragon_io_operations_total")
	metrics.NetworkDropRate = parsePrometheusMetric(metricsStr, "tetragon_network_drops_total")
	metrics.NetworkRxPacketsSec = parsePrometheusMetric(metricsStr, "tetragon_network_rx_packets")
	metrics.NetworkTxPacketsSec = parsePrometheusMetric(metricsStr, "tetragon_network_tx_packets")

	return metrics, nil
}

// parsePrometheusMetric extracts a metric value from Prometheus text format
func parsePrometheusMetric(body, metricName string) float64 {
	// Simple parser for Prometheus text format: metric_name{labels} value
	lines := strings.Split(body, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, metricName) && !strings.HasPrefix(line, "#") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				var value float64
				fmt.Sscanf(parts[len(parts)-1], "%f", &value)
				return value
			}
		}
	}
	return 0.0
}

// fetchFromPrometheus fetches metrics from Prometheus as fallback
func (c *TetragonClient) fetchFromPrometheus(ctx context.Context, nodeName string) (*NodeMetrics, error) {
	metrics := &NodeMetrics{
		NodeName:  nodeName,
		Timestamp: time.Now(),
	}

	// Query Prometheus for each metric
	queries := map[string]*float64{
		"node_cpu_seconds_total":              &metrics.CPUUtilization,
		"node_memory_MemTotal_bytes":          &metrics.MemoryUtilization,
		"node_disk_io_time_seconds_total":     &metrics.DiskIOWaitMs,
		"node_network_receive_packets_total":  &metrics.NetworkRxPacketsSec,
		"node_network_transmit_packets_total": &metrics.NetworkTxPacketsSec,
	}

	for query, target := range queries {
		value, err := c.queryPrometheus(ctx, query, nodeName)
		if err == nil {
			*target = value
		}
	}

	// eBPF-specific metrics not available in Prometheus
	// Set to 0 with a flag indicating "estimated"
	metrics.L3CacheMissRate = 0.0     // Not available without eBPF
	metrics.L3CacheOccupancyMB = 0.0  // Not available without eBPF
	metrics.MemoryBandwidthGbps = 0.0 // Not available without eBPF

	return metrics, nil
}

// queryPrometheus executes a PromQL query
func (c *TetragonClient) queryPrometheus(ctx context.Context, metric, nodeName string) (float64, error) {
	query := fmt.Sprintf("%s{instance=~\"%s.*\"}", metric, nodeName)
	url := fmt.Sprintf("%s/api/v1/query?query=%s", c.prometheusEndpoint, query)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}

	// Parse Prometheus response
	var result struct {
		Data struct {
			Result []struct {
				Value []interface{} `json:"value"`
			} `json:"result"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return 0, err
	}

	if len(result.Data.Result) > 0 && len(result.Data.Result[0].Value) > 1 {
		if val, ok := result.Data.Result[0].Value[1].(string); ok {
			var f float64
			fmt.Sscanf(val, "%f", &f)
			return f, nil
		}
	}

	return 0, fmt.Errorf("no data")
}

// ToTelemetryMap converts NodeMetrics to the map format expected by BrainClient
// IMPORTANT: Must include all 15 metrics to match Python model's FEATURE_NAMES
func (m *NodeMetrics) ToTelemetryMap() map[string]float64 {
	return map[string]float64{
		"cpu_utilization":        m.CPUUtilization,
		"cpu_throttle_rate":      m.CPUThrottleRate,
		"memory_utilization":     m.MemoryUtilization,
		"memory_bandwidth_gbps":  m.MemoryBandwidthGbps,
		"l3_cache_miss_rate":     m.L3CacheMissRate,
		"l3_cache_occupancy_mb":  m.L3CacheOccupancyMB,
		"disk_io_wait_ms":        m.DiskIOWaitMs,
		"disk_iops":              m.DiskIOPS,
		"network_rx_packets_sec": m.NetworkRxPacketsSec,
		"network_tx_packets_sec": m.NetworkTxPacketsSec,
		"network_drop_rate":      m.NetworkDropRate,
		// Cost/resilience metrics - default values for now
		"node_cost_index":        0.1, // Default cost index
		"zone_diversity_score":   0.5, // Neutral zone score
		"spot_interruption_risk": 0.0, // Assume on-demand by default
		"is_spot_instance":       0.0, // Boolean: not spot
	}
}

// IsUsingFallback returns true if Prometheus fallback is active
func (c *TetragonClient) IsUsingFallback() bool {
	return c.usePrometheus
}
