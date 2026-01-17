/*
Package collector watches Kubernetes scheduling events and collects
training data for the KubeAttention model.

It captures:
1. Pod scheduling decisions (which node was chosen)
2. Node telemetry at decision time
3. Outcome after 5 minutes (success, OOM, eviction)
*/
package collector

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

// SchedulingEvent represents a single scheduling decision with outcome
type SchedulingEvent struct {
	Timestamp          time.Time              `json:"timestamp"`
	EventID            string                 `json:"event_id"`
	PodUID             string                 `json:"pod_uid"`
	PodName            string                 `json:"pod_name"`
	PodNamespace       string                 `json:"pod_namespace"`
	PodLabels          map[string]string      `json:"pod_labels"`
	CPURequestMilli    int64                  `json:"cpu_request_milli"`
	MemoryRequestBytes int64                  `json:"memory_request_bytes"`
	CandidateNodes     []string               `json:"candidate_nodes"`
	NodeTelemetry      map[string]NodeMetrics `json:"node_telemetry"`
	ChosenNode         string                 `json:"chosen_node"`
	SchedulerName      string                 `json:"scheduler_name"`
	// Outcome fields - filled after 5 minutes
	Outcome          string    `json:"outcome"` // "running", "oom_killed", "evicted", "failed"
	OutcomeTimestamp time.Time `json:"outcome_timestamp"`
	P99LatencyMs     float64   `json:"p99_latency_ms"`
}

// NodeMetrics represents telemetry snapshot at scheduling time
// IMPORTANT: Must include all 15 metrics to match Python model's FEATURE_NAMES
type NodeMetrics struct {
	NodeName            string    `json:"node_name"`
	Timestamp           time.Time `json:"timestamp"`
	CPUUtilization      float64   `json:"cpu_utilization"`
	MemoryUtilization   float64   `json:"memory_utilization"`
	L3CacheMissRate     float64   `json:"l3_cache_miss_rate"`
	L3CacheOccupancyMB  float64   `json:"l3_cache_occupancy_mb"`
	MemoryBandwidthGbps float64   `json:"memory_bandwidth_gbps"`
	DiskIOWaitMs        float64   `json:"disk_io_wait_ms"`
	DiskIOPS            float64   `json:"disk_iops"`
	NetworkRxPacketsSec float64   `json:"network_rx_packets_sec"`
	NetworkTxPacketsSec float64   `json:"network_tx_packets_sec"`
	NetworkDropRate     float64   `json:"network_drop_rate"`
	CPUThrottleRate     float64   `json:"cpu_throttle_rate"`
	// Cost/resilience metrics (Phase 2 & 4)
	NodeCostIndex        float64 `json:"node_cost_index"`
	ZoneDiversityScore   float64 `json:"zone_diversity_score"`
	SpotInterruptionRisk float64 `json:"spot_interruption_risk"`
	IsSpotInstance       float64 `json:"is_spot_instance"`
}

// EventCollector watches scheduling events and collects training data
type EventCollector struct {
	client          kubernetes.Interface
	metricsClient   metricsv.Interface
	restConfig      *rest.Config
	outputPath      string
	outcomeWaitTime time.Duration
	mu              sync.Mutex
	pendingOutcomes map[string]*SchedulingEvent // pod UID -> event
	eventFile       *os.File
	eventCount      int64
}

// NewEventCollector creates a collector that logs to JSONL file
func NewEventCollector(outputPath string) (*EventCollector, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	file, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open output file: %w", err)
	}

	metricsClient, err := metricsv.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics client: %w", err)
	}

	return &EventCollector{
		client:          client,
		metricsClient:   metricsClient,
		restConfig:      config,
		outputPath:      outputPath,
		outcomeWaitTime: 30 * time.Second, // Reduced for faster data collection
		pendingOutcomes: make(map[string]*SchedulingEvent),
		eventFile:       file,
	}, nil
}

// Start begins watching for scheduling events
func (c *EventCollector) Start(ctx context.Context) error {
	fmt.Println("🔍 Starting scheduling event collector...")

	// Watch pod events
	watcher, err := c.client.CoreV1().Pods("").Watch(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to watch pods: %w", err)
	}

	// Process events
	go c.processEvents(ctx, watcher)

	// Check pending outcomes periodically
	go c.checkOutcomes(ctx)

	<-ctx.Done()
	return nil
}

// processEvents handles incoming pod events
func (c *EventCollector) processEvents(ctx context.Context, watcher watch.Interface) {
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-watcher.ResultChan():
			if !ok {
				return
			}

			pod, ok := event.Object.(*corev1.Pod)
			if !ok {
				continue
			}

			switch event.Type {
			case watch.Modified:
				// Check if pod was just scheduled
				if pod.Spec.NodeName != "" && pod.Status.Phase == corev1.PodPending {
					c.recordSchedulingDecision(ctx, pod)
				}
			}
		}
	}
}

// recordSchedulingDecision captures a scheduling decision
func (c *EventCollector) recordSchedulingDecision(ctx context.Context, pod *corev1.Pod) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Skip if already recorded
	if _, exists := c.pendingOutcomes[string(pod.UID)]; exists {
		return
	}

	// Calculate resource requests
	var cpuMilli, memBytes int64
	for _, container := range pod.Spec.Containers {
		if cpu := container.Resources.Requests.Cpu(); cpu != nil {
			cpuMilli += cpu.MilliValue()
		}
		if mem := container.Resources.Requests.Memory(); mem != nil {
			memBytes += mem.Value()
		}
	}

	// Get node telemetry (would call TetragonClient in production)
	telemetry := c.collectNodeTelemetry(ctx)

	event := &SchedulingEvent{
		Timestamp:          time.Now(),
		EventID:            fmt.Sprintf("%s-%d", pod.UID, time.Now().UnixNano()),
		PodUID:             string(pod.UID),
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		PodLabels:          pod.Labels,
		CPURequestMilli:    cpuMilli,
		MemoryRequestBytes: memBytes,
		ChosenNode:         pod.Spec.NodeName,
		NodeTelemetry:      telemetry,
		SchedulerName:      pod.Spec.SchedulerName,
		Outcome:            "pending", // Will be updated after 5 minutes
	}

	c.pendingOutcomes[string(pod.UID)] = event
	c.eventCount++

	fmt.Printf("📝 Recorded scheduling event #%d: %s/%s -> %s\n",
		c.eventCount, pod.Namespace, pod.Name, pod.Spec.NodeName)
}

// collectNodeTelemetry gets current telemetry for all nodes from metrics-server
func (c *EventCollector) collectNodeTelemetry(ctx context.Context) map[string]NodeMetrics {
	nodes, err := c.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil
	}

	telemetry := make(map[string]NodeMetrics)

	// Get node metrics from metrics-server using official client
	nodeMetricsList, err := c.metricsClient.MetricsV1beta1().NodeMetricses().List(ctx, metav1.ListOptions{})

	// Build lookup map
	nodeMetricsMap := make(map[string]struct {
		CPUNano  int64
		MemBytes int64
	})
	if err == nil && nodeMetricsList != nil {
		for _, item := range nodeMetricsList.Items {
			nodeMetricsMap[item.Name] = struct {
				CPUNano  int64
				MemBytes int64
			}{
				CPUNano:  item.Usage.Cpu().MilliValue() * 1000000, // milli to nano
				MemBytes: item.Usage.Memory().Value(),
			}
		}
	}

	for _, node := range nodes.Items {
		metrics := NodeMetrics{
			NodeName:  node.Name,
			Timestamp: time.Now(),
			// Cost/resilience defaults (Phase 2 & 4)
			NodeCostIndex:        0.1, // Default cost index
			ZoneDiversityScore:   0.5, // Neutral zone score
			SpotInterruptionRisk: 0.0, // Assume on-demand
			IsSpotInstance:       0.0, // Boolean: not spot
		}

		// Get allocatable resources for percentage calculation
		allocCPU := node.Status.Allocatable.Cpu().MilliValue()
		allocMem := node.Status.Allocatable.Memory().Value()

		// Look up actual usage from metrics-server
		if m, ok := nodeMetricsMap[node.Name]; ok {
			// Calculate utilization percentages
			if allocCPU > 0 {
				cpuMilli := m.CPUNano / 1000000 // nano to milli
				metrics.CPUUtilization = float64(cpuMilli) / float64(allocCPU)
			}
			if allocMem > 0 {
				metrics.MemoryUtilization = float64(m.MemBytes) / float64(allocMem)
			}
		}

		telemetry[node.Name] = metrics
	}

	return telemetry
}

// checkOutcomes periodically checks pod outcomes
func (c *EventCollector) checkOutcomes(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.evaluatePendingOutcomes(ctx)
		}
	}
}

// evaluatePendingOutcomes checks if pods have succeeded or failed
func (c *EventCollector) evaluatePendingOutcomes(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for podUID, event := range c.pendingOutcomes {
		// Wait for outcome window
		if now.Sub(event.Timestamp) < c.outcomeWaitTime {
			continue
		}

		// Get current pod status
		pod, err := c.client.CoreV1().Pods(event.PodNamespace).Get(ctx, event.PodName, metav1.GetOptions{})
		if err != nil {
			event.Outcome = "deleted"
		} else {
			event.Outcome = c.evaluatePodOutcome(pod)
		}

		event.OutcomeTimestamp = now

		// Write to file
		c.writeEvent(event)

		// Remove from pending
		delete(c.pendingOutcomes, podUID)

		fmt.Printf("Outcome for %s/%s: %s\n", event.PodNamespace, event.PodName, event.Outcome)
	}
}

// evaluatePodOutcome determines if scheduling was successful
func (c *EventCollector) evaluatePodOutcome(pod *corev1.Pod) string {
	// Check container statuses for OOM or other failures
	for _, status := range pod.Status.ContainerStatuses {
		if status.State.Terminated != nil {
			if status.State.Terminated.Reason == "OOMKilled" {
				return "oom_killed"
			}
			return "terminated"
		}
		if status.RestartCount > 0 {
			return "restarted"
		}
	}

	// Check if pod was evicted
	if pod.Status.Phase == corev1.PodFailed {
		if pod.Status.Reason == "Evicted" {
			return "evicted"
		}
		return "failed"
	}

	if pod.Status.Phase == corev1.PodRunning {
		return "running"
	}

	return "unknown"
}

// writeEvent writes an event to the JSONL file
func (c *EventCollector) writeEvent(event *SchedulingEvent) {
	data, err := json.Marshal(event)
	if err != nil {
		fmt.Printf("Error marshaling event: %v\n", err)
		return
	}

	c.eventFile.Write(append(data, '\n'))
}

// Close closes the collector
func (c *EventCollector) Close() error {
	return c.eventFile.Close()
}

// GetEventCount returns the number of events collected
func (c *EventCollector) GetEventCount() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.eventCount
}
