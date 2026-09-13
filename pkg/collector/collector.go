// Package collector records Kubernetes scheduling decisions and measured outcomes.
package collector

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

const featureSchemaVersion = "node-pod-v1"

// SchedulingEvent is a durable scheduling decision or its later outcome update.
type SchedulingEvent struct {
	Timestamp          time.Time              `json:"timestamp"`
	EventID            string                 `json:"event_id"`
	EvidenceSource     string                 `json:"evidence_source"`
	PodUID             string                 `json:"pod_uid"`
	PodName            string                 `json:"pod_name"`
	PodNamespace       string                 `json:"pod_namespace"`
	PodLabels          map[string]string      `json:"pod_labels"`
	Priority           int32                  `json:"priority"`
	Criticality        string                 `json:"criticality"`
	WorkloadType       string                 `json:"workload_type"`
	CPURequestMilli    int64                  `json:"cpu_request_milli"`
	MemoryRequestBytes int64                  `json:"memory_request_bytes"`
	CandidateNodes     []string               `json:"candidate_nodes"`
	NodeTelemetry      map[string]NodeMetrics `json:"node_telemetry"`
	ChosenNode         string                 `json:"chosen_node"`
	SchedulerName      string                 `json:"scheduler_name"`
	Outcome            string                 `json:"outcome"`
	OutcomeTimestamp   time.Time              `json:"outcome_timestamp,omitempty"`
	P99LatencyMs       *float64               `json:"p99_latency_ms,omitempty"`
}

// NodeMetrics is one measured node sample with explicit availability.
type NodeMetrics struct {
	NodeName             string    `json:"node_name"`
	Timestamp            time.Time `json:"timestamp"`
	ObservationWindowMs  int64     `json:"observation_window_ms"`
	TelemetrySource      string    `json:"telemetry_source"`
	AvailableMetrics     []string  `json:"available_metrics"`
	DegradationReason    string    `json:"degradation_reason,omitempty"`
	SchemaVersion        string    `json:"schema_version"`
	CPUUtilization       float64   `json:"cpu_utilization"`
	CPUThrottleRate      float64   `json:"cpu_throttle_rate"`
	MemoryUtilization    float64   `json:"memory_utilization"`
	MemoryBandwidthGbps  float64   `json:"memory_bandwidth_gbps"`
	L3CacheMissRate      float64   `json:"l3_cache_miss_rate"`
	L3CacheOccupancyMB   float64   `json:"l3_cache_occupancy_mb"`
	DiskIOWaitMs         float64   `json:"disk_io_wait_ms"`
	DiskIOPS             float64   `json:"disk_iops"`
	NetworkRxPacketsSec  float64   `json:"network_rx_packets_sec"`
	NetworkTxPacketsSec  float64   `json:"network_tx_packets_sec"`
	NetworkDropRate      float64   `json:"network_drop_rate"`
	CostPerHour          float64   `json:"cost_per_hour"`
	ZoneDiversityScore   float64   `json:"zone_diversity_score"`
	SpotInterruptionRisk float64   `json:"spot_interruption_risk"`
	IsSpotInstance       bool      `json:"is_spot_instance"`
}

type EventCollector struct {
	client          kubernetes.Interface
	metricsClient   metricsv.Interface
	outputPath      string
	outcomeWaitTime time.Duration

	mu              sync.Mutex
	pendingOutcomes map[string]*SchedulingEvent
	completed       map[string]struct{}
	eventFile       *os.File
	eventCount      int64
}

// NewEventCollector constructs an in-cluster collector and restores durable state.
func NewEventCollector(outputPath string, outcomeWaitTime time.Duration) (*EventCollector, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("get in-cluster config: %w", err)
	}
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("create kubernetes client: %w", err)
	}
	metricsClient, err := metricsv.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("create metrics client: %w", err)
	}
	return newEventCollector(client, metricsClient, outputPath, outcomeWaitTime)
}

func newEventCollector(client kubernetes.Interface, metricsClient metricsv.Interface, outputPath string, outcomeWaitTime time.Duration) (*EventCollector, error) {
	if outcomeWaitTime <= 0 {
		return nil, fmt.Errorf("outcome wait time must be positive")
	}
	file, err := os.OpenFile(outputPath, os.O_CREATE|os.O_RDWR, 0o640)
	if err != nil {
		return nil, fmt.Errorf("open output file: %w", err)
	}
	collector := &EventCollector{
		client:          client,
		metricsClient:   metricsClient,
		outputPath:      outputPath,
		outcomeWaitTime: outcomeWaitTime,
		pendingOutcomes: make(map[string]*SchedulingEvent),
		completed:       make(map[string]struct{}),
		eventFile:       file,
	}
	if err := collector.restoreState(); err != nil {
		file.Close()
		return nil, err
	}
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		file.Close()
		return nil, fmt.Errorf("seek output file: %w", err)
	}
	return collector, nil
}

func (c *EventCollector) restoreState() error {
	if _, err := c.eventFile.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek event history: %w", err)
	}
	scanner := bufio.NewScanner(c.eventFile)
	buffer := make([]byte, 64*1024)
	scanner.Buffer(buffer, 10*1024*1024)
	for scanner.Scan() {
		var event SchedulingEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return fmt.Errorf("decode event history: %w", err)
		}
		if event.PodUID == "" {
			continue
		}
		if event.Outcome == "pending" {
			eventCopy := event
			c.pendingOutcomes[event.PodUID] = &eventCopy
			delete(c.completed, event.PodUID)
		} else {
			delete(c.pendingOutcomes, event.PodUID)
			c.completed[event.PodUID] = struct{}{}
		}
		c.eventCount++
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read event history: %w", err)
	}
	return nil
}

// Start watches pods until cancellation, reconnecting after watch closure or error.
func (c *EventCollector) Start(ctx context.Context) error {
	go c.checkOutcomes(ctx)
	backoff := time.Second
	for ctx.Err() == nil {
		err := c.watchOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		fmt.Fprintf(os.Stderr, "pod watch interrupted: %v; reconnecting in %s\n", err, backoff)
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-timer.C:
		}
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
	return nil
}

func (c *EventCollector) watchOnce(ctx context.Context) error {
	pods, err := c.client.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods: %w", err)
	}
	for index := range pods.Items {
		if err := c.observePod(ctx, &pods.Items[index]); err != nil {
			fmt.Fprintf(os.Stderr, "record scheduling decision: %v\n", err)
		}
	}
	watcher, err := c.client.CoreV1().Pods("").Watch(ctx, metav1.ListOptions{
		ResourceVersion:     pods.ResourceVersion,
		AllowWatchBookmarks: true,
	})
	if err != nil {
		return fmt.Errorf("watch pods: %w", err)
	}
	defer watcher.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case event, open := <-watcher.ResultChan():
			if !open {
				return fmt.Errorf("pod watch closed")
			}
			if event.Type == watch.Error {
				return apierrors.FromObject(event.Object)
			}
			pod, ok := event.Object.(*corev1.Pod)
			if !ok || (event.Type != watch.Added && event.Type != watch.Modified) {
				continue
			}
			if err := c.observePod(ctx, pod); err != nil {
				fmt.Fprintf(os.Stderr, "record scheduling decision: %v\n", err)
			}
		}
	}
}

func (c *EventCollector) observePod(ctx context.Context, pod *corev1.Pod) error {
	if pod.Spec.NodeName == "" || pod.UID == "" || pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return nil
	}
	uid := string(pod.UID)
	c.mu.Lock()
	_, pending := c.pendingOutcomes[uid]
	_, completed := c.completed[uid]
	c.mu.Unlock()
	if pending || completed {
		return nil
	}

	telemetry, candidateNodes, err := c.collectNodeTelemetry(ctx)
	if err != nil {
		return err
	}
	cpuMilli, memoryBytes := podRequests(pod)
	var priority int32
	if pod.Spec.Priority != nil {
		priority = *pod.Spec.Priority
	}
	event := &SchedulingEvent{
		Timestamp:          time.Now().UTC(),
		EventID:            uid,
		EvidenceSource:     "measured",
		PodUID:             uid,
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		PodLabels:          pod.Labels,
		Priority:           priority,
		Criticality:        pod.Annotations["kubeattention.io/criticality"],
		WorkloadType:       pod.Labels["kubeattention.io/workload-type"],
		CPURequestMilli:    cpuMilli,
		MemoryRequestBytes: memoryBytes,
		CandidateNodes:     candidateNodes,
		NodeTelemetry:      telemetry,
		ChosenNode:         pod.Spec.NodeName,
		SchedulerName:      pod.Spec.SchedulerName,
		Outcome:            "pending",
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.pendingOutcomes[uid]; exists {
		return nil
	}
	if _, exists := c.completed[uid]; exists {
		return nil
	}
	if err := c.writeEventLocked(event); err != nil {
		return err
	}
	c.pendingOutcomes[uid] = event
	c.eventCount++
	return nil
}

func podRequests(pod *corev1.Pod) (int64, int64) {
	var regularCPU, regularMemory int64
	for _, container := range pod.Spec.Containers {
		regularCPU += container.Resources.Requests.Cpu().MilliValue()
		regularMemory += container.Resources.Requests.Memory().Value()
	}
	var initCPU, initMemory int64
	for _, container := range pod.Spec.InitContainers {
		if value := container.Resources.Requests.Cpu().MilliValue(); value > initCPU {
			initCPU = value
		}
		if value := container.Resources.Requests.Memory().Value(); value > initMemory {
			initMemory = value
		}
	}
	if initCPU > regularCPU {
		regularCPU = initCPU
	}
	if initMemory > regularMemory {
		regularMemory = initMemory
	}
	if pod.Spec.Overhead != nil {
		regularCPU += pod.Spec.Overhead.Cpu().MilliValue()
		regularMemory += pod.Spec.Overhead.Memory().Value()
	}
	return regularCPU, regularMemory
}

func (c *EventCollector) collectNodeTelemetry(ctx context.Context) (map[string]NodeMetrics, []string, error) {
	nodes, err := c.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("list candidate nodes: %w", err)
	}
	type usage struct {
		cpuMilli  int64
		memory    int64
		timestamp time.Time
		window    time.Duration
	}
	usageByNode := make(map[string]usage, len(nodes.Items))
	metricsList, metricsErr := c.metricsClient.MetricsV1beta1().NodeMetricses().List(ctx, metav1.ListOptions{})
	if metricsErr == nil {
		for _, item := range metricsList.Items {
			usageByNode[item.Name] = usage{
				cpuMilli:  item.Usage.Cpu().MilliValue(),
				memory:    item.Usage.Memory().Value(),
				timestamp: item.Timestamp.Time,
				window:    item.Window.Duration,
			}
		}
	}

	telemetry := make(map[string]NodeMetrics, len(nodes.Items))
	candidateNodes := make([]string, 0, len(nodes.Items))
	for _, node := range nodes.Items {
		candidateNodes = append(candidateNodes, node.Name)
		sample := NodeMetrics{
			NodeName:          node.Name,
			SchemaVersion:     featureSchemaVersion,
			DegradationReason: "metrics.k8s.io sample unavailable",
			IsSpotInstance: node.Labels["kubernetes.io/lifecycle"] == "spot" ||
				node.Labels["karpenter.sh/capacity-type"] == "spot",
		}
		if measured, ok := usageByNode[node.Name]; ok {
			sample.Timestamp = measured.timestamp
			sample.ObservationWindowMs = measured.window.Milliseconds()
			sample.TelemetrySource = "metrics.k8s.io/v1beta1"
			sample.AvailableMetrics = []string{"cpu_utilization", "memory_utilization"}
			sample.DegradationReason = "optional hardware contention measurements unavailable"
			if capacity := node.Status.Allocatable.Cpu().MilliValue(); capacity > 0 {
				sample.CPUUtilization = clamp(float64(measured.cpuMilli) / float64(capacity))
			}
			if capacity := node.Status.Allocatable.Memory().Value(); capacity > 0 {
				sample.MemoryUtilization = clamp(float64(measured.memory) / float64(capacity))
			}
		}
		telemetry[node.Name] = sample
	}
	return telemetry, candidateNodes, nil
}

func clamp(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func (c *EventCollector) checkOutcomes(ctx context.Context) {
	interval := c.outcomeWaitTime / 4
	if interval > 30*time.Second {
		interval = 30 * time.Second
	}
	if interval < time.Second {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
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

func (c *EventCollector) evaluatePendingOutcomes(ctx context.Context) {
	now := time.Now().UTC()
	c.mu.Lock()
	due := make([]*SchedulingEvent, 0, len(c.pendingOutcomes))
	for _, event := range c.pendingOutcomes {
		if now.Sub(event.Timestamp) >= c.outcomeWaitTime {
			due = append(due, event)
		}
	}
	c.mu.Unlock()

	for _, event := range due {
		pod, err := c.client.CoreV1().Pods(event.PodNamespace).Get(ctx, event.PodName, metav1.GetOptions{})
		outcome := ""
		switch {
		case apierrors.IsNotFound(err):
			outcome = "deleted"
		case err != nil:
			fmt.Fprintf(os.Stderr, "read outcome for %s/%s: %v\n", event.PodNamespace, event.PodName, err)
			continue
		default:
			outcome = evaluatePodOutcome(pod)
		}
		if outcome == "pending" {
			continue
		}
		completed := *event
		completed.Outcome = outcome
		completed.OutcomeTimestamp = now

		c.mu.Lock()
		current, exists := c.pendingOutcomes[event.PodUID]
		if !exists || current.EventID != event.EventID {
			c.mu.Unlock()
			continue
		}
		if err := c.writeEventLocked(&completed); err != nil {
			fmt.Fprintf(os.Stderr, "write outcome for %s/%s: %v\n", event.PodNamespace, event.PodName, err)
			c.mu.Unlock()
			continue
		}
		delete(c.pendingOutcomes, event.PodUID)
		c.completed[event.PodUID] = struct{}{}
		c.eventCount++
		c.mu.Unlock()
	}
}

func evaluatePodOutcome(pod *corev1.Pod) string {
	if pod.Status.Reason == "Evicted" {
		return "evicted"
	}
	statuses := append([]corev1.ContainerStatus(nil), pod.Status.InitContainerStatuses...)
	statuses = append(statuses, pod.Status.ContainerStatuses...)
	for _, status := range statuses {
		if status.State.Terminated != nil && status.State.Terminated.Reason == "OOMKilled" {
			return "oom_killed"
		}
		if status.LastTerminationState.Terminated != nil && status.LastTerminationState.Terminated.Reason == "OOMKilled" {
			return "oom_killed"
		}
	}
	for _, status := range statuses {
		if status.RestartCount > 0 {
			return "restarted"
		}
	}
	switch pod.Status.Phase {
	case corev1.PodRunning:
		return "running"
	case corev1.PodSucceeded:
		return "succeeded"
	case corev1.PodFailed:
		return "failed"
	default:
		return "pending"
	}
}

func (c *EventCollector) writeEventLocked(event *SchedulingEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}
	if _, err := c.eventFile.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("append event: %w", err)
	}
	if err := c.eventFile.Sync(); err != nil {
		return fmt.Errorf("sync event: %w", err)
	}
	return nil
}

func (c *EventCollector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.eventFile.Sync(); err != nil {
		return err
	}
	return c.eventFile.Close()
}

func (c *EventCollector) GetEventCount() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.eventCount
}
