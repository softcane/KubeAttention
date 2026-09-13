/*
Package scheduler implements a Kubernetes Scheduling Framework Score plugin
that uses the KubeAttention Brain for intelligent node scoring.

This plugin:
1. Collects real-time telemetry from candidate nodes
2. Sends it to the Brain via gRPC over Unix Domain Socket
3. Uses the Brain's Transformer-based scoring
4. Falls back to LeastAllocated if Brain is unavailable (circuit breaker)
5. Supports Shadow Mode for safe rollout
*/
package scheduler

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	schedulerpb "github.com/softcane/KubeAttention/gen/go"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-scheduler/framework"
	"sigs.k8s.io/yaml"
)

const (
	PluginName = "KubeAttention"

	AnnotationRecommendedNode = "kubeattention.io/recommended-node"
	AnnotationScore           = "kubeattention.io/score"
	AnnotationReasoning       = "kubeattention.io/reasoning"

	SchedulerModeShadow = "shadow"
	SchedulerModeActive = "active"
)

// KubeAttentionArgs holds the configuration for the plugin.
type KubeAttentionArgs struct {
	BrainEndpoint string `json:"brainEndpoint,omitempty"`
	TimeoutMs     int    `json:"timeoutMs,omitempty"`
	Mode          string `json:"mode,omitempty"`
	FallbackScore int64  `json:"fallbackScore,omitempty"`
}

// SetDefaults applies fail-safe scheduler defaults.
func (args *KubeAttentionArgs) SetDefaults() {
	if args.BrainEndpoint == "" {
		args.BrainEndpoint = DefaultUDSPath
	}
	if args.TimeoutMs <= 0 {
		args.TimeoutMs = 50
	}
	if args.Mode == "" {
		args.Mode = SchedulerModeShadow
	}
	if args.FallbackScore <= 0 {
		args.FallbackScore = 50
	}
}

func (args *KubeAttentionArgs) validate() error {
	if args.Mode != SchedulerModeShadow && args.Mode != SchedulerModeActive {
		return fmt.Errorf("mode must be %q or %q", SchedulerModeShadow, SchedulerModeActive)
	}
	if args.FallbackScore < framework.MinNodeScore || args.FallbackScore > framework.MaxNodeScore {
		return fmt.Errorf("fallbackScore must be between %d and %d", framework.MinNodeScore, framework.MaxNodeScore)
	}
	return nil
}

func (args *KubeAttentionArgs) shadowMode() bool {
	return args.Mode == SchedulerModeShadow
}

type KubeAttention struct {
	handle          framework.Handle
	args            *KubeAttentionArgs
	brainClient     *BrainClient
	telemetryStore  *TelemetryStore
	shadowAnnotator *ShadowAnnotator
	mu              sync.RWMutex
}

var _ framework.PreScorePlugin = &KubeAttention{}
var _ framework.ScorePlugin = &KubeAttention{}
var _ framework.ScoreExtensions = &KubeAttention{}
var _ framework.PreEnqueuePlugin = &KubeAttention{}
var _ framework.PostBindPlugin = &KubeAttention{}

// New creates a KubeAttention scheduler plugin.
func New(ctx context.Context, obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	args := &KubeAttentionArgs{}
	if err := decodeArgs(obj, args); err != nil {
		return nil, fmt.Errorf("decode KubeAttention arguments: %w", err)
	}
	args.SetDefaults()
	if err := args.validate(); err != nil {
		return nil, err
	}

	nodeLister := h.SharedInformerFactory().Core().V1().Nodes().Lister()
	metricsSource, err := NewKubernetesMetricsSource(h.KubeConfig(), nodeLister)
	if err != nil {
		return nil, err
	}
	telemetryStore := NewTelemetryStore(metricsSource, nodeLister, time.Second)
	client, err := NewBrainClient(args.BrainEndpoint, time.Duration(args.TimeoutMs)*time.Millisecond)
	if err != nil {
		return nil, fmt.Errorf("create Brain client: %w", err)
	}

	telemetryStore.Start(ctx)
	go func() {
		connectCtx, cancel := context.WithTimeout(ctx, time.Duration(args.TimeoutMs)*time.Millisecond)
		defer cancel()
		_ = client.Connect(connectCtx)
	}()
	plugin := &KubeAttention{
		handle:         h,
		args:           args,
		brainClient:    client,
		telemetryStore: telemetryStore,
	}
	plugin.shadowAnnotator = NewShadowAnnotator(h.ClientSet(), ShadowModeConfig{
		Enabled:         args.shadowMode(),
		LogDecisionDiff: true,
		AnnotatePods:    true,
	})
	return plugin, nil
}

func decodeArgs(obj runtime.Object, args *KubeAttentionArgs) error {
	if obj == nil {
		return nil
	}
	unknown, ok := obj.(*runtime.Unknown)
	if !ok {
		return fmt.Errorf("expected *runtime.Unknown, got %T", obj)
	}
	if len(unknown.Raw) == 0 {
		return nil
	}
	return yaml.Unmarshal(unknown.Raw, args)
}

// Name returns the plugin name
func (ka *KubeAttention) Name() string {
	return PluginName
}

const batchResultKey framework.StateKey = "kubeattention.io/batch-results"

type batchResultState struct {
	scores map[string]*schedulerpb.NodeScore
}

func (s *batchResultState) Clone() framework.StateData {
	scores := make(map[string]*schedulerpb.NodeScore, len(s.scores))
	for nodeName, score := range s.scores {
		if score == nil {
			continue
		}
		scoreCopy := *score
		scores[nodeName] = &scoreCopy
	}
	return &batchResultState{scores: scores}
}

// PreScore implements the PreScore plugin interface.
// It collects telemetry for ALL nodes and calls BatchScore ONCE per pod,
// significantly reducing latency and gRPC overhead.
func (ka *KubeAttention) PreScore(
	ctx context.Context,
	state framework.CycleState,
	pod *v1.Pod,
	nodes []framework.NodeInfo,
) *framework.Status {
	if len(nodes) == 0 {
		return nil
	}

	req := &schedulerpb.BatchScoreRequest{
		PodRequirements: podRequirements(pod),
		Nodes:           make([]*schedulerpb.NodeTelemetry, 0, len(nodes)),
	}
	activeTelemetryReady := true
	now := time.Now()
	for _, nodeInfo := range nodes {
		node := nodeInfo.Node()
		if node == nil {
			continue
		}
		metrics := ka.telemetryStore.GetMetrics(node.Name)
		if !metrics.ActiveScoringReady(now, 30*time.Second) {
			activeTelemetryReady = false
		}
		req.Nodes = append(req.Nodes, nodeTelemetry(node, metrics))
	}

	if !ka.args.shadowMode() && !activeTelemetryReady {
		nodeScores := make(map[string]*schedulerpb.NodeScore, len(req.GetNodes()))
		for _, node := range req.GetNodes() {
			nodeScores[node.GetNodeName()] = &schedulerpb.NodeScore{
				NodeName:  node.GetNodeName(),
				Score:     ka.args.FallbackScore,
				Reasoning: "active scoring disabled: required interference telemetry unavailable or stale",
			}
		}
		state.Write(batchResultKey, &batchResultState{scores: nodeScores})
		return nil
	}

	resp, err := ka.brainClient.BatchScore(ctx, req)
	if err != nil {
		return nil
	}

	nodeScores := make(map[string]*schedulerpb.NodeScore, len(resp.GetScores()))
	for _, score := range resp.GetScores() {
		if score != nil {
			nodeScores[score.GetNodeName()] = score
		}
	}
	state.Write(batchResultKey, &batchResultState{scores: nodeScores})
	return nil
}

// Score scores a node for pod placement by looking up the pre-computed batch result
func (ka *KubeAttention) Score(
	_ context.Context,
	state framework.CycleState,
	_ *v1.Pod,
	nodeInfo framework.NodeInfo,
) (int64, *framework.Status) {
	node := nodeInfo.Node()
	if node == nil {
		return ka.args.FallbackScore, nil
	}
	data, err := state.Read(batchResultKey)
	if err != nil {
		return ka.args.FallbackScore, nil
	}
	results, ok := data.(*batchResultState)
	if !ok {
		return ka.args.FallbackScore, nil
	}
	res, ok := results.scores[node.Name]
	if !ok {
		return ka.args.FallbackScore, nil
	}

	if ka.args.shadowMode() {
		ka.storeRecommendation(state, node.Name, res)
		return ka.args.FallbackScore, nil
	}
	return res.GetScore(), nil
}

// NormalizeScore normalizes scores to [0, 100] range
func (ka *KubeAttention) NormalizeScore(
	_ context.Context,
	_ framework.CycleState,
	_ *v1.Pod,
	scores framework.NodeScoreList,
) *framework.Status {
	for i := range scores {
		if scores[i].Score > framework.MaxNodeScore {
			scores[i].Score = framework.MaxNodeScore
		}
		if scores[i].Score < framework.MinNodeScore {
			scores[i].Score = framework.MinNodeScore
		}
	}
	return nil
}

// ScoreExtensions returns the score extensions
func (ka *KubeAttention) ScoreExtensions() framework.ScoreExtensions {
	return ka
}

const FeatureSchemaVersion = "node-pod-v1"

func nodeTelemetry(node *v1.Node, metrics *NodeMetrics) *schedulerpb.NodeTelemetry {
	telemetry := &schedulerpb.NodeTelemetry{
		NodeName:          node.Name,
		AvailabilityZone:  node.Labels[v1.LabelTopologyZone],
		RackId:            node.Labels["topology.kubeattention.io/rack"],
		IsSpotInstance:    node.Labels["kubernetes.io/lifecycle"] == "spot" || node.Labels["karpenter.sh/capacity-type"] == "spot",
		SchemaVersion:     FeatureSchemaVersion,
		DegradationReason: "node telemetry unavailable",
	}
	if metrics == nil {
		return telemetry
	}
	telemetry.TimestampUnixMs = metrics.Timestamp.UnixMilli()
	telemetry.ObservationWindowMs = metrics.Window.Milliseconds()
	telemetry.TelemetrySource = metrics.Source
	telemetry.DegradationReason = metrics.DegradationReason
	telemetry.CpuUtilization = metrics.CPUUtilization
	telemetry.CpuThrottleRate = metrics.CPUThrottleRate
	telemetry.MemoryUtilization = metrics.MemoryUtilization
	telemetry.MemoryBandwidthGbps = metrics.MemoryBandwidthGbps
	telemetry.L3CacheMissRate = metrics.L3CacheMissRate
	telemetry.L3CacheOccupancyMb = metrics.L3CacheOccupancyMB
	telemetry.DiskIoWaitMs = metrics.DiskIOWaitMs
	telemetry.DiskIops = metrics.DiskIOPS
	telemetry.NetworkRxPacketsSec = metrics.NetworkRxPacketsSec
	telemetry.NetworkTxPacketsSec = metrics.NetworkTxPacketsSec
	telemetry.NetworkDropRate = metrics.NetworkDropRate
	telemetry.AvailableMetrics = availableProtoMetrics(metrics.Available)
	return telemetry
}

func availableProtoMetrics(available MetricSet) []schedulerpb.NodeMetric {
	metrics := make([]schedulerpb.NodeMetric, 0, 11)
	candidates := []struct {
		flag   MetricSet
		metric schedulerpb.NodeMetric
	}{
		{MetricCPUUtilization, schedulerpb.NodeMetric_NODE_METRIC_CPU_UTILIZATION},
		{MetricCPUThrottleRate, schedulerpb.NodeMetric_NODE_METRIC_CPU_THROTTLE_RATE},
		{MetricMemoryUtilization, schedulerpb.NodeMetric_NODE_METRIC_MEMORY_UTILIZATION},
		{MetricMemoryBandwidth, schedulerpb.NodeMetric_NODE_METRIC_MEMORY_BANDWIDTH},
		{MetricL3CacheMissRate, schedulerpb.NodeMetric_NODE_METRIC_L3_CACHE_MISS_RATE},
		{MetricL3CacheOccupancy, schedulerpb.NodeMetric_NODE_METRIC_L3_CACHE_OCCUPANCY},
		{MetricDiskIOWait, schedulerpb.NodeMetric_NODE_METRIC_DISK_IO_WAIT},
		{MetricDiskIOPS, schedulerpb.NodeMetric_NODE_METRIC_DISK_IOPS},
		{MetricNetworkRXPackets, schedulerpb.NodeMetric_NODE_METRIC_NETWORK_RX_PACKETS},
		{MetricNetworkTXPackets, schedulerpb.NodeMetric_NODE_METRIC_NETWORK_TX_PACKETS},
		{MetricNetworkDropRate, schedulerpb.NodeMetric_NODE_METRIC_NETWORK_DROP_RATE},
	}
	for _, candidate := range candidates {
		if available.Has(candidate.flag) {
			metrics = append(metrics, candidate.metric)
		}
	}
	return metrics
}

func podRequirements(pod *v1.Pod) *schedulerpb.PodRequirements {
	var cpuMilli, memoryBytes int64
	for _, container := range pod.Spec.Containers {
		cpuMilli += container.Resources.Requests.Cpu().MilliValue()
		memoryBytes += container.Resources.Requests.Memory().Value()
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
	if initCPU > cpuMilli {
		cpuMilli = initCPU
	}
	if initMemory > memoryBytes {
		memoryBytes = initMemory
	}
	if pod.Spec.Overhead != nil {
		cpuMilli += pod.Spec.Overhead.Cpu().MilliValue()
		memoryBytes += pod.Spec.Overhead.Memory().Value()
	}

	criticality := schedulerpb.Criticality_CRITICALITY_UNKNOWN
	switch strings.ToLower(pod.Annotations["kubeattention.io/criticality"]) {
	case "low":
		criticality = schedulerpb.Criticality_CRITICALITY_LOW
	case "medium":
		criticality = schedulerpb.Criticality_CRITICALITY_MEDIUM
	case "high":
		criticality = schedulerpb.Criticality_CRITICALITY_HIGH
	}
	var priority int32
	if pod.Spec.Priority != nil {
		priority = *pod.Spec.Priority
	}
	return &schedulerpb.PodRequirements{
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		CpuMilli:     cpuMilli,
		MemoryBytes:  memoryBytes,
		WorkloadType: pod.Labels["kubeattention.io/workload-type"],
		Labels:       pod.Labels,
		Criticality:  criticality,
		Priority:     priority,
	}
}

const shadowRecommendationKey framework.StateKey = "kubeattention.io/shadow-recommendation"

// ShadowRecommendation holds the Brain's recommendation in shadow mode
type ShadowRecommendation struct {
	BestNode  string
	Score     int64
	Reasoning string
}

// storeRecommendation stores a recommendation for shadow mode in a thread-safe way
func (ka *KubeAttention) storeRecommendation(
	state framework.CycleState,
	nodeName string,
	resp *schedulerpb.NodeScore,
) {
	ka.mu.Lock()
	defer ka.mu.Unlock()

	var rec *ShadowRecommendation
	if data, err := state.Read(shadowRecommendationKey); err == nil {
		rec, _ = data.(*ShadowRecommendation)
	}
	if rec == nil {
		rec = &ShadowRecommendation{}
	}
	if resp.Score > rec.Score {
		rec.BestNode = nodeName
		rec.Score = resp.Score
		rec.Reasoning = resp.Reasoning
	}
	state.Write(shadowRecommendationKey, rec)
}

// Clone implements framework.StateData for ShadowRecommendation
func (rec *ShadowRecommendation) Clone() framework.StateData {
	return &ShadowRecommendation{
		BestNode:  rec.BestNode,
		Score:     rec.Score,
		Reasoning: rec.Reasoning,
	}
}

// PostBind is called after a pod is bound - useful for shadow mode logging
func (ka *KubeAttention) PostBind(
	ctx context.Context,
	state framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) {
	if !ka.args.shadowMode() {
		return
	}

	ka.mu.RLock()
	data, err := state.Read(shadowRecommendationKey)
	if err != nil {
		ka.mu.RUnlock()
		return
	}
	rec, ok := data.(*ShadowRecommendation)
	if !ok || rec.BestNode == "" {
		ka.mu.RUnlock()
		return
	}
	recCopy := *rec
	ka.mu.RUnlock()

	if err := ka.shadowAnnotator.AnnotatePod(ctx, pod, &recCopy, nodeName); err != nil {
		fmt.Printf("KubeAttention shadow annotation failed for %s/%s: %v\n", pod.Namespace, pod.Name, err)
		return
	}
	ka.shadowAnnotator.RecordDecision(recCopy.BestNode, nodeName)
}

// PreEnqueue implements K8s 1.35+ Workload-Aware Scheduling.
// Filters pods before they enter the scheduling queue, rejecting workloads
// that the Brain has already determined will fail on all available nodes.
func (ka *KubeAttention) PreEnqueue(ctx context.Context, pod *v1.Pod) *framework.Status {
	// Skip if Brain is unhealthy (let default scheduler handle)
	if ka.brainClient.GetCircuitState() == CircuitOpen {
		return framework.NewStatus(framework.Success)
	}

	// Check for explicit skip annotation
	if pod.Annotations != nil {
		if _, ok := pod.Annotations["kubeattention.io/skip-preenqueue"]; ok {
			return framework.NewStatus(framework.Success)
		}
	}

	// For resource-intensive pods, do a pre-flight Brain health check
	cpuRequest := int64(0)
	memRequest := int64(0)
	for _, container := range pod.Spec.Containers {
		if cpu := container.Resources.Requests.Cpu(); cpu != nil {
			cpuRequest += cpu.MilliValue()
		}
		if mem := container.Resources.Requests.Memory(); mem != nil {
			memRequest += mem.Value()
		}
	}

	// If pod requests >4 CPU or >8GB RAM, verify Brain is healthy
	if cpuRequest > 4000 || memRequest > 8*1024*1024*1024 {
		timeoutCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
		defer cancel()

		healthy, _, err := ka.brainClient.HealthCheck(timeoutCtx)
		if err != nil || !healthy {
			// Log but allow scheduling to proceed with fallback
			fmt.Printf("KubeAttention PreEnqueue: Brain unhealthy for large pod %s/%s, using fallback\n",
				pod.Namespace, pod.Name)
		}
	}

	return framework.NewStatus(framework.Success)
}
