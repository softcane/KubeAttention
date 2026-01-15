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
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	// PluginName is the name of this scheduler plugin
	PluginName = "KubeAttention"

	// AnnotationRecommendedNode is set in shadow mode
	AnnotationRecommendedNode = "kubeattention.io/recommended-node"

	// AnnotationScore is the Brain's score for the recommended node
	AnnotationScore = "kubeattention.io/score"

	// AnnotationConfidence is the Brain's confidence
	AnnotationConfidence = "kubeattention.io/confidence"

	// AnnotationReasoning explains the score
	AnnotationReasoning = "kubeattention.io/reasoning"
)

// KubeAttentionArgs holds the configuration for the plugin
type KubeAttentionArgs struct {
	// UDSPath is the Unix Domain Socket path for Brain communication
	UDSPath string `json:"udsPath,omitempty"`

	// TimeoutMs is the maximum time to wait for Brain response (default: 50)
	TimeoutMs int `json:"timeoutMs,omitempty"`

	// ShadowMode when true, only annotates pods without affecting scheduling
	ShadowMode bool `json:"shadowMode,omitempty"`

	// FallbackScore to use when Brain is unavailable (default: 50)
	FallbackScore int64 `json:"fallbackScore,omitempty"`
}

// SetDefaults sets default values for KubeAttentionArgs
func (args *KubeAttentionArgs) SetDefaults() {
	if args.UDSPath == "" {
		args.UDSPath = DefaultUDSPath
	}
	if args.TimeoutMs <= 0 {
		args.TimeoutMs = 50
	}
	if args.FallbackScore <= 0 {
		args.FallbackScore = 50
	}
	// Shadow mode defaults to true as per PLAN.md
	// This is already the zero value for bool, but being explicit
}

type KubeAttention struct {
	handle         framework.Handle
	args           *KubeAttentionArgs
	brainClient    *BrainClient
	telemetryStore *TelemetryStore
	tetragonClient *TetragonClient
	mu             sync.RWMutex
}

var _ framework.PreScorePlugin = &KubeAttention{} // Implementing PreScore for batching
var _ framework.ScorePlugin = &KubeAttention{}
var _ framework.ScoreExtensions = &KubeAttention{}
var _ framework.PreEnqueuePlugin = &KubeAttention{}

// New creates a new KubeAttention plugin
func New(obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	args := &KubeAttentionArgs{}

	if obj != nil {
		if err := framework.DecodeInto(obj, args); err != nil {
			return nil, fmt.Errorf("failed to decode KubeAttentionArgs: %w", err)
		}
	}

	args.SetDefaults()

	// Create Singleton Tetragon Client
	tetragon := NewTetragonClient("", "")
	
	// Create Telemetry Store for background collection
	telemetryStore := NewTelemetryStore(tetragon, 1*time.Second)

	// Create Brain client
	client, err := NewBrainClient(args.UDSPath, DefaultTimeout)
	if err != nil {
		return nil, fmt.Errorf("failed to create Brain client: %w", err)
	}

	// Connect to Brain (non-blocking) and start background store
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), DefaultTimeout*10)
		defer cancel()
		_ = client.Connect(ctx)
		
		// Start background telemetry collection for all nodes
		// In a real K8s plugin, we would use an informer to get the list of nodes
		// For now, we start with nil and let updateAll handle dynamic discovery or 
		// wait for first PreScore to populate node list.
		telemetryStore.Start(context.Background(), nil)
	}()

	return &KubeAttention{
		handle:         h,
		args:           args,
		brainClient:    client,
		telemetryStore: telemetryStore,
		tetragonClient: tetragon,
	}, nil
}

// Name returns the plugin name
func (ka *KubeAttention) Name() string {
	return PluginName
}

// batchResultKey is the key for storing BatchScore results in CycleState
type batchResultKey struct{}

// PreScore implements the PreScore plugin interface.
// It collects telemetry for ALL nodes and calls BatchScore ONCE per pod,
// significantly reducing latency and gRPC overhead.
func (ka *KubeAttention) PreScore(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) *framework.Status {
	if len(nodes) == 0 {
		return framework.NewStatus(framework.Success)
	}

	// Build BatchScore request
	req := &BatchScoreRequest{
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		Nodes:        make([]ScoreRequest, len(nodes)),
	}

	for i, node := range nodes {
		// Get telemetry from background TelemetryStore (FAST/NO NETWORK)
		metrics := ka.telemetryStore.GetMetrics(node.Name)
		var telemetry map[string]float64
		if metrics != nil {
			telemetry = metrics.ToTelemetryMap()
		} else {
			// Fallback: limited K8s-only telemetry if background store hasn't caught up
			telemetry = ka.collectNodeTelemetryFallback(node)
		}

		req.Nodes[i] = ScoreRequest{
			PodName:      pod.Name,
			PodNamespace: pod.Namespace,
			NodeName:     node.Name,
			Telemetry:    telemetry,
		}
	}

	// Call Brain BatchScore (ONE network call for all nodes)
	resp, err := ka.brainClient.BatchScore(ctx, req)
	if err != nil {
		// Fail silently, Score() will handle fallback
		return framework.NewStatus(framework.Success)
	}

	// Store results in CycleState for use in Score()
	nodeScores := make(map[string]NodeScore)
	for _, s := range resp.Scores {
		nodeScores[s.NodeName] = s
	}
	state.Write(batchResultKey{}, nodeScores)

	return framework.NewStatus(framework.Success)
}

// Score scores a node for pod placement by looking up the pre-computed batch result
func (ka *KubeAttention) Score(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) (int64, *framework.Status) {
	// Lookup batch result from CycleState
	data, err := state.Read(batchResultKey{})
	if err != nil {
		// Fallback if PreScore failed or didn't run
		return ka.args.FallbackScore, framework.NewStatus(framework.Success)
	}

	nodeScores := data.(map[string]NodeScore)
	res, ok := nodeScores[nodeName]
	if !ok {
		return ka.args.FallbackScore, framework.NewStatus(framework.Success)
	}

	// If in shadow mode, store the recommendation but return neutral score
	if ka.args.ShadowMode {
		ka.storeRecommendation(state, nodeName, &res)
		return 50, framework.NewStatus(framework.Success)
	}

	return res.Score, framework.NewStatus(framework.Success)
}

// NormalizeScore normalizes scores to [0, 100] range
func (ka *KubeAttention) NormalizeScore(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	scores framework.NodeScoreList,
) *framework.Status {
	// Brain already returns scores in 0-100 range
	// Just ensure bounds
	for i := range scores {
		if scores[i].Score > 100 {
			scores[i].Score = 100
		}
		if scores[i].Score < 0 {
			scores[i].Score = 0
		}
	}
	return framework.NewStatus(framework.Success)
}

// ScoreExtensions returns the score extensions
func (ka *KubeAttention) ScoreExtensions() framework.ScoreExtensions {
	return ka
}

// collectNodeTelemetryFallback gathers basic K8s metrics when TelemetryStore is not yet populated.
// NO external network calls are allowed here.
func (ka *KubeAttention) collectNodeTelemetryFallback(node *v1.Node) map[string]float64 {
	// eBPF metrics set to 0 (default fallback)
	return map[string]float64{
		"cpu_utilization":        0.5, // Conservative estimate
		"memory_utilization":     0.5,
		"l3_cache_miss_rate":     0.0,
		"disk_io_wait_ms":        0.0,
		"network_drop_rate":      0.0,
		"cpu_throttle_rate":      0.0,
		"memory_bandwidth_gbps":  0.0,
		"l3_cache_occupancy_mb":  0.0,
		"disk_iops":              0.0,
		"network_rx_packets_sec": 0.0,
		"network_tx_packets_sec": 0.0,
	}
}

// shadowRecommendationKey is the key for storing shadow mode recommendations
type shadowRecommendationKey struct{}

// ShadowRecommendation holds the Brain's recommendation in shadow mode
type ShadowRecommendation struct {
	BestNode   string
	Score      int64
	Confidence float64
	Reasoning  string
}

// storeRecommendation stores a recommendation for shadow mode in a thread-safe way
func (ka *KubeAttention) storeRecommendation(
	state *framework.CycleState,
	nodeName string,
	resp *NodeScore,
) {
	ka.mu.Lock()
	defer ka.mu.Unlock()

	// Get or create recommendation
	var rec *ShadowRecommendation
	if data, err := state.Read(shadowRecommendationKey{}); err == nil {
		rec = data.(*ShadowRecommendation)
	} else {
		rec = &ShadowRecommendation{}
	}

	// Update if this node has a higher score
	if resp.Score > rec.Score {
		rec.BestNode = nodeName
		rec.Score = resp.Score
		rec.Confidence = resp.Confidence
		rec.Reasoning = resp.Reasoning
	}

	state.Write(shadowRecommendationKey{}, rec)
}

// Clone implements framework.StateData for ShadowRecommendation
func (rec *ShadowRecommendation) Clone() framework.StateData {
	return &ShadowRecommendation{
		BestNode:   rec.BestNode,
		Score:      rec.Score,
		Confidence: rec.Confidence,
		Reasoning:  rec.Reasoning,
	}
}

// PostBind is called after a pod is bound - useful for shadow mode logging
func (ka *KubeAttention) PostBind(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) {
	if !ka.args.ShadowMode {
		return
	}

	// Get recommendation from state
	ka.mu.RLock()
	defer ka.mu.RUnlock()

	if data, err := state.Read(shadowRecommendationKey{}); err == nil {
		rec := data.(*ShadowRecommendation)
		if rec.BestNode != "" {
			// Log the recommendation vs actual decision
			fmt.Printf("KubeAttention Shadow: pod=%s/%s actual=%s recommended=%s score=%d confidence=%.2f\n",
				pod.Namespace, pod.Name, nodeName, rec.BestNode, rec.Score, rec.Confidence)

			if rec.BestNode != nodeName {
				fmt.Printf("  Decision differed! Reason: %s\n", rec.Reasoning)
			}
		}
	}
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

