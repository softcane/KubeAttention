/*
Package scheduler implements shadow mode for KubeAttention.

Shadow mode allows the scheduler to calculate and log recommendations
without actually affecting pod placement. This enables safe rollout
and comparison against the default scheduler.
*/
package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// ShadowModeConfig holds shadow mode configuration
type ShadowModeConfig struct {
	// Enabled toggles shadow mode
	Enabled bool

	// LogDecisionDiff logs when KubeAttention would have made a different decision
	LogDecisionDiff bool

	// AnnotatePods adds annotations to pods with recommendations
	AnnotatePods bool

	// MetricsEnabled exports shadow mode comparison metrics
	MetricsEnabled bool
}

// ShadowModeStats tracks shadow mode statistics
type ShadowModeStats struct {
	TotalDecisions     int64
	MatchingDecisions  int64
	DifferentDecisions int64
	BetterByBrain      int64 // Times Brain's choice would have been better
	WorseByBrain       int64 // Times Brain's choice would have been worse
}

// ShadowAnnotator handles pod annotation in shadow mode
type ShadowAnnotator struct {
	handle framework.Handle
	config ShadowModeConfig
	stats  ShadowModeStats
}

// NewShadowAnnotator creates a new shadow annotator
func NewShadowAnnotator(h framework.Handle, config ShadowModeConfig) *ShadowAnnotator {
	return &ShadowAnnotator{
		handle: h,
		config: config,
	}
}

// AnnotatePod adds shadow mode annotations to a pod
func (sa *ShadowAnnotator) AnnotatePod(
	ctx context.Context,
	pod *v1.Pod,
	recommendation *ShadowRecommendation,
	actualNode string,
) error {
	if !sa.config.AnnotatePods {
		return nil
	}

	// Build patch for annotations
	annotations := map[string]string{
		AnnotationRecommendedNode: recommendation.BestNode,
		AnnotationScore:           fmt.Sprintf("%d", recommendation.Score),
		AnnotationConfidence:      fmt.Sprintf("%.3f", recommendation.Confidence),
		AnnotationReasoning:       recommendation.Reasoning,
		"kubeattention.io/mode":   "shadow",
		"kubeattention.io/time":   time.Now().UTC().Format(time.RFC3339),
	}

	// Add decision comparison
	if recommendation.BestNode == actualNode {
		annotations["kubeattention.io/decision"] = "match"
	} else {
		annotations["kubeattention.io/decision"] = "differ"
		annotations["kubeattention.io/actual-node"] = actualNode
	}

	// Create JSON patch
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"annotations": annotations,
		},
	}

	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return fmt.Errorf("failed to marshal patch: %w", err)
	}

	// Apply patch
	_, err = sa.handle.ClientSet().CoreV1().Pods(pod.Namespace).Patch(
		ctx,
		pod.Name,
		types.MergePatchType,
		patchBytes,
		metav1.PatchOptions{},
	)

	return err
}

// RecordDecision records a scheduling decision for metrics
func (sa *ShadowAnnotator) RecordDecision(
	recommendedNode string,
	actualNode string,
) {
	sa.stats.TotalDecisions++

	if recommendedNode == actualNode {
		sa.stats.MatchingDecisions++
	} else {
		sa.stats.DifferentDecisions++

		if sa.config.LogDecisionDiff {
			fmt.Printf("🔍 Shadow Mode: Brain recommended %s, but scheduler chose %s\n",
				recommendedNode, actualNode)
		}
	}
}

// GetStats returns current shadow mode statistics
func (sa *ShadowAnnotator) GetStats() ShadowModeStats {
	return sa.stats
}

// MatchRate returns the percentage of matching decisions
func (sa *ShadowAnnotator) MatchRate() float64 {
	if sa.stats.TotalDecisions == 0 {
		return 0
	}
	return float64(sa.stats.MatchingDecisions) / float64(sa.stats.TotalDecisions) * 100
}
