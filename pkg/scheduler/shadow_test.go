package scheduler

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestShadowAnnotatorWritesRecommendation(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "probe", Namespace: "default"}}
	client := fake.NewSimpleClientset(pod)
	annotator := NewShadowAnnotator(client, ShadowModeConfig{Enabled: true, AnnotatePods: true})
	recommendation := &ShadowRecommendation{BestNode: "node-b", Score: 87, Reasoning: "lower contention"}

	if err := annotator.AnnotatePod(context.Background(), pod, recommendation, "node-a"); err != nil {
		t.Fatalf("annotate pod: %v", err)
	}
	updated, err := client.CoreV1().Pods("default").Get(context.Background(), "probe", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if got := updated.Annotations[AnnotationRecommendedNode]; got != "node-b" {
		t.Fatalf("recommended node = %q, want node-b", got)
	}
	if got := updated.Annotations[AnnotationScore]; got != "87" {
		t.Fatalf("score = %q, want 87", got)
	}
	if got := updated.Annotations["kubeattention.io/decision"]; got != "differ" {
		t.Fatalf("decision = %q, want differ", got)
	}
}
