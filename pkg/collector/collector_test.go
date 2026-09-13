package collector

import (
	"context"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	kubernetesfake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

func TestCollectorReconnectsClosedPodWatch(t *testing.T) {
	client := kubernetesfake.NewSimpleClientset()
	var watchCount atomic.Int32
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	client.Fake.PrependWatchReactor("pods", func(k8stesting.Action) (bool, watch.Interface, error) {
		count := watchCount.Add(1)
		watcher := watch.NewRaceFreeFake()
		watcher.Stop()
		if count == 2 {
			cancel()
		}
		return true, watcher, nil
	})
	collector, err := newEventCollector(
		client,
		metricsfake.NewSimpleClientset(),
		filepath.Join(t.TempDir(), "events.jsonl"),
		time.Second,
	)
	if err != nil {
		t.Fatalf("create collector: %v", err)
	}
	defer collector.Close()

	if err := collector.Start(ctx); err != nil {
		t.Fatalf("start collector: %v", err)
	}
	if got := watchCount.Load(); got < 2 {
		t.Fatalf("watch attempts = %d, want at least 2", got)
	}
}

func TestCollectorRestoresPendingAndCompletedDeduplication(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	client := kubernetesfake.NewSimpleClientset()
	metrics := metricsfake.NewSimpleClientset()
	collector, err := newEventCollector(client, metrics, path, time.Minute)
	if err != nil {
		t.Fatalf("create collector: %v", err)
	}
	pending := &SchedulingEvent{
		Timestamp:      time.Now().UTC(),
		EventID:        "uid-1",
		PodUID:         "uid-1",
		PodName:        "pod-a",
		PodNamespace:   "default",
		EvidenceSource: "measured",
		Outcome:        "pending",
	}
	collector.mu.Lock()
	if err := collector.writeEventLocked(pending); err != nil {
		collector.mu.Unlock()
		t.Fatalf("write pending: %v", err)
	}
	collector.pendingOutcomes[pending.PodUID] = pending
	collector.mu.Unlock()
	if err := collector.Close(); err != nil {
		t.Fatalf("close collector: %v", err)
	}

	restored, err := newEventCollector(client, metrics, path, time.Minute)
	if err != nil {
		t.Fatalf("restore collector: %v", err)
	}
	if _, ok := restored.pendingOutcomes["uid-1"]; !ok {
		t.Fatal("pending outcome was not restored")
	}
	completed := *pending
	completed.Outcome = "running"
	restored.mu.Lock()
	if err := restored.writeEventLocked(&completed); err != nil {
		restored.mu.Unlock()
		t.Fatalf("write completed: %v", err)
	}
	restored.mu.Unlock()
	if err := restored.Close(); err != nil {
		t.Fatalf("close restored collector: %v", err)
	}

	final, err := newEventCollector(client, metrics, path, time.Minute)
	if err != nil {
		t.Fatalf("restore completed collector: %v", err)
	}
	defer final.Close()
	if _, ok := final.pendingOutcomes["uid-1"]; ok {
		t.Fatal("completed event restored as pending")
	}
	if _, ok := final.completed["uid-1"]; !ok {
		t.Fatal("completed UID missing from durable deduplication state")
	}
}

func TestEvaluatePodOutcomeClassification(t *testing.T) {
	tests := []struct {
		name string
		pod  *corev1.Pod
		want string
	}{
		{
			name: "eviction",
			pod:  &corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodFailed, Reason: "Evicted"}},
			want: "evicted",
		},
		{
			name: "current OOM",
			pod: &corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{
				State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "OOMKilled"}},
			}}}},
			want: "oom_killed",
		},
		{
			name: "previous init OOM",
			pod: &corev1.Pod{Status: corev1.PodStatus{InitContainerStatuses: []corev1.ContainerStatus{{
				LastTerminationState: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "OOMKilled"}},
				RestartCount:         1,
			}}}},
			want: "oom_killed",
		},
		{
			name: "restart",
			pod: &corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, ContainerStatuses: []corev1.ContainerStatus{{
				RestartCount: 1,
			}}}},
			want: "restarted",
		},
		{
			name: "succeeded",
			pod:  &corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodSucceeded}},
			want: "succeeded",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := evaluatePodOutcome(test.pod); got != test.want {
				t.Fatalf("outcome = %q, want %q", got, test.want)
			}
		})
	}
}

func TestWriteEventReportsStorageFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	collector, err := newEventCollector(
		kubernetesfake.NewSimpleClientset(),
		metricsfake.NewSimpleClientset(),
		path,
		time.Minute,
	)
	if err != nil {
		t.Fatalf("create collector: %v", err)
	}
	if err := collector.eventFile.Close(); err != nil {
		t.Fatalf("close event file: %v", err)
	}
	if err := collector.writeEventLocked(&SchedulingEvent{PodUID: "uid"}); err == nil {
		t.Fatal("write to closed event file succeeded")
	}
}

func TestObservePodDeduplicatesUID(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-a", Namespace: "default", UID: types.UID("uid-a")},
		Spec:       corev1.PodSpec{NodeName: "node-a", SchedulerName: "kubeattention-scheduler"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}
	client := kubernetesfake.NewSimpleClientset(pod, node)
	metrics := metricsfake.NewSimpleClientset()
	metrics.Fake.PrependReactor("list", "nodes", func(k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, nil
	})
	collector, err := newEventCollector(client, metrics, filepath.Join(t.TempDir(), "events.jsonl"), time.Minute)
	if err != nil {
		t.Fatalf("create collector: %v", err)
	}
	defer collector.Close()
	if err := collector.observePod(context.Background(), pod); err != nil {
		t.Fatalf("first observation: %v", err)
	}
	if err := collector.observePod(context.Background(), pod); err != nil {
		t.Fatalf("second observation: %v", err)
	}
	if len(collector.pendingOutcomes) != 1 {
		t.Fatalf("pending outcomes = %d, want 1", len(collector.pendingOutcomes))
	}
	contents, err := os.ReadFile(collector.outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	lines := 0
	for _, value := range contents {
		if value == '\n' {
			lines++
		}
	}
	if lines != 1 {
		t.Fatalf("durable records = %d, want 1", lines)
	}
}
