package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

type fakeMetricsSource struct{}

func (fakeMetricsSource) GetNodeMetrics(_ context.Context, nodeName string) (*NodeMetrics, error) {
	if nodeName == "" {
		return nil, fmt.Errorf("missing node name")
	}
	return &NodeMetrics{NodeName: nodeName, Timestamp: time.Now(), CPUUtilization: 0.25}, nil
}

func waitForMetrics(t *testing.T, store *TelemetryStore, nodeName string, present bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if (store.GetMetrics(nodeName) != nil) == present {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("node %s presence did not become %v", nodeName, present)
}

func TestTelemetryStoreDiscoversAddedAndRemovedNodes(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := indexer.Add(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}); err != nil {
		t.Fatalf("add node-a: %v", err)
	}
	lister := corelisters.NewNodeLister(indexer)
	store := NewTelemetryStore(fakeMetricsSource{}, lister, 10*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer store.Stop()
	store.Start(ctx)

	waitForMetrics(t, store, "node-a", true)
	if err := indexer.Add(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-b"}}); err != nil {
		t.Fatalf("add node-b: %v", err)
	}
	waitForMetrics(t, store, "node-b", true)
	if err := indexer.Delete(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}); err != nil {
		t.Fatalf("delete node-a: %v", err)
	}
	waitForMetrics(t, store, "node-a", false)
}
