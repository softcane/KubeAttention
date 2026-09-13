package scheduler

import (
	"context"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
)

// MetricsSource fetches one node's measurements.
type MetricsSource interface {
	GetNodeMetrics(context.Context, string) (*NodeMetrics, error)
}

// TelemetryStore maintains a background-updated cache of node metrics.
type TelemetryStore struct {
	source     MetricsSource
	nodes      corelisters.NodeLister
	mu         sync.RWMutex
	metrics    map[string]*NodeMetrics
	stopped    chan struct{}
	stopOnce   sync.Once
	updateFreq time.Duration
}

// NewTelemetryStore creates a store backed by the scheduler's node informer.
func NewTelemetryStore(source MetricsSource, nodes corelisters.NodeLister, updateFreq time.Duration) *TelemetryStore {
	if updateFreq <= 0 {
		updateFreq = time.Second
	}
	return &TelemetryStore{
		source:     source,
		nodes:      nodes,
		metrics:    make(map[string]*NodeMetrics),
		stopped:    make(chan struct{}),
		updateFreq: updateFreq,
	}
}

// Start begins node discovery and periodic collection.
func (s *TelemetryStore) Start(ctx context.Context) {
	go func() {
		s.updateAll(ctx)
		ticker := time.NewTicker(s.updateFreq)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-s.stopped:
				return
			case <-ticker.C:
				s.updateAll(ctx)
			}
		}
	}()
}

func (s *TelemetryStore) updateAll(ctx context.Context) {
	if s.nodes == nil || s.source == nil {
		return
	}
	nodes, err := s.nodes.List(labels.Everything())
	if err != nil {
		return
	}

	liveNodes := make(map[string]struct{}, len(nodes))
	var wg sync.WaitGroup
	for _, node := range nodes {
		if node == nil {
			continue
		}
		name := node.Name
		liveNodes[name] = struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			fetchCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
			defer cancel()
			metrics, fetchErr := s.source.GetNodeMetrics(fetchCtx, name)
			if fetchErr != nil || metrics == nil {
				return
			}
			s.mu.Lock()
			s.metrics[name] = metrics
			s.mu.Unlock()
		}()
	}
	wg.Wait()

	s.mu.Lock()
	for name := range s.metrics {
		if _, ok := liveNodes[name]; !ok {
			delete(s.metrics, name)
		}
	}
	s.mu.Unlock()
}

// GetMetrics returns the latest cached measurement for a node.
func (s *TelemetryStore) GetMetrics(nodeName string) *NodeMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metrics[nodeName]
}

// Stop stops background collection. It is safe to call more than once.
func (s *TelemetryStore) Stop() {
	s.stopOnce.Do(func() { close(s.stopped) })
}
