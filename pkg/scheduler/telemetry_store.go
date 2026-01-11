package scheduler

import (
	"context"
	"sync"
	"time"
)

// TelemetryStore maintains a background-updated cache of node metrics
// to prevent synchronous network calls during scheduling.
type TelemetryStore struct {
	client     *TetragonClient
	mu         sync.RWMutex
	metrics    map[string]*NodeMetrics
	stopped    chan struct{}
	updateFreq time.Duration
}

// NewTelemetryStore creates and starts a background collector
func NewTelemetryStore(client *TetragonClient, updateFreq time.Duration) *TelemetryStore {
	if updateFreq <= 0 {
		updateFreq = 1 * time.Second
	}
	
	s := &TelemetryStore{
		client:     client,
		metrics:    make(map[string]*NodeMetrics),
		stopped:    make(chan struct{}),
		updateFreq: updateFreq,
	}
	
	return s
}

// Start begins periodic background collection
func (s *TelemetryStore) Start(ctx context.Context, nodeNames []string) {
	go func() {
		ticker := time.NewTicker(s.updateFreq)
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-s.stopped:
				return
			case <-ticker.C:
				s.updateAll(ctx, nodeNames)
			}
		}
	}()
}

// updateAll fetches metrics for all nodes in parallel
func (s *TelemetryStore) updateAll(ctx context.Context, nodeNames []string) {
	var wg sync.WaitGroup
	for _, name := range nodeNames {
		wg.Add(1)
		go func(node string) {
			defer wg.Done()
			// Fast timeout for each node fetch
			fetchCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
			defer cancel()
			
			if m, err := s.client.GetNodeMetrics(fetchCtx, node); err == nil {
				s.mu.Lock()
				s.metrics[node] = m
				s.mu.Unlock()
			}
		}(name)
	}
	wg.Wait()
}

// GetMetrics returns cached metrics for a node
func (s *TelemetryStore) GetMetrics(nodeName string) *NodeMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metrics[nodeName]
}

// Stop stops the background collector
func (s *TelemetryStore) Stop() {
	close(s.stopped)
}
