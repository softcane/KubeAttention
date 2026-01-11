/*
Package scheduler implements the KubeAttention gRPC client for the Brain.

This client communicates with the Python Brain server over Unix Domain Socket,
implementing circuit breaker pattern with 50ms timeout fallback.
*/
package scheduler

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// DefaultUDSPath is the production Unix socket path
	DefaultUDSPath = "/var/run/kubeattention/brain.sock"
	
	// DevUDSPath is for local development/testing
	DevUDSPath = "/tmp/kubeattention-brain.sock"
	
	// DefaultTimeout is the circuit breaker timeout (50ms as per PLAN.md)
	DefaultTimeout = 50 * time.Millisecond
	
	// HealthCheckInterval for circuit breaker
	HealthCheckInterval = 5 * time.Second
	
	// CircuitBreakerThreshold - failures before opening circuit
	CircuitBreakerThreshold = 3
)

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
	CircuitClosed CircuitState = iota // Normal operation
	CircuitOpen                       // Failing, use fallback
	CircuitHalfOpen                   // Testing if service recovered
)

// BrainClient wraps the gRPC connection to the Brain with circuit breaker
type BrainClient struct {
	conn         *grpc.ClientConn
	udsPath      string
	timeout      time.Duration
	
	// Circuit breaker state
	mu              sync.RWMutex
	circuitState    CircuitState
	failureCount    int
	lastFailureTime time.Time
	lastLatencyMs   int64
}

// ScoreRequest mirrors the proto ScoreRequest
type ScoreRequest struct {
	PodName      string
	PodNamespace string
	NodeName     string
	Telemetry    map[string]float64
}

// ScoreResponse mirrors the proto ScoreResponse
type ScoreResponse struct {
	Score      int64
	Reasoning  string
	Confidence float64
}

// NodeScore matches proto NodeScore
type NodeScore struct {
	NodeName   string
	Score      int64
	Reasoning  string
	Confidence float64
}

// BatchScoreRequest matches proto BatchScoreRequest
type BatchScoreRequest struct {
	PodName      string
	PodNamespace string
	Nodes        []ScoreRequest // Contains per-node telemetry
}

// BatchScoreResponse matches proto BatchScoreResponse
type BatchScoreResponse struct {
	Scores []NodeScore
}

// NewBrainClient creates a new client with circuit breaker
func NewBrainClient(udsPath string, timeout time.Duration) (*BrainClient, error) {
	if udsPath == "" {
		udsPath = DevUDSPath
	}
	if timeout == 0 {
		timeout = DefaultTimeout
	}
	
	client := &BrainClient{
		udsPath:      udsPath,
		timeout:      timeout,
		circuitState: CircuitClosed,
	}
	
	return client, nil
}

// Connect establishes the gRPC connection
func (c *BrainClient) Connect(ctx context.Context) error {
	dialer := func(ctx context.Context, addr string) (net.Conn, error) {
		return net.Dial("unix", c.udsPath)
	}
	
	conn, err := grpc.DialContext(ctx,
		c.udsPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(dialer),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to Brain at %s: %w", c.udsPath, err)
	}
	
	c.conn = conn
	return nil
}

// Close closes the gRPC connection
func (c *BrainClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Score sends a scoring request to the Brain with circuit breaker
func (c *BrainClient) Score(ctx context.Context, req *ScoreRequest) (*ScoreResponse, error) {
	// Check circuit breaker
	c.mu.RLock()
	state := c.circuitState
	c.mu.RUnlock()
	
	if state == CircuitOpen {
		// Check if we should try half-open
		c.mu.Lock()
		if time.Since(c.lastFailureTime) > HealthCheckInterval {
			c.circuitState = CircuitHalfOpen
			state = CircuitHalfOpen
		}
		c.mu.Unlock()
		
		if state == CircuitOpen {
			return c.fallbackScore(req), nil
		}
	}
	
	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	
	start := time.Now()
	
	// Make the actual gRPC call
	resp, err := c.doScore(timeoutCtx, req)
	
	elapsed := time.Since(start)
	c.lastLatencyMs = elapsed.Milliseconds()
	
	if err != nil {
		c.recordFailure()
		
		// If circuit is half-open, revert to open
		c.mu.Lock()
		if c.circuitState == CircuitHalfOpen {
			c.circuitState = CircuitOpen
		}
		c.mu.Unlock()
		
		// Return fallback
		return c.fallbackScore(req), nil
	}
	
	// Success - reset circuit breaker
	c.mu.Lock()
	c.failureCount = 0
	c.circuitState = CircuitClosed
	c.mu.Unlock()
	
	return resp, nil
}

// BatchScore sends a batch of nodes to the Brain for parallel scoring
func (c *BrainClient) BatchScore(ctx context.Context, req *BatchScoreRequest) (*BatchScoreResponse, error) {
	c.mu.RLock()
	state := c.circuitState
	c.mu.RUnlock()

	if state == CircuitOpen {
		return c.fallbackBatchScore(req), nil
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	start := time.Now()
	resp := new(BatchScoreResponse)
	err := c.conn.Invoke(timeoutCtx, "/scheduler.Brain/BatchScore", req, resp)
	
	c.lastLatencyMs = time.Since(start).Milliseconds()

	if err != nil {
		c.recordFailure()
		return c.fallbackBatchScore(req), nil
	}

	return resp, nil
}

func (c *BrainClient) fallbackBatchScore(req *BatchScoreRequest) *BatchScoreResponse {
	resp := &BatchScoreResponse{
		Scores: make([]NodeScore, len(req.Nodes)),
	}
	for i, node := range req.Nodes {
		fallback := c.fallbackScore(&node)
		resp.Scores[i] = NodeScore{
			NodeName:   node.NodeName,
			Score:      fallback.Score,
			Reasoning:  fallback.Reasoning,
			Confidence: fallback.Confidence,
		}
	}
	return resp
}

// doScore performs the actual gRPC call
func (c *BrainClient) doScore(ctx context.Context, req *ScoreRequest) (*ScoreResponse, error) {
	if c.conn == nil {
		return nil, fmt.Errorf("not connected")
	}
	
	// Use raw gRPC call (would use generated stubs after buf generate)
	// For handshake test, we send a minimal request
	
	out := new(ScoreResponse)
	err := c.conn.Invoke(ctx, "/scheduler.Brain/Score", req, out)
	if err != nil {
		return nil, err
	}
	
	return out, nil
}

// recordFailure updates circuit breaker state
func (c *BrainClient) recordFailure() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.failureCount++
	c.lastFailureTime = time.Now()
	
	if c.failureCount >= CircuitBreakerThreshold {
		c.circuitState = CircuitOpen
	}
}

// fallbackScore returns LeastAllocated-style score as fallback
func (c *BrainClient) fallbackScore(req *ScoreRequest) *ScoreResponse {
	// Simple fallback: use CPU utilization if available
	score := int64(50) // Default middle score
	
	if cpuUtil, ok := req.Telemetry["cpu_utilization"]; ok {
		// Invert utilization: lower usage = higher score
		score = int64((1.0 - cpuUtil) * 100)
	}
	
	return &ScoreResponse{
		Score:      score,
		Reasoning:  "Fallback: Brain unavailable, using LeastAllocated strategy",
		Confidence: 0.5, // Lower confidence for fallback
	}
}

// GetCircuitState returns current circuit breaker state
func (c *BrainClient) GetCircuitState() CircuitState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.circuitState
}

// GetLastLatency returns the last measured latency in milliseconds
func (c *BrainClient) GetLastLatency() int64 {
	return c.lastLatencyMs
}

// HealthCheck performs a lightweight health check on the Brain
func (c *BrainClient) HealthCheck(ctx context.Context) (bool, int64, error) {
	if c.conn == nil {
		return false, 0, fmt.Errorf("not connected")
	}
	
	timeoutCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()
	
	start := time.Now()
	
	// Invoke health check RPC
	type healthResp struct {
		Healthy      bool
		LatencyMs    int64
		ModelVersion string
	}
	
	resp := new(healthResp)
	err := c.conn.Invoke(timeoutCtx, "/scheduler.Brain/HealthCheck", struct{}{}, resp)
	
	elapsed := time.Since(start)
	
	if err != nil {
		return false, elapsed.Milliseconds(), err
	}
	
	return resp.Healthy, elapsed.Milliseconds(), nil
}
