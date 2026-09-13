/*
Package scheduler implements the fail-safe gRPC client used by the scheduler.
*/
package scheduler

import (
	"context"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	schedulerpb "github.com/softcane/KubeAttention/gen/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	DefaultUDSPath          = "/var/run/kubeattention/brain.sock"
	DevUDSPath              = "/tmp/kubeattention-brain.sock"
	DefaultTimeout          = 50 * time.Millisecond
	CircuitBreakerThreshold = 3
	HealthCheckInterval     = 5 * time.Second
)

// CircuitState represents the state of the Brain circuit breaker.
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// BrainClient wraps the generated Brain client with bounded fallback and recovery.
type BrainClient struct {
	endpoint         string
	timeout          time.Duration
	recoveryInterval time.Duration

	connectMu sync.Mutex
	mu        sync.RWMutex
	conn      *grpc.ClientConn
	client    schedulerpb.BrainClient

	circuitState    CircuitState
	failureCount    int
	lastFailureTime time.Time
	lastLatencyMs   int64
}

// NewBrainClient creates a Brain client. The endpoint may be a Unix socket path,
// unix:///path, or a TCP gRPC target such as brain:50051.
func NewBrainClient(endpoint string, timeout time.Duration) (*BrainClient, error) {
	if endpoint == "" {
		endpoint = DevUDSPath
	}
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &BrainClient{endpoint: endpoint, timeout: timeout, recoveryInterval: HealthCheckInterval}, nil
}

// Connect establishes and verifies the gRPC connection.
func (c *BrainClient) Connect(ctx context.Context) error {
	c.connectMu.Lock()
	defer c.connectMu.Unlock()

	if client := c.snapshotClient(); client != nil {
		if err := c.checkHealth(ctx, client); err == nil {
			c.recordSuccess()
			return nil
		}
	}

	target, dialOptions := c.dialConfig()
	conn, err := grpc.DialContext(ctx, target, dialOptions...)
	if err != nil {
		c.recordFailure()
		return fmt.Errorf("connect to Brain at %s: %w", c.endpoint, err)
	}
	client := schedulerpb.NewBrainClient(conn)
	if err := c.checkHealth(ctx, client); err != nil {
		_ = conn.Close()
		c.recordFailure()
		return fmt.Errorf("verify Brain at %s: %w", c.endpoint, err)
	}

	c.mu.Lock()
	oldConn := c.conn
	c.conn = conn
	c.client = client
	c.failureCount = 0
	c.circuitState = CircuitClosed
	c.mu.Unlock()
	if oldConn != nil {
		_ = oldConn.Close()
	}
	return nil
}

func (c *BrainClient) dialConfig() (string, []grpc.DialOption) {
	options := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	}
	if strings.HasPrefix(c.endpoint, "/") || strings.HasPrefix(c.endpoint, "unix://") {
		path := strings.TrimPrefix(c.endpoint, "unix://")
		dialer := &net.Dialer{}
		options = append(options, grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", path)
		}))
		return "passthrough:///kubeattention-brain", options
	}
	return c.endpoint, options
}

// Close closes the current gRPC connection.
func (c *BrainClient) Close() error {
	c.mu.Lock()
	conn := c.conn
	c.conn = nil
	c.client = nil
	c.mu.Unlock()
	if conn != nil {
		return conn.Close()
	}
	return nil
}

// Score sends a single-node request or returns a neutral fallback.
func (c *BrainClient) Score(ctx context.Context, req *schedulerpb.ScoreRequest) (*schedulerpb.ScoreResponse, error) {
	client, ok := c.availableClient(ctx)
	if !ok {
		return fallbackScore(req), nil
	}

	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	start := time.Now()
	resp, err := client.Score(callCtx, req)
	c.setLatency(time.Since(start))
	if err != nil {
		c.recordFailure()
		return fallbackScore(req), nil
	}
	c.recordSuccess()
	return resp, nil
}

// BatchScore sends all candidates in one request or returns neutral scores.
func (c *BrainClient) BatchScore(ctx context.Context, req *schedulerpb.BatchScoreRequest) (*schedulerpb.BatchScoreResponse, error) {
	client, ok := c.availableClient(ctx)
	if !ok {
		return fallbackBatchScore(req), nil
	}

	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	start := time.Now()
	resp, err := client.BatchScore(callCtx, req)
	c.setLatency(time.Since(start))
	if err != nil {
		c.recordFailure()
		return fallbackBatchScore(req), nil
	}
	c.recordSuccess()
	return resp, nil
}

func (c *BrainClient) availableClient(ctx context.Context) (schedulerpb.BrainClient, bool) {
	c.mu.RLock()
	client := c.client
	state := c.circuitState
	lastFailure := c.lastFailureTime
	c.mu.RUnlock()

	if client != nil && state != CircuitOpen {
		return client, true
	}
	if state == CircuitOpen && time.Since(lastFailure) < c.recoveryInterval {
		return nil, false
	}

	connectCtx, cancel := c.callContext(ctx)
	defer cancel()
	if err := c.Connect(connectCtx); err != nil {
		return nil, false
	}
	return c.snapshotClient(), true
}

func (c *BrainClient) snapshotClient() schedulerpb.BrainClient {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.client
}

func (c *BrainClient) callContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, c.timeout)
}

func (c *BrainClient) checkHealth(ctx context.Context, client schedulerpb.BrainClient) error {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	resp, err := client.HealthCheck(callCtx, &schedulerpb.HealthCheckRequest{})
	if err != nil {
		return err
	}
	if !resp.GetHealthy() {
		return fmt.Errorf("Brain reports unhealthy")
	}
	return nil
}

func (c *BrainClient) recordFailure() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failureCount++
	c.lastFailureTime = time.Now()
	if c.failureCount >= CircuitBreakerThreshold {
		c.circuitState = CircuitOpen
	}
}

func (c *BrainClient) recordSuccess() {
	c.mu.Lock()
	c.failureCount = 0
	c.circuitState = CircuitClosed
	c.mu.Unlock()
}

func (c *BrainClient) setLatency(elapsed time.Duration) {
	c.mu.Lock()
	c.lastLatencyMs = elapsed.Milliseconds()
	c.mu.Unlock()
}

func fallbackScore(req *schedulerpb.ScoreRequest) *schedulerpb.ScoreResponse {
	return &schedulerpb.ScoreResponse{
		Score:     50,
		Reasoning: "Brain unavailable or unhealthy; neutral score",
	}
}

func fallbackBatchScore(req *schedulerpb.BatchScoreRequest) *schedulerpb.BatchScoreResponse {
	if req == nil {
		return &schedulerpb.BatchScoreResponse{}
	}
	scores := make([]*schedulerpb.NodeScore, 0, len(req.GetNodes()))
	for _, node := range req.GetNodes() {
		scores = append(scores, &schedulerpb.NodeScore{
			NodeName:  node.GetNodeName(),
			Score:     50,
			Reasoning: "Brain unavailable or unhealthy; neutral score",
		})
	}
	return &schedulerpb.BatchScoreResponse{Scores: scores}
}

// GetCircuitState returns the current circuit state.
func (c *BrainClient) GetCircuitState() CircuitState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.circuitState
}

// GetLastLatency returns the last measured RPC latency in milliseconds.
func (c *BrainClient) GetLastLatency() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lastLatencyMs
}

// HealthCheck verifies that Brain is reachable and has a usable model.
func (c *BrainClient) HealthCheck(ctx context.Context) (bool, int64, error) {
	client, ok := c.availableClient(ctx)
	if !ok {
		return false, c.GetLastLatency(), fmt.Errorf("Brain unavailable")
	}
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	start := time.Now()
	resp, err := client.HealthCheck(callCtx, &schedulerpb.HealthCheckRequest{})
	c.setLatency(time.Since(start))
	if err != nil {
		c.recordFailure()
		return false, c.GetLastLatency(), err
	}
	if !resp.GetHealthy() {
		c.recordFailure()
		return false, c.GetLastLatency(), fmt.Errorf("Brain reports unhealthy")
	}
	c.recordSuccess()
	return true, c.GetLastLatency(), nil
}
