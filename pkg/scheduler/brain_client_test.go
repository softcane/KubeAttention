package scheduler

import (
	"context"
	"net"
	"sync/atomic"
	"testing"
	"time"

	schedulerpb "github.com/softcane/KubeAttention/gen/go"
	"google.golang.org/grpc"
)

type testBrainServer struct {
	schedulerpb.UnimplementedBrainServer
	healthy atomic.Bool
}

func (s *testBrainServer) Score(_ context.Context, req *schedulerpb.ScoreRequest) (*schedulerpb.ScoreResponse, error) {
	if !s.healthy.Load() {
		return &schedulerpb.ScoreResponse{Score: 50}, nil
	}
	return &schedulerpb.ScoreResponse{Score: 87, Reasoning: req.GetNodeName()}, nil
}

func (s *testBrainServer) BatchScore(_ context.Context, req *schedulerpb.BatchScoreRequest) (*schedulerpb.BatchScoreResponse, error) {
	if !s.healthy.Load() {
		return nil, grpc.ErrServerStopped
	}
	scores := make([]*schedulerpb.NodeScore, 0, len(req.GetNodes()))
	for _, node := range req.GetNodes() {
		scores = append(scores, &schedulerpb.NodeScore{NodeName: node.GetNodeName(), Score: 87})
	}
	return &schedulerpb.BatchScoreResponse{Scores: scores}, nil
}

func (s *testBrainServer) HealthCheck(context.Context, *schedulerpb.HealthCheckRequest) (*schedulerpb.HealthCheckResponse, error) {
	return &schedulerpb.HealthCheckResponse{Healthy: s.healthy.Load(), ModelVersion: "test"}, nil
}

func startTestBrain(t *testing.T) (string, *testBrainServer) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	brain := &testBrainServer{}
	brain.healthy.Store(true)
	server := grpc.NewServer()
	schedulerpb.RegisterBrainServer(server, brain)
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(func() {
		server.Stop()
		_ = listener.Close()
	})
	return listener.Addr().String(), brain
}

func oneNodeRequest() *schedulerpb.BatchScoreRequest {
	return &schedulerpb.BatchScoreRequest{
		PodRequirements: &schedulerpb.PodRequirements{PodName: "probe", PodNamespace: "default"},
		Nodes:           []*schedulerpb.NodeTelemetry{{NodeName: "node-a", TimestampUnixMs: time.Now().UnixMilli()}},
	}
}

func TestBatchScoreBeforeConnectionReturnsNeutral(t *testing.T) {
	client, err := NewBrainClient("127.0.0.1:1", 10*time.Millisecond)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	resp, err := client.BatchScore(context.Background(), oneNodeRequest())
	if err != nil {
		t.Fatalf("batch score: %v", err)
	}
	if len(resp.GetScores()) != 1 || resp.GetScores()[0].GetScore() != 50 {
		t.Fatalf("expected one neutral score, got %#v", resp.GetScores())
	}
}

func TestClientUsesGeneratedProtobuf(t *testing.T) {
	endpoint, _ := startTestBrain(t)
	client, err := NewBrainClient(endpoint, 250*time.Millisecond)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	defer client.Close()
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	resp, err := client.BatchScore(context.Background(), oneNodeRequest())
	if err != nil {
		t.Fatalf("batch score: %v", err)
	}
	if len(resp.GetScores()) != 1 || resp.GetScores()[0].GetScore() != 87 {
		t.Fatalf("expected Brain score, got %#v", resp.GetScores())
	}
}

func TestBatchCircuitRecoversWhenBrainReturns(t *testing.T) {
	endpoint, brain := startTestBrain(t)
	client, err := NewBrainClient(endpoint, 250*time.Millisecond)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	client.recoveryInterval = 10 * time.Millisecond
	defer client.Close()
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}

	brain.healthy.Store(false)
	for range CircuitBreakerThreshold {
		resp, callErr := client.BatchScore(context.Background(), oneNodeRequest())
		if callErr != nil {
			t.Fatalf("fallback call: %v", callErr)
		}
		if resp.GetScores()[0].GetScore() != 50 {
			t.Fatalf("expected neutral fallback, got %d", resp.GetScores()[0].GetScore())
		}
	}
	if client.GetCircuitState() != CircuitOpen {
		t.Fatalf("expected open circuit, got %v", client.GetCircuitState())
	}

	brain.healthy.Store(true)
	time.Sleep(2 * client.recoveryInterval)
	resp, err := client.BatchScore(context.Background(), oneNodeRequest())
	if err != nil {
		t.Fatalf("recovered call: %v", err)
	}
	if resp.GetScores()[0].GetScore() != 87 {
		t.Fatalf("expected recovered Brain score, got %d", resp.GetScores()[0].GetScore())
	}
	if client.GetCircuitState() != CircuitClosed {
		t.Fatalf("expected closed circuit, got %v", client.GetCircuitState())
	}
}
