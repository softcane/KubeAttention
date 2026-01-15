/*
Collector CLI for collecting training data from a Kubernetes cluster.

Usage:
    go run ./cmd/collector --output /data/events.jsonl
*/
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/softcane/KubeAttention/pkg/collector"
)

func main() {
	outputPath := flag.String("output", "events.jsonl", "Output JSONL file path")
	flag.Parse()

	fmt.Println("KubeAttention Event Collector")
	fmt.Printf("Output: %s\n", *outputPath)

	c, err := collector.NewEventCollector(*outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create collector: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	// Handle graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Start collector
	if err := c.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Collector error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Collected %d events\n", c.GetEventCount())
}
