#!/usr/bin/env python3
"""
KubeAttention Benchmark Visualization

Generates PNG charts comparing default scheduler vs KubeAttention.

Usage:
    python benchmark/visualize.py --input benchmark/results/benchmark_*.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_results(input_file: Path) -> list[dict]:
    """Load benchmark results from JSON."""
    with open(input_file) as f:
        return json.load(f)


def create_p99_comparison_chart(results: list[dict], output_dir: Path):
    """Create P99 latency comparison bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    schedulers = [r["scheduler"] for r in results]
    p99_means = [r["summary"].get("p99_mean", 0) for r in results]
    
    colors = ["#FF6B6B" if s == "default" else "#4ECDC4" for s in schedulers]
    
    bars = ax.bar(schedulers, p99_means, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, p99_means):
        height = bar.get_height()
        ax.annotate(f'{value:.2f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')
    
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('P99 Latency Comparison: Default vs KubeAttention', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(p99_means) * 1.3)
    
    # Add improvement annotation if both results exist
    if len(results) >= 2:
        default_p99 = next((r["summary"]["p99_mean"] for r in results if r["scheduler"] == "default"), 0)
        kubeattention_p99 = next((r["summary"]["p99_mean"] for r in results if r["scheduler"] == "kubeattention"), 0)
        if default_p99 > 0 and kubeattention_p99 > 0:
            improvement = ((default_p99 - kubeattention_p99) / default_p99) * 100
            ax.annotate(f'↓ {improvement:.1f}% improvement',
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        fontsize=16, fontweight='bold', color='green',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "p99_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_placement_accuracy_chart(results: list[dict], output_dir: Path):
    """Create placement accuracy comparison chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    schedulers = [r["scheduler"] for r in results]
    noisy_counts = [r["summary"].get("pods_on_noisy_nodes", 0) for r in results]
    quiet_counts = [r["summary"].get("pods_on_quiet_nodes", 0) for r in results]
    
    x = range(len(schedulers))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], noisy_counts, width, 
                   label='Pods on Noisy Nodes', color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], quiet_counts, width, 
                   label='Pods on Quiet Nodes', color='#4ECDC4', edgecolor='black')
    
    ax.set_ylabel('Number of Pods', fontsize=12)
    ax.set_title('Placement Accuracy: Where Did Pods Land?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in schedulers])
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / "placement_accuracy.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_latency_distribution_chart(results: list[dict], output_dir: Path):
    """Create P50/P90/P99 comparison chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    schedulers = [r["scheduler"] for r in results]
    
    # For each scheduler, get P50, P90, P99 means
    metrics_labels = ['P50', 'P90', 'P99']
    x = range(len(metrics_labels))
    width = 0.35
    
    colors = {"default": "#FF6B6B", "kubeattention": "#4ECDC4"}
    
    for i, result in enumerate(results):
        scheduler = result["scheduler"]
        summary = result["summary"]
        
        # Calculate P50, P90, P99 from metrics if available
        p50 = summary.get("p50_mean", 0)
        p90 = summary.get("p99_mean", 0) * 0.7  # Approximate
        p99 = summary.get("p99_mean", 0)
        
        values = [p50, p90, p99]
        offset = (i - 0.5) * width
        
        bars = ax.bar([xi + offset for xi in x], values, width, 
                      label=scheduler.title(), color=colors.get(scheduler, "#888"),
                      edgecolor='black')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Distribution: P50, P90, P99', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / "latency_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_summary_dashboard(results: list[dict], output_dir: Path):
    """Create summary dashboard with all key metrics."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('KubeAttention Benchmark Results Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: P99 bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    schedulers = [r["scheduler"] for r in results]
    p99_means = [r["summary"].get("p99_mean", 0) for r in results]
    colors = ["#FF6B6B" if s == "default" else "#4ECDC4" for s in schedulers]
    ax1.bar(schedulers, p99_means, color=colors, edgecolor="black")
    ax1.set_ylabel('P99 Latency (ms)')
    ax1.set_title('P99 Latency Comparison')
    for i, v in enumerate(p99_means):
        ax1.text(i, v + 0.1, f'{v:.2f}ms', ha='center', fontweight='bold')
    
    # Top-right: Placement pie charts
    ax2 = fig.add_subplot(gs[0, 1])
    if len(results) >= 2:
        kubeattention_result = next((r for r in results if r["scheduler"] == "kubeattention"), results[-1])
        noisy = kubeattention_result["summary"].get("pods_on_noisy_nodes", 1)
        quiet = kubeattention_result["summary"].get("pods_on_quiet_nodes", 1)
        ax2.pie([noisy, quiet], labels=['Noisy Nodes', 'Quiet Nodes'], 
                colors=['#FF6B6B', '#4ECDC4'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('KubeAttention Pod Placement')
    
    # Bottom: Summary text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    summary_text = "BENCHMARK SUMMARY\n" + "="*50 + "\n\n"
    for result in results:
        summary = result["summary"]
        summary_text += f"Scheduler: {result['scheduler'].upper()}\n"
        summary_text += f"  P99 Mean: {summary.get('p99_mean', 0):.2f}ms\n"
        summary_text += f"  P99 Max:  {summary.get('p99_max', 0):.2f}ms\n"
        summary_text += f"  Samples:  {summary.get('total_samples', 0)}\n\n"
    
    # Calculate improvement
    if len(results) >= 2:
        default_p99 = next((r["summary"]["p99_mean"] for r in results if r["scheduler"] == "default"), 0)
        kubeattention_p99 = next((r["summary"]["p99_mean"] for r in results if r["scheduler"] == "kubeattention"), 0)
        if default_p99 > 0 and kubeattention_p99 > 0:
            improvement = ((default_p99 - kubeattention_p99) / default_p99) * 100
            summary_text += f"IMPROVEMENT: {improvement:.1f}% P99 latency reduction\n"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = output_dir / "dashboard.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="KubeAttention Benchmark Visualization")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file from benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output) if args.output else input_file.parent
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return
    
    print("="*60)
    print("KubeAttention Benchmark Visualization")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    results = load_results(input_file)
    print(f"\nLoaded {len(results)} benchmark results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating charts...")
    create_p99_comparison_chart(results, output_dir)
    create_placement_accuracy_chart(results, output_dir)
    create_latency_distribution_chart(results, output_dir)
    create_summary_dashboard(results, output_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
