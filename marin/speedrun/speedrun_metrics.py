"""Scaling law metrics for speedrun analysis."""

import argparse
import json
from dataclasses import dataclass

import fsspec


@dataclass
class ScalingLawParams:
    alpha: float = 0.53  # Power law exponent from paper
    A: float = 0.29  # Coefficient from paper


def compute_expected_flops(target_bpb: float, params: ScalingLawParams | None = None) -> float:
    """Compute expected FLOPs needed to achieve a target BPB according to scaling law."""
    if params is None:
        params = ScalingLawParams()

    # N*(C) = AC^a from the paper, solving for C
    return (target_bpb / params.A) ** (1/params.alpha)


def compute_speedup(actual_flops: float, achieved_bpb: float, params: ScalingLawParams | None = None) -> float:
    """Compute speedup as ratio of expected FLOPs from scaling law to actual FLOPs used."""
    expected_flops = compute_expected_flops(achieved_bpb, params)
    return expected_flops / actual_flops


def analyze_speedrun_results(results_path: str) -> dict:
    """Analyze speedrun results and compute speedup metrics."""
    with fsspec.open(results_path, "r") as f:
        data = json.load(f)

    metrics = {}
    for run in data.get("runs", []):
        stats = run.get("run_stats", {})
        bpb = stats.get("eval/paloma/c4_en/bpb")
        flops = stats.get("total_training_flops")

        if bpb is not None and flops is not None:
            speedup = compute_speedup(flops, bpb)
            metrics[run.get("metadata", {}).get("wandb_run_id", "unknown")] = {
                "bpb": bpb,
                "flops": flops,
                "speedup": speedup,
            }

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze speedrun results using scaling laws")
    parser.add_argument("results_path", type=str, help="Path to speedrun_results.json file")
    args = parser.parse_args()

    metrics = analyze_speedrun_results(args.results_path)
    print("\nSpeedup Analysis:")
    print("-" * 50)
    for run_id, stats in metrics.items():
        print(f"Run: {run_id}")
        print(f"BPB: {stats['bpb']:.4f}")
        print(f"FLOPs: {stats['flops']:.2e}")
        print(f"Speedup: {stats['speedup']:.2f}x")
        print("-" * 50)
