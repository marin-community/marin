# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the complete, audited eight-run faithful SkyRL comparison."""

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

mpl.use("Agg")


def allocated_hours(attempts: list[dict], cutoff_ms: int) -> float:
    return sum(
        max(0, min(int(row["finished_at_ms"]), cutoff_ms) - int(row["started_at_ms"])) * int(row["gpus"]) / 3_600_000
        for row in attempts
    )


def paired_interval(differences: np.ndarray) -> tuple[float, list[float]]:
    assert differences.shape == (3,)
    critical = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    mean = float(differences.mean())
    half_width = critical * differences.std(ddof=1) / math.sqrt(3)
    return mean, [float(mean - half_width), float(mean + half_width)]


def summarize(config: dict, audit: dict, cache: Path, ledger: dict) -> dict:
    run_id = config["run"]["id"]
    assert audit["run_id"] == run_id and audit["audit_passed"]
    assert audit["matches_authors_completion_grading"]
    assert audit["runtime"]["launcher_commit"] == config["runtime"]["launcher_commit"]
    assert audit["completed_updates"] == 300 and audit["primary_endpoint_completed_updates"] == 299
    assert audit["full_completion_grade_differences_in_native_archives"] == {"train": 0, "eval": 0}
    metrics = [
        json.loads((cache / "exports/training_metrics" / f"train-{step:08d}.json").read_text()) for step in range(1, 301)
    ]
    inventory = {row["Key"].rsplit("/", 1)[-1]: row for row in json.loads((cache / "object_inventory.json").read_text())}
    attempts = [row for row in ledger["attempts"] if row["task_id"].rsplit("/", 1)[0] == "/romain/" + run_id]
    assert attempts and all(int(row["finished_at_ms"]) for row in attempts)
    points = []
    for point in audit["quality"]:
        step = point["completed_updates"]
        assert point["responses"] == 512
        stamp = inventory[f"eval-{step:08d}.json"]["last_modified_ms"]
        points.append(
            {
                **point,
                "consumed_loss_tokens": sum(m["async/performance/consumed_loss_tokens"] for m in metrics[:step]),
                "allocated_h100_hours": allocated_hours(attempts, stamp),
            }
        )
    endpoint = next(point for point in points if point["completed_updates"] == 299)
    return {
        "run_id": run_id,
        "source": config["runtime"]["launcher_commit"],
        "seed": config["run"]["seed"],
        "noise": config["skyrl"]["generator"]["sampler_weight_noise_scale"],
        "score_centering": config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"],
        "evaluation_membership_sha256": audit["evaluation_membership_sha256"],
        "endpoint": endpoint,
        "quality_points": points,
        "completed_updates": 300,
        "nonzero_gradient_updates": audit["nonzero_gradient_updates"],
        "total_consumed_loss_tokens": sum(m["async/performance/consumed_loss_tokens"] for m in metrics),
        "total_allocated_h100_hours": sum(row["allocated_h100_hours"] for row in attempts),
        "cost_basis": (
            "All recorded allocation intervals, including setup, evaluation, checkpointing and failed attempts."
        ),
        "gradient_norms": [m["policy/raw_grad_norm"] for m in metrics],
        "omitted_mass_samples": [point["mean_omitted_mass"] for point in audit["learner_evidence"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.costs.read_text())
    configs = [yaml.safe_load(path.read_text()) for path in sorted(args.config_dir.glob("*.yaml"))]
    assert len(configs) == 8
    runs = []
    for config in configs:
        run_id = config["run"]["id"]
        audit = json.loads((args.audit_dir / f"{run_id}.json").read_text())
        runs.append(summarize(config, audit, args.cache_dir / run_id, ledger))
    assert len({run["source"] for run in runs}) == 1
    pairs = []
    for seed in (0, 1, 2):
        pair = {run["score_centering"]: run for run in runs if run["seed"] == seed and run["noise"] == 0.05}
        assert set(pair) == {False, True}
        assert pair[False]["evaluation_membership_sha256"] == pair[True]["evaluation_membership_sha256"]
        tis, sc = pair[False]["endpoint"]["quality"], pair[True]["endpoint"]["quality"]
        pairs.append({"seed": seed, "tis": tis, "sc": sc, "difference": sc - tis})
    differences = np.asarray([pair["difference"] for pair in pairs])
    mean, interval = paired_interval(differences)
    report = {
        "replication_unit": "paired training seed",
        "paired_seeds": 3,
        "primary_endpoint": "Clean evaluation after299 updates, before update300; first64 shuffled training prompts.",
        "supplemental_endpoint": "Update300 evaluation is separate from the primary endpoint and charged to cost.",
        "pairs": pairs,
        "mean_difference": mean,
        "paired_t_95_interval": interval,
        "interval_note": "Unbounded paired t interval with two degrees of freedom; not a probability range.",
        "runs": runs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "skyrl_faithful_results.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for run in runs:
        color = "#007b83" if run["score_centering"] else "#c36c44"
        label = f"{'TIS+SC' if run['score_centering'] else 'TIS'} seed{run['seed']}"
        points = [point for point in run["quality_points"] if point["completed_updates"] != 300]
        panel = axes[0, 0] if run["noise"] else axes[0, 1]
        panel.plot(
            [point["completed_updates"] for point in points],
            [100 * point["quality"] for point in points],
            color=color,
            alpha=0.7,
            label=label,
        )
        if run["noise"]:
            axes[0, 2].plot(range(1, 301), run["gradient_norms"], color=color, alpha=0.7)
            axes[1, 0].plot(
                [point["consumed_loss_tokens"] for point in points],
                [100 * point["quality"] for point in points],
                color=color,
                alpha=0.7,
            )
            axes[1, 1].plot(
                [point["allocated_h100_hours"] for point in points],
                [100 * point["quality"] for point in points],
                color=color,
                alpha=0.7,
            )
            axes[1, 2].plot(range(1, 301), run["omitted_mass_samples"], color=color, alpha=0.7)
    titles = [
        "Noise0.05: three paired seeds",
        "No-noise sanity pair",
        "Gradient norm",
        "Quality against consumed loss tokens",
        "Quality against allocated H100-hours",
        "Sampled omitted mass",
    ]
    labels = [
        "Completed updates",
        "Completed updates",
        "Completed updates",
        "Consumed loss tokens",
        "Allocated H100-hours",
        "Completed updates",
    ]
    for panel, title, label in zip(axes.flat, titles, labels, strict=True):
        panel.set_title(title)
        panel.set_xlabel(label)
        panel.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output / "skyrl_faithful_quality_stability.png", dpi=170)
    figure.savefig(args.output / "skyrl_faithful_quality_stability.svg")
    plt.close(figure)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}))


if __name__ == "__main__":
    main()
