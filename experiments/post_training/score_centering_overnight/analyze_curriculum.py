# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze complete paired curriculum runs using training seeds as replicates."""

import argparse
import copy
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from analysis_json import read_json, write_json
from analyze_skyrl import allocated_hours, paired_interval

mpl.use("Agg")


def subset_quality(rows: list[dict], members: set[int]) -> float:
    selected = [row for row in rows if row["source_row"] in members]
    assert len(selected) == len(members) and selected
    return sum(row["quality_sum"] for row in selected) / sum(row["responses"] for row in selected)


def paired_recipe(config: dict) -> dict:
    recipe = copy.deepcopy({"inputs": config["inputs"], "skyrl": config["skyrl"]})
    recipe["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"] = False
    return json.loads(json.dumps(recipe).replace(config["run"]["id"], "PAIRED_RUN"))


def summarize(config: dict, audit: dict, cache: Path, costs: dict, *, recovery_run_ids: tuple[str, ...] = ()) -> dict:
    run_id = config["run"]["id"]
    updates = config["skyrl"]["trainer"]["max_steps"]
    assert audit["run_id"] == run_id and audit["audit_passed"]
    assert audit["audit_scope"] == "complete native curriculum run"
    assert audit["source"] == config["runtime"]["launcher_commit"]
    assert audit["completed_updates"] == updates
    allocation_jobs = {"/romain/" + identity for identity in (run_id, *recovery_run_ids)}
    attempts = [row for row in costs["attempts"] if row["task_id"].rsplit("/", 1)[0] in allocation_jobs]
    assert attempts and all(int(row["finished_at_ms"]) for row in attempts)
    metrics = [
        json.loads((cache / "exports/training_metrics" / f"train-{step:08d}.json").read_text())
        for step in range(1, updates + 1)
    ]
    assert all(math.isfinite(row["policy/raw_grad_norm"]) for row in metrics)
    metric_prefix = urlparse(config["artifacts"]["export_root"]).path.lstrip("/").rstrip("/") + "/training_metrics/"
    inventory = {
        row["Key"].removeprefix(metric_prefix): row
        for row in json.loads((cache / "object_inventory.json").read_text())
        if row["Key"].startswith(metric_prefix)
    }
    row_points = {point["completed_updates"]: point["rows"] for point in audit["evaluation_rows"]}
    points = []
    for point in audit["quality"]:
        step = point["completed_updates"]
        rows = row_points[step]
        assert sum(row["responses"] for row in rows) == point["responses"]
        assert math.isclose(subset_quality(rows, {row["source_row"] for row in rows}), point["quality"], abs_tol=1e-12)
        stamp = inventory[f"eval-{step:08d}.json"]["last_modified_ms"]
        points.append(
            {
                **point,
                "elapsed_seconds": (stamp - min(int(row["started_at_ms"]) for row in attempts)) / 1000,
                "consumed_loss_tokens": sum(row["async/performance/consumed_loss_tokens"] for row in metrics[:step]),
                "allocated_h100_hours": allocated_hours(attempts, stamp),
                "errors": sum(row["errors"] for row in rows),
            }
        )
    assert points[-1]["completed_updates"] == updates
    return {
        "run_id": run_id,
        "source": audit["source"],
        "seed": config["run"]["seed"],
        "score_centering": audit["score_centering"],
        "completed_updates": updates,
        "evaluation_membership_sha256": audit["evaluation_membership_sha256"],
        "points": points,
        "endpoint": points[-1],
        "evaluation_rows": audit["evaluation_rows"],
        "gradient_norms": [row["policy/raw_grad_norm"] for row in metrics],
        "ppo_clip_fractions": [row["policy/ppo_clip_ratio"] for row in metrics],
        "learner_evidence": audit["learner_evidence"],
        "total_allocated_h100_hours": sum(row["allocated_h100_hours"] for row in attempts),
        "recovery_run_ids": list(recovery_run_ids),
        "source_task_errors": audit["source_task_errors"],
        "cost_basis": (
            "All recorded task allocation intervals, including failed attempts, setup, evaluation and checkpointing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.yaml")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "main"), required=True)
    parser.add_argument("--common-source-rows", type=Path)
    parser.add_argument("--recovery-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    costs = json.loads(args.costs.read_text())
    configs = [yaml.safe_load(path.read_text()) for path in sorted(args.config_dir.glob(args.pattern))]
    recoveries = {config["run"]["id"]: [] for config in configs}
    for path in args.recovery_config:
        previous = yaml.safe_load(path.read_text())
        matches = [
            config
            for config in configs
            if config["run"]["seed"] == previous["run"]["seed"]
            and config["runtime"]["launcher_commit"] == previous["runtime"]["launcher_commit"]
            and config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
            == previous["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
            and paired_recipe(config) == paired_recipe(previous)
        ]
        assert len(matches) == 1, f"Recovery configuration has no unique matching scientific recipe: {path}"
        identity = previous["run"]["id"]
        assert identity not in recoveries and all(identity not in values for values in recoveries.values())
        recoveries[matches[0]["run"]["id"]].append(identity)
    seeds = (0,) if args.mode == "pilot" else (0, 1, 2)
    assert len(configs) == 2 * len(seeds)
    if args.mode == "pilot":
        assert all(cfg["skyrl"]["trainer"]["max_steps"] == 40 for cfg in configs)
    else:
        assert all(cfg["skyrl"]["trainer"]["max_steps"] >= 300 for cfg in configs)
    runs = []
    for config in configs:
        run_id = config["run"]["id"]
        path = args.audit_dir / f"{run_id}.json"
        audit = read_json(path if path.exists() else path.with_suffix(".json.gz"))
        runs.append(summarize(config, audit, args.cache_dir / run_id, costs, recovery_run_ids=tuple(recoveries[run_id])))
    assert len({run["source"] for run in runs}) == 1
    members = set(json.loads(args.common_source_rows.read_text())) if args.common_source_rows else None
    if members is not None:
        for run in runs:
            run["common_quality_points"] = [
                {"completed_updates": point["completed_updates"], "quality": subset_quality(point["rows"], members)}
                for point in run["evaluation_rows"]
            ]
    pairs = []
    for seed in seeds:
        configured_pair = {
            cfg["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]: cfg
            for cfg in configs
            if cfg["run"]["seed"] == seed
        }
        assert set(configured_pair) == {False, True}
        assert paired_recipe(configured_pair[False]) == paired_recipe(configured_pair[True])
        pair = {run["score_centering"]: run for run in runs if run["seed"] == seed}
        assert set(pair) == {False, True}
        assert pair[False]["evaluation_membership_sha256"] == pair[True]["evaluation_membership_sha256"]
        assert pair[False]["completed_updates"] == pair[True]["completed_updates"]
        tis, sc = pair[False]["endpoint"]["quality"], pair[True]["endpoint"]["quality"]
        item = {"seed": seed, "tis": tis, "sc": sc, "difference": sc - tis}
        if members is not None:
            item["common_tis"] = subset_quality(pair[False]["evaluation_rows"][-1]["rows"], members)
            item["common_sc"] = subset_quality(pair[True]["evaluation_rows"][-1]["rows"], members)
            item["common_difference"] = item["common_sc"] - item["common_tis"]
        pairs.append(item)
    differences = np.asarray([pair["difference"] for pair in pairs])
    report = {"mode": args.mode, "replication_unit": "paired training seed", "pairs": pairs, "runs": runs}
    if args.mode == "main":
        report["mean_difference"], report["paired_t_95_interval"] = paired_interval(differences)
        if members is not None:
            report["common_mean_difference"], report["common_paired_t_95_interval"] = paired_interval(
                np.asarray([pair["common_difference"] for pair in pairs])
            )
        report["interval_note"] = (
            "Unbounded paired t interval, two degrees of freedom; no equivalence or acceptable-loss margin."
        )
    else:
        report["pilot_difference"] = float(differences[0])
        report["scope"] = "One exploratory paired seed; no population interval or equivalence claim."
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "paired_curriculum_results.json.gz", report)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for run in runs:
        color = "#007b83" if run["score_centering"] else "#c36c44"
        label = f"{'TIS+SC' if run['score_centering'] else 'TIS'} seed{run['seed']}"
        for axis, key in zip(
            axes[0], ("completed_updates", "consumed_loss_tokens", "allocated_h100_hours"), strict=True
        ):
            axis.plot(
                [p[key] for p in run["points"]],
                [100 * p["quality"] for p in run["points"]],
                color=color,
                alpha=0.7,
                label=label,
            )
        axes[1, 0].plot(range(1, run["completed_updates"] + 1), run["gradient_norms"], color=color, alpha=0.7)
        axes[1, 1].plot(range(1, run["completed_updates"] + 1), run["ppo_clip_fractions"], color=color, alpha=0.7)
        points = run["learner_evidence"]
        axes[1, 2].plot(
            [p["step"] for p in points],
            [p.get("mean_omitted_mass", float("nan")) for p in points],
            color=color,
            alpha=0.7,
        )
    for axis, title in zip(
        axes.flat,
        (
            "Quality / updates",
            "Quality / loss tokens",
            "Quality / H100-hours",
            "Gradient norm",
            "PPO clipping",
            "Sampled omitted mass",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.grid(alpha=0.2)
    for axis, label in zip(axes[0], ("Completed updates", "Consumed loss tokens", "Allocated H100-hours"), strict=True):
        axis.set_xlabel(label)
        axis.set_ylabel("Native clean-policy quality (%)")
    for axis, label in zip(
        axes[1], ("Raw gradient norm", "Clipped-token fraction", "Mean omitted mass (sample)"), strict=True
    ):
        axis.set_xlabel("Completed updates")
        axis.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output / "curriculum_quality_and_stability.png", dpi=180)
    figure.savefig(args.output / "curriculum_quality_and_stability.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
