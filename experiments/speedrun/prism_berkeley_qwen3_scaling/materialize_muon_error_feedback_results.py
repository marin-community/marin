# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the completed Qwen3 130M error-aware Muon sweep."""

from __future__ import annotations

import dataclasses
import json
import statistics
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from rigging.filesystem import open_url

MANIFEST_PATH = "gs://marin-us-central1/experiments/muon_error_feedback_sweep-d76bb7.json"
IRIS_JOB_ID = "/kaiyuew/muon-error-feedback-130m-cubic30-central1-20260710-163320"
DATA_BROWSER_URL = (
    "https://marin.community/data-browser/experiment?"
    "path=gs%3A//marin-us-central1/experiments/muon_error_feedback_sweep-d76bb7.json"
)
METRIC = "eval/paloma/c4_en/bpb"
LOSS_METRIC = "eval/paloma/c4_en/loss"
LEARNING_RATES = (0.008, 0.012, 0.016, 0.020, 0.024)
VARIANTS = (
    ("muon", 0.0),
    ("blend", 0.05),
    ("blend", 0.15),
    ("blend", 0.3),
    ("blend", 0.5),
    ("hesscorr", 0.1),
    ("hesscorr", 0.3),
    ("hesscorr", 1.0),
)
OUTPUT_PATH = Path(__file__).with_name("muon_error_feedback_results.json")


@dataclass(frozen=True)
class SweepRun:
    """One completed cell in the error-aware Muon sweep."""

    run_name: str
    state: str
    policy: str
    gain: float
    learning_rate: float
    adam_learning_rate: float
    c4_en_bpb: float
    c4_en_loss: float
    training_time: float
    seed: int
    num_train_steps: int
    train_batch_size: int
    source_results_path: str
    wandb_url: str
    run_completion_timestamp: str


def _load_json(path: str) -> dict[str, Any]:
    with open_url(path, "r") as f:
        return json.load(f)


def _wandb_path(url: str) -> str:
    parts = urllib.parse.urlparse(url).path.strip("/").split("/")
    if len(parts) != 4 or parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL: {url}")
    return f"{parts[0]}/{parts[1]}/{parts[3]}"


def _gain(policy: str, optimizer: dict[str, Any]) -> float:
    if policy == "muon":
        return 0.0
    if policy == "blend":
        return float(optimizer["blend_gain"])
    if policy == "hesscorr":
        return float(optimizer["correction_gain"])
    raise ValueError(f"Unexpected policy in completed sweep: {policy!r}")


def _fetch_runs() -> list[SweepRun]:
    manifest = _load_json(MANIFEST_PATH)
    api = wandb.Api(timeout=30)
    runs = []
    for step in manifest["steps"]:
        name = step["name"]
        if not name.startswith("speedrun/qwen3_130m_error_aware_muon_") or not name.endswith("-speedrun_results"):
            continue

        source_results_path = step["config"]["output_path"]
        run_info = _load_json(source_results_path)["runs"][0]["run_info"]
        train_config = run_info["train_config"]
        optimizer = train_config["optimizer_config"]
        policy = str(optimizer["policy"])
        wandb_url = str(run_info["wandb_run_link"])
        wandb_run = api.run(_wandb_path(wandb_url))
        loss = wandb_run.summary.get(LOSS_METRIC)
        if loss is None:
            raise ValueError(f"Finished run {wandb_run.id} has no {LOSS_METRIC}")

        runs.append(
            SweepRun(
                run_name=str(wandb_run.id),
                state=str(wandb_run.state),
                policy=policy,
                gain=_gain(policy, optimizer),
                learning_rate=float(optimizer["learning_rate"]),
                adam_learning_rate=float(optimizer["adam_lr"]),
                c4_en_bpb=float(run_info[METRIC]),
                c4_en_loss=float(loss),
                training_time=float(run_info["training_time"]),
                seed=int(wandb_run.config["trainer"]["seed"]),
                num_train_steps=int(train_config["num_train_steps"]),
                train_batch_size=int(train_config["train_batch_size"]),
                source_results_path=source_results_path,
                wandb_url=wandb_url,
                run_completion_timestamp=str(run_info["run_completion_timestamp"]),
            )
        )
    return sorted(runs, key=lambda run: (VARIANTS.index((run.policy, run.gain)), run.learning_rate))


def _validate_grid(runs: list[SweepRun]) -> None:
    expected_cells = {(policy, gain, learning_rate) for policy, gain in VARIANTS for learning_rate in LEARNING_RATES}
    actual_cells = {(run.policy, run.gain, run.learning_rate) for run in runs}
    if len(runs) != len(expected_cells):
        raise ValueError(f"Expected {len(expected_cells)} runs, found {len(runs)}")
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        unexpected = sorted(actual_cells - expected_cells)
        raise ValueError(f"Sweep grid mismatch: missing={missing}, unexpected={unexpected}")
    unfinished = [run.run_name for run in runs if run.state != "finished"]
    if unfinished:
        raise ValueError(f"Sweep contains unfinished runs: {unfinished}")


def _paired_summary(runs: list[SweepRun]) -> list[dict[str, Any]]:
    baseline = {run.learning_rate: run for run in runs if run.policy == "muon"}
    summaries = []
    for policy, gain in VARIANTS:
        if policy == "muon":
            continue
        variant_runs = [run for run in runs if (run.policy, run.gain) == (policy, gain)]
        deltas = [run.c4_en_bpb - baseline[run.learning_rate].c4_en_bpb for run in variant_runs]
        time_overheads = [run.training_time / baseline[run.learning_rate].training_time - 1.0 for run in variant_runs]
        summaries.append(
            {
                "policy": policy,
                "gain": gain,
                "mean_delta_vs_muon": statistics.fmean(deltas),
                "median_delta_vs_muon": statistics.median(deltas),
                "num_wins": sum(delta < 0.0 for delta in deltas),
                "mean_training_time_overhead_fraction": statistics.fmean(time_overheads),
                "delta_by_learning_rate": {
                    f"{run.learning_rate:.3f}": delta for run, delta in zip(variant_runs, deltas, strict=True)
                },
            }
        )
    return summaries


def build_payload(runs: list[SweepRun]) -> dict[str, Any]:
    """Validate the completed grid and build its checked-in summary."""
    _validate_grid(runs)
    best = min(runs, key=lambda run: run.c4_en_bpb)
    paired_muon = next(run for run in runs if run.policy == "muon" and run.learning_rate == best.learning_rate)
    best_record = dataclasses.asdict(best)
    best_record.update(
        {
            "paired_muon_bpb": paired_muon.c4_en_bpb,
            "paired_muon_run_name": paired_muon.run_name,
            "paired_delta_vs_muon": best.c4_en_bpb - paired_muon.c4_en_bpb,
            "paired_training_time_overhead_fraction": best.training_time / paired_muon.training_time - 1.0,
        }
    )
    return {
        "experiment": {
            "manifest_path": MANIFEST_PATH,
            "iris_job_id": IRIS_JOB_ID,
            "data_browser_url": DATA_BROWSER_URL,
            "num_training_runs": len(runs),
            "num_result_steps": len(runs),
            "num_succeeded": len(runs),
        },
        "metric": {"name": METRIC, "lower_is_better": True},
        "setup": {
            "model": "Qwen3 130M",
            "model_size": 154_147_328,
            "dataset": "FineWeb-Edu 10B pretokenized cache",
            "tpu_type": "v5p-8",
            "seed": 0,
            "num_train_steps": 4_959,
            "train_batch_size": 128,
            "sequence_length": 4_096,
            "total_tokens": 2_599_944_192,
            "momentum": 0.95,
            "nesterov": False,
            "weight_decay": 0.1,
            "adam_lr_ratio": 0.2,
            "quintic_steps": 5,
            "cubic_steps": 30,
        },
        "grid": {
            "learning_rates": list(LEARNING_RATES),
            "variants": [{"policy": policy, "gain": gain} for policy, gain in VARIANTS],
        },
        "best_observed_run": best_record,
        "paired_summary": _paired_summary(runs),
        "runs": [dataclasses.asdict(run) for run in runs],
        "reference_runs": {
            "historical_muon": {
                "context_only": True,
                "bpb": 1.1662893295288086,
                "learning_rate": 0.016,
                "url": "https://wandb.ai/marin-community/marin/runs/qwen3_130m_muon_4096-04770b",
                "comparison_caveat": (
                    "Uses Nesterov momentum, constant decoupled weight decay, Muon epsilon 1e-5, "
                    "a different cache instance, and v5p-32 hardware."
                ),
            },
            "prism_berkeley": {
                "context_only": True,
                "bpb": 1.1702665090560913,
                "learning_rate": 0.016,
                "url": ("https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_prism_berkeley_o5_4096_lrx1-2fd229"),
                "comparison_caveat": "Fresh in-sweep Muon outperforms this run at the shared LR of 0.016.",
            },
        },
    }


def main() -> None:
    payload = build_payload(_fetch_runs())
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    best = payload["best_observed_run"]
    print(f"Wrote {OUTPUT_PATH}: {len(payload['runs'])} runs; best={best['run_name']} bpb={best['c4_en_bpb']:.6f}")


if __name__ == "__main__":
    main()
