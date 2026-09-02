# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the completed Qwen3 error-aware Muon experiments."""

from __future__ import annotations

import dataclasses
import json
import math
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
FOLLOWUP_300M_JOB_ID = "/kaiyuew/muon-error-feedback-300m-hesscorr-stable-20260828-222720"
FOLLOWUP_300M_VERSION = "2026.08.28.3"
REFERENCE_300M_VERSION = "2026.07.23.4"
FOLLOWUP_300M_GLOBAL_STEP = 11_443
FOLLOWUP_300M_LEARNING_RATES = (0.004, 0.006, 0.008, 0.010, 0.012)
FOLLOWUP_300M_VARIANTS = (("hesscorr", 0.1), ("hesscorr", 0.3), ("hesscorr", 1.0))
REFERENCE_300M_VARIANTS = (("muon", 0.0), ("blend", 0.05))


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


@dataclass(frozen=True)
class FollowupRun:
    """One completed 300M Hesscorr cell or paired reference cell."""

    run_name: str
    state: str
    policy: str
    gain: float
    learning_rate: float
    c4_en_bpb: float
    training_time: float
    train_loss: float
    global_step: int
    seed: int
    num_train_steps: int
    train_batch_size: int
    artifact_version: str
    source_results_path: str
    checkpoint_metadata_path: str
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


def _number_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _run_stem(policy: str, gain: float, learning_rate: float) -> str:
    if policy == "muon":
        variant = "muon"
    else:
        variant = f"{policy}-g{_number_slug(gain)}"
    return f"qwen3_300m_error_aware_muon_{variant}_lr{_number_slug(learning_rate)}"


def _fetch_followup_run(
    api: wandb.Api,
    *,
    policy: str,
    gain: float,
    learning_rate: float,
    artifact_version: str,
) -> FollowupRun:
    run_stem = _run_stem(policy, gain, learning_rate)
    checkpoint_root = f"gs://marin-us-central1/checkpoints/speedrun/{run_stem}/{artifact_version}"
    source_results_path = f"{checkpoint_root}/speedrun_results.json"
    checkpoint_metadata_path = f"{checkpoint_root}/checkpoints/step-{FOLLOWUP_300M_GLOBAL_STEP}/metadata.json"
    run_info = _load_json(source_results_path)["runs"][0]["run_info"]
    _load_json(checkpoint_metadata_path)
    wandb_url = str(run_info["wandb_run_link"])
    wandb_run = api.run(_wandb_path(wandb_url))
    train_loss = wandb_run.summary.get("train/loss")
    global_step = wandb_run.summary.get("global_step")
    if train_loss is None or global_step is None:
        raise ValueError(f"Finished run {wandb_run.id} has no final train/loss or global_step")

    train_config = run_info["train_config"]
    return FollowupRun(
        run_name=str(wandb_run.id),
        state=str(wandb_run.state),
        policy=policy,
        gain=gain,
        learning_rate=learning_rate,
        c4_en_bpb=float(run_info[METRIC]),
        training_time=float(run_info["training_time"]),
        train_loss=float(train_loss),
        global_step=int(global_step),
        seed=int(wandb_run.config["trainer"]["seed"]),
        num_train_steps=int(train_config["num_train_steps"]),
        train_batch_size=int(train_config["train_batch_size"]),
        artifact_version=artifact_version,
        source_results_path=source_results_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        wandb_url=wandb_url,
        run_completion_timestamp=str(run_info["run_completion_timestamp"]),
    )


def _fetch_300m_runs() -> tuple[list[FollowupRun], list[FollowupRun]]:
    api = wandb.Api(timeout=30)
    followup_runs = [
        _fetch_followup_run(
            api,
            policy=policy,
            gain=gain,
            learning_rate=learning_rate,
            artifact_version=FOLLOWUP_300M_VERSION,
        )
        for policy, gain in FOLLOWUP_300M_VARIANTS
        for learning_rate in FOLLOWUP_300M_LEARNING_RATES
    ]
    reference_runs = [
        _fetch_followup_run(
            api,
            policy=policy,
            gain=gain,
            learning_rate=learning_rate,
            artifact_version=REFERENCE_300M_VERSION,
        )
        for policy, gain in REFERENCE_300M_VARIANTS
        for learning_rate in FOLLOWUP_300M_LEARNING_RATES
    ]
    return followup_runs, reference_runs


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
                "url": "https://wandb.ai/understanding-sam/marin/runs/qwen3_130m_prism_berkeley_o5_4096_lrx1-2fd229",
                "comparison_caveat": "Fresh in-sweep Muon outperforms this run at the shared LR of 0.016.",
            },
        },
    }


def _validate_300m_runs(followup_runs: list[FollowupRun], reference_runs: list[FollowupRun]) -> None:
    expected_followup = {
        (policy, gain, learning_rate)
        for policy, gain in FOLLOWUP_300M_VARIANTS
        for learning_rate in FOLLOWUP_300M_LEARNING_RATES
    }
    expected_references = {
        (policy, gain, learning_rate)
        for policy, gain in REFERENCE_300M_VARIANTS
        for learning_rate in FOLLOWUP_300M_LEARNING_RATES
    }
    actual_followup = {(run.policy, run.gain, run.learning_rate) for run in followup_runs}
    actual_references = {(run.policy, run.gain, run.learning_rate) for run in reference_runs}
    if len(followup_runs) != len(expected_followup) or actual_followup != expected_followup:
        raise ValueError(f"300M Hesscorr grid mismatch: expected={expected_followup}, actual={actual_followup}")
    if len(reference_runs) != len(expected_references) or actual_references != expected_references:
        raise ValueError(f"300M reference grid mismatch: expected={expected_references}, actual={actual_references}")

    incomplete = [
        run.run_name
        for run in [*followup_runs, *reference_runs]
        if run.state != "finished" or run.global_step != FOLLOWUP_300M_GLOBAL_STEP
    ]
    if incomplete:
        raise ValueError(f"300M comparison contains incomplete runs: {incomplete}")
    non_finite = [
        run.run_name
        for run in [*followup_runs, *reference_runs]
        if not all(math.isfinite(value) for value in (run.c4_en_bpb, run.training_time, run.train_loss))
    ]
    if non_finite:
        raise ValueError(f"300M comparison contains non-finite metrics: {non_finite}")


def _comparison_summary(
    followup_runs: list[FollowupRun],
    reference_runs: list[FollowupRun],
) -> list[dict[str, Any]]:
    muon = {run.learning_rate: run for run in reference_runs if (run.policy, run.gain) == REFERENCE_300M_VARIANTS[0]}
    blend = {run.learning_rate: run for run in reference_runs if (run.policy, run.gain) == REFERENCE_300M_VARIANTS[1]}
    summaries = []
    for policy, gain in FOLLOWUP_300M_VARIANTS:
        variant_runs = [run for run in followup_runs if (run.policy, run.gain) == (policy, gain)]
        muon_deltas = [run.c4_en_bpb - muon[run.learning_rate].c4_en_bpb for run in variant_runs]
        blend_deltas = [run.c4_en_bpb - blend[run.learning_rate].c4_en_bpb for run in variant_runs]
        muon_overheads = [run.training_time / muon[run.learning_rate].training_time - 1.0 for run in variant_runs]
        blend_overheads = [run.training_time / blend[run.learning_rate].training_time - 1.0 for run in variant_runs]
        summaries.append(
            {
                "policy": policy,
                "gain": gain,
                "mean_delta_vs_muon": statistics.fmean(muon_deltas),
                "median_delta_vs_muon": statistics.median(muon_deltas),
                "num_wins_vs_muon": sum(delta < 0.0 for delta in muon_deltas),
                "mean_delta_vs_blend_0p05": statistics.fmean(blend_deltas),
                "median_delta_vs_blend_0p05": statistics.median(blend_deltas),
                "num_wins_vs_blend_0p05": sum(delta < 0.0 for delta in blend_deltas),
                "mean_training_time_overhead_vs_muon_fraction": statistics.fmean(muon_overheads),
                "mean_training_time_overhead_vs_blend_0p05_fraction": statistics.fmean(blend_overheads),
                "delta_vs_muon_by_learning_rate": {
                    f"{run.learning_rate:.3f}": delta for run, delta in zip(variant_runs, muon_deltas, strict=True)
                },
                "delta_vs_blend_0p05_by_learning_rate": {
                    f"{run.learning_rate:.3f}": delta for run, delta in zip(variant_runs, blend_deltas, strict=True)
                },
            }
        )
    return summaries


def build_300m_payload(
    followup_runs: list[FollowupRun],
    reference_runs: list[FollowupRun],
) -> dict[str, Any]:
    """Validate and summarize the stabilized 300M Hesscorr follow-up."""
    _validate_300m_runs(followup_runs, reference_runs)
    best = min(followup_runs, key=lambda run: run.c4_en_bpb)
    paired_muon = next(run for run in reference_runs if run.policy == "muon" and run.learning_rate == best.learning_rate)
    paired_blend = next(
        run for run in reference_runs if run.policy == "blend" and run.learning_rate == best.learning_rate
    )
    best_record = dataclasses.asdict(best)
    best_record.update(
        {
            "paired_muon_bpb": paired_muon.c4_en_bpb,
            "paired_delta_vs_muon": best.c4_en_bpb - paired_muon.c4_en_bpb,
            "paired_blend_0p05_bpb": paired_blend.c4_en_bpb,
            "paired_delta_vs_blend_0p05": best.c4_en_bpb - paired_blend.c4_en_bpb,
        }
    )
    return {
        "experiment": {
            "iris_job_id": FOLLOWUP_300M_JOB_ID,
            "artifact_version": FOLLOWUP_300M_VERSION,
            "num_hesscorr_runs": len(followup_runs),
            "num_reference_runs": len(reference_runs),
            "num_succeeded": len(followup_runs),
            "aggregate_preemptions": 124,
        },
        "setup": {
            "model": "Qwen3 300M",
            "dataset": "FineWeb-Edu 10B pretokenized cache",
            "tpu_type": "v5p-8",
            "seed": 0,
            "num_train_steps": 11_444,
            "final_global_step": FOLLOWUP_300M_GLOBAL_STEP,
            "train_batch_size": 128,
            "sequence_length": 4_096,
            "total_tokens": 5_999_951_872,
            "momentum": 0.98,
            "correction_warmup_steps": 50,
            "sylvester_steps": 400,
            "inverse_steps": 60,
        },
        "grid": {
            "learning_rates": list(FOLLOWUP_300M_LEARNING_RATES),
            "variants": [{"policy": policy, "gain": gain} for policy, gain in FOLLOWUP_300M_VARIANTS],
        },
        "best_observed_hesscorr_run": best_record,
        "comparison_summary": _comparison_summary(followup_runs, reference_runs),
        "runs": [dataclasses.asdict(run) for run in followup_runs],
        "paired_reference_runs": [dataclasses.asdict(run) for run in reference_runs],
    }


def main() -> None:
    payload = build_payload(_fetch_runs())
    followup_runs, reference_runs = _fetch_300m_runs()
    payload["three_hundred_million_hesscorr_followup"] = build_300m_payload(followup_runs, reference_runs)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    best = payload["best_observed_run"]
    best_300m = payload["three_hundred_million_hesscorr_followup"]["best_observed_hesscorr_run"]
    print(
        f"Wrote {OUTPUT_PATH}: {len(payload['runs'])} 130M runs and {len(followup_runs)} 300M runs; "
        f"best_130m={best['run_name']} bpb={best['c4_en_bpb']:.6f}; "
        f"best_300m={best_300m['run_name']} bpb={best_300m['c4_en_bpb']:.6f}"
    )


if __name__ == "__main__":
    main()
