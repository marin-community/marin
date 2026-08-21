# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wandb>=0.21",
# ]
# ///

"""Materialize the operational assessment of the WSD80 gradient-conflict canary."""

from __future__ import annotations

import csv
import datetime
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
CANARY_DESIGN_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260810"
LAUNCH_DESIGN_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_canary_results_20260811"
EXACT_REPORT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_20260810/exact_state_canary_20260810.json"
)
CHECKPOINT_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_20260810/trajectories"
)
RUN_IDS = (
    "gcf_p1_r3d28260_m100a_common-tied-035_s2026081000",
    "gcf_p1_r3d28260_m100a_common-tied-035_s2026081001",
)
CHECKPOINT_STEPS = (
    2_826,
    2_842,
    7_065,
    11_304,
    15_543,
    19_782,
    22_352,
    22_544,
    22_608,
    22_672,
    22_864,
    25_434,
    28_259,
)
TERMINAL_STEP = 28_259
METRICS = {
    "programming_languages_bpb": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    "c4_english_bpb": "eval/paloma/c4_en-llama3/bpb",
    "wikipedia_english_bpb": "eval/uncheatable_eval/wikipedia_english-llama3/bpb",
    "uncheatable_bpb": "eval/uncheatable_eval/bpb",
    "paloma_macro_bpb": "eval/paloma/macro_bpb",
    "aggregate_eval_bpb": "eval/bpb",
    "tokens_per_second": "throughput/tokens_per_second",
    "mfu_percent": "throughput/mfu",
}


def _parse_time(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _gcloud_json(path: str) -> dict[str, Any]:
    result = subprocess.run(
        ["gcloud", "storage", "cat", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _gcloud_size(path: str) -> int:
    result = subprocess.run(
        ["gcloud", "storage", "du", "--summarize", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])


def _gcloud_checkpoint_steps(path: str) -> tuple[int, ...]:
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{path}/"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = f"{path}/step-"
    return tuple(
        sorted(
            int(line.removeprefix(prefix).rstrip("/")) for line in result.stdout.splitlines() if line.startswith(prefix)
        )
    )


def _run_rows() -> list[dict[str, Any]]:
    api = wandb.Api(timeout=60)
    rows: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        run = api.run(f"marin-community/marin/{run_id}")
        if run.state != "finished":
            raise ValueError(f"Canary run is not finished: {run_id}: {run.state}")
        if int(run.summary["global_step"]) != TERMINAL_STEP:
            raise ValueError(f"Canary run did not reach step {TERMINAL_STEP}: {run_id}")
        created = _parse_time(run._attrs["createdAt"])
        heartbeat = _parse_time(run._attrs["heartbeatAt"])
        row = {
            "run_id": run_id,
            "wandb_url": run.url,
            "state": run.state,
            "global_step": int(run.summary["global_step"]),
            "wall_clock_hours": (heartbeat - created).total_seconds() / 3_600,
        }
        for label, key in METRICS.items():
            row[label] = float(run.summary[key])
        total_tokens = float(run.summary["throughput/total_tokens"])
        row["active_training_hours"] = total_tokens / row["tokens_per_second"] / 3_600
        rows.append(row)
    return rows


def _checkpoint_sizes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        checkpoint_root = f"{CHECKPOINT_ROOT}/{run_id}/2026.08.10/checkpoints"
        observed_steps = _gcloud_checkpoint_steps(checkpoint_root)
        if observed_steps != CHECKPOINT_STEPS:
            raise ValueError(f"Canary permanent checkpoint set drifted for {run_id}: {observed_steps}")
        for step in CHECKPOINT_STEPS:
            path = f"{checkpoint_root}/step-{step}"
            rows.append(
                {
                    "run_id": run_id,
                    "checkpoint_step": step,
                    "physical_bytes": _gcloud_size(path),
                    "path": path,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Metric | Seed 0 | Seed 1 | Mean | Absolute seed difference |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in METRICS:
        pair = [float(row[label]) for row in rows]
        lines.append(
            f"| {label.replace('_', ' ')} | {pair[0]:.9f} | {pair[1]:.9f} | "
            f"{statistics.fmean(pair):.9f} | {abs(pair[0] - pair[1]):.9f} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = _run_rows()
    size_rows = _checkpoint_sizes()
    exact_report = _gcloud_json(EXACT_REPORT)
    canary_design = json.loads((CANARY_DESIGN_DIR / "design_manifest.json").read_text())
    launch_design = json.loads((LAUNCH_DESIGN_DIR / "design_manifest.json").read_text())

    nonterminal_sizes = [int(row["physical_bytes"]) for row in size_rows if int(row["checkpoint_step"]) != TERMINAL_STEP]
    terminal_sizes = [int(row["physical_bytes"]) for row in size_rows if int(row["checkpoint_step"]) == TERMINAL_STEP]
    representative_state_bytes = int(statistics.median(nonterminal_sizes))
    representative_terminal_bytes = int(statistics.median(terminal_sizes))
    checkpoint_count = int(launch_design["checkpoint_count"])
    trajectory_count = int(launch_design["trajectory_count"])
    projected_bytes = checkpoint_count * representative_state_bytes
    provisioned_bytes = int(projected_bytes * 1.35)
    if int(canary_design["checkpoint_count"]) != 2_542:
        raise ValueError("Canary design checkpoint count drifted")

    exact_pairs = exact_report.get("comparisons", [])
    if len(exact_pairs) != 2:
        raise ValueError(f"Expected two exact-state comparisons, got {len(exact_pairs)}")
    if any(row.get("missing_from_parent") or row.get("missing_from_fork") for row in exact_pairs):
        raise ValueError("Exact-state report contains missing keys")
    if any(int(row.get("value_mismatch_count", 0)) != 0 or row.get("value_mismatches") for row in exact_pairs):
        raise ValueError("Exact-state report contains value mismatches")

    _write_csv(OUTPUT_DIR / "endpoint_metrics.csv", run_rows)
    _write_csv(OUTPUT_DIR / "checkpoint_sizes.csv", size_rows)
    summary = {
        "analysis_scope": "operational_canary_only",
        "scientific_h2_evidence": False,
        "run_count": len(run_rows),
        "terminal_step": TERMINAL_STEP,
        "permanent_checkpoint_steps": list(CHECKPOINT_STEPS),
        "permanent_checkpoint_set_exact": True,
        "exact_state_pair_count": len(exact_pairs),
        "representative_nonterminal_state_bytes": representative_state_bytes,
        "legacy_duplicate_terminal_path_bytes": representative_terminal_bytes,
        "terminal_duplicate_write_fixed_for_full_panel": True,
        "projected_full_panel_checkpoint_bytes": projected_bytes,
        "checkpoint_storage_envelope_bytes_with_35pct_headroom": provisioned_bytes,
    }
    (OUTPUT_DIR / "canary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    mean_active = statistics.fmean(float(row["active_training_hours"]) for row in run_rows)
    mean_wall = statistics.fmean(float(row["wall_clock_hours"]) for row in run_rows)
    report = f"""# StarCoder WSD80 gradient-conflict canary assessment

## Verdict

The two-parent canary passes its operational purpose. Both parents completed the
28,260-update trajectory, retained exactly all 13 preregistered permanent states
with no extras, and reproduced a separately retained state byte-for-byte after
an exact 16-update fork. The two endpoint measurements and runtime rates are
close across seeds.

This is **not** evidence for or against the gradient-conflict hypotheses. No
source or target gradients were computed, and two seeds cannot estimate the
preregistered 24-seed H2 effect. The canary outcomes must not set numerical
reliability thresholds or alter the H2 decision rule.

## Endpoint and runtime agreement

{_metric_table(run_rows)}

The mean active-training estimate is {mean_active:.2f} hours per parent at the
reported throughput. Mean W&B creation-to-final-heartbeat time is
{mean_wall:.2f} hours; the difference includes two recovered infrastructure
preemption waves and restart overhead.

## Exact continuation

Both exact forks loaded the complete parent state at step 2,826 and replayed 16
updates to step 2,842. Each comparison covered 175 logical keys and
2,337,162,393 logical bytes with zero missing keys and zero value mismatches.
This verifies model, optimizer, RNG, logical-step, and schedule continuation for
the tested configuration. It does not yet validate source-switch rollouts or
the gradient-probe implementation.

## Storage

A nonterminal full state occupies approximately
{representative_state_bytes / 1e9:.3f} GB. The legacy terminal path occupies
{representative_terminal_bytes / 1e9:.3f} GB. Inspection showed one OCDBT state
with unreachable duplicate physical objects, not a separate model export: the
terminal step matched both a permanent step policy and the trainer's
forced-final hook. The checkpointer now waits for the existing permanent write
rather than writing that path twice. Applying one measured state to
{checkpoint_count:,} review-v6 permanent states across {trajectory_count}
trajectories projects
{projected_bytes / 1e12:.3f} TB. A 35% operating envelope is
{provisioned_bytes / 1e12:.3f} TB. Temporary rolling states remain separate and
must retain only one recovery checkpoint per trajectory.

## Remaining staged evidence

Training and probe fanout now have separate readiness gates. Training may stage
after the decay-crossing exact fork, full runtime-config and regional-cache
audits, and independent review. The frozen 112-row two-seed numerical preflight
must still validate gradient signs, norm correction, block stability,
optimizer-update reproduction, and the 0.0001-BPB H4 resolution target before
probe and rollout fanout, without opening H2a signs.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


if __name__ == "__main__":
    main()
