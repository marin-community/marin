# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "wandb",
# ]
# ///
"""Inventory smooth-target trajectories available for temporal-state identification."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path

import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "intermediate_target_trajectory_audit_20260731"

WSD80_SOURCE = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
PAIRED_TRAJECTORY_SOURCE = REFERENCE_OUTPUTS / "tied_two_phase_trajectory_audit_20260726" / "wandb_histories.csv"

WANDB_PATH = "marin-community/marin"
WSD80_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
UNCHEATABLE_TARGET = "eval/uncheatable_eval/bpb"
WSD80_PHASE_BOUNDARY_STEP = 3040
WSD80_FINAL_STEP = 3813
THREE_HUNDRED_M_PHASE_BOUNDARY_STEP = 18_310
THREE_HUNDRED_M_FINAL_STEP = 22_887


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-wsd80", action="store_true")
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def wsd80_manifest() -> pd.DataFrame:
    frame = pd.read_csv(WSD80_SOURCE).dropna(subset=["wandb_run_id", "wsd80_bpb"]).copy()
    columns = [
        "wandb_run_id",
        "wandb_run_name",
        "wandb_url",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "wsd80_bpb",
        "panel",
    ]
    return frame[columns].drop_duplicates("wandb_run_id").reset_index(drop=True)


def fetch_wsd80_history(row: object) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=90)
    run = api.run(f"{WANDB_PATH}/{row.wandb_run_id}")
    history = run.history(
        keys=["global_step", "run_progress", WSD80_TARGET, UNCHEATABLE_TARGET],
        samples=10_000,
        pandas=True,
    )
    available = [
        column for column in ("global_step", "run_progress", WSD80_TARGET, UNCHEATABLE_TARGET) if column in history
    ]
    history = history[available].copy()
    history = history.loc[history[WSD80_TARGET].notna()].copy()
    history["wandb_run_id"] = run.id
    history["wandb_run_name"] = run.name
    history["phase_0_starcoder"] = row.phase_0_starcoder
    history["phase_1_starcoder"] = row.phase_1_starcoder
    history["panel"] = row.panel

    checkpointer = run.config.get("trainer", {}).get("checkpointer", {})
    config = {
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_state": run.state,
        "checkpoint_base_path": checkpointer.get("base_path"),
        "checkpoint_keep": json.dumps(checkpointer.get("keep")),
        "hf_save_path": run.config.get("hf_save_path"),
        "hf_save_steps": run.config.get("hf_save_steps"),
    }
    return history, config


def collect_wsd80(
    manifest: pd.DataFrame,
    history_path: Path,
    config_path: Path,
    *,
    refresh: bool,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history_path.exists() and config_path.exists() and not refresh:
        return pd.read_csv(history_path), pd.read_csv(config_path)

    histories: list[pd.DataFrame] = []
    configs: list[dict[str, object]] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(fetch_wsd80_history, row): row for row in manifest.itertuples(index=False)}
        for index, future in enumerate(futures.as_completed(pending), start=1):
            row = pending[future]
            try:
                history, config = future.result()
            except Exception as error:
                raise RuntimeError(f"Failed to fetch WSD80 run {row.wandb_run_id}") from error
            histories.append(history)
            configs.append(config)
            if index % 50 == 0 or index == len(pending):
                print(f"Fetched {index}/{len(pending)} WSD80 histories", flush=True)

    combined = pd.concat(histories, ignore_index=True)
    config_frame = pd.DataFrame(configs)
    combined.to_csv(history_path, index=False)
    config_frame.to_csv(config_path, index=False)
    return combined, config_frame


def run_inventory(
    histories: pd.DataFrame,
    *,
    panel: str,
    target: str,
    boundary_step: int,
    final_step: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, block in histories.groupby("wandb_run_id", sort=True):
        block = block.loc[block[target].notna()].sort_values("global_step")
        steps = block["global_step"].astype(int).tolist()
        rows.append(
            {
                "panel": panel,
                "wandb_run_id": run_id,
                "n_target_evaluations": len(block),
                "first_step": min(steps),
                "last_step": max(steps),
                "step_sequence": ";".join(str(step) for step in steps),
                "has_pre_switch": any(step < boundary_step for step in steps),
                "has_at_switch": boundary_step in steps,
                "has_post_switch_before_final": any(boundary_step < step < final_step for step in steps),
                "has_final": final_step in steps,
            }
        )
    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame) -> dict[str, object]:
    step_sequences = inventory["step_sequence"].value_counts()
    return {
        "runs": len(inventory),
        "target_evaluations": int(inventory["n_target_evaluations"].sum()),
        "min_evaluations_per_run": int(inventory["n_target_evaluations"].min()),
        "median_evaluations_per_run": float(inventory["n_target_evaluations"].median()),
        "max_evaluations_per_run": int(inventory["n_target_evaluations"].max()),
        "runs_with_pre_switch": int(inventory["has_pre_switch"].sum()),
        "runs_with_at_switch": int(inventory["has_at_switch"].sum()),
        "runs_with_post_switch_before_final": int(inventory["has_post_switch_before_final"].sum()),
        "runs_with_final": int(inventory["has_final"].sum()),
        "most_common_step_sequence": str(step_sequences.index[0]),
        "runs_with_most_common_step_sequence": int(step_sequences.iloc[0]),
    }


def render_report(
    wsd_summary: dict[str, object],
    three_hundred_m_summary: dict[str, object],
    wsd_configs: pd.DataFrame,
) -> str:
    finished = int(wsd_configs["wandb_state"].eq("finished").sum())
    checkpoint_keep = sorted(wsd_configs["checkpoint_keep"].dropna().astype(str).unique().tolist())
    hf_save_steps = sorted(wsd_configs["hf_save_steps"].dropna().astype(int).unique().tolist())
    wsd_coverage = coverage_row("WSD80 StarCoder", wsd_summary)
    three_hundred_m_coverage = coverage_row("300M paired policies", three_hundred_m_summary)
    return f"""# Intermediate Smooth-Target Trajectory Audit

## Decision

Existing data are sufficient to identify a temporal state **partially**, without using final endpoint
targets to choose its transition law:

- The 300M 39-bucket paired panel has dense fixed-distribution Uncheatable trajectories through both
  phases.
- WSD80 has three smooth-target measurements during phase 0 and one endpoint after phase 1. It can
  identify acquisition during phase 0 and the net phase-1 update, but not the shape of forgetting or
  consolidation inside phase 1.
- Native Table-9 is endpoint-only. It cannot identify transition dynamics and must remain a downstream
  target for a frozen state representation.

Training loss is not substituted for a smooth target. It is on-policy and changes meaning when the
sampled data distribution switches.

## Coverage

| Panel | Runs | Target rows | Evals/run (min/median/max) | Pre-switch | At switch | Post-switch before final | Final |
|---|---:|---:|---:|---:|---:|---:|---:|
{wsd_coverage}
{three_hundred_m_coverage}

The most common WSD80 sequence is `{wsd_summary["most_common_step_sequence"]}` across
{wsd_summary["runs_with_most_common_step_sequence"]} runs; the phase boundary is step
{WSD80_PHASE_BOUNDARY_STEP}. The most common 300M sequence is
`{three_hundred_m_summary["most_common_step_sequence"]}` across
{three_hundred_m_summary["runs_with_most_common_step_sequence"]} runs; the phase boundary lies between
steps 18,000 and 19,000.

## Persistence Audit

- WSD80 W&B coverage: {finished}/{len(wsd_configs)} audited runs are finished.
- WSD80 checkpoint keep configurations: `{checkpoint_keep}`; `hf_save_steps` values:
  `{hf_save_steps}`.
- Narrow GCS checks on a representative WSD80 run and a representative 300M run found only the final
  model checkpoint (`step-3813` and `step-22887`, respectively), plus `eval_metrics.jsonl`.
- The smooth trajectories are nevertheless durable in both W&B histories and the persisted
  `eval_metrics.jsonl` files. Intermediate model weights are not retained.

## Identification Consequences

1. Fit the aggregate spine only on physically tied endpoints.
2. Freeze a low-dimensional transition law using 300M Uncheatable trajectories and WSD80 phase-0
   trajectories plus the net phase-1 update.
3. Fit target-specific response coefficients only after the latent transition is frozen.
4. Use Table-9 only to test whether that frozen state transfers; do not tune the transition against
   Table-9 endpoints.
5. Do not claim within-phase-1 retention or forgetting is separately identified. Existing WSD80 has no
   smooth target between the switch and endpoint.

## Provenance

- WSD80 manifest: `{WSD80_SOURCE.relative_to(SCRIPT_DIR.parents[4])}`
- 300M cached histories: `{PAIRED_TRAJECTORY_SOURCE.relative_to(SCRIPT_DIR.parents[4])}`
- WSD80 target: `{WSD80_TARGET}`
- 300M target: `{UNCHEATABLE_TARGET}`
"""


def coverage_row(label: str, summary: dict[str, object]) -> str:
    evaluations = (
        f"{summary['min_evaluations_per_run']}/{summary['median_evaluations_per_run']:.0f}/"
        f"{summary['max_evaluations_per_run']}"
    )
    values = [
        label,
        summary["runs"],
        summary["target_evaluations"],
        evaluations,
        summary["runs_with_pre_switch"],
        summary["runs_with_at_switch"],
        summary["runs_with_post_switch_before_final"],
        summary["runs_with_final"],
    ]
    return "| " + " | ".join(str(value) for value in values) + " |"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = wsd80_manifest()
    history_path = args.output_dir / "wsd80_target_histories.csv"
    config_path = args.output_dir / "wsd80_run_configs.csv"
    wsd_histories, wsd_configs = collect_wsd80(
        manifest,
        history_path,
        config_path,
        refresh=args.refresh_wsd80,
        max_workers=args.max_workers,
    )
    wsd_inventory = run_inventory(
        wsd_histories,
        panel="starcoder_wsd80",
        target=WSD80_TARGET,
        boundary_step=WSD80_PHASE_BOUNDARY_STEP,
        final_step=WSD80_FINAL_STEP,
    )

    three_hundred_m_histories = pd.read_csv(PAIRED_TRAJECTORY_SOURCE)
    three_hundred_m_histories = three_hundred_m_histories.loc[three_hundred_m_histories["scale_key"].eq("300m")].copy()
    three_hundred_m_inventory = run_inventory(
        three_hundred_m_histories,
        panel="300m_paired",
        target=UNCHEATABLE_TARGET,
        boundary_step=THREE_HUNDRED_M_PHASE_BOUNDARY_STEP,
        final_step=THREE_HUNDRED_M_FINAL_STEP,
    )

    inventory = pd.concat([wsd_inventory, three_hundred_m_inventory], ignore_index=True)
    inventory.to_csv(args.output_dir / "trajectory_inventory.csv", index=False)

    summary = {
        "starcoder_wsd80": summarize_inventory(wsd_inventory),
        "300m_paired": summarize_inventory(three_hundred_m_inventory),
        "table9": {
            "availability": "endpoint_only",
            "admissible_use": "frozen-state transfer evaluation",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = render_report(summary["starcoder_wsd80"], summary["300m_paired"], wsd_configs)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
