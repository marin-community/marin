# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Build DCLM Core eval state rows for the 300M proportional-noise baseline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.launch_300m_dclm_core_evals import (
    DEFAULT_EXPECTED_300M_STEP,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    DCLM_GENERATION_SMOOTH_MODE,
    DCLMEvalCandidate,
    DCLMEvalMode,
    DCLMEvalSpec,
    _checkpoint_region,
    _launch_decision,
    _metric_coverage_by_root,
    _modes_for_aliases,
    _row_mode_for_task_mode,
    _slug,
    _step_name_for_eval_key,
    _task_has_dclm_smooth_metric,
    dclm_core_task_aliases,
    split_state_rows_by_task_alias,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import _string_value
from experiments.evals.dclm_core import task_by_alias

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
RAW_MATRIX_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "raw_metric_matrix_300m"
DEFAULT_INPUT_CSV = RAW_MATRIX_DIR / "noise_baseline_proportional_variable_subset_300m_6b.csv"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "300m_dclm_core_completion"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "300m_dclm_proportional_noise_eval_state_20260614.csv"
EXPECTED_REPEAT_COUNT = 10
PANEL = "proportional_variable_subset_noise_300m_6b"
COHORT = "proportional_noise"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--eval-key-suffix", default="dclm_noise_20260614")
    parser.add_argument("--split-task-alias-rows", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _expected_step_from_noise_row(row: pd.Series) -> int:
    for column in ("target_final_checkpoint_step", "target_eval_step", "max_checkpoint_step"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return int(value)
    return DEFAULT_EXPECTED_300M_STEP


def _completed_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" in frame.columns:
        return frame["status"].astype(str).str.lower().eq("completed")
    return frame["checkpoint_root"].notna()


def proportional_noise_candidates(
    input_csv: Path,
    *,
    allow_incomplete: bool = False,
) -> list[DCLMEvalCandidate]:
    """Return completed proportional-noise 300M checkpoints as DCLM candidates."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing proportional-noise matrix: {input_csv}")
    frame = pd.read_csv(input_csv, low_memory=False)
    required_columns = {"run_name", "registry_run_key", "source_experiment", "checkpoint_root"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Proportional-noise matrix missing columns: {missing_columns}")
    if len(frame) != EXPECTED_REPEAT_COUNT and not allow_incomplete:
        raise ValueError(f"Expected {EXPECTED_REPEAT_COUNT} proportional-noise rows, found {len(frame)}")
    frame = frame.loc[_completed_mask(frame)].copy()
    if len(frame) != EXPECTED_REPEAT_COUNT and not allow_incomplete:
        raise ValueError(f"Expected {EXPECTED_REPEAT_COUNT} completed proportional-noise rows, found {len(frame)}")
    if frame["checkpoint_root"].duplicated().any():
        dupes = frame.loc[frame["checkpoint_root"].duplicated(keep=False), ["run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate checkpoint roots:\n{dupes.to_string(index=False)}")

    candidates: list[DCLMEvalCandidate] = []
    for _, row in frame.sort_values("run_name").iterrows():
        checkpoint_root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not checkpoint_root:
            if allow_incomplete:
                continue
            raise ValueError(f"Missing checkpoint_root for run {row.get('run_name')!r}")
        candidates.append(
            DCLMEvalCandidate(
                panel=PANEL,
                run_name=_string_value(row["run_name"]),
                registry_key=_string_value(row["registry_run_key"]),
                source_experiment=_string_value(row["source_experiment"]),
                cohort=COHORT,
                checkpoint_root=checkpoint_root,
                expected_checkpoint_step=_expected_step_from_noise_row(row),
            )
        )
    return candidates


def _smooth_coverage_by_root(paths: list[Path | str], task_aliases: tuple[str, ...]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for path in paths:
        if isinstance(path, Path) and not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False) if isinstance(path, Path) else pd.read_csv(path)
        if "checkpoint_root" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            covered = coverage.setdefault(root, set())
            for alias in task_aliases:
                if _task_has_dclm_smooth_metric(row, alias):
                    covered.add(alias)
    return coverage


def build_state_rows_for_candidates(
    candidates: list[DCLMEvalCandidate],
    *,
    mode: DCLMEvalMode,
    eval_key_suffix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    coverage_paths: list[Path | str],
) -> list[DCLMEvalSpec]:
    """Build DCLM eval state rows for an explicit candidate set."""
    requested_aliases = dclm_core_task_aliases(mode=mode)
    coverage = (
        _smooth_coverage_by_root(coverage_paths, requested_aliases)
        if mode == DCLMEvalMode.SMOOTH
        else _metric_coverage_by_root(coverage_paths, requested_aliases)
    )
    rows: list[DCLMEvalSpec] = []
    for idx, candidate in enumerate(candidates):
        latest_hf_checkpoint = f"{candidate.checkpoint_root.rstrip('/')}/hf/step-{candidate.expected_checkpoint_step}"
        checkpoint_region = _checkpoint_region(candidate.checkpoint_root)
        is_region_local = checkpoint_region in {"", tpu_region}
        existing_tasks = sorted(coverage.get(candidate.checkpoint_root, set()))
        for task_mode in _modes_for_aliases(requested_aliases):
            mode_aliases = tuple(alias for alias in requested_aliases if task_by_alias(alias).mode == task_mode)
            existing_mode_tasks = [alias for alias in existing_tasks if alias in mode_aliases]
            missing_tasks = [alias for alias in mode_aliases if alias not in existing_mode_tasks]
            has_all_tasks = not missing_tasks
            eligible, decision = _launch_decision(
                checkpoint_root=candidate.checkpoint_root,
                has_exact_hf_checkpoint=True,
                is_region_local=is_region_local,
                has_all_tasks=has_all_tasks,
            )
            row_mode = _row_mode_for_task_mode(task_mode, mode)
            suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
            eval_key = f"dclm300m_{idx:03d}_{row_mode}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
            rows.append(
                DCLMEvalSpec(
                    eval_key=eval_key,
                    mode=row_mode,
                    panel=candidate.panel,
                    run_name=candidate.run_name,
                    registry_key=candidate.registry_key,
                    source_experiment=candidate.source_experiment,
                    cohort=candidate.cohort,
                    checkpoint_root=candidate.checkpoint_root,
                    expected_checkpoint_step=candidate.expected_checkpoint_step,
                    hf_checkpoint_count=1,
                    hf_checkpoint_latest=latest_hf_checkpoint,
                    hf_checkpoint_latest_step=candidate.expected_checkpoint_step,
                    has_exact_hf_checkpoint=True,
                    checkpoint_region=checkpoint_region,
                    is_region_local=is_region_local,
                    existing_artifact_count=len(existing_mode_tasks),
                    existing_tasks=";".join(existing_mode_tasks),
                    missing_task_count=len(missing_tasks),
                    missing_tasks=";".join(missing_tasks),
                    has_all_tasks=has_all_tasks,
                    task_aliases=";".join(mode_aliases),
                    launch_tpu_type=tpu_type,
                    launch_tpu_region=tpu_region,
                    launch_tpu_zone=tpu_zone,
                    eligible=eligible,
                    launch_decision=decision,
                    step_name=_step_name_for_eval_key(row_mode, eval_key),
                    result_path=f"executor_output:{eval_key}",
                )
            )
    return rows


def build_combined_state_rows(
    *,
    input_csv: Path,
    eval_key_suffix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    split_task_alias_rows: bool,
    allow_incomplete: bool = False,
) -> list[DCLMEvalSpec]:
    """Build hard plus smooth DCLM rows for proportional-noise repeats.

    MCQ hard rows also emit MCQ smooth scalars when the DCLM launcher is run
    with ``--sample-smooth-metrics``, so only generation tasks need separate
    ``generation_smooth`` rows.
    """
    candidates = proportional_noise_candidates(input_csv, allow_incomplete=allow_incomplete)
    coverage_paths = []
    hard_rows = build_state_rows_for_candidates(
        candidates,
        mode=DCLMEvalMode.ALL,
        eval_key_suffix=eval_key_suffix,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        coverage_paths=coverage_paths,
    )
    smooth_rows = build_state_rows_for_candidates(
        candidates,
        mode=DCLMEvalMode.SMOOTH,
        eval_key_suffix=f"{eval_key_suffix}_smooth",
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        coverage_paths=coverage_paths,
    )
    smooth_rows = [row for row in smooth_rows if row.mode == DCLM_GENERATION_SMOOTH_MODE]
    rows = [
        replace(row, launch_decision="launch", eligible=row.eligible)
        for row in [*hard_rows, *smooth_rows]
        if row.eligible
    ]
    return split_state_rows_by_task_alias(rows) if split_task_alias_rows else rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    rows = build_combined_state_rows(
        input_csv=args.input_csv,
        eval_key_suffix=args.eval_key_suffix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        split_task_alias_rows=args.split_task_alias_rows,
        allow_incomplete=args.allow_incomplete,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in rows]).to_csv(args.output_csv, index=False)
    launch_count = sum(row.launch_decision == "launch" for row in rows)
    logger.info("Wrote %d DCLM proportional-noise state rows to %s", len(rows), args.output_csv)
    logger.info("Launch rows: %d", launch_count)


if __name__ == "__main__":
    main()
