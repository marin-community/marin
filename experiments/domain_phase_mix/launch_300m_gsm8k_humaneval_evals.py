# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch missing GSM8K/HumanEval evals for 300M data-mixture rows.

This covers the 300M/6B qsplit-core signal panel plus the run_00097 fixed-seed
noise panel needed for signal-to-noise estimates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.rl.placement import marin_prefix_for_region
import pandas as pd

from experiments.domain_phase_mix.launch_baseline_scaling_downstream_evals import (
    GENERATION_ENGINE_KWARGS,
    _read_eval_metrics,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
METRICS_WIDE_CSV = METRIC_REGISTRY_DIR / "metrics_wide.csv"
RUN_REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
BASELINE_SCALING_DOWNSTREAM_METRICS_CSV = (
    SCRIPT_DIR / "exploratory" / "paper_plots" / "img" / "baseline_scaling_downstream_eval_metrics_merged.csv"
)
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_gsm8k_humaneval_completion"
STATE_CSV = OUTPUT_DIR / "300m_gsm8k_humaneval_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_gsm8k_humaneval_eval_launch_manifest.csv"

RUN00097_300M_FIXED_SUBSET_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_300m_6b_fixed_subset/collect_results-605e6a/results.csv"
)
RUN00097_300M_FIXED_SUBSET_CHECKPOINT_PREFIX = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset"
)

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_gsm8k_humaneval_evals"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_EXPECTED_300M_STEP = 22887
RESULTS_CSV = "300m_gsm8k_humaneval_eval_results.csv"
STATE_OUTPUT_CSV = "300m_gsm8k_humaneval_eval_state.csv"

GSM8K_METRIC = "lm_eval/gsm8k/exact_match,flexible-extract"
HUMANEVAL_METRIC = "lm_eval/humaneval/pass@1,create_test"
BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "has_gsm8k", "has_humaneval", "eligible"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
    "existing_artifact_count",
}


@dataclass(frozen=True)
class EvalCandidate:
    """One candidate checkpoint to evaluate."""

    panel: str
    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int
    has_gsm8k_metric: bool
    has_humaneval_metric: bool


@dataclass(frozen=True)
class EvalSpec:
    """One downstream-eval state row and potential launch unit."""

    eval_key: str
    panel: str
    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int
    hf_checkpoint_count: int
    hf_checkpoint_latest: str
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    existing_artifact_count: int
    existing_tasks: str
    has_gsm8k: bool
    has_humaneval: bool
    task_aliases: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class Collect300MGsm8kHumanEvalResultsConfig:
    """Config for collecting 300M GSM8K/HumanEval outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _executor_prefix(executor_prefix: str | None, default_region: str) -> str | None:
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_region), executor_prefix)


def _exact_hf_checkpoint(checkpoint_root: str, expected_step: int) -> str:
    if not checkpoint_root.startswith("gs://") or expected_step < 0:
        return ""
    checkpoint = f"{checkpoint_root.rstrip('/')}/hf/step-{expected_step}"
    tokenizer_config = f"{checkpoint}/tokenizer_config.json"
    config = f"{checkpoint}/config.json"
    try:
        with fsspec.open(tokenizer_config, "rb"):
            pass
        with fsspec.open(config, "rb"):
            pass
    except OSError:
        return ""
    return checkpoint


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _metric_coverage_by_root(paths: list[str | Path]) -> dict[str, dict[str, bool]]:
    coverage: dict[str, dict[str, bool]] = {}
    for path in paths:
        if isinstance(path, Path) and not path.exists():
            continue
        frame = _read_csv(path)
        if "checkpoint_root" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            entry = coverage.setdefault(root, {"gsm8k": False, "humaneval": False})
            if GSM8K_METRIC in frame.columns:
                entry["gsm8k"] = entry["gsm8k"] or pd.notna(row.get(GSM8K_METRIC))
            if HUMANEVAL_METRIC in frame.columns:
                entry["humaneval"] = entry["humaneval"] or pd.notna(row.get(HUMANEVAL_METRIC))
    return coverage


def _metric_registry_candidates(coverage: dict[str, dict[str, bool]]) -> list[EvalCandidate]:
    if not METRICS_WIDE_CSV.exists():
        raise FileNotFoundError(f"Missing metric registry {METRICS_WIDE_CSV}")
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    signal = frame[frame["scale"].eq("300m_6b")].copy()
    candidates: list[EvalCandidate] = []
    for _, row in signal.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root:
            continue
        covered = coverage.get(root, {})
        candidates.append(
            EvalCandidate(
                panel="signal_300m_6b",
                run_name=_string_value(row.get("run_name")),
                registry_key=_string_value(row.get("registry_run_key")),
                source_experiment=_string_value(row.get("source_experiment")),
                cohort=_string_value(row.get("cohort")),
                checkpoint_root=root,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
                has_gsm8k_metric=covered.get("gsm8k", False),
                has_humaneval_metric=covered.get("humaneval", False),
            )
        )
    return candidates


def _fixed_seed_noise_candidates(coverage: dict[str, dict[str, bool]]) -> list[EvalCandidate]:
    frame = _read_csv(RUN00097_300M_FIXED_SUBSET_RESULTS_URI)
    if "cohort" not in frame.columns:
        raise ValueError(f"Fixed-seed noise CSV missing required columns: {RUN00097_300M_FIXED_SUBSET_RESULTS_URI}")
    seed_rows = frame[frame["cohort"].eq("seed_sweep")].copy()
    if len(seed_rows) != 10:
        raise ValueError(f"Expected 10 fixed-seed noise rows, found {len(seed_rows)}")
    candidates: list[EvalCandidate] = []
    for _, row in seed_rows.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root:
            wandb_run_id = _string_value(row.get("wandb_run_id")).rstrip("/")
            if wandb_run_id:
                root = f"{RUN00097_300M_FIXED_SUBSET_CHECKPOINT_PREFIX}/{wandb_run_id}"
        if not root:
            raise ValueError(f"Fixed-seed row missing checkpoint_root:\n{row.to_string()}")
        covered = coverage.get(root, {})
        candidates.append(
            EvalCandidate(
                panel="fixed_seed_noise_300m_6b",
                run_name=_string_value(row.get("run_name")),
                registry_key=f"fixed_seed_noise_300m_6b:{_string_value(row.get('run_name'))}",
                source_experiment="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset",
                cohort=_string_value(row.get("cohort")),
                checkpoint_root=root,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
                has_gsm8k_metric=covered.get("gsm8k", False),
                has_humaneval_metric=covered.get("humaneval", False),
            )
        )
    return candidates


def _expected_step_from_run_registry(row: pd.Series) -> int:
    for column in ("target_final_checkpoint_step", "target_eval_step", "max_checkpoint_step"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return int(value)
    return DEFAULT_EXPECTED_300M_STEP


def _registry_extra_candidates(
    coverage: dict[str, dict[str, bool]],
    existing_roots: set[str],
) -> list[EvalCandidate]:
    if not RUN_REGISTRY_CSV.exists():
        return []
    frame = pd.read_csv(RUN_REGISTRY_CSV, low_memory=False)
    ready = frame[
        frame["scale"].eq("300m_6b")
        & frame["is_perplexity_ready"].map(_bool_value)
        & pd.to_numeric(frame["target_budget_multiplier"], errors="coerce").eq(1.0)
    ].copy()
    candidates: list[EvalCandidate] = []
    for _, row in ready.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root or root in existing_roots:
            continue
        covered = coverage.get(root, {})
        candidates.append(
            EvalCandidate(
                panel="registry_extra_300m_6b",
                run_name=_string_value(row.get("run_name")),
                registry_key=_string_value(row.get("registry_id")),
                source_experiment=_string_value(row.get("source_experiment")),
                cohort=_string_value(row.get("study_cohort")),
                checkpoint_root=root,
                expected_checkpoint_step=_expected_step_from_run_registry(row),
                has_gsm8k_metric=covered.get("gsm8k", False),
                has_humaneval_metric=covered.get("humaneval", False),
            )
        )
    return candidates


def _candidate_records() -> list[EvalCandidate]:
    coverage = _metric_coverage_by_root([METRICS_WIDE_CSV, BASELINE_SCALING_DOWNSTREAM_METRICS_CSV])
    candidates = _metric_registry_candidates(coverage)
    candidates.extend(_fixed_seed_noise_candidates(coverage))
    roots = {candidate.checkpoint_root for candidate in candidates}
    candidates.extend(_registry_extra_candidates(coverage, roots))

    by_root: dict[str, EvalCandidate] = {}
    for candidate in candidates:
        if candidate.checkpoint_root in by_root:
            continue
        by_root[candidate.checkpoint_root] = candidate
    return sorted(by_root.values(), key=lambda row: (row.panel, row.run_name))


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    has_gsm8k: bool,
    has_humaneval: bool,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if has_gsm8k and has_humaneval:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
) -> list[EvalSpec]:
    """Build state rows for 300M GSM8K/HumanEval completion."""
    rows: list[EvalSpec] = []
    for idx, candidate in enumerate(_candidate_records()):
        latest_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        latest_hf_step = candidate.expected_checkpoint_step if latest_hf_checkpoint else -1
        has_exact_hf_checkpoint = bool(latest_hf_checkpoint)
        has_gsm8k = candidate.has_gsm8k_metric
        has_humaneval = candidate.has_humaneval_metric
        eligible, decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            has_gsm8k=has_gsm8k,
            has_humaneval=has_humaneval,
        )
        suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
        eval_key = f"gsmhe300m_{idx:03d}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            EvalSpec(
                eval_key=eval_key,
                panel=candidate.panel,
                run_name=candidate.run_name,
                registry_key=candidate.registry_key,
                source_experiment=candidate.source_experiment,
                cohort=candidate.cohort,
                checkpoint_root=candidate.checkpoint_root,
                expected_checkpoint_step=candidate.expected_checkpoint_step,
                hf_checkpoint_count=int(bool(latest_hf_checkpoint)),
                hf_checkpoint_latest=latest_hf_checkpoint,
                hf_checkpoint_latest_step=latest_hf_step,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                existing_artifact_count=0,
                existing_tasks="",
                has_gsm8k=has_gsm8k,
                has_humaneval=has_humaneval,
                task_aliases="gsm8k_5shot;humaneval_10shot",
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=decision,
                step_name=f"evaluation/lm_evaluation_harness/{eval_key}",
                result_path=f"executor_output:{eval_key}",
            )
        )
    return rows


def _write_local_outputs(rows: list[EvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = pd.DataFrame.from_records([asdict(row) for row in rows])
    state.to_csv(STATE_CSV, index=False)
    state[state["launch_decision"] == "launch"].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _load_state_rows(path: Path | str) -> list[EvalSpec]:
    frame = _read_csv(path)
    missing_columns = sorted(field.name for field in fields(EvalSpec) if field.name not in frame.columns)
    if missing_columns:
        raise ValueError(f"State CSV {path} is missing columns: {missing_columns}")
    rows: list[EvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(EvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(EvalSpec(**kwargs))
    return rows


def _metric_rows_from_result_paths(
    state_rows: list[EvalSpec], results_by_eval_key: dict[str, InputName]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        result_path = results_by_eval_key.get(row.eval_key)
        if result_path is None:
            record["collection_status"] = "not_launched"
            records.append(record)
            continue
        metrics, error = _read_eval_metrics(result_path)
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)
    return records


def collect_eval_results(config: Collect300MGsm8kHumanEvalResultsConfig) -> None:
    """Collect downstream eval outputs into one normalized CSV."""
    state_rows = [EvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    records = _metric_rows_from_result_paths(state_rows, config.results_by_eval_key)
    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    state_rows: list[EvalSpec], max_eval_instances: int | None
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build GSM8K/HumanEval eval steps for rows requiring launch."""
    from experiments.evals.evals import evaluate_lm_evaluation_harness
    from experiments.evals.task_configs import GSM8K_5_SHOT, HUMANEVAL_10_SHOT

    eval_steps: list[ExecutorStep] = []
    results_by_eval_key: dict[str, InputName] = {}
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone,
        )
        eval_step = evaluate_lm_evaluation_harness(
            model_name=row.eval_key,
            model_path=row.checkpoint_root,
            evals=[GSM8K_5_SHOT, HUMANEVAL_10_SHOT],
            max_eval_instances=max_eval_instances,
            engine_kwargs=GENERATION_ENGINE_KWARGS,
            resource_config=resource_config,
            discover_latest_checkpoint=True,
            wandb_tags=["300m", "gsm8k-humaneval", "snr"],
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[EvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 300M GSM8K/HumanEval eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=Collect300MGsm8kHumanEvalResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--state-csv")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.state_csv is None:
        state_rows = build_state_rows(
            default_tpu_type=args.tpu_type,
            default_tpu_region=args.tpu_region,
            default_tpu_zone=args.tpu_zone,
            eval_key_suffix=args.eval_key_suffix,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d GSM8K/HumanEval eval steps", launch_count)
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(state_rows, args.max_eval_instances)
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*eval_steps, collect_step],
        description=f"{args.name_prefix}: 300M GSM8K/HumanEval completion for signal/noise SNR",
    )


if __name__ == "__main__":
    main()
