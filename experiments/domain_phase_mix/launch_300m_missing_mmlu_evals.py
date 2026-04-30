# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch missing MMLU evals for target-ready 300M data-mixture runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

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
    _checkpoint_hf_checkpoints,
    _checkpoint_lm_eval_artifacts,
    _checkpoint_step_from_path,
    _read_eval_metrics,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRICS_WIDE_CSV = TWO_PHASE_MANY_DIR / "metric_registry" / "metrics_wide.csv"
RUN_REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"
BASELINE_SCALING_METRICS_CSV = (
    SCRIPT_DIR / "exploratory" / "paper_plots" / "img" / "baseline_scaling_downstream_eval_metrics_merged.csv"
)
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "300m_missing_mmlu_recovery"
STATE_CSV = OUTPUT_DIR / "300m_missing_mmlu_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_missing_mmlu_eval_launch_manifest.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_missing_mmlu_evals"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 256
RESULTS_CSV = "300m_missing_mmlu_eval_results.csv"
STATE_OUTPUT_CSV = "300m_missing_mmlu_eval_state.csv"
STANDARD_MMLU_ACC = "lm_eval/mmlu_5shot/acc"
SL_VERB_MMLU_ACC = "lm_eval/mmlu_sl_verb_5shot/acc"


@dataclass(frozen=True)
class MissingMmluEvalSpec:
    """One missing 300M MMLU eval row."""

    eval_key: str
    run_name: str
    registry_key: str
    family: str
    scale: str
    cohort: str
    source_experiment: str
    target_budget_multiplier: str
    checkpoint_root: str
    expected_checkpoint_step: int
    hf_checkpoint_count: int
    hf_checkpoint_latest: str
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    existing_artifact_count: int
    existing_artifact_tasks: str
    has_mmlu_5shot: bool
    has_mmlu_sl_verb_5shot: bool
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    launch_decision: str
    result_path: str


@dataclass(frozen=True)
class Collect300MMissingMmluResultsConfig:
    """Config for collecting 300M missing-MMLU eval outputs."""

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


def _load_metric_coverage() -> dict[str, dict[str, bool]]:
    coverage: dict[str, dict[str, bool]] = {}
    if METRICS_WIDE_CSV.exists():
        frame = pd.read_csv(
            METRICS_WIDE_CSV,
            usecols=lambda column: column in {"checkpoint_root", STANDARD_MMLU_ACC, SL_VERB_MMLU_ACC},
            low_memory=False,
        )
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            entry = coverage.setdefault(root, {"mmlu_5shot": False, "mmlu_sl_verb_5shot": False})
            entry["mmlu_5shot"] = entry["mmlu_5shot"] or pd.notna(row.get(STANDARD_MMLU_ACC))
            entry["mmlu_sl_verb_5shot"] = entry["mmlu_sl_verb_5shot"] or pd.notna(row.get(SL_VERB_MMLU_ACC))

    if BASELINE_SCALING_METRICS_CSV.exists():
        frame = pd.read_csv(
            BASELINE_SCALING_METRICS_CSV,
            usecols=lambda column: column in {"checkpoint_root", STANDARD_MMLU_ACC, SL_VERB_MMLU_ACC},
            low_memory=False,
        )
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            entry = coverage.setdefault(root, {"mmlu_5shot": False, "mmlu_sl_verb_5shot": False})
            entry["mmlu_5shot"] = entry["mmlu_5shot"] or pd.notna(row.get(STANDARD_MMLU_ACC))
            entry["mmlu_sl_verb_5shot"] = entry["mmlu_sl_verb_5shot"] or pd.notna(row.get(SL_VERB_MMLU_ACC))
    return coverage


def _expected_step_from_run_registry(row: pd.Series) -> int:
    for column in ("target_final_checkpoint_step", "target_eval_step", "max_checkpoint_step"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return int(value)
    return -1


def _candidate_records() -> list[dict[str, object]]:
    if not METRICS_WIDE_CSV.exists():
        raise FileNotFoundError(f"Missing metric registry {METRICS_WIDE_CSV}")
    if not RUN_REGISTRY_CSV.exists():
        raise FileNotFoundError(f"Missing run registry {RUN_REGISTRY_CSV}")

    coverage = _load_metric_coverage()
    records_by_root: dict[str, dict[str, object]] = {}

    metric_frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    qsplit_core = metric_frame[metric_frame["scale"].eq("300m_6b")].copy()
    for _, row in qsplit_core.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root:
            continue
        covered = coverage.get(root, {})
        if covered.get("mmlu_5shot", False):
            continue
        records_by_root[root] = {
            "run_name": _string_value(row.get("run_name")),
            "registry_key": _string_value(row.get("registry_run_key")),
            "family": "qsplit_core_300m_1x",
            "scale": "300m_6b",
            "cohort": _string_value(row.get("cohort")),
            "source_experiment": _string_value(row.get("source_experiment")),
            "target_budget_multiplier": "1.0",
            "checkpoint_root": root,
            "expected_checkpoint_step": 22887,
            "has_mmlu_5shot": covered.get("mmlu_5shot", False),
            "has_mmlu_sl_verb_5shot": covered.get("mmlu_sl_verb_5shot", False),
        }

    run_registry = pd.read_csv(RUN_REGISTRY_CSV, low_memory=False)
    ready = run_registry[
        run_registry["scale"].eq("300m_6b") & run_registry["is_perplexity_ready"].map(_bool_value)
    ].copy()
    for _, row in ready.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root:
            continue
        covered = coverage.get(root, {})
        if covered.get("mmlu_5shot", False):
            continue
        records_by_root[root] = {
            "run_name": _string_value(row.get("run_name")),
            "registry_key": _string_value(row.get("registry_id")),
            "family": _string_value(row.get("family")),
            "scale": _string_value(row.get("scale")),
            "cohort": _string_value(row.get("study_cohort")),
            "source_experiment": _string_value(row.get("source_experiment")),
            "target_budget_multiplier": _string_value(row.get("target_budget_multiplier")),
            "checkpoint_root": root,
            "expected_checkpoint_step": _expected_step_from_run_registry(row),
            "has_mmlu_5shot": covered.get("mmlu_5shot", False),
            "has_mmlu_sl_verb_5shot": covered.get("mmlu_sl_verb_5shot", False),
        }

    return sorted(
        records_by_root.values(),
        key=lambda record: (
            float(record["target_budget_multiplier"] or 0),
            str(record["family"]),
            str(record["run_name"]),
        ),
    )


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
) -> list[MissingMmluEvalSpec]:
    rows: list[MissingMmluEvalSpec] = []
    for idx, record in enumerate(_candidate_records()):
        checkpoint_root = str(record["checkpoint_root"])
        artifact_paths, artifact_tasks = _checkpoint_lm_eval_artifacts(checkpoint_root)
        hf_checkpoints = sorted(
            _checkpoint_hf_checkpoints(checkpoint_root),
            key=lambda path: (_checkpoint_step_from_path(path), path),
        )
        latest_hf_checkpoint = hf_checkpoints[-1] if hf_checkpoints else ""
        latest_hf_step = _checkpoint_step_from_path(latest_hf_checkpoint) if latest_hf_checkpoint else -1
        expected_step = int(record["expected_checkpoint_step"])
        has_exact_hf_checkpoint = expected_step >= 0 and any(
            _checkpoint_step_from_path(checkpoint) == expected_step for checkpoint in hf_checkpoints
        )
        has_mmlu_5shot = bool(record["has_mmlu_5shot"]) or "mmlu_5shot" in artifact_tasks
        has_mmlu_sl_verb_5shot = bool(record["has_mmlu_sl_verb_5shot"]) or "mmlu_sl_verb_5shot" in artifact_tasks
        if not has_exact_hf_checkpoint:
            launch_decision = "defer_missing_exact_hf_checkpoint"
        elif has_mmlu_5shot:
            launch_decision = "skip_existing"
        else:
            launch_decision = "launch"

        suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
        eval_key = (
            f"mmlu300m_{idx:03d}_{_slug(str(record['run_name']))}_"
            f"{_slug(str(record['target_budget_multiplier']))}x{suffix}"
        )
        result_path = ""
        if launch_decision == "launch":
            result_path = f"{eval_key}/results.json"
        rows.append(
            MissingMmluEvalSpec(
                eval_key=eval_key,
                run_name=str(record["run_name"]),
                registry_key=str(record["registry_key"]),
                family=str(record["family"]),
                scale=str(record["scale"]),
                cohort=str(record["cohort"]),
                source_experiment=str(record["source_experiment"]),
                target_budget_multiplier=str(record["target_budget_multiplier"]),
                checkpoint_root=checkpoint_root,
                expected_checkpoint_step=expected_step,
                hf_checkpoint_count=len(hf_checkpoints),
                hf_checkpoint_latest=latest_hf_checkpoint,
                hf_checkpoint_latest_step=latest_hf_step,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                existing_artifact_count=len(artifact_paths),
                existing_artifact_tasks=";".join(sorted(artifact_tasks)),
                has_mmlu_5shot=has_mmlu_5shot,
                has_mmlu_sl_verb_5shot=has_mmlu_sl_verb_5shot,
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                launch_decision=launch_decision,
                result_path=result_path,
            )
        )
    return rows


def write_local_state(state_rows: list[MissingMmluEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in state_rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _flatten_eval_results(payload: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task_name, task_results in payload.get("results", {}).items():
        if not isinstance(task_results, dict):
            continue
        for metric_name, metric_value in task_results.items():
            metric_key = metric_name.removesuffix(",none")
            if isinstance(metric_value, int | float):
                flat[f"lm_eval/{task_name}/{metric_key}"] = float(metric_value)
    for metric_name, metric_value in payload.get("averages", {}).items():
        metric_key = metric_name.removesuffix(",none")
        if isinstance(metric_value, int | float):
            flat[f"lm_eval/averages/{metric_key}"] = float(metric_value)
    return flat


def collect_eval_results(config: Collect300MMissingMmluResultsConfig) -> None:
    state_by_key = {row["eval_key"]: row for row in json.loads(config.state_rows_json)}
    records: list[dict[str, object]] = []
    for eval_key, state in state_by_key.items():
        record = dict(state)
        result_path = config.results_by_eval_key.get(eval_key)
        if result_path is None:
            record["collection_status"] = "not_launched"
            records.append(record)
            continue
        record["result_path"] = str(result_path)
        metrics, error = _read_eval_metrics(result_path)
        if metrics:
            record.update(metrics)
            record["collection_status"] = "collected"
        else:
            try:
                with fsspec.open(result_path, "rt") as handle:
                    payload = json.load(handle)
                flat = _flatten_eval_results(payload)
                record.update(flat)
                record["collection_status"] = "collected" if flat else "missing_metrics"
            except (OSError, json.JSONDecodeError) as exc:
                record["collection_status"] = "missing_metrics"
                error = str(exc)
        record["collection_error"] = error
        records.append(record)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(config.output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records(list(state_by_key.values())).to_csv(handle, index=False)


def build_eval_steps(state_rows: list[MissingMmluEvalSpec]) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
    from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_SL_VERB_5_SHOT

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
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=row.eval_key,
            model_path=row.checkpoint_root,
            evals=[MMLU_5_SHOT, MMLU_SL_VERB_5_SHOT],
            resource_config=resource_config,
            discover_latest_checkpoint=True,
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step, "results.json")
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[MissingMmluEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 300M missing-MMLU eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=Collect300MMissingMmluResultsConfig(
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
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    state_rows = build_state_rows(
        default_tpu_type=args.tpu_type,
        default_tpu_region=args.tpu_region,
        default_tpu_zone=args.tpu_zone,
        eval_key_suffix=args.eval_key_suffix,
    )
    write_local_state(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote local eval state to %s", STATE_CSV)
    logger.info("Wrote local launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d missing-MMLU eval steps", launch_count)
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(state_rows)
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*eval_steps, collect_step],
        description=f"{args.name_prefix}: recover missing 300M MMLU metrics",
    )


if __name__ == "__main__":
    main()
