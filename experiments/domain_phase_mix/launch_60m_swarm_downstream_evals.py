# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch GSM8K/HumanEval evals for the 60M fit-swarm panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from fray.cluster import ResourceConfig
from marin.rl.placement import marin_prefix_for_region
import pandas as pd

from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.launch_baseline_scaling_downstream_evals import (
    GENERATION_ENGINE_KWARGS,
    _checkpoint_hf_checkpoints,
    _checkpoint_lm_eval_artifacts,
    _checkpoint_step_from_path,
    _has_task,
    _read_eval_metrics,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
METRICS_WIDE_CSV = METRIC_REGISTRY_DIR / "metrics_wide.csv"
FIT_DATASET_CSV = (
    METRIC_REGISTRY_DIR / "fit_datasets" / "eval_uncheatable_eval_bpb__60m_1p2b__signal__fit_swarm_60m_default.csv"
)
ANALYSIS_DATASET_CSV = TWO_PHASE_MANY_DIR / "analysis_dataset" / "nd_scale_runs.csv"
BASELINE_SCALING_DOWNSTREAM_METRICS_CSV = (
    SCRIPT_DIR / "exploratory" / "paper_plots" / "img" / "baseline_scaling_downstream_eval_metrics_merged.csv"
)
OUTPUT_DIR = METRIC_REGISTRY_DIR / "benchmark_aggregate_60m"
ACCOUNTING_CSV = OUTPUT_DIR / "60m_fit_swarm_benchmark_accounting.csv"
STATE_CSV = OUTPUT_DIR / "60m_fit_swarm_downstream_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "60m_fit_swarm_downstream_eval_launch_manifest.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_60m_swarm_downstream_evals"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_EXPECTED_STEP = 4576
EXPECTED_FIT_SWARM_ROWS = 242
RESULTS_CSV = "60m_swarm_downstream_eval_results.csv"
STATE_OUTPUT_CSV = "60m_swarm_downstream_eval_state.csv"

GSM8K_METRIC = "lm_eval/gsm8k/exact_match,flexible-extract"
HUMANEVAL_METRIC = "lm_eval/humaneval/pass@1,create_test"
MMLU_ACC_METRIC = "lm_eval/mmlu_5shot/acc"


class SwarmEvalSuite(StrEnum):
    """Eval suites supported by the 60M swarm launcher."""

    GSM8K_HUMANEVAL = "gsm8k_humaneval"


BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "has_gsm8k", "has_humaneval", "has_mmlu", "eligible"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
    "existing_artifact_count",
}


@dataclass(frozen=True)
class SwarmEvalSpec:
    """One fit-swarm eval state row."""

    eval_key: str
    registry_run_key: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    final_checkpoint_path: str
    expected_checkpoint_step: int
    hf_checkpoint_count: int
    hf_checkpoint_latest: str
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    existing_artifact_count: int
    existing_tasks: str
    has_gsm8k: bool
    has_humaneval: bool
    has_mmlu: bool
    existing_result_sources: str
    task_suite: str
    task_aliases: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class Collect60MSwarmDownstreamEvalResultsConfig:
    """Config for collecting 60M swarm downstream eval outputs."""

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


def _executor_prefix(executor_prefix: str | None, default_tpu_region: str) -> str | None:
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_tpu_region), executor_prefix)


def _load_fit_swarm() -> pd.DataFrame:
    if not FIT_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing fit-swarm dataset {FIT_DATASET_CSV}. "
            "Build the metric registry fit dataset before launching downstream evals."
        )
    frame = pd.read_csv(FIT_DATASET_CSV)
    if len(frame) != EXPECTED_FIT_SWARM_ROWS:
        raise ValueError(f"Expected {EXPECTED_FIT_SWARM_ROWS} fit-swarm rows, found {len(frame)}")
    duplicate_roots = frame["checkpoint_root"].duplicated(keep=False)
    if duplicate_roots.any():
        duplicates = frame.loc[duplicate_roots, ["run_name", "checkpoint_root"]]
        raise ValueError(f"Duplicate checkpoint roots in fit-swarm rows:\n{duplicates.to_string(index=False)}")
    return frame


def _load_analysis_metadata() -> pd.DataFrame:
    if not ANALYSIS_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing analysis dataset {ANALYSIS_DATASET_CSV}")
    columns = [
        "checkpoint_root",
        "final_checkpoint_path",
        "final_checkpoint_step",
        "is_complete_checkpoint",
        "scale",
        "target_budget_multiplier",
        "is_target_step_label",
    ]
    frame = pd.read_csv(ANALYSIS_DATASET_CSV, usecols=lambda column: column in columns, low_memory=False)
    frame = frame[frame["scale"] == "60m_1p2b"].copy()
    return frame.drop_duplicates(subset=["checkpoint_root"], keep="first")


def _baseline_existing_results() -> dict[str, dict[str, str]]:
    if not BASELINE_SCALING_DOWNSTREAM_METRICS_CSV.exists():
        return {}
    frame = pd.read_csv(BASELINE_SCALING_DOWNSTREAM_METRICS_CSV)
    out: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        checkpoint_root = _string_value(row.get("checkpoint_root"))
        if not checkpoint_root:
            continue
        sources: list[str] = []
        has_gsm8k = pd.notna(row.get(GSM8K_METRIC))
        has_humaneval = pd.notna(row.get(HUMANEVAL_METRIC))
        has_mmlu = pd.notna(row.get(MMLU_ACC_METRIC))
        for metric in (GSM8K_METRIC, HUMANEVAL_METRIC, MMLU_ACC_METRIC):
            source_col = metric.replace("/", "_").replace(",", "_").replace("@", "_").replace("-", "_")
            source_col = f"{source_col}__source_path"
            source = _string_value(row.get(source_col))
            if source:
                sources.append(source)
        out[checkpoint_root] = {
            "has_gsm8k": str(has_gsm8k),
            "has_humaneval": str(has_humaneval),
            "has_mmlu": str(has_mmlu),
            "sources": ";".join(sorted(set(sources))),
        }
    return out


def _metric_registry_mmlu_roots() -> set[str]:
    if not METRICS_WIDE_CSV.exists():
        return set()
    frame = pd.read_csv(METRICS_WIDE_CSV, usecols=lambda column: column in {"checkpoint_root", MMLU_ACC_METRIC})
    if MMLU_ACC_METRIC not in frame.columns:
        return set()
    return set(frame.loc[frame[MMLU_ACC_METRIC].notna(), "checkpoint_root"].astype(str))


def _expected_step(row: pd.Series) -> int:
    value = pd.to_numeric(pd.Series([row.get("final_checkpoint_step")]), errors="coerce").iloc[0]
    if pd.notna(value):
        return int(value)
    return DEFAULT_EXPECTED_STEP


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
) -> list[SwarmEvalSpec]:
    """Build benchmark accounting and eval state rows for the 60M fit swarm."""
    fit_frame = _load_fit_swarm()
    analysis = _load_analysis_metadata()
    existing_result_by_root = _baseline_existing_results()
    mmlu_metric_roots = _metric_registry_mmlu_roots()
    merged = fit_frame.merge(analysis, on="checkpoint_root", how="left", suffixes=("", "_analysis"))
    if merged["final_checkpoint_step"].isna().any():
        missing = merged.loc[merged["final_checkpoint_step"].isna(), ["run_name", "checkpoint_root"]]
        raise ValueError(f"Missing analysis metadata for fit-swarm rows:\n{missing.to_string(index=False)}")

    rows: list[SwarmEvalSpec] = []
    for idx, row in merged.iterrows():
        checkpoint_root = _string_value(row.get("checkpoint_root"))
        artifact_paths, checkpoint_tasks = _checkpoint_lm_eval_artifacts(checkpoint_root)
        hf_checkpoints = sorted(
            _checkpoint_hf_checkpoints(checkpoint_root),
            key=lambda path: (_checkpoint_step_from_path(path), path),
        )
        latest_hf_checkpoint = hf_checkpoints[-1] if hf_checkpoints else ""
        latest_hf_step = _checkpoint_step_from_path(latest_hf_checkpoint) if latest_hf_checkpoint else -1
        expected_step = _expected_step(row)
        has_exact_hf_checkpoint = expected_step >= 0 and any(
            _checkpoint_step_from_path(checkpoint) == expected_step for checkpoint in hf_checkpoints
        )

        existing_result = existing_result_by_root.get(checkpoint_root, {})
        has_gsm8k = _has_task(checkpoint_tasks, "gsm8k") or existing_result.get("has_gsm8k") == "True"
        has_humaneval = _has_task(checkpoint_tasks, "humaneval") or existing_result.get("has_humaneval") == "True"
        has_mmlu = (
            _has_task(checkpoint_tasks, "mmlu")
            or existing_result.get("has_mmlu") == "True"
            or checkpoint_root in mmlu_metric_roots
        )
        eligible, decision = _launch_decision(
            checkpoint_root=checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            has_gsm8k=has_gsm8k,
            has_humaneval=has_humaneval,
        )
        run_name = _string_value(row.get("run_name"))
        suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
        eval_key = f"swarm60m_{idx:03d}_{_slug(run_name)}_gsm8k_humaneval{suffix}"
        rows.append(
            SwarmEvalSpec(
                eval_key=eval_key,
                registry_run_key=_string_value(row.get("registry_run_key")),
                run_name=run_name,
                source_experiment=_string_value(row.get("source_experiment")),
                checkpoint_root=checkpoint_root,
                final_checkpoint_path=_string_value(row.get("final_checkpoint_path")),
                expected_checkpoint_step=expected_step,
                hf_checkpoint_count=len(hf_checkpoints),
                hf_checkpoint_latest=latest_hf_checkpoint,
                hf_checkpoint_latest_step=latest_hf_step,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                existing_artifact_count=len(artifact_paths),
                existing_tasks=";".join(sorted(checkpoint_tasks)),
                has_gsm8k=has_gsm8k,
                has_humaneval=has_humaneval,
                has_mmlu=has_mmlu,
                existing_result_sources=existing_result.get("sources", ""),
                task_suite=SwarmEvalSuite.GSM8K_HUMANEVAL.value,
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


def _write_local_outputs(rows: list[SwarmEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    state = pd.DataFrame.from_records(records)
    accounting_columns = [
        "registry_run_key",
        "run_name",
        "source_experiment",
        "checkpoint_root",
        "final_checkpoint_path",
        "expected_checkpoint_step",
        "hf_checkpoint_count",
        "hf_checkpoint_latest",
        "hf_checkpoint_latest_step",
        "has_exact_hf_checkpoint",
        "existing_artifact_count",
        "existing_tasks",
        "has_gsm8k",
        "has_humaneval",
        "has_mmlu",
        "existing_result_sources",
        "eligible",
        "launch_decision",
    ]
    state[accounting_columns].to_csv(ACCOUNTING_CSV, index=False)
    state.to_csv(STATE_CSV, index=False)
    state[state["launch_decision"] == "launch"].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _load_state_rows(path: Path | str) -> list[SwarmEvalSpec]:
    frame = pd.read_csv(path)
    missing_columns = sorted(field.name for field in fields(SwarmEvalSpec) if field.name not in frame.columns)
    if missing_columns:
        raise ValueError(f"State CSV {path} is missing columns: {missing_columns}")
    rows: list[SwarmEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(SwarmEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(SwarmEvalSpec(**kwargs))
    return rows


def _with_eval_key_suffix(rows: list[SwarmEvalSpec], eval_key_suffix: str) -> list[SwarmEvalSpec]:
    if not eval_key_suffix:
        return rows
    out: list[SwarmEvalSpec] = []
    marker = "_gsm8k_humaneval"
    for row in rows:
        base = row.eval_key.split(marker, maxsplit=1)[0] + marker
        eval_key = f"{base}_{eval_key_suffix}"
        out.append(
            replace(
                row,
                eval_key=eval_key,
                step_name=f"evaluation/lm_evaluation_harness/{eval_key}",
                result_path=f"executor_output:{eval_key}",
            )
        )
    return out


def _metric_rows_from_result_paths(
    state_rows: list[SwarmEvalSpec], results_by_eval_key: dict[str, InputName]
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


def collect_eval_results(config: Collect60MSwarmDownstreamEvalResultsConfig) -> None:
    """Collect downstream eval outputs into one normalized CSV."""
    import fsspec

    state_rows = [SwarmEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    records = _metric_rows_from_result_paths(state_rows, config.results_by_eval_key)
    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    state_rows: list[SwarmEvalSpec], max_eval_instances: int | None
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
            wandb_tags=["60m-swarm", "benchmark-aggregate", "gsm8k-humaneval"],
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[SwarmEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 60M fit-swarm downstream eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=Collect60MSwarmDownstreamEvalResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--state-csv")
    return parser.parse_known_args()


def _has_iris_context() -> bool:
    try:
        from iris.client.client import get_iris_ctx
    except ImportError:
        return False
    return get_iris_ctx() is not None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")
    sys.argv = [sys.argv[0], *remaining]
    try:
        if args.state_csv is None:
            state_rows = build_state_rows(
                default_tpu_type=args.tpu_type,
                default_tpu_region=args.tpu_region,
                default_tpu_zone=args.tpu_zone,
                eval_key_suffix=args.eval_key_suffix,
            )
        else:
            state_rows = _with_eval_key_suffix(_load_state_rows(args.state_csv), args.eval_key_suffix)
    except FileNotFoundError:
        if args.state_csv is not None or not STATE_CSV.exists():
            raise
        logger.warning("Falling back to precomputed state CSV because registry inputs are unavailable: %s", STATE_CSV)
        state_rows = _load_state_rows(STATE_CSV)
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote accounting to %s", ACCOUNTING_CSV)
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d downstream eval steps", launch_count)
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
        description=f"{args.name_prefix}: 60M fit-swarm GSM8K/HumanEval evals",
    )


if __name__ == "__main__":
    main()
