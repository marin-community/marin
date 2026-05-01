# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch downstream evals for central baseline-scaling plot checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import fsspec
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

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR / "exploratory" / "paper_plots"
PAPER_PLOTS_IMG_DIR = PAPER_PLOTS_DIR / "img"
BASELINE_MANIFEST_CSV = str(PAPER_PLOTS_IMG_DIR / "baseline_scaling_trajectories_manifest.csv")
LOCAL_STATE_CSV = PAPER_PLOTS_IMG_DIR / "baseline_scaling_downstream_eval_state.csv"
LOCAL_LAUNCH_MANIFEST_CSV = PAPER_PLOTS_IMG_DIR / "baseline_scaling_downstream_eval_launch_manifest.csv"
RESULTS_CSV = "baseline_scaling_downstream_eval_results.csv"
STATE_CSV = "baseline_scaling_downstream_eval_state.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_downstream_evals"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT: int | None = None
GENERATION_ENGINE_KWARGS = {"max_num_batched_tokens": 1024}
LM_EVAL_RESULTS_GLOB = "lm_eval_artifacts/lm_eval_harness_results*.json"
LM_EVAL_METRIC_SUFFIX = ",none"
HF_CONFIG_GLOB = "**/config.json"
HF_TOKENIZER_CONFIG = "tokenizer_config.json"


class DownstreamEvalSuite(StrEnum):
    """Eval suites supported by the baseline-scaling downstream launcher."""

    ALL = "all"
    GSM8K_HUMANEVAL = "gsm8k_humaneval"
    MMLU = "mmlu"


@dataclass(frozen=True)
class BaselineEvalSpec:
    """One downstream-eval state row and potential launch unit."""

    eval_key: str
    method_id: str
    method: str
    scale: str
    scale_label: str
    run_name: str
    source_experiment: str
    checkpoint_root: str
    hf_checkpoint_count: int
    hf_checkpoint_latest: str
    expected_checkpoint_step: int
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    cell_status: str
    target_ready: bool
    task_suite: str
    task_aliases: str
    existing_artifact_count: int
    existing_tasks: str
    has_gsm8k: bool
    has_humaneval: bool
    has_mmlu: bool
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class CollectBaselineScalingDownstreamEvalResultsConfig:
    """Config for collecting baseline-scaling downstream eval outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _suite_requested(requested: DownstreamEvalSuite, suite: DownstreamEvalSuite) -> bool:
    return requested == DownstreamEvalSuite.ALL or requested == suite


def _artifact_step(path: str) -> int:
    match = re.search(r"lm_eval_harness_results\.(\d+)\.json$", path)
    if match is None:
        return -1
    return int(match.group(1))


def _checkpoint_lm_eval_artifacts(checkpoint_root: str) -> tuple[list[str], set[str]]:
    if not checkpoint_root.startswith("gs://"):
        return [], set()

    pattern = checkpoint_root.rstrip("/") + f"/{LM_EVAL_RESULTS_GLOB}"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    matches = [match if str(match).startswith("gs://") else f"gs://{match}" for match in fs.glob(pattern)]
    tasks: set[str] = set()
    for path in sorted(matches, key=lambda item: (_artifact_step(item), item)):
        try:
            with fsspec.open(path, "rt") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results", {})
        if isinstance(results, dict):
            tasks.update(str(task) for task in results)
    return sorted(matches), tasks


def _checkpoint_hf_checkpoints(checkpoint_root: str) -> list[str]:
    if not checkpoint_root.startswith("gs://"):
        return []

    pattern = checkpoint_root.rstrip("/") + f"/{HF_CONFIG_GLOB}"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    checkpoints: list[str] = []
    for match in fs.glob(pattern):
        config_path = match if str(match).startswith("gs://") else f"gs://{match}"
        checkpoint_path = os.path.dirname(config_path)
        tokenizer_config_path = os.path.join(checkpoint_path, HF_TOKENIZER_CONFIG)
        try:
            with fsspec.open(tokenizer_config_path, "rb"):
                checkpoints.append(checkpoint_path)
        except OSError:
            continue
    return sorted(set(checkpoints))


def _checkpoint_step_from_path(path: str) -> int:
    match = re.search(r"/step-(\d+)(?:/|$)", path)
    if match is None:
        return -1
    return int(match.group(1))


def _expected_checkpoint_step(row: pd.Series) -> int:
    for key in ("target_final_checkpoint_step", "target_eval_step", "max_checkpoint_step"):
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return int(value)
    return -1


def _has_task(tasks: set[str], task_name: str) -> bool:
    lower_task_name = task_name.lower()
    return any(lower_task_name in task.lower() for task in tasks)


def _checkpoint_region(checkpoint_root: str, default_region: str) -> str:
    if not checkpoint_root.startswith("gs://"):
        return default_region
    bucket = checkpoint_root.removeprefix("gs://").split("/", 1)[0]
    prefix = "marin-"
    if bucket.startswith(prefix):
        return bucket.removeprefix(prefix)
    return default_region


def _zone_for_region(region: str, default_region: str, default_zone: str) -> str:
    if region == default_region:
        return default_zone
    return ""


def _model_slug(row: pd.Series, suite: DownstreamEvalSuite, eval_key_suffix: str) -> str:
    parts = [
        "baseline_scaling",
        _string_value(row.get("method_id")),
        _string_value(row.get("scale")),
        suite.value,
        eval_key_suffix,
    ]
    return "_".join(part for part in parts if part).replace("/", "_")


def _launch_decision(
    *,
    suite: DownstreamEvalSuite,
    target_ready: bool,
    cell_status: str,
    checkpoint_root: str,
    has_hf_checkpoint: bool,
    include_diagnostic: bool,
    has_gsm8k: bool,
    has_humaneval: bool,
    has_mmlu: bool,
    has_exact_hf_checkpoint: bool,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if cell_status != "target_ready" and not include_diagnostic:
        return False, "defer_not_target_ready"
    if not target_ready and not include_diagnostic:
        return False, "defer_not_target_ready"
    if not has_hf_checkpoint:
        return False, "defer_missing_hf_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if suite == DownstreamEvalSuite.GSM8K_HUMANEVAL and has_gsm8k and has_humaneval:
        return True, "skip_existing"
    if suite == DownstreamEvalSuite.MMLU and has_mmlu:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    manifest_csv: str,
    suite: DownstreamEvalSuite,
    include_diagnostic: bool,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
    method_id: str | None,
    scale: str | None,
) -> list[BaselineEvalSpec]:
    """Build the downstream-eval state table from the baseline plot manifest."""
    manifest = pd.read_csv(manifest_csv)
    if method_id is not None:
        manifest = manifest[manifest["method_id"] == method_id]
    if scale is not None:
        manifest = manifest[manifest["scale"] == scale]
    if manifest.empty:
        raise ValueError(f"No manifest rows matched method_id={method_id!r}, scale={scale!r}")

    rows: list[BaselineEvalSpec] = []
    for _, row in manifest.iterrows():
        checkpoint_root = _string_value(row.get("checkpoint_root"))
        artifact_paths, existing_tasks = _checkpoint_lm_eval_artifacts(checkpoint_root)
        hf_checkpoints = _checkpoint_hf_checkpoints(checkpoint_root)
        hf_checkpoints = sorted(hf_checkpoints, key=lambda path: (_checkpoint_step_from_path(path), path))
        expected_step = _expected_checkpoint_step(row)
        latest_hf_checkpoint = hf_checkpoints[-1] if hf_checkpoints else ""
        latest_hf_step = _checkpoint_step_from_path(latest_hf_checkpoint) if latest_hf_checkpoint else -1
        has_exact_hf_checkpoint = expected_step >= 0 and any(
            _checkpoint_step_from_path(checkpoint) == expected_step for checkpoint in hf_checkpoints
        )
        has_gsm8k = _has_task(existing_tasks, "gsm8k")
        has_humaneval = _has_task(existing_tasks, "humaneval")
        has_mmlu = _has_task(existing_tasks, "mmlu")
        launch_region = _checkpoint_region(checkpoint_root, default_tpu_region)
        launch_zone = _zone_for_region(launch_region, default_tpu_region, default_tpu_zone)
        target_ready = _bool_value(row.get("target_ready"))
        cell_status = _string_value(row.get("cell_status"))

        for task_suite, task_aliases in (
            (DownstreamEvalSuite.GSM8K_HUMANEVAL, "gsm8k_5shot;humaneval_10shot"),
            (DownstreamEvalSuite.MMLU, "mmlu_5shot"),
        ):
            if not _suite_requested(suite, task_suite):
                continue
            eligible, decision = _launch_decision(
                suite=task_suite,
                target_ready=target_ready,
                cell_status=cell_status,
                checkpoint_root=checkpoint_root,
                has_hf_checkpoint=bool(hf_checkpoints),
                include_diagnostic=include_diagnostic,
                has_gsm8k=has_gsm8k,
                has_humaneval=has_humaneval,
                has_mmlu=has_mmlu,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            )
            slug = _model_slug(row, task_suite, eval_key_suffix)
            rows.append(
                BaselineEvalSpec(
                    eval_key=slug,
                    method_id=_string_value(row.get("method_id")),
                    method=_string_value(row.get("method")),
                    scale=_string_value(row.get("scale")),
                    scale_label=_string_value(row.get("scale_label")),
                    run_name=_string_value(row.get("run_name")),
                    source_experiment=_string_value(row.get("source_experiment")),
                    checkpoint_root=checkpoint_root,
                    hf_checkpoint_count=len(hf_checkpoints),
                    hf_checkpoint_latest=latest_hf_checkpoint,
                    expected_checkpoint_step=expected_step,
                    hf_checkpoint_latest_step=latest_hf_step,
                    has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                    cell_status=cell_status,
                    target_ready=target_ready,
                    task_suite=task_suite.value,
                    task_aliases=task_aliases,
                    existing_artifact_count=len(artifact_paths),
                    existing_tasks=";".join(sorted(existing_tasks)),
                    has_gsm8k=has_gsm8k,
                    has_humaneval=has_humaneval,
                    has_mmlu=has_mmlu,
                    launch_tpu_type=default_tpu_type,
                    launch_tpu_region=launch_region,
                    launch_tpu_zone=launch_zone,
                    eligible=eligible,
                    launch_decision=decision,
                    step_name=f"evaluation/lm_evaluation_harness/{slug}",
                    result_path=f"executor_output:{slug}",
                )
            )
    return rows


def _write_local_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def write_local_state(rows: list[BaselineEvalSpec]) -> None:
    """Write local planning state for the paper-plot workflow."""
    records = [asdict(row) for row in rows]
    _write_local_csv(LOCAL_STATE_CSV, records)
    _write_local_csv(
        LOCAL_LAUNCH_MANIFEST_CSV,
        [record for record in records if record["launch_decision"] == "launch"],
    )


def _metric_rows_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = payload.get("results", {})
    if isinstance(results, dict):
        for task_key, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_key, value in task_metrics.items():
                metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
                if metric_name.endswith("_stderr"):
                    continue
                if isinstance(value, int | float):
                    metrics[f"lm_eval/{task_key}/{metric_name}"] = float(value)
    averages = payload.get("averages", {})
    if isinstance(averages, dict):
        for metric_key, value in averages.items():
            metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
            if isinstance(value, int | float):
                metrics[f"lm_eval/averages/{metric_name}"] = float(value)
    return metrics


def _candidate_result_json_paths(path: str) -> list[str]:
    if path.endswith(".json"):
        return [path]
    fs, _, _ = fsspec.get_fs_token_paths(path)
    root = path.rstrip("/")
    patterns = [
        f"{root}/results.json",
        f"{root}/**/results*.json",
        f"{root}/**/lm_eval_harness_results*.json",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in fs.glob(pattern):
            normalized = match if str(match).startswith("gs://") else f"gs://{match}"
            if normalized not in matches:
                matches.append(normalized)
    return sorted(matches)


def _read_eval_metrics(path: object) -> tuple[dict[str, float], str]:
    result_path = str(path)
    errors: list[str] = []
    merged: dict[str, float] = {}
    for candidate in _candidate_result_json_paths(result_path):
        try:
            with fsspec.open(candidate, "rt") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}")
            continue
        metrics = _metric_rows_from_payload(payload)
        if metrics:
            merged.update(metrics)
    if merged:
        return merged, ""
    return {}, ";".join(errors) if errors else "no_lm_eval_results_found"


def collect_eval_results(config: CollectBaselineScalingDownstreamEvalResultsConfig) -> None:
    """Collect downstream eval output rows into one normalized CSV."""
    state_rows = [BaselineEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        result_path = config.results_by_eval_key.get(row.eval_key)
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

    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    *,
    state_rows: list[BaselineEvalSpec],
    max_eval_instances: int | None,
    only_launch_tpu_region: str | None,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build eval steps for state rows whose launch decision is `launch`."""
    from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness, evaluate_lm_evaluation_harness
    from experiments.evals.task_configs import GSM8K_5_SHOT, HUMANEVAL_10_SHOT, MMLU_5_SHOT

    eval_steps: list[ExecutorStep] = []
    results_by_eval_key: dict[str, InputName] = {}
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        if only_launch_tpu_region is not None and row.launch_tpu_region != only_launch_tpu_region:
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone or None,
        )
        if row.task_suite == DownstreamEvalSuite.GSM8K_HUMANEVAL.value:
            eval_step = evaluate_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=[GSM8K_5_SHOT, HUMANEVAL_10_SHOT],
                max_eval_instances=max_eval_instances,
                engine_kwargs=GENERATION_ENGINE_KWARGS,
                resource_config=resource_config,
                discover_latest_checkpoint=True,
                wandb_tags=["baseline-scaling", "gsm8k-humaneval"],
            )
            results_by_eval_key[row.eval_key] = output_path_of(eval_step)
        elif row.task_suite == DownstreamEvalSuite.MMLU.value:
            eval_step = evaluate_levanter_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=[MMLU_5_SHOT],
                resource_config=resource_config,
                max_eval_instances=max_eval_instances,
                discover_latest_checkpoint=True,
            )
            results_by_eval_key[row.eval_key] = output_path_of(eval_step, "results.json")
        else:
            raise ValueError(f"Unsupported task suite: {row.task_suite}")
        eval_steps.append(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[BaselineEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect baseline-scaling downstream eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=CollectBaselineScalingDownstreamEvalResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", default=BASELINE_MANIFEST_CSV)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument(
        "--suite",
        type=DownstreamEvalSuite,
        choices=list(DownstreamEvalSuite),
        default=DownstreamEvalSuite.ALL,
    )
    parser.add_argument("--include-diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--only-launch-tpu-region")
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--method-id")
    parser.add_argument("--scale")
    return parser.parse_known_args()


def _executor_prefix(executor_prefix: str | None, default_tpu_region: str) -> str | None:
    """Return an absolute executor prefix so eval outputs are uploaded to GCS."""
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_tpu_region), executor_prefix)


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    state_rows = build_state_rows(
        manifest_csv=args.manifest_csv,
        suite=args.suite,
        include_diagnostic=args.include_diagnostic,
        default_tpu_type=args.tpu_type,
        default_tpu_region=args.tpu_region,
        default_tpu_zone=args.tpu_zone,
        eval_key_suffix=args.eval_key_suffix,
        method_id=args.method_id,
        scale=args.scale,
    )
    write_local_state(state_rows)
    launch_count = sum(
        row.launch_decision == "launch"
        and (args.only_launch_tpu_region is None or row.launch_tpu_region == args.only_launch_tpu_region)
        for row in state_rows
    )
    logger.info("Wrote local eval state to %s", LOCAL_STATE_CSV)
    logger.info("Wrote local launch manifest to %s", LOCAL_LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d downstream eval steps", launch_count)
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(
        state_rows=state_rows,
        max_eval_instances=args.max_eval_instances,
        only_launch_tpu_region=args.only_launch_tpu_region,
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)

    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*eval_steps, collect_step],
        description=f"{args.name_prefix}: baseline-scaling downstream evals",
    )


if __name__ == "__main__":
    main()
