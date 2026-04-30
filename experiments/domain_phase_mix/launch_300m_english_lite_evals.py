# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch English-lite lm-eval expansion for 300M data-mixture rows.

This covers the same 300M/6B signal and fixed-seed noise population as the
GSM8K/HumanEval completion launcher, but runs English MCQ/logprob-style tasks
that are useful for benchmark-optimization SNR accounting. It intentionally
excludes generation-heavy and frontier-hard suites.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import logging
import os
from pathlib import Path
import re
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
    versioned,
)
import pandas as pd
from marin.execution.remote import remote

from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    METRICS_WIDE_CSV,
    _bool_value,
    _candidate_records,
    _exact_hf_checkpoint,
    _executor_prefix,
    _read_csv,
    _slug,
    _string_value,
)
from experiments.domain_phase_mix.launch_baseline_scaling_downstream_evals import _read_eval_metrics

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_english_lite_completion"
STATE_CSV = OUTPUT_DIR / "300m_english_lite_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_english_lite_eval_launch_manifest.csv"
RETRY_STATE_CSV = OUTPUT_DIR / "300m_english_lite_eval_retry_state.csv"
MERGED_RESULTS_CSV = OUTPUT_DIR / "300m_english_lite_eval_results_merged.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_english_lite_evals_20260429"
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/300m-english-lite-v1"
RESULTS_CSV = "300m_english_lite_eval_results.csv"
STATE_OUTPUT_CSV = "300m_english_lite_eval_state.csv"
EXECUTOR_STATUS_FILE = ".executor_status"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
EVAL_OUTPUT_RE = re.compile(r"/lmeval_debug_(?P<eval_key>.+)-[0-9a-f]{6}/\.executor_status$")

OLMO_BASE_ENGLISH_LITE_ALIASES = (
    "arc_easy_5shot",
    "arc_challenge_5shot",
    "csqa_5shot",
    "hellaswag_5shot",
    "winogrande_5shot",
    "socialiqa_5shot",
    "piqa_5shot",
    "sciq_5shot",
    "lambada_0shot",
    "medmcqa_5shot",
)

ADDITIONAL_ENGLISH_LITE_ALIASES = (
    "boolq_10shot",
    "openbookqa_0shot",
    "copa_0shot",
    "wsc273_0shot",
    "swag_0shot",
    "truthfulqa_mc1_0shot",
    "truthfulqa_mc2_0shot",
)

ENGLISH_LITE_TASK_ALIASES = OLMO_BASE_ENGLISH_LITE_ALIASES + ADDITIONAL_ENGLISH_LITE_ALIASES

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "has_all_tasks", "is_region_local", "eligible"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
    "existing_artifact_count",
    "missing_task_count",
}


@dataclass(frozen=True)
class EnglishLiteEvalSpec:
    """One English-lite eval state row and potential launch unit."""

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
    checkpoint_region: str
    is_region_local: bool
    existing_artifact_count: int
    existing_tasks: str
    missing_task_count: int
    missing_tasks: str
    has_all_tasks: bool
    task_aliases: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class Collect300MEnglishLiteResultsConfig:
    """Config for collecting 300M English-lite eval outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def english_lite_task_aliases() -> tuple[str, ...]:
    """Return the exact metric aliases covered by this launcher."""
    return ENGLISH_LITE_TASK_ALIASES


def english_lite_tasks():
    """Return EvalTaskConfig objects for the English-lite suite.

    Imports stay local so dry-run accounting remains lightweight on machines
    that do not have the full evaluation dependency stack imported yet.
    """
    from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_TASKS
    from marin.evaluation.evaluation_config import EvalTaskConfig

    olmo_without_mmlu = [task for task in OLMO_BASE_EASY_OVERLAP_TASKS if (task.task_alias or task.name) != "mmlu_5shot"]
    extra_tasks = [
        EvalTaskConfig("boolq", 10, task_alias="boolq_10shot"),
        EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
        EvalTaskConfig("copa", 0, task_alias="copa_0shot"),
        EvalTaskConfig("wsc273", 0, task_alias="wsc273_0shot"),
        EvalTaskConfig("swag", 0, task_alias="swag_0shot"),
        EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_mc1_0shot"),
        EvalTaskConfig("truthfulqa_mc2", 0, task_alias="truthfulqa_mc2_0shot"),
    ]
    task_aliases = tuple(task.task_alias or task.name for task in [*olmo_without_mmlu, *extra_tasks])
    if task_aliases != ENGLISH_LITE_TASK_ALIASES:
        raise ValueError(
            "English-lite task alias mismatch:\n" f"expected={ENGLISH_LITE_TASK_ALIASES}\n" f"actual={task_aliases}"
        )
    return [*olmo_without_mmlu, *extra_tasks]


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("lm_eval/"))


def _metric_coverage_by_root(paths: list[str | Path], task_aliases: tuple[str, ...]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for path in paths:
        if isinstance(path, Path) and not path.exists():
            continue
        frame = _read_csv(path)
        if "checkpoint_root" not in frame.columns:
            continue
        metric_columns = _metric_columns(frame)
        if not metric_columns:
            continue
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            covered = coverage.setdefault(root, set())
            for alias in task_aliases:
                prefix = f"lm_eval/{alias}/"
                if any(column.startswith(prefix) and pd.notna(row.get(column)) for column in metric_columns):
                    covered.add(alias)
    return coverage


def _checkpoint_region(checkpoint_root: str) -> str:
    if not checkpoint_root.startswith("gs://"):
        return ""
    bucket = checkpoint_root.removeprefix("gs://").split("/", maxsplit=1)[0]
    prefix = "marin-"
    if bucket.startswith(prefix):
        return bucket.removeprefix(prefix)
    return ""


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    is_region_local: bool,
    has_all_tasks: bool,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if not is_region_local:
        return False, "defer_checkpoint_region_mismatch"
    if has_all_tasks:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
) -> list[EnglishLiteEvalSpec]:
    """Build state rows for 300M English-lite eval completion."""
    task_aliases = english_lite_task_aliases()
    coverage = _metric_coverage_by_root([METRICS_WIDE_CSV], task_aliases)
    rows: list[EnglishLiteEvalSpec] = []
    for idx, candidate in enumerate(_candidate_records()):
        latest_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        latest_hf_step = candidate.expected_checkpoint_step if latest_hf_checkpoint else -1
        has_exact_hf_checkpoint = bool(latest_hf_checkpoint)
        checkpoint_region = _checkpoint_region(candidate.checkpoint_root)
        is_region_local = checkpoint_region in {"", default_tpu_region}
        existing_tasks = sorted(coverage.get(candidate.checkpoint_root, set()))
        missing_tasks = [alias for alias in task_aliases if alias not in existing_tasks]
        has_all_tasks = not missing_tasks
        eligible, decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            is_region_local=is_region_local,
            has_all_tasks=has_all_tasks,
        )
        suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
        eval_key = f"englishlite300m_{idx:03d}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            EnglishLiteEvalSpec(
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
                checkpoint_region=checkpoint_region,
                is_region_local=is_region_local,
                existing_artifact_count=len(existing_tasks),
                existing_tasks=";".join(existing_tasks),
                missing_task_count=len(missing_tasks),
                missing_tasks=";".join(missing_tasks),
                has_all_tasks=has_all_tasks,
                task_aliases=";".join(task_aliases),
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=decision,
                step_name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{eval_key}",
                result_path=f"executor_output:{eval_key}",
            )
        )
    return rows


def _write_local_outputs(rows: list[EnglishLiteEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = pd.DataFrame.from_records([asdict(row) for row in rows])
    state.to_csv(STATE_CSV, index=False)
    state[state["launch_decision"] == "launch"].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _load_state_rows(path: Path | str) -> list[EnglishLiteEvalSpec]:
    frame = _read_csv(path)
    missing_columns = sorted(field.name for field in fields(EnglishLiteEvalSpec) if field.name not in frame.columns)
    if missing_columns:
        raise ValueError(f"State CSV {path} is missing columns: {missing_columns}")
    rows: list[EnglishLiteEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(EnglishLiteEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(EnglishLiteEvalSpec(**kwargs))
    return rows


def _metric_rows_from_result_paths(
    state_rows: list[EnglishLiteEvalSpec], results_by_eval_key: dict[str, InputName]
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


def _normalize_gcs_path(path: object) -> str:
    value = str(path)
    return value if value.startswith("gs://") else f"gs://{value}"


def _read_text(path: str) -> str:
    with fsspec.open(path, "rt") as handle:
        return handle.read().strip()


def _eval_key_from_status_path(status_path: str) -> str | None:
    match = EVAL_OUTPUT_RE.search(status_path)
    if match is None:
        return None
    return match.group("eval_key")


def _status_paths_under_prefix(prefix: str) -> list[str]:
    root = prefix.rstrip("/")
    pattern = f"{root}/evaluation/lm_evaluation_harness_levanter/lmeval_debug_*/{EXECUTOR_STATUS_FILE}"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    return sorted(_normalize_gcs_path(path) for path in fs.glob(pattern))


def _scan_eval_statuses(prefixes: list[str]) -> dict[str, dict[str, str]]:
    """Return latest status/output-path metadata by eval key for one or more executor prefixes."""
    statuses: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        for status_path in _status_paths_under_prefix(prefix):
            eval_key = _eval_key_from_status_path(status_path)
            if eval_key is None:
                continue
            try:
                status = _read_text(status_path)
            except OSError as exc:
                logger.warning("Failed to read executor status %s: %s", status_path, exc)
                continue
            output_path = status_path.rsplit(f"/{EXECUTOR_STATUS_FILE}", maxsplit=1)[0]
            previous = statuses.get(eval_key)
            if previous is None or (previous["status"] != STATUS_SUCCESS and status == STATUS_SUCCESS):
                statuses[eval_key] = {
                    "eval_key": eval_key,
                    "output_path": output_path,
                    "prefix": prefix,
                    "status": status,
                    "status_path": status_path,
                }
    return statuses


def write_retry_state_from_prefix(
    *,
    prefix: str,
    state_rows: list[EnglishLiteEvalSpec],
    output_csv: Path,
) -> list[EnglishLiteEvalSpec]:
    """Write a state CSV containing only failed/missing eval rows from an executor prefix."""
    statuses = _scan_eval_statuses([prefix])
    retry_rows: list[EnglishLiteEvalSpec] = []
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        entry = statuses.get(row.eval_key)
        if entry is not None and entry["status"] == STATUS_SUCCESS:
            continue
        retry_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in retry_rows]).to_csv(output_csv, index=False)
    status_counts: dict[str, int] = {}
    for row in retry_rows:
        status = statuses.get(row.eval_key, {}).get("status", "MISSING")
        status_counts[status] = status_counts.get(status, 0) + 1
    logger.info(
        "Wrote %d retry rows to %s from %s; status_counts=%s",
        len(retry_rows),
        output_csv,
        prefix,
        status_counts,
    )
    return retry_rows


def collect_eval_results_from_prefixes(
    *,
    prefixes: list[str],
    state_rows: list[EnglishLiteEvalSpec],
    output_csv: Path,
) -> pd.DataFrame:
    """Collect successful English-lite child outputs from executor prefixes into one CSV."""
    statuses = _scan_eval_statuses(prefixes)
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        entry = statuses.get(row.eval_key)
        if entry is None:
            record["collection_status"] = "missing_executor_status"
            record["collection_error"] = "no_executor_status_found"
            records.append(record)
            continue
        record["executor_status"] = entry["status"]
        record["result_path"] = entry["output_path"]
        record["status_path"] = entry["status_path"]
        if entry["status"] != STATUS_SUCCESS:
            record["collection_status"] = "executor_not_success"
            record["collection_error"] = entry["status"]
            records.append(record)
            continue
        metrics, error = _read_eval_metrics(entry["output_path"])
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    logger.info(
        "Wrote %d collected rows to %s from %d prefixes; collection_status=%s",
        len(frame),
        output_csv,
        len(prefixes),
        frame["collection_status"].value_counts(dropna=False).to_dict(),
    )
    return frame


def collect_eval_results(config: Collect300MEnglishLiteResultsConfig) -> None:
    """Collect English-lite eval outputs into one normalized CSV."""
    state_rows = [EnglishLiteEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    records = _metric_rows_from_result_paths(state_rows, config.results_by_eval_key)
    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[EnglishLiteEvalSpec],
    max_eval_instances: int | None,
    eval_datasets_cache_path: str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build English-lite eval steps for rows requiring launch."""
    from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
    from marin.evaluation.eval_dataset_cache import (
        CacheEvalDatasetsConfig,
        HF_CACHE_LAYOUT_VERSION,
        _cache_eval_datasets,
    )

    eval_tasks = english_lite_tasks()
    cache_eval_step = ExecutorStep(
        name=f"{name_prefix}/cache_eval_datasets",
        description="Pre-cache English-lite evaluation datasets to east5 GCS",
        fn=remote(
            _cache_eval_datasets,
            resources=ResourceConfig.with_cpu(
                cpu=1,
                ram="8g",
                disk="32g",
                regions=[DEFAULT_TPU_REGION],
            ),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=CacheEvalDatasetsConfig(
            eval_tasks=tuple(eval_tasks),
            gcs_path=eval_datasets_cache_path,
            cache_layout_version=versioned(HF_CACHE_LAYOUT_VERSION),
        ),
    )
    cache_dependency = output_path_of(cache_eval_step, ".eval_datasets_manifest.json")

    eval_steps: list[ExecutorStep] = [cache_eval_step]
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
            evals=eval_tasks,
            resource_config=resource_config,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=True,
            eval_datasets_cache_path=eval_datasets_cache_path,
            eval_datasets_cache_dependency=cache_dependency,
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[EnglishLiteEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 300M English-lite eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=Collect300MEnglishLiteResultsConfig(
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
    parser.add_argument("--eval-datasets-cache-path", default=DEFAULT_EVAL_DATASETS_CACHE_PATH)
    parser.add_argument("--write-retry-state-from-prefix")
    parser.add_argument("--retry-state-output", default=str(RETRY_STATE_CSV))
    parser.add_argument("--collect-from-prefix", action="append", default=[])
    parser.add_argument("--collect-output-csv", default=str(MERGED_RESULTS_CSV))
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.write_retry_state_from_prefix is not None:
        state_rows = _load_state_rows(args.state_csv or STATE_CSV)
        write_retry_state_from_prefix(
            prefix=args.write_retry_state_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.retry_state_output),
        )
        return
    if args.collect_from_prefix:
        state_rows = _load_state_rows(args.state_csv or STATE_CSV)
        collect_eval_results_from_prefixes(
            prefixes=args.collect_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.collect_output_csv),
        )
        return

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
    logger.info(
        "Prepared %d English-lite eval steps over %d candidate checkpoints and %d task aliases",
        launch_count,
        len(state_rows),
        len(ENGLISH_LITE_TASK_ALIASES),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        max_eval_instances=args.max_eval_instances,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
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
        description=f"{args.name_prefix}: 300M English-lite eval expansion for signal/noise SNR",
    )


if __name__ == "__main__":
    main()
