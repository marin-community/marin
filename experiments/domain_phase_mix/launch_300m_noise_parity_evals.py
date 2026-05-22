# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Backfill parity eval aliases for 300M run_00097 fixed/variable noise rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import HF_CACHE_LAYOUT_VERSION, CacheEvalDatasetsConfig, _cache_eval_datasets
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    _candidate_records,
    _exact_hf_checkpoint,
    _read_csv,
    _slug,
    _string_value,
)
from experiments.domain_phase_mix.parity_eval_rerun_common import flatten_parity_eval_results
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT, MMLU_SL_VERB_5_SHOT

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
OUTPUT_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "300m_noise_parity_completion"
STATE_CSV = OUTPUT_DIR / "300m_noise_parity_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_noise_parity_eval_launch_manifest.csv"
RESULTS_CSV_LOCAL = OUTPUT_DIR / "300m_noise_parity_eval_results.csv"
RAW_MATRIX_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "raw_metric_matrix_300m"
FIXED_NOISE_MATRIX_CSV = RAW_MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv"
VARIABLE_NOISE_MATRIX_CSV = RAW_MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_noise_parity_evals_20260501"
DEFAULT_EXECUTOR_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_noise_parity_evals_20260501"
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/300m-noise-parity-v1"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 20
RESULTS_JSON = "results.json"
RESULTS_CSV = "300m_noise_parity_eval_results.csv"
STATE_OUTPUT_CSV = "300m_noise_parity_eval_state.csv"
NOISE_PANELS = {"fixed_seed_noise_300m_6b", "variable_subset_noise_300m_6b"}
PROPORTIONAL_NOISE_PANELS = {
    "proportional_variable_subset_noise_60m_1p2b",
    "proportional_variable_subset_noise_300m_6b",
}
PROPORTIONAL_PERTURBATION_PANELS = {
    "proportional_perturbation_60m_1p2b",
    "proportional_perturbation_300m_6b",
    "proportional_baseline_anchor_60m_1p2b",
    "proportional_baseline_anchor_300m_6b",
    "proportional_controllability_300m_6b",
}
EXECUTOR_STATUS_FILE = ".executor_status"
STATUS_SUCCESS = "SUCCESS"
EVAL_OUTPUT_RE = re.compile(r"/lmeval_debug_(?P<eval_key>noiseparity300m_.+)-[0-9a-f]{6}/\.executor_status$")

TASKS_BY_ALIAS = {
    "mmlu_5shot": MMLU_5_SHOT,
    "mmlu_sl_verb_5shot": MMLU_SL_VERB_5_SHOT,
    "mmlu_pro_5shot": MMLU_PRO_5_SHOT,
    "arc_easy": EvalTaskConfig("arc_easy", 10),
    "piqa": EvalTaskConfig("piqa", 10),
    "sciq_0shot": EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
    "hellaswag_0shot": EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
}
TASK_ALIASES = tuple(TASKS_BY_ALIAS)

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "eligible"}
INT_STATE_FIELDS = {"expected_checkpoint_step", "hf_checkpoint_count", "hf_checkpoint_latest_step", "missing_task_count"}


@dataclass(frozen=True)
class NoiseParityEvalSpec:
    """One fixed/variable noise-row parity eval state row."""

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
    existing_tasks: str
    missing_task_count: int
    missing_tasks: str
    task_aliases: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class CollectNoiseParityResultsConfig:
    """Config for collecting 300M noise parity eval outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


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


def _executor_prefix(executor_prefix: str | None, default_region: str) -> str | None:
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_region), executor_prefix)


def _normalize_gcs_path(path: object) -> str:
    value = str(path)
    return value if value.startswith("gs://") else f"gs://{value}"


def _read_text(path: str) -> str:
    with fsspec.open(path, "rt") as handle:
        return handle.read().strip()


def _status_paths_under_prefix(prefix: str) -> list[str]:
    root = prefix.rstrip("/")
    pattern = f"{root}/evaluation/lm_evaluation_harness_levanter/lmeval_debug_noiseparity300m_*/{EXECUTOR_STATUS_FILE}"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    return sorted(_normalize_gcs_path(path) for path in fs.glob(pattern))


def _scan_eval_statuses(prefixes: list[str]) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        for status_path in _status_paths_under_prefix(prefix):
            match = EVAL_OUTPUT_RE.search(status_path)
            if match is None:
                continue
            eval_key = match.group("eval_key")
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


def _task_metric_columns(alias: str) -> tuple[str, ...]:
    return (
        f"lm_eval/{alias}/bpb",
        f"lm_eval/{alias}/acc",
        f"lm_eval/{alias}/acc_norm",
        f"lm_eval/{alias}/choice_prob_norm",
        f"lm_eval/{alias}/choice_logprob_norm",
    )


def _coverage_from_csv(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    if "checkpoint_root" not in frame.columns:
        return {}
    coverage: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        root = _string_value(row.get("checkpoint_root")).rstrip("/")
        if not root:
            continue
        covered = coverage.setdefault(root, set())
        for alias in TASK_ALIASES:
            if any(column in frame.columns and pd.notna(row.get(column)) for column in _task_metric_columns(alias)):
                covered.add(alias)
    return coverage


def _metric_coverage_by_root() -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for path in (FIXED_NOISE_MATRIX_CSV, VARIABLE_NOISE_MATRIX_CSV, RESULTS_CSV_LOCAL):
        for root, aliases in _coverage_from_csv(path).items():
            coverage.setdefault(root, set()).update(aliases)
    return coverage


def _noise_candidates() -> list[Any]:
    allowed_panels = NOISE_PANELS | PROPORTIONAL_NOISE_PANELS | PROPORTIONAL_PERTURBATION_PANELS
    return [candidate for candidate in _candidate_records() if candidate.panel in allowed_panels]


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    missing_tasks: list[str],
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if not missing_tasks:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
    task_aliases: tuple[str, ...] = TASK_ALIASES,
) -> list[NoiseParityEvalSpec]:
    """Build fixed/variable noise parity eval rows."""
    coverage = _metric_coverage_by_root()
    rows: list[NoiseParityEvalSpec] = []
    for idx, candidate in enumerate(_noise_candidates()):
        latest_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        has_exact_hf_checkpoint = bool(latest_hf_checkpoint)
        existing_tasks = sorted(coverage.get(candidate.checkpoint_root, set()))
        missing_tasks = [alias for alias in task_aliases if alias not in existing_tasks]
        eligible, launch_decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            missing_tasks=missing_tasks,
        )
        suffix = f"_{_slug(eval_key_suffix)}" if eval_key_suffix else ""
        eval_key = f"noiseparity300m_{idx:03d}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            NoiseParityEvalSpec(
                eval_key=eval_key,
                panel=candidate.panel,
                run_name=candidate.run_name,
                registry_key=candidate.registry_key,
                source_experiment=candidate.source_experiment,
                cohort=candidate.cohort,
                checkpoint_root=candidate.checkpoint_root,
                expected_checkpoint_step=candidate.expected_checkpoint_step,
                hf_checkpoint_count=int(has_exact_hf_checkpoint),
                hf_checkpoint_latest=latest_hf_checkpoint,
                hf_checkpoint_latest_step=candidate.expected_checkpoint_step if has_exact_hf_checkpoint else -1,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                existing_tasks=";".join(existing_tasks),
                missing_task_count=len(missing_tasks),
                missing_tasks=";".join(missing_tasks),
                task_aliases=";".join(task_aliases),
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=launch_decision,
                step_name=f"evaluation/lm_evaluation_harness_levanter/noise_parity_{eval_key}",
                result_path=f"executor_output:{eval_key}",
            )
        )
    return rows


def _write_local_outputs(rows: list[NoiseParityEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = pd.DataFrame.from_records([asdict(row) for row in rows])
    state.to_csv(STATE_CSV, index=False)
    state[state["launch_decision"] == "launch"].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _load_state_rows(path: Path | str) -> list[NoiseParityEvalSpec]:
    frame = _read_csv(path)
    missing_columns = sorted(field.name for field in fields(NoiseParityEvalSpec) if field.name not in frame.columns)
    if missing_columns:
        raise ValueError(f"State CSV {path} is missing columns: {missing_columns}")
    rows: list[NoiseParityEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(NoiseParityEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(NoiseParityEvalSpec(**kwargs))
    return rows


def _tasks_for_row(row: NoiseParityEvalSpec) -> list[EvalTaskConfig]:
    aliases = [alias for alias in row.missing_tasks.split(";") if alias]
    return [TASKS_BY_ALIAS[alias] for alias in aliases]


def _metric_rows_from_result_paths(
    state_rows: list[NoiseParityEvalSpec], results_by_eval_key: dict[str, InputName]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        result_path = results_by_eval_key.get(row.eval_key)
        if result_path is None:
            record["collection_status"] = "not_launched"
            records.append(record)
            continue
        try:
            with fsspec.open(result_path, "r") as handle:
                payload = json.load(handle)
        except OSError as exc:
            record["collection_status"] = "missing_metrics"
            record["collection_error"] = str(exc)
            records.append(record)
            continue
        record.update(flatten_parity_eval_results(payload))
        record["collection_status"] = "collected"
        record["collection_error"] = ""
        record["result_path"] = str(result_path)
        records.append(record)
    return records


def collect_eval_results_from_prefixes(
    *,
    prefixes: list[str],
    state_rows: list[NoiseParityEvalSpec],
    output_csv: Path,
) -> pd.DataFrame:
    """Collect successful noise parity child outputs from executor prefixes into one CSV."""
    statuses = _scan_eval_statuses(prefixes)

    def _status_entry_for_row(row: NoiseParityEvalSpec) -> dict[str, str] | None:
        entry = statuses.get(row.eval_key)
        if entry is not None:
            return entry

        # Retry and targeted backfill jobs can be produced from a different
        # local candidate enumeration than the collector. Match on stable run
        # names so recovery does not depend on the transient numeric prefix.
        run_slug = _slug(row.run_name)
        matches = [
            status
            for eval_key, status in statuses.items()
            if f"_{run_slug}_" in eval_key or eval_key.endswith(f"_{run_slug}")
        ]
        success_matches = [status for status in matches if status["status"] == STATUS_SUCCESS]
        if len(success_matches) == 1:
            return success_matches[0]
        if len(matches) == 1:
            return matches[0]
        return None

    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        entry = _status_entry_for_row(row)
        if entry is None:
            record["collection_status"] = "missing_executor_status"
            record["collection_error"] = "no_executor_status_found"
            records.append(record)
            continue
        record["executor_status"] = entry["status"]
        record["executor_eval_key"] = entry["eval_key"]
        record["result_path"] = entry["output_path"]
        record["status_path"] = entry["status_path"]
        result_path = os.path.join(entry["output_path"], RESULTS_JSON)
        try:
            with fsspec.open(result_path, "r") as handle:
                payload = json.load(handle)
        except OSError as exc:
            if entry["status"] != STATUS_SUCCESS:
                record["collection_status"] = "executor_not_success"
                record["collection_error"] = entry["status"]
                records.append(record)
                continue
            record["collection_status"] = "missing_metrics"
            record["collection_error"] = str(exc)
            records.append(record)
            continue
        record.update(flatten_parity_eval_results(payload))
        record["collection_status"] = (
            "collected" if entry["status"] == STATUS_SUCCESS else f"collected_executor_status_{entry['status'].lower()}"
        )
        record["collection_error"] = "" if entry["status"] == STATUS_SUCCESS else entry["status"]
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


def collect_eval_results(config: CollectNoiseParityResultsConfig) -> None:
    """Collect noise parity eval outputs into one normalized CSV."""
    state_rows = [NoiseParityEvalSpec(**row) for row in json.loads(config.state_rows_json)]
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
    state_rows: list[NoiseParityEvalSpec],
    eval_datasets_cache_path: str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build parity eval steps for rows requiring launch."""
    from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness

    launch_tasks_by_alias: dict[str, EvalTaskConfig] = {}
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        for alias in row.missing_tasks.split(";"):
            if alias:
                launch_tasks_by_alias[alias] = TASKS_BY_ALIAS[alias]

    if not launch_tasks_by_alias:
        return [], {}

    cache_eval_step = ExecutorStep(
        name=f"{name_prefix}/cache_eval_datasets",
        description="Pre-cache 300M noise parity evaluation datasets to east5 GCS",
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
            eval_tasks=versioned(tuple(launch_tasks_by_alias.values())),
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
        row_tasks = _tasks_for_row(row)
        if not row_tasks:
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone,
        )
        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=row.eval_key,
            model_path=row.checkpoint_root,
            evals=row_tasks,
            resource_config=resource_config,
            discover_latest_checkpoint=True,
            eval_datasets_cache_path=eval_datasets_cache_path,
            eval_datasets_cache_dependency=cache_dependency,
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step, RESULTS_JSON)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[NoiseParityEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 300M noise parity eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=CollectNoiseParityResultsConfig(
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
    parser.add_argument("--executor-prefix", default=DEFAULT_EXECUTOR_PREFIX)
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--state-csv")
    parser.add_argument("--eval-datasets-cache-path", default=DEFAULT_EVAL_DATASETS_CACHE_PATH)
    parser.add_argument(
        "--task-alias",
        action="append",
        choices=TASK_ALIASES,
        default=[],
        help="Limit the parity backfill to one or more task aliases. Defaults to all parity aliases.",
    )
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument("--collect-from-prefix", action="append", default=[])
    parser.add_argument("--collect-output-csv", default=str(RESULTS_CSV_LOCAL))
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
            task_aliases=tuple(args.task_alias) if args.task_alias else TASK_ALIASES,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    if args.include_run_name:
        include_run_names = set(args.include_run_name)
        state_rows = [row for row in state_rows if row.run_name in include_run_names]
        missing_run_names = sorted(include_run_names - {row.run_name for row in state_rows})
        if missing_run_names:
            raise ValueError(f"Requested run names not present in noise parity state: {missing_run_names}")
    if args.collect_from_prefix:
        collect_eval_results_from_prefixes(
            prefixes=args.collect_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.collect_output_csv),
        )
        return
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d noise parity eval steps over %d candidate checkpoints", launch_count, len(state_rows))
    if args.dry_run or os.getenv("CI") is not None:
        return

    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
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
        description=f"{args.name_prefix}: 300M run_00097 fixed/variable noise parity eval completion",
    )


if __name__ == "__main__":
    main()
