# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch DCLM Core eval completion for 300M data-mixture rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
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
from marin.execution.remote import remote
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.domain_phase_mix.launch_300m_english_lite_evals import _checkpoint_region
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_EXPECTED_300M_STEP,
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
from experiments.domain_phase_mix.launch_baseline_scaling_downstream_evals import (
    GENERATION_ENGINE_KWARGS,
    _read_eval_metrics,
)
from experiments.evals.dclm_core import (
    TaskMode,
    dclm_core_centered_accuracy,
    eval_tasks_for_aliases,
    launchable_task_aliases,
    task_aliases_for_mode,
    task_by_alias,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_dclm_core_completion"
STATE_CSV = OUTPUT_DIR / "300m_dclm_core_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_dclm_core_eval_launch_manifest.csv"
RETRY_STATE_CSV = OUTPUT_DIR / "300m_dclm_core_eval_retry_state.csv"
MERGED_RESULTS_CSV = OUTPUT_DIR / "300m_dclm_core_eval_results_merged.csv"
QSPLIT300M_COMPLETED_CSV = TWO_PHASE_MANY_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_dclm_core_evals_20260605"
DEFAULT_EVAL_DATASETS_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/300m-dclm-core-v1"
DEFAULT_MAX_CONCURRENT = 512
DEFAULT_INCLUDED_PANELS = ("signal_300m_6b",)
DCLM_MAX_MODEL_LEN = 2048
# BigBench generation tasks in the pinned lm-eval fork use 128 generated tokens;
# repeat-copy style tasks can be truncated by the old 32-token override.
DCLM_GENERATION_PARAMS = {"max_gen_toks": 128}
DCLM_EXTRACTIVE_GENERATION_PARAMS = {"max_gen_toks": 32}


def _api_model_kwargs(generation_params: dict[str, int]) -> dict[str, object]:
    return {
        **GENERATION_ENGINE_KWARGS,
        # Some lm-eval tasks, notably squadv2, construct request-local generation
        # kwargs directly and do not preserve simple_evaluate(gen_kwargs). Make the
        # local-completions model default match the requested DCLM cap.
        "max_gen_toks": generation_params["max_gen_toks"],
    }


DCLM_GENERATION_API_MODEL_KWARGS = _api_model_kwargs(DCLM_GENERATION_PARAMS)
DCLM_EXTRACTIVE_API_MODEL_KWARGS = _api_model_kwargs(DCLM_EXTRACTIVE_GENERATION_PARAMS)
DCLM_GENERATION_ENGINE_KWARGS = {
    **DCLM_GENERATION_API_MODEL_KWARGS,
    "max_model_len": DCLM_MAX_MODEL_LEN,
    "max_length": DCLM_MAX_MODEL_LEN,
    # lm-eval can only left-truncate generation prompts when it sends token ids.
    "truncate": True,
    "tokenized_requests": True,
    "max_num_batched_tokens": 4096,
}
DCLM_EXTRACTIVE_ENGINE_KWARGS = DCLM_EXTRACTIVE_API_MODEL_KWARGS
RESULTS_CSV = "300m_dclm_core_eval_results.csv"
STATE_OUTPUT_CSV = "300m_dclm_core_eval_state.csv"
EXECUTOR_STATUS_FILE = ".executor_status"
STATUS_SUCCESS = "SUCCESS"
EVAL_OUTPUT_RE = re.compile(r"/(?:lmeval_debug_)?(?P<eval_key>dclm300m_.+)-[0-9a-f]{6}/\.executor_status$")
SIGNAL_PANEL = "signal_300m_6b"
STRATIFIED_RUN_NAME = "baseline_stratified"
DCLM_DERIVED_METRIC_PREFIX = "lm_eval/dclm_core/"
DCLM_SMOOTH_METRICS = frozenset(
    {
        "bpb",
        "choice_logprob",
        "choice_logprob_norm",
        "choice_prob_norm",
        "logprob",
    }
)
DCLM_GENERATION_SMOOTH_MODE = "generation_smooth"
LEVENTER_DCLM_SMOOTH_MODES = frozenset({TaskMode.MCQ.value, DCLM_GENERATION_SMOOTH_MODE})
SQUAD_GENERATION_SMOOTH_TASK = "squad_smooth_loglikelihood"
COQA_GENERATION_SMOOTH_TASK = "coqa_smooth_loglikelihood"
GENERATION_SMOOTH_METRIC_LIST = (
    {"metric": "perplexity", "aggregation": "perplexity", "higher_is_better": False},
    {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
)
STRATIFIED_300M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b"
STRATIFIED_300M_CHECKPOINT_ROOT = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_stratified_300m_6b/baseline_stratified-050d85"
)

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "has_all_tasks", "is_region_local", "eligible"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
    "existing_artifact_count",
    "missing_task_count",
}


class DCLMEvalMode(StrEnum):
    """Launcher mode selector."""

    ALL = "all"
    EXTRACTIVE = "extractive"
    MCQ = "mcq"
    GENERATION = "generation"
    SMOOTH = "smooth"


@dataclass(frozen=True)
class DCLMEvalSpec:
    """One DCLM Core eval state row and potential launch unit."""

    eval_key: str
    mode: str
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
class DCLMEvalCandidate:
    """One candidate checkpoint for DCLM Core eval completion."""

    panel: str
    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int


@dataclass(frozen=True)
class Collect300MDCLMCoreResultsConfig:
    """Config for collecting 300M DCLM Core eval outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def dclm_core_task_aliases(
    *,
    mode: DCLMEvalMode = DCLMEvalMode.ALL,
    included_aliases: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return launchable DCLM Core aliases filtered by launcher mode."""
    if mode == DCLMEvalMode.MCQ:
        aliases = task_aliases_for_mode(TaskMode.MCQ)
    elif mode == DCLMEvalMode.EXTRACTIVE:
        aliases = task_aliases_for_mode(TaskMode.EXTRACTIVE)
    elif mode == DCLMEvalMode.GENERATION:
        aliases = task_aliases_for_mode(TaskMode.GENERATION)
    elif mode == DCLMEvalMode.SMOOTH:
        aliases = (*task_aliases_for_mode(TaskMode.MCQ), *task_aliases_for_mode(TaskMode.GENERATION))
    else:
        aliases = launchable_task_aliases()
    if included_aliases is None:
        return aliases
    unknown_aliases = sorted(set(included_aliases) - set(launchable_task_aliases()))
    if unknown_aliases:
        raise ValueError(f"Unknown or non-launchable DCLM Core aliases requested: {unknown_aliases}")
    return tuple(alias for alias in aliases if alias in set(included_aliases))


def _modes_for_aliases(task_aliases: tuple[str, ...]) -> tuple[TaskMode, ...]:
    modes = sorted({task_by_alias(alias).mode for alias in task_aliases}, key=lambda mode: mode.value)
    return tuple(modes)


def _row_mode_for_task_mode(task_mode: TaskMode, eval_mode: DCLMEvalMode) -> str:
    if eval_mode == DCLMEvalMode.SMOOTH and task_mode == TaskMode.GENERATION:
        return DCLM_GENERATION_SMOOTH_MODE
    return task_mode.value


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("lm_eval/"))


def _task_has_metric(row: pd.Series, alias: str) -> bool:
    task = task_by_alias(alias)
    for metric_path in task.metric_paths():
        if metric_path in row.index and pd.notna(row.get(metric_path)):
            return True
    prefix = f"lm_eval/{alias}/"
    return any(str(column).startswith(prefix) and pd.notna(row.get(column)) for column in row.index)


def _task_has_alias_exact_metric(row: pd.Series, alias: str) -> bool:
    task = task_by_alias(alias)
    metrics = task.metric_candidates or (task.primary_metric,)
    for metric in metrics:
        metric_path = f"lm_eval/{alias}/{metric}"
        if metric_path in row.index and pd.notna(row.get(metric_path)):
            return True
    derived_prefix = f"{DCLM_DERIVED_METRIC_PREFIX}{alias}/"
    return any(str(column).startswith(derived_prefix) and pd.notna(row.get(column)) for column in row.index)


def _task_has_dclm_smooth_metric(row: pd.Series, alias: str) -> bool:
    prefix = f"lm_eval/{alias}/"
    for column in row.index:
        column = str(column)
        if not column.startswith(prefix) or pd.isna(row.get(column)):
            continue
        metric = column[len(prefix) :]
        if metric.startswith("native_") or metric in DCLM_SMOOTH_METRICS:
            return True
    return False


def _metric_coverage_by_root(paths: list[str | Path], task_aliases: tuple[str, ...]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for path in paths:
        if isinstance(path, Path) and not path.exists():
            continue
        frame = _read_csv(path)
        if "checkpoint_root" not in frame.columns or not _metric_columns(frame):
            continue
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if not root:
                continue
            covered = coverage.setdefault(root, set())
            for alias in task_aliases:
                if _task_has_alias_exact_metric(row, alias):
                    covered.add(alias)
    return coverage


def _expected_step_from_row(row: pd.Series) -> int:
    for column in ("target_final_checkpoint_step", "target_eval_step", "max_checkpoint_step"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return int(value)
    return DEFAULT_EXPECTED_300M_STEP


def _qsplit300m_name() -> str:
    from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import NAME

    return NAME


def build_qsplit300m_run_specs() -> list[Any]:
    from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import build_run_specs

    return build_run_specs()


def resolve_latest_checkpoint_root(**kwargs: Any) -> str | None:
    from experiments.domain_phase_mix.qsplit240_replay import resolve_latest_checkpoint_root as resolve

    return resolve(**kwargs)


def _olmix_uncheatable_run_name() -> str:
    from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import RUN_NAME

    return RUN_NAME


def _resolve_qsplit300m_checkpoint_root(run_name: str) -> str:
    root = resolve_latest_checkpoint_root(
        experiment_name_prefix=_qsplit300m_name(),
        run_name=run_name,
        checkpoint_regions=(DEFAULT_TPU_REGION,),
    )
    if root is None:
        return ""
    return root.removesuffix("/checkpoints")


def _qsplit300m_completed_checkpoint_roots() -> dict[str, str]:
    if not QSPLIT300M_COMPLETED_CSV.exists():
        return {}
    frame = pd.read_csv(QSPLIT300M_COMPLETED_CSV, usecols=["run_name", "checkpoint_root"])
    return {
        str(row["run_name"]): str(row["checkpoint_root"]).rstrip("/")
        for _, row in frame.iterrows()
        if pd.notna(row.get("checkpoint_root")) and str(row.get("checkpoint_root")).strip()
    }


def _signal_candidate_records(included_run_names: set[str] | None = None) -> list[DCLMEvalCandidate]:
    candidates: list[DCLMEvalCandidate] = []
    qsplit300m_name = _qsplit300m_name()
    olmix_uncheatable_run_name = _olmix_uncheatable_run_name()
    completed_roots = _qsplit300m_completed_checkpoint_roots()
    for spec in build_qsplit300m_run_specs():
        if included_run_names is not None and spec.run_name not in included_run_names:
            continue
        checkpoint_root = completed_roots.get(spec.run_name) or _resolve_qsplit300m_checkpoint_root(spec.run_name)
        candidates.append(
            DCLMEvalCandidate(
                panel=SIGNAL_PANEL,
                run_name=spec.run_name,
                registry_key=f"qsplit240_300m_6b:{spec.run_name}",
                source_experiment=qsplit300m_name,
                cohort="signal",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
            )
        )

    if included_run_names is None or olmix_uncheatable_run_name in included_run_names:
        checkpoint_root = completed_roots.get(olmix_uncheatable_run_name) or _resolve_qsplit300m_checkpoint_root(
            olmix_uncheatable_run_name
        )
        candidates.append(
            DCLMEvalCandidate(
                panel=SIGNAL_PANEL,
                run_name=olmix_uncheatable_run_name,
                registry_key=f"qsplit240_300m_6b:{olmix_uncheatable_run_name}",
                source_experiment=qsplit300m_name,
                cohort="signal",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
            )
        )
    if included_run_names is None or STRATIFIED_RUN_NAME in included_run_names:
        candidates.append(
            DCLMEvalCandidate(
                panel=SIGNAL_PANEL,
                run_name=STRATIFIED_RUN_NAME,
                registry_key=f"stratified_300m_6b:{STRATIFIED_RUN_NAME}",
                source_experiment=STRATIFIED_300M_SOURCE_EXPERIMENT,
                cohort="signal",
                checkpoint_root=STRATIFIED_300M_CHECKPOINT_ROOT,
                expected_checkpoint_step=DEFAULT_EXPECTED_300M_STEP,
            )
        )
    return candidates


def _extra_panel_candidate_records(
    included_panels: set[str],
    included_run_names: set[str] | None,
) -> list[DCLMEvalCandidate]:
    candidates: list[DCLMEvalCandidate] = []
    for candidate in _candidate_records():
        if candidate.panel == SIGNAL_PANEL or candidate.panel not in included_panels:
            continue
        if included_run_names is not None and candidate.run_name not in included_run_names:
            continue
        candidates.append(
            DCLMEvalCandidate(
                panel=candidate.panel,
                run_name=candidate.run_name,
                registry_key=candidate.registry_key,
                source_experiment=candidate.source_experiment,
                cohort=candidate.cohort,
                checkpoint_root=candidate.checkpoint_root,
                expected_checkpoint_step=candidate.expected_checkpoint_step,
            )
        )
    return candidates


def _dclm_candidate_records(
    included_panels: set[str] | None,
    included_run_names: set[str] | None = None,
) -> list[DCLMEvalCandidate]:
    panels = included_panels or set(DEFAULT_INCLUDED_PANELS)
    candidates: list[DCLMEvalCandidate] = []
    if SIGNAL_PANEL in panels:
        candidates.extend(_signal_candidate_records(included_run_names))
    candidates.extend(_extra_panel_candidate_records(panels, included_run_names))
    by_root: dict[str, DCLMEvalCandidate] = {}
    for candidate in candidates:
        if candidate.checkpoint_root in by_root:
            continue
        by_root[candidate.checkpoint_root] = candidate
    return sorted(by_root.values(), key=lambda row: (row.panel, row.run_name))


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    is_region_local: bool,
    has_all_tasks: bool,
    force_launch: bool = False,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if not is_region_local:
        return False, "defer_checkpoint_region_mismatch"
    if force_launch:
        return True, "launch"
    if has_all_tasks:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
    mode: DCLMEvalMode,
    task_aliases: tuple[str, ...] | None,
    included_run_names: set[str] | None,
    included_panels: set[str] | None,
    assume_exact_hf_checkpoints: bool = False,
    force_launch: bool = False,
) -> list[DCLMEvalSpec]:
    """Build state rows for 300M DCLM Core eval completion."""
    requested_aliases = dclm_core_task_aliases(mode=mode, included_aliases=task_aliases)
    coverage = _metric_coverage_by_root([METRICS_WIDE_CSV, MERGED_RESULTS_CSV], requested_aliases)
    rows: list[DCLMEvalSpec] = []
    for idx, candidate in enumerate(_dclm_candidate_records(included_panels, included_run_names)):
        if assume_exact_hf_checkpoints and candidate.checkpoint_root and candidate.expected_checkpoint_step >= 0:
            latest_hf_checkpoint = (
                f"{candidate.checkpoint_root.rstrip('/')}/hf/step-{candidate.expected_checkpoint_step}"
            )
        else:
            latest_hf_checkpoint = _exact_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        latest_hf_step = candidate.expected_checkpoint_step if latest_hf_checkpoint else -1
        has_exact_hf_checkpoint = bool(latest_hf_checkpoint)
        checkpoint_region = _checkpoint_region(candidate.checkpoint_root)
        is_region_local = checkpoint_region in {"", default_tpu_region}
        existing_tasks = sorted(coverage.get(candidate.checkpoint_root, set()))
        for task_mode in _modes_for_aliases(requested_aliases):
            mode_aliases = tuple(alias for alias in requested_aliases if task_by_alias(alias).mode == task_mode)
            existing_mode_tasks = [alias for alias in existing_tasks if alias in mode_aliases]
            missing_tasks = list(mode_aliases) if force_launch else [
                alias for alias in mode_aliases if alias not in existing_mode_tasks
            ]
            has_all_tasks = not missing_tasks
            eligible, decision = _launch_decision(
                checkpoint_root=candidate.checkpoint_root,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                is_region_local=is_region_local,
                has_all_tasks=has_all_tasks,
                force_launch=force_launch,
            )
            suffix = f"_{eval_key_suffix}" if eval_key_suffix else ""
            row_mode = _row_mode_for_task_mode(task_mode, mode)
            eval_key = f"dclm300m_{idx:03d}_{row_mode}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
            step_name = _step_name_for_eval_key(row_mode, eval_key)
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
                    hf_checkpoint_count=int(bool(latest_hf_checkpoint)),
                    hf_checkpoint_latest=latest_hf_checkpoint,
                    hf_checkpoint_latest_step=latest_hf_step,
                    has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                    checkpoint_region=checkpoint_region,
                    is_region_local=is_region_local,
                    existing_artifact_count=len(existing_mode_tasks),
                    existing_tasks=";".join(existing_mode_tasks),
                    missing_task_count=len(missing_tasks),
                    missing_tasks=";".join(missing_tasks),
                    has_all_tasks=has_all_tasks,
                    task_aliases=";".join(mode_aliases),
                    launch_tpu_type=default_tpu_type,
                    launch_tpu_region=default_tpu_region,
                    launch_tpu_zone=default_tpu_zone,
                    eligible=eligible,
                    launch_decision=decision,
                    step_name=step_name,
                    result_path=f"executor_output:{eval_key}",
                )
            )
    return rows


def _write_local_outputs(rows: list[DCLMEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = pd.DataFrame.from_records([asdict(row) for row in rows])
    state.to_csv(STATE_CSV, index=False)
    state[state["launch_decision"] == "launch"].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _load_state_rows(path: Path | str) -> list[DCLMEvalSpec]:
    frame = _read_csv(path)
    missing_columns = sorted(field.name for field in fields(DCLMEvalSpec) if field.name not in frame.columns)
    if missing_columns:
        raise ValueError(f"State CSV {path} is missing columns: {missing_columns}")
    rows: list[DCLMEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(DCLMEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(DCLMEvalSpec(**kwargs))
    return rows


def _normalize_gcs_path(path: object) -> str:
    value = str(path)
    return value if value.startswith("gs://") else f"gs://{value}"


def _read_text(path: str) -> str:
    with fsspec.open(path, "rt") as handle:
        return handle.read().strip()


def _status_paths_under_prefix(prefix: str) -> list[str]:
    root = prefix.rstrip("/")
    patterns = [
        f"{root}/evaluation/lm_evaluation_harness_levanter/lmeval_debug_dclm300m_*/{EXECUTOR_STATUS_FILE}",
        f"{root}/evaluation/lm_evaluation_harness/dclm300m_*/{EXECUTOR_STATUS_FILE}",
    ]
    paths: list[str] = []
    for pattern in patterns:
        fs, _, _ = fsspec.get_fs_token_paths(pattern)
        paths.extend(_normalize_gcs_path(path) for path in fs.glob(pattern))
    return sorted(set(paths))


def _eval_key_from_status_path(status_path: str) -> str | None:
    match = EVAL_OUTPUT_RE.search(status_path)
    if match is None:
        return None
    return match.group("eval_key")


def _scan_eval_statuses(prefixes: list[str]) -> dict[str, dict[str, str]]:
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


def _ensure_eval_dataset_cache_manifest(eval_datasets_cache_path: str) -> None:
    manifest_path = f"{eval_datasets_cache_path.rstrip('/')}/.eval_datasets_manifest.json"
    fs, _, _ = fsspec.get_fs_token_paths(manifest_path)
    if not fs.exists(manifest_path):
        raise FileNotFoundError(
            f"Requested existing eval dataset cache is missing its manifest: {manifest_path}"
        )


def write_retry_state_from_prefix(
    *,
    prefix: str,
    state_rows: list[DCLMEvalSpec],
    output_csv: Path,
) -> list[DCLMEvalSpec]:
    """Write a state CSV containing only failed/missing eval rows from an executor prefix."""
    statuses = _scan_eval_statuses([prefix])
    retry_rows: list[DCLMEvalSpec] = []
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        entry = statuses.get(row.eval_key)
        if entry is not None and entry["status"] == STATUS_SUCCESS:
            continue
        retry_rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in retry_rows]).to_csv(output_csv, index=False)
    return retry_rows


def write_gapfill_state_from_results(
    *,
    state_rows: list[DCLMEvalSpec],
    results_csv: Path,
    output_csv: Path,
    eval_key_suffix: str,
    task_aliases: tuple[str, ...] | None = None,
) -> list[DCLMEvalSpec]:
    """Write launch rows for DCLM aliases still missing from a collected result table."""
    results = _read_csv(results_csv)
    if "checkpoint_root" not in results.columns:
        raise ValueError(f"Results CSV {results_csv} is missing checkpoint_root")
    result_by_root = {
        _string_value(row.get("checkpoint_root")).rstrip("/"): row
        for _, row in results.iterrows()
        if _string_value(row.get("checkpoint_root")).rstrip("/")
    }
    requested_aliases = set(task_aliases) if task_aliases is not None else set(launchable_task_aliases())
    gapfill_rows: list[DCLMEvalSpec] = []
    for row in state_rows:
        if not row.eligible or row.launch_decision == "defer_checkpoint_region_mismatch":
            continue
        result = result_by_root.get(row.checkpoint_root.rstrip("/"))
        if result is None:
            continue
        aliases = [
            alias
            for alias in row.task_aliases.split(";")
            if alias and alias in requested_aliases and not _task_has_alias_exact_metric(result, alias)
        ]
        if not aliases:
            continue
        eval_key = f"{row.eval_key}_{eval_key_suffix}"
        step_group = "lm_evaluation_harness_levanter" if row.mode == TaskMode.MCQ.value else "lm_evaluation_harness"
        step_name = f"evaluation/{step_group}/{eval_key}"
        if row.mode == TaskMode.MCQ.value:
            step_name = f"evaluation/{step_group}/lmeval_debug_{eval_key}"
        gapfill_rows.append(
            replace(
                row,
                eval_key=eval_key,
                existing_artifact_count=0,
                existing_tasks="",
                missing_task_count=len(aliases),
                missing_tasks=";".join(aliases),
                has_all_tasks=False,
                task_aliases=";".join(aliases),
                launch_decision="launch",
                step_name=step_name,
                result_path=f"executor_output:{eval_key}",
            )
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in gapfill_rows]).to_csv(output_csv, index=False)
    return gapfill_rows


def write_native_smooth_gapfill_state_from_results(
    *,
    state_rows: list[DCLMEvalSpec],
    results_csv: Path,
    output_csv: Path,
    eval_key_suffix: str,
    task_aliases: tuple[str, ...] | None = None,
) -> list[DCLMEvalSpec]:
    """Write MCQ launch rows for aliases still missing native smooth metrics."""
    results = _read_csv(results_csv)
    if "checkpoint_root" not in results.columns:
        raise ValueError(f"Results CSV {results_csv} is missing checkpoint_root")
    result_by_root = {
        _string_value(row.get("checkpoint_root")).rstrip("/"): row
        for _, row in results.iterrows()
        if _string_value(row.get("checkpoint_root")).rstrip("/")
    }
    requested_aliases = set(task_aliases) if task_aliases is not None else set(task_aliases_for_mode(TaskMode.MCQ))
    gapfill_rows: list[DCLMEvalSpec] = []
    for row in state_rows:
        if row.mode != TaskMode.MCQ.value:
            continue
        if not row.eligible or row.launch_decision == "defer_checkpoint_region_mismatch":
            continue
        result = result_by_root.get(row.checkpoint_root.rstrip("/"))
        if result is None:
            aliases = [alias for alias in row.task_aliases.split(";") if alias and alias in requested_aliases]
        else:
            aliases = [
                alias
                for alias in row.task_aliases.split(";")
                if alias and alias in requested_aliases and not _task_has_dclm_smooth_metric(result, alias)
            ]
        if not aliases:
            continue
        eval_key = f"{row.eval_key}_{eval_key_suffix}"
        gapfill_rows.append(
            replace(
                row,
                eval_key=eval_key,
                existing_artifact_count=0,
                existing_tasks="",
                missing_task_count=len(aliases),
                missing_tasks=";".join(aliases),
                has_all_tasks=False,
                task_aliases=";".join(aliases),
                launch_decision="launch",
                step_name=_step_name_for_eval_key(row.mode, eval_key),
                result_path=f"executor_output:{eval_key}",
            )
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in gapfill_rows]).to_csv(output_csv, index=False)
    return gapfill_rows


def write_smooth_gapfill_state_from_results(
    *,
    state_rows: list[DCLMEvalSpec],
    results_csv: Path,
    output_csv: Path,
    eval_key_suffix: str,
    task_aliases: tuple[str, ...] | None = None,
) -> list[DCLMEvalSpec]:
    """Write launch rows for aliases still missing any smooth scalar.

    This is broader than the legacy native-MCQ helper: it also includes
    generation-smooth rows, where generation tasks are evaluated with
    teacher-forced target loglikelihood rather than hard generation.
    """
    results = _read_csv(results_csv)
    if "checkpoint_root" not in results.columns:
        raise ValueError(f"Results CSV {results_csv} is missing checkpoint_root")
    result_by_root = {
        _string_value(row.get("checkpoint_root")).rstrip("/"): row
        for _, row in results.iterrows()
        if _string_value(row.get("checkpoint_root")).rstrip("/")
    }
    requested_aliases = set(task_aliases) if task_aliases is not None else set(dclm_core_task_aliases(mode=DCLMEvalMode.SMOOTH))
    gapfill_rows: list[DCLMEvalSpec] = []
    for row in state_rows:
        if row.mode not in LEVENTER_DCLM_SMOOTH_MODES:
            continue
        if not row.eligible or row.launch_decision == "defer_checkpoint_region_mismatch":
            continue
        result = result_by_root.get(row.checkpoint_root.rstrip("/"))
        if result is None:
            aliases = [alias for alias in row.task_aliases.split(";") if alias and alias in requested_aliases]
        else:
            aliases = [
                alias
                for alias in row.task_aliases.split(";")
                if alias and alias in requested_aliases and not _task_has_dclm_smooth_metric(result, alias)
            ]
        if not aliases:
            continue
        eval_key = f"{row.eval_key}_{eval_key_suffix}"
        gapfill_rows.append(
            replace(
                row,
                eval_key=eval_key,
                existing_artifact_count=0,
                existing_tasks="",
                missing_task_count=len(aliases),
                missing_tasks=";".join(aliases),
                has_all_tasks=False,
                task_aliases=";".join(aliases),
                launch_decision="launch",
                step_name=_step_name_for_eval_key(row.mode, eval_key),
                result_path=f"executor_output:{eval_key}",
            )
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([asdict(row) for row in gapfill_rows]).to_csv(output_csv, index=False)
    return gapfill_rows


def _status_entry_for_row(row: DCLMEvalSpec, statuses: dict[str, dict[str, str]]) -> dict[str, str] | None:
    entry = statuses.get(row.eval_key)
    if entry is not None:
        return entry
    run_slug = _slug(row.run_name)
    mode_token = f"_{row.mode}_"
    matches = [
        status
        for eval_key, status in statuses.items()
        if mode_token in eval_key and (f"_{run_slug}_" in eval_key or eval_key.endswith(f"_{run_slug}"))
    ]
    success_matches = [status for status in matches if status["status"] == STATUS_SUCCESS]
    if len(success_matches) == 1:
        return success_matches[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _metric_rows_from_result_paths(
    state_rows: list[DCLMEvalSpec], results_by_eval_key: dict[str, InputName]
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
        record.update(dclm_core_centered_accuracy(record))
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)
    return records


def _append_semicolon_value(record: dict[str, Any], key: str, value: object) -> None:
    value_string = _string_value(value)
    if not value_string:
        return
    parts = [part for part in _string_value(record.get(key)).split(";") if part]
    for value_part in value_string.split(";"):
        value_part = value_part.strip()
        if value_part and value_part not in parts:
            parts.append(value_part)
    record[key] = ";".join(parts)


def _merge_records_by_checkpoint(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_root: dict[str, dict[str, Any]] = {}
    metric_prefixes = ("lm_eval/",)
    union_fields = (
        "mode",
        "eval_key",
        "task_aliases",
        "missing_tasks",
        "existing_tasks",
        "step_name",
        "result_path",
        "status_path",
        "executor_eval_key",
        "collection_status",
        "collection_error",
    )
    for record in records:
        root = _string_value(record.get("checkpoint_root")).rstrip("/")
        if not root:
            continue
        merged = by_root.setdefault(
            root,
            {key: value for key, value in record.items() if not str(key).startswith(metric_prefixes)},
        )
        for field in union_fields:
            _append_semicolon_value(merged, field, record.get(field))
        for key, value in record.items():
            if str(key).startswith(metric_prefixes) and pd.notna(value):
                merged[key] = value
            elif key not in union_fields and pd.notna(value) and not _string_value(merged.get(key)):
                merged[key] = value
    for merged in by_root.values():
        merged.update(dclm_core_centered_accuracy(merged))
    return list(by_root.values())


def _existing_metric_records_for_roots(roots: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not roots:
        return records
    for path in (METRICS_WIDE_CSV, MERGED_RESULTS_CSV):
        if isinstance(path, Path) and not path.exists():
            continue
        frame = _read_csv(path)
        if "checkpoint_root" not in frame.columns:
            continue
        metric_columns = [
            column
            for column in _metric_columns(frame)
            if not str(column).startswith(DCLM_DERIVED_METRIC_PREFIX)
        ]
        if not metric_columns:
            continue
        for _, row in frame.iterrows():
            root = _string_value(row.get("checkpoint_root")).rstrip("/")
            if root not in roots:
                continue
            record: dict[str, Any] = {"checkpoint_root": root}
            for column in metric_columns:
                value = row.get(column)
                if pd.notna(value):
                    record[column] = value
            if len(record) > 1:
                records.append(record)
    return records


def collect_eval_results_from_prefixes(
    *,
    prefixes: list[str],
    state_rows: list[DCLMEvalSpec],
    output_csv: Path,
) -> pd.DataFrame:
    """Collect successful DCLM Core child outputs from executor prefixes into one CSV."""
    statuses = _scan_eval_statuses(prefixes)
    roots = {_string_value(row.checkpoint_root).rstrip("/") for row in state_rows if row.checkpoint_root}
    records: list[dict[str, Any]] = _existing_metric_records_for_roots(roots)
    for row in state_rows:
        record = asdict(row)
        entry = _status_entry_for_row(row, statuses)
        if entry is None:
            record["collection_status"] = "missing_executor_status"
            record["collection_error"] = "no_executor_status_found"
            records.append(record)
            continue
        record["executor_status"] = entry["status"]
        record["executor_eval_key"] = entry["eval_key"]
        record["result_path"] = entry["output_path"]
        record["status_path"] = entry["status_path"]
        metrics, error = _read_eval_metrics(entry["output_path"])
        record.update(metrics)
        record["collection_status"] = (
            "collected" if metrics and entry["status"] == STATUS_SUCCESS else f"executor_status_{entry['status']}"
        )
        record["collection_error"] = error if metrics else entry["status"]
        records.append(record)
    frame = pd.DataFrame.from_records(_merge_records_by_checkpoint(records))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    logger.info("Wrote %d collected checkpoint rows to %s", len(frame), output_csv)
    return frame


def collect_eval_results(config: Collect300MDCLMCoreResultsConfig) -> None:
    """Collect DCLM Core eval outputs into one normalized CSV."""
    state_rows = [DCLMEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    roots = {_string_value(row.checkpoint_root).rstrip("/") for row in state_rows if row.checkpoint_root}
    records = _merge_records_by_checkpoint(
        [
            *_existing_metric_records_for_roots(roots),
            *_metric_rows_from_result_paths(state_rows, config.results_by_eval_key),
        ]
    )
    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def _row_missing_aliases(row: DCLMEvalSpec) -> tuple[str, ...]:
    return tuple(alias for alias in row.missing_tasks.split(";") if alias)


def _step_name_for_eval_key(mode: str, eval_key: str) -> str:
    step_group = "lm_evaluation_harness_levanter" if mode in LEVENTER_DCLM_SMOOTH_MODES else "lm_evaluation_harness"
    if mode in LEVENTER_DCLM_SMOOTH_MODES:
        return f"evaluation/{step_group}/lmeval_debug_{eval_key}"
    return f"evaluation/{step_group}/{eval_key}"


def _generation_smooth_eval_tasks(task_aliases: tuple[str, ...]) -> list[EvalTaskConfig]:
    eval_tasks: list[EvalTaskConfig] = []
    for alias in task_aliases:
        base = task_by_alias(alias).eval_config()
        if alias == "coqa_0shot":
            eval_tasks.append(
                EvalTaskConfig(
                    name=COQA_GENERATION_SMOOTH_TASK,
                    num_fewshot=base.num_fewshot,
                    task_alias=base.task_alias,
                    task_kwargs={
                        "dataset_path": "EleutherAI/coqa",
                        "training_split": "train",
                        "validation_split": "validation",
                        "fewshot_split": "train",
                        "output_type": "loglikelihood",
                        "doc_to_text": (
                            "{{story}}\n\n"
                            "{% for question in questions.input_text %}"
                            "Q: {{question}}\n\n"
                            "{% if loop.last %}A:{% else %}"
                            "A: {{answers.input_text[loop.index0]}}\n\n"
                            "{% endif %}"
                            "{% endfor %}"
                        ),
                        "doc_to_target": "{{answers.input_text[(questions.input_text | length) - 1]}}",
                        "metric_list": [dict(metric) for metric in GENERATION_SMOOTH_METRIC_LIST],
                        "metadata": {
                            "version": 1.0,
                            "source": "CoQA smooth proxy using story, conversation history, and primary answer",
                        },
                    },
                )
            )
            continue
        if alias == "squad_10shot":
            eval_tasks.append(
                EvalTaskConfig(
                    name=SQUAD_GENERATION_SMOOTH_TASK,
                    num_fewshot=base.num_fewshot,
                    task_alias=base.task_alias,
                    task_kwargs={
                        "dataset_path": "hazyresearch/based-squad",
                        "validation_split": "validation",
                        "fewshot_split": "validation",
                        "output_type": "loglikelihood",
                        "doc_to_text": "{{text}}",
                        "doc_to_target": "{{value}}",
                        "metric_list": [dict(metric) for metric in GENERATION_SMOOTH_METRIC_LIST],
                        "metadata": {
                            "version": 1.0,
                            "source": "squad_completion smooth proxy using based-squad prompt and answer fields",
                        },
                    },
                )
            )
            continue
        task_kwargs = dict(base.task_kwargs or {})
        task_kwargs.pop("generation_kwargs", None)
        task_kwargs.pop("filter_list", None)
        task_kwargs["output_type"] = "loglikelihood"
        task_kwargs["metric_list"] = [dict(metric) for metric in GENERATION_SMOOTH_METRIC_LIST]
        eval_tasks.append(
            EvalTaskConfig(
                name=base.name,
                num_fewshot=base.num_fewshot,
                task_alias=base.task_alias,
                task_kwargs=task_kwargs,
            )
        )
    return eval_tasks


def split_state_rows_by_task_alias(state_rows: list[DCLMEvalSpec]) -> list[DCLMEvalSpec]:
    """Split launch rows so each child eval handles exactly one missing task alias."""
    split_rows: list[DCLMEvalSpec] = []
    for row in state_rows:
        aliases = _row_missing_aliases(row)
        if row.launch_decision != "launch" or len(aliases) <= 1:
            split_rows.append(row)
            continue
        for alias in aliases:
            eval_key = f"{row.eval_key}_{_slug(alias)}"
            split_rows.append(
                replace(
                    row,
                    eval_key=eval_key,
                    missing_task_count=1,
                    missing_tasks=alias,
                    task_aliases=alias,
                    step_name=_step_name_for_eval_key(row.mode, eval_key),
                    result_path=f"executor_output:{eval_key}",
                )
            )
    return split_rows


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[DCLMEvalSpec],
    max_eval_instances: int | None,
    eval_datasets_cache_path: str | None,
    log_samples: bool = False,
    sample_log_all: bool = False,
    max_logged_samples_per_task: int | None = None,
    sample_smooth_metrics: bool = False,
    drop_samples_after_metrics: bool = False,
    use_wandb_tracker: bool = True,
    child_preemptible: bool = True,
    reuse_existing_eval_dataset_cache: bool = False,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build DCLM Core eval steps for rows requiring launch."""
    from marin.evaluation.eval_dataset_cache import (
        HF_CACHE_LAYOUT_VERSION,
        CacheEvalDatasetsConfig,
        _cache_eval_datasets,
    )

    from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness, evaluate_lm_evaluation_harness

    eval_steps: list[ExecutorStep] = []
    results_by_eval_key: dict[str, InputName] = {}
    mcq_aliases = tuple(
        alias
        for alias in task_aliases_for_mode(TaskMode.MCQ)
        if any(alias in _row_missing_aliases(row) for row in state_rows if row.launch_decision == "launch")
    )
    cache_dependency: InputName | None = None
    if mcq_aliases and eval_datasets_cache_path is not None:
        if reuse_existing_eval_dataset_cache:
            _ensure_eval_dataset_cache_manifest(eval_datasets_cache_path)
        else:
            cache_eval_step = ExecutorStep(
                name=f"{name_prefix}/cache_eval_datasets",
                description="Pre-cache DCLM Core MCQ evaluation datasets to east5 GCS",
                fn=remote(
                    _cache_eval_datasets,
                    resources=ResourceConfig.with_cpu(cpu=1, ram="8g", disk="32g", regions=[DEFAULT_TPU_REGION]),
                    pip_dependency_groups=["eval", "cpu"],
                ),
                config=CacheEvalDatasetsConfig(
                    eval_tasks=tuple(eval_tasks_for_aliases(mcq_aliases)),
                    gcs_path=eval_datasets_cache_path,
                    cache_layout_version=versioned(HF_CACHE_LAYOUT_VERSION),
                ),
            )
            eval_steps.append(cache_eval_step)
            cache_dependency = output_path_of(cache_eval_step, ".eval_datasets_manifest.json")

    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        aliases = _row_missing_aliases(row)
        if not aliases:
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone,
            preemptible=child_preemptible,
        )
        eval_tasks = eval_tasks_for_aliases(aliases)
        if row.mode == TaskMode.MCQ.value:
            eval_step = evaluate_levanter_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=eval_tasks,
                resource_config=resource_config,
                max_eval_instances=max_eval_instances,
                discover_latest_checkpoint=True,
                eval_datasets_cache_path=eval_datasets_cache_path,
                eval_datasets_cache_dependency=cache_dependency,
                log_samples=log_samples,
                sample_log_all=sample_log_all,
                max_logged_samples_per_task=max_logged_samples_per_task,
                sample_smooth_metrics=sample_smooth_metrics,
                drop_samples_after_metrics=drop_samples_after_metrics,
                use_wandb_tracker=use_wandb_tracker,
            )
        elif row.mode == DCLM_GENERATION_SMOOTH_MODE:
            eval_step = evaluate_levanter_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=_generation_smooth_eval_tasks(aliases),
                resource_config=resource_config,
                max_eval_instances=max_eval_instances,
                discover_latest_checkpoint=True,
                log_samples=log_samples,
                sample_log_all=sample_log_all,
                max_logged_samples_per_task=max_logged_samples_per_task,
                sample_smooth_metrics=sample_smooth_metrics,
                drop_samples_after_metrics=drop_samples_after_metrics,
                use_wandb_tracker=use_wandb_tracker,
            )
        elif row.mode == TaskMode.GENERATION.value:
            eval_step = evaluate_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=eval_tasks,
                resource_config=resource_config,
                max_eval_instances=max_eval_instances,
                engine_kwargs=DCLM_GENERATION_ENGINE_KWARGS,
                generation_params=DCLM_GENERATION_PARAMS,
                discover_latest_checkpoint=True,
            )
        elif row.mode == TaskMode.EXTRACTIVE.value:
            eval_step = evaluate_lm_evaluation_harness(
                model_name=row.eval_key,
                model_path=row.checkpoint_root,
                evals=eval_tasks,
                resource_config=resource_config,
                max_eval_instances=max_eval_instances,
                engine_kwargs=DCLM_EXTRACTIVE_ENGINE_KWARGS,
                generation_params=DCLM_EXTRACTIVE_GENERATION_PARAMS,
                discover_latest_checkpoint=True,
            )
        else:
            raise ValueError(f"Unknown DCLM eval row mode: {row.mode}")
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[DCLMEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final result collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect 300M DCLM Core eval results for {len(results_by_eval_key)} eval steps",
        fn=collect_eval_results,
        config=Collect300MDCLMCoreResultsConfig(
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
    parser.add_argument("--skip-eval-dataset-cache", action="store_true")
    parser.add_argument(
        "--reuse-existing-eval-dataset-cache",
        action="store_true",
        help=(
            "Use --eval-datasets-cache-path in eval children without scheduling a cache build step. "
            "Requires an existing .eval_datasets_manifest.json at the cache path."
        ),
    )
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--sample-log-all", action="store_true")
    parser.add_argument("--max-logged-samples-per-task", type=int)
    parser.add_argument("--sample-smooth-metrics", action="store_true")
    parser.add_argument("--drop-samples-after-metrics", action="store_true")
    parser.add_argument(
        "--disable-wandb-tracker",
        action="store_true",
        help=(
            "Use a no-op Levanter tracker for MCQ eval children. Results still upload to GCS; "
            "this avoids W&B teardown failures leaving completed evals marked RUNNING."
        ),
    )
    parser.add_argument(
        "--split-task-alias-rows",
        action="store_true",
        help=(
            "Split launch state rows into one row per missing task alias before building eval steps. "
            "This creates shorter child eval jobs while preserving checkpoint-level collection. "
            "When writing retries from a prefix, only use this with prefixes that were also launched split."
        ),
    )
    parser.add_argument(
        "--child-no-preemptible",
        action="store_true",
        help="Require non-preemptible TPU workers for child eval tasks; does not alter the Iris parent job.",
    )
    parser.add_argument(
        "--force-launch",
        action="store_true",
        help="Launch requested rows even when hard DCLM metrics already exist for the checkpoint.",
    )
    parser.add_argument(
        "--assume-exact-hf-checkpoints",
        action="store_true",
        help=(
            "Trust completed checkpoint metadata and use checkpoint_root/hf/step-<expected> without "
            "probing every HF export on GCS. Use only for audited completed 300M rows."
        ),
    )
    parser.add_argument("--mode", choices=[mode.value for mode in DCLMEvalMode], default=DCLMEvalMode.ALL.value)
    parser.add_argument(
        "--task-alias",
        action="append",
        choices=launchable_task_aliases(),
        default=[],
        help="Limit this launch to one or more launchable DCLM Core task aliases.",
    )
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument(
        "--include-panel",
        action="append",
        default=[],
        help="Candidate panels to include. Defaults to the 300M signal swarm only.",
    )
    parser.add_argument("--write-retry-state-from-prefix")
    parser.add_argument("--retry-state-output", default=str(RETRY_STATE_CSV))
    parser.add_argument("--write-gapfill-state-from-results", action="store_true")
    parser.add_argument("--write-smooth-gapfill-state-from-results", action="store_true")
    parser.add_argument("--write-native-smooth-gapfill-state-from-results", action="store_true")
    parser.add_argument("--gapfill-results-csv", default=str(MERGED_RESULTS_CSV))
    parser.add_argument("--gapfill-state-output", default=str(RETRY_STATE_CSV))
    parser.add_argument("--gapfill-eval-key-suffix", default="gapfill")
    parser.add_argument(
        "--gapfill-task-alias",
        action="append",
        choices=launchable_task_aliases(),
        default=[],
        help="Limit generated gap-fill rows to one or more DCLM Core aliases.",
    )
    parser.add_argument("--collect-from-prefix", action="append", default=[])
    parser.add_argument("--collect-output-csv", default=str(MERGED_RESULTS_CSV))
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
            mode=DCLMEvalMode(args.mode),
            task_aliases=tuple(args.task_alias) if args.task_alias else None,
            included_run_names=set(args.include_run_name) if args.include_run_name else None,
            included_panels=set(args.include_panel) if args.include_panel else None,
            assume_exact_hf_checkpoints=args.assume_exact_hf_checkpoints,
            force_launch=args.force_launch,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    if args.split_task_alias_rows:
        state_rows = split_state_rows_by_task_alias(state_rows)
    if args.write_retry_state_from_prefix is not None:
        write_retry_state_from_prefix(
            prefix=args.write_retry_state_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.retry_state_output),
        )
        return
    if args.write_gapfill_state_from_results:
        write_gapfill_state_from_results(
            state_rows=state_rows,
            results_csv=Path(args.gapfill_results_csv),
            output_csv=Path(args.gapfill_state_output),
            eval_key_suffix=args.gapfill_eval_key_suffix,
            task_aliases=tuple(args.gapfill_task_alias) if args.gapfill_task_alias else None,
        )
        return
    if args.write_native_smooth_gapfill_state_from_results:
        write_native_smooth_gapfill_state_from_results(
            state_rows=state_rows,
            results_csv=Path(args.gapfill_results_csv),
            output_csv=Path(args.gapfill_state_output),
            eval_key_suffix=args.gapfill_eval_key_suffix,
            task_aliases=tuple(args.gapfill_task_alias) if args.gapfill_task_alias else None,
        )
        return
    if args.write_smooth_gapfill_state_from_results:
        write_smooth_gapfill_state_from_results(
            state_rows=state_rows,
            results_csv=Path(args.gapfill_results_csv),
            output_csv=Path(args.gapfill_state_output),
            eval_key_suffix=args.gapfill_eval_key_suffix,
            task_aliases=tuple(args.gapfill_task_alias) if args.gapfill_task_alias else None,
        )
        return
    if args.collect_from_prefix:
        collect_eval_results_from_prefixes(
            prefixes=args.collect_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.collect_output_csv),
        )
        return
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    launch_aliases = sorted(
        {alias for row in state_rows if row.launch_decision == "launch" for alias in _row_missing_aliases(row)}
    )
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info(
        "Prepared %d DCLM Core eval rows over %d state rows and %d launch aliases",
        launch_count,
        len(state_rows),
        len(launch_aliases),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return
    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        max_eval_instances=args.max_eval_instances,
        eval_datasets_cache_path=None if args.skip_eval_dataset_cache else args.eval_datasets_cache_path,
        log_samples=args.log_samples,
        sample_log_all=args.sample_log_all,
        max_logged_samples_per_task=args.max_logged_samples_per_task,
        sample_smooth_metrics=args.sample_smooth_metrics,
        drop_samples_after_metrics=args.drop_samples_after_metrics,
        use_wandb_tracker=not args.disable_wandb_tracker,
        child_preemptible=not args.child_no_preemptible,
        reuse_existing_eval_dataset_cache=args.reuse_existing_eval_dataset_cache,
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
        description=f"{args.name_prefix}: 300M DCLM Core eval completion",
    )


if __name__ == "__main__":
    main()
