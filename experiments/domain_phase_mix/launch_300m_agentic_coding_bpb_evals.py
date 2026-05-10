# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch agentic-coding assistant-action BPB evals for 300M signal/noise rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from marin.evaluation.perplexity_gap import GapFinderModelConfig, model_perplexity_scores, raw_text_dataset
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.agentic_coding_eval_dataset import (
    DEFAULT_OUTPUT_URI,
    MANIFEST_FILENAME,
    agentic_coding_eval_bundle_step,
    agentic_eval_slices,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    _bool_value,
    _candidate_records,
    _exact_hf_checkpoint,
    _executor_prefix,
    _slug,
    _string_value,
)
from experiments.marin_models import marin_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_agentic_coding_bpb"
STATE_CSV = OUTPUT_DIR / "300m_agentic_coding_bpb_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_agentic_coding_bpb_launch_manifest.csv"
RESULTS_CSV_LOCAL = OUTPUT_DIR / "300m_agentic_coding_bpb_results.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_agentic_coding_bpb_evals_20260504"
RESULTS_CSV = "300m_agentic_coding_bpb_results.csv"
STATE_OUTPUT_CSV = "300m_agentic_coding_bpb_state.csv"
SUMMARY_JSON = "summary.json"
MAX_EVAL_LENGTH = 4096
MAX_DOC_BYTES = 32_768
MAX_DOCS_PER_DATASET = 512
CANARY_DOCS_PER_DATASET = 32
EXPECTED_CANDIDATE_COUNT = 262
ALLOWED_PANELS = {
    "signal_300m_6b",
    "fixed_seed_noise_300m_6b",
    "variable_subset_noise_300m_6b",
    "proportional_variable_subset_noise_60m_1p2b",
    "proportional_variable_subset_noise_300m_6b",
    "proportional_perturbation_60m_1p2b",
    "proportional_perturbation_300m_6b",
    "proportional_baseline_anchor_60m_1p2b",
    "proportional_baseline_anchor_300m_6b",
}
SIGNAL_SOURCE_EXPERIMENTS = {
    "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b",
    "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
}

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "eligible", "has_existing_results", "uses_east5_checkpoint"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
}

PRIMARY_METRIC = "eval/agentic_coding/success_macro_bpb"
CODERFORGE_PRIMARY_METRIC = "eval/agentic_coding/coderforge_success_macro_bpb"


@dataclass(frozen=True)
class AgenticBpbEvalSpec:
    """One agentic-coding BPB eval state row and potential launch unit."""

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
    uses_east5_checkpoint: bool
    has_existing_results: bool
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class CollectAgenticBpbResultsConfig:
    """Config for collecting agentic-coding BPB outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _existing_result_roots(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "checkpoint_root" not in frame.columns or PRIMARY_METRIC not in frame.columns:
        return set()
    return {
        _string_value(row.get("checkpoint_root")).rstrip("/")
        for _, row in frame.iterrows()
        if pd.notna(row.get(PRIMARY_METRIC))
    }


def _uses_east5_checkpoint(checkpoint_root: str) -> bool:
    return checkpoint_root.startswith("gs://marin-us-east5/")


def _east5_checkpoint_root(checkpoint_root: str) -> str:
    if checkpoint_root.startswith("gs://marin-us-central1/"):
        return "gs://marin-us-east5/" + checkpoint_root.removeprefix("gs://marin-us-central1/")
    return checkpoint_root


def _east5_hf_checkpoint(checkpoint_root: str, expected_step: int) -> str:
    east5_root = _east5_checkpoint_root(checkpoint_root)
    if not _uses_east5_checkpoint(east5_root):
        return ""
    return _exact_hf_checkpoint(east5_root, expected_step)


def _launch_decision(
    *,
    checkpoint_root: str,
    has_exact_hf_checkpoint: bool,
    uses_east5_checkpoint: bool,
    has_existing_results: bool,
) -> tuple[bool, str]:
    if not checkpoint_root:
        return False, "defer_missing_checkpoint"
    if not has_exact_hf_checkpoint:
        return False, "defer_missing_exact_hf_checkpoint"
    if not uses_east5_checkpoint:
        return False, "defer_checkpoint_not_east5"
    if has_existing_results:
        return True, "skip_existing"
    return True, "launch"


def build_state_rows(
    *,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
    eval_key_suffix: str,
    allow_partial: bool,
) -> list[AgenticBpbEvalSpec]:
    """Build state rows for 300M agentic-coding BPB evals."""
    existing_roots = _existing_result_roots(RESULTS_CSV_LOCAL)
    candidates = [
        candidate
        for candidate in _candidate_records()
        if candidate.panel in ALLOWED_PANELS
        and (candidate.panel != "signal_300m_6b" or candidate.source_experiment in SIGNAL_SOURCE_EXPERIMENTS)
    ]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT and not allow_partial:
        counts = pd.Series([candidate.panel for candidate in candidates]).value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} 300M signal/noise candidates, found {len(candidates)}; "
            f"panel counts={counts}. Pass --allow-partial only for debugging."
        )

    rows: list[AgenticBpbEvalSpec] = []
    for idx, candidate in enumerate(candidates):
        exact_hf_checkpoint = _east5_hf_checkpoint(candidate.checkpoint_root, candidate.expected_checkpoint_step)
        has_exact_hf_checkpoint = bool(exact_hf_checkpoint)
        uses_east5_checkpoint = bool(exact_hf_checkpoint)
        has_existing_results = candidate.checkpoint_root.rstrip("/") in existing_roots
        eligible, launch_decision = _launch_decision(
            checkpoint_root=candidate.checkpoint_root,
            has_exact_hf_checkpoint=has_exact_hf_checkpoint,
            uses_east5_checkpoint=uses_east5_checkpoint,
            has_existing_results=has_existing_results,
        )
        suffix = f"_{_slug(eval_key_suffix)}" if eval_key_suffix else ""
        eval_key = f"agentic300m_{idx:03d}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            AgenticBpbEvalSpec(
                eval_key=eval_key,
                panel=candidate.panel,
                run_name=candidate.run_name,
                registry_key=candidate.registry_key,
                source_experiment=candidate.source_experiment,
                cohort=candidate.cohort,
                checkpoint_root=candidate.checkpoint_root,
                expected_checkpoint_step=candidate.expected_checkpoint_step,
                hf_checkpoint_count=1 if has_exact_hf_checkpoint else 0,
                hf_checkpoint_latest=exact_hf_checkpoint,
                hf_checkpoint_latest_step=candidate.expected_checkpoint_step if has_exact_hf_checkpoint else -1,
                has_exact_hf_checkpoint=has_exact_hf_checkpoint,
                uses_east5_checkpoint=uses_east5_checkpoint,
                has_existing_results=has_existing_results,
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=launch_decision,
                step_name=f"analysis/model_perplexity_scores/agentic_coding_bpb/{eval_key}",
                result_path="",
            )
        )
    return rows


def _load_state_rows(path: str | Path) -> list[AgenticBpbEvalSpec]:
    frame = _read_csv(path)
    expected_fields = {field.name for field in fields(AgenticBpbEvalSpec)}
    missing = expected_fields - set(frame.columns)
    if missing:
        raise ValueError(f"State CSV {path} is missing columns: {sorted(missing)}")
    rows: list[AgenticBpbEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(AgenticBpbEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(AgenticBpbEvalSpec(**kwargs))
    return rows


def _write_local_outputs(rows: list[AgenticBpbEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _agentic_datasets(*, materializer_step: ExecutorStep | None, bundle_uri: str):
    datasets = {}
    for spec in agentic_eval_slices(bundle_uri):
        tags = (
            "agentic_coding",
            f"agentic_coding/{spec.outcome}",
            f"agentic_coding/{spec.source_dataset}",
            f"agentic_coding/{spec.source_dataset}_{spec.outcome}",
        )
        source = materializer_step / spec.output_name if materializer_step is not None else spec.path
        dataset_name = f"agentic_coding/{spec.output_name.removesuffix('.jsonl')}"
        datasets[dataset_name] = raw_text_dataset(source, text_key="text", tags=tags)
    return datasets


def _slice_metric_name(dataset_name: str, leaf: str) -> str:
    return f"eval/{dataset_name}/{leaf}"


def _summary_rows(summary_path: str) -> list[dict[str, Any]]:
    with fsspec.open(summary_path, "rt") as handle:
        summary = json.load(handle)
    rows = summary.get("datasets", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected model-perplexity summary format in {summary_path}")
    return [row for row in rows if isinstance(row, dict)]


def _metrics_from_summary_path(summary_path: str) -> tuple[dict[str, float], str]:
    try:
        rows = _summary_rows(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {}, str(exc)

    metrics: dict[str, float] = {}
    success_bpbs: list[float] = []
    failed_bpbs: list[float] = []
    coderforge_success_bpbs: list[float] = []
    for row in rows:
        dataset_name = _string_value(row.get("name"))
        if not dataset_name.startswith("agentic_coding/"):
            continue
        bpb = row.get("bpb")
        if not isinstance(bpb, int | float):
            continue
        bpb_float = float(bpb)
        metrics[_slice_metric_name(dataset_name, "bpb")] = bpb_float
        for leaf in ("documents", "bytes", "bits"):
            value = row.get(leaf)
            if isinstance(value, int | float):
                metrics[_slice_metric_name(dataset_name, leaf)] = float(value)
        if dataset_name.endswith("_success"):
            success_bpbs.append(bpb_float)
            if dataset_name.startswith("agentic_coding/coderforge_"):
                coderforge_success_bpbs.append(bpb_float)
        if dataset_name.endswith("_fail"):
            failed_bpbs.append(bpb_float)

    if success_bpbs:
        metrics[PRIMARY_METRIC] = sum(success_bpbs) / len(success_bpbs)
    if coderforge_success_bpbs:
        metrics[CODERFORGE_PRIMARY_METRIC] = sum(coderforge_success_bpbs) / len(coderforge_success_bpbs)
    if failed_bpbs:
        metrics["eval/agentic_coding/failed_macro_bpb"] = sum(failed_bpbs) / len(failed_bpbs)
    if success_bpbs and failed_bpbs:
        metrics["eval/agentic_coding/success_minus_failed_bpb"] = (
            metrics[PRIMARY_METRIC] - metrics["eval/agentic_coding/failed_macro_bpb"]
        )
    return metrics, ""


def collect_agentic_bpb_results(config: CollectAgenticBpbResultsConfig) -> None:
    """Collect agentic-coding BPB outputs into one normalized CSV."""
    state_rows = [AgenticBpbEvalSpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        result_path = config.results_by_eval_key.get(row.eval_key)
        if result_path is None:
            record["collection_status"] = "not_launched"
            records.append(record)
            continue
        summary_path = os.path.join(str(result_path).rstrip("/"), SUMMARY_JSON)
        metrics, error = _metrics_from_summary_path(summary_path)
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)

    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[AgenticBpbEvalSpec],
    bundle_uri: str,
    materializer_step: ExecutorStep | None,
    max_docs_per_dataset: int | None,
    max_eval_length: int,
    max_doc_bytes: int,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build agentic-coding model-perplexity eval steps."""
    eval_steps: list[ExecutorStep] = []
    results_by_eval_key: dict[str, InputName] = {}
    datasets = _agentic_datasets(materializer_step=materializer_step, bundle_uri=bundle_uri)
    for row in state_rows:
        if row.launch_decision != "launch":
            continue
        resource_config = ResourceConfig.with_tpu(
            row.launch_tpu_type,
            regions=[row.launch_tpu_region],
            zone=row.launch_tpu_zone,
        )
        eval_step = model_perplexity_scores(
            model=GapFinderModelConfig(
                checkpoint_path=row.hf_checkpoint_latest,
                checkpoint_is_hf=True,
                tokenizer=marin_tokenizer,
            ),
            datasets=datasets,
            resource_config=resource_config,
            per_device_batch_size=4,
            max_eval_length=max_eval_length,
            max_docs_per_dataset=max_docs_per_dataset,
            max_doc_bytes=max_doc_bytes,
            name=f"agentic_coding_bpb/{row.eval_key}",
            wandb_tags=["300m", "agentic-coding-bpb", "signal-noise"],
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[AgenticBpbEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final agentic-coding BPB collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect agentic-coding BPB results for {len(results_by_eval_key)} eval steps",
        fn=collect_agentic_bpb_results,
        config=CollectAgenticBpbResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _bundle_manifest_exists(bundle_uri: str) -> bool:
    try:
        with fsspec.open(f"{bundle_uri.rstrip('/')}/{MANIFEST_FILENAME}", "rt"):
            return True
    except OSError:
        return False


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-docs-per-dataset", type=int, default=MAX_DOCS_PER_DATASET)
    parser.add_argument("--max-eval-length", type=int, default=MAX_EVAL_LENGTH)
    parser.add_argument("--max-doc-bytes", type=int, default=MAX_DOC_BYTES)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-key-suffix", default="")
    parser.add_argument("--state-csv")
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument("--bundle-uri", default=DEFAULT_OUTPUT_URI)
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--include-materializer", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_known_args()


def _apply_requested_subset(
    rows: list[AgenticBpbEvalSpec],
    *,
    include_run_names: list[str],
    canary: bool,
) -> list[AgenticBpbEvalSpec]:
    if include_run_names:
        include = set(include_run_names)
        selected = [row for row in rows if row.run_name in include]
        missing = sorted(include - {row.run_name for row in selected})
        if missing:
            raise ValueError(f"Requested run names not present in agentic BPB state: {missing}")
        return selected
    if not canary:
        return rows
    launch_rows = [row for row in rows if row.launch_decision == "launch"]
    selected_eval_keys = {row.eval_key for row in launch_rows[:2]}
    return [row for row in rows if row.eval_key in selected_eval_keys]


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
            allow_partial=args.allow_partial,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    state_rows = _apply_requested_subset(state_rows, include_run_names=args.include_run_name, canary=args.canary)
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d agentic-coding BPB eval steps over %d candidate checkpoints", launch_count, len(state_rows))
    if args.dry_run or os.getenv("CI") is not None:
        return

    max_docs_per_dataset = CANARY_DOCS_PER_DATASET if args.canary else args.max_docs_per_dataset
    materializer_step = (
        agentic_coding_eval_bundle_step(args.bundle_uri)
        if args.include_materializer and not args.skip_materialize
        else None
    )
    if materializer_step is None and not _bundle_manifest_exists(args.bundle_uri):
        raise ValueError(
            f"Missing agentic eval bundle manifest under {args.bundle_uri}; "
            "materialize the bundle first or pass --include-materializer with a same-region bundle URI"
        )
    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        bundle_uri=args.bundle_uri,
        materializer_step=materializer_step,
        max_docs_per_dataset=max_docs_per_dataset,
        max_eval_length=args.max_eval_length,
        max_doc_bytes=args.max_doc_bytes,
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        results_by_eval_key=results_by_eval_key,
    )
    steps = [*([materializer_step] if materializer_step is not None else []), *eval_steps, collect_step]
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{args.name_prefix}: 300M agentic-coding assistant-action BPB evals",
    )


if __name__ == "__main__":
    main()
