# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch raw-PPL evals for the 300M signal and run_00097 noise rows."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from marin.evaluation.perplexity_gap import GapFinderModelConfig, RawTextEvaluationDataset, model_perplexity_scores
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.utils import fsspec_glob

from experiments.bio_chem_notation import bio_chem_raw_validation_sets
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    EXTRA_CANDIDATES_ENV,
    PANEL_FILTER_ENV,
    _bool_value,
    _exact_hf_checkpoint,
    _executor_prefix,
    _slug,
    _string_value,
)
from experiments.evals.asr_ocr_noisy_ppl import noisy_asr_ocr_raw_validation_sets
from experiments.evals.exp5053_lm_eval_bridge import lm_eval_bridge_raw_validation_sets
from experiments.evals.exp5057_binary_network_security_evals import binary_network_security_raw_validation_sets
from experiments.evals.exp5061_package_metadata_evals import package_metadata_raw_validation_sets
from experiments.evals.exp5062_game_music_evals import game_music_raw_validation_sets
from experiments.evals.fineweb2_multilingual import (
    FINEWEB2_MULTILINGUAL_EVAL_CONFIGS,
    fineweb2_multilingual_raw_validation_sets,
)
from experiments.evals.formal_methods_ppl import formal_methods_hardware_rtl_raw_validation_sets
from experiments.evals.gh_archive_structured_output import gh_archive_structured_output_raw_validation_sets
from experiments.evals.long_tail_ppl_runnable import runnable_long_tail_raw_validation_sets
from experiments.evals.raw_web_markup_ppl import raw_web_markup_raw_validation_sets
from experiments.marin_models import marin_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_raw_ppl_completion"
STATE_CSV = OUTPUT_DIR / "300m_raw_ppl_eval_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_raw_ppl_eval_launch_manifest.csv"
RESULTS_CSV_LOCAL = OUTPUT_DIR / "300m_raw_ppl_eval_results.csv"
RAW_MATRIX_DIR = METRIC_REGISTRY_DIR / "raw_metric_matrix_300m"
SIGNAL_MATRIX_CSV = RAW_MATRIX_DIR / "raw_metric_matrix_300m.csv"
FIXED_NOISE_MATRIX_CSV = RAW_MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv"
VARIABLE_NOISE_MATRIX_CSV = RAW_MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_raw_ppl_evals_20260515"
RESULTS_CSV = "300m_raw_ppl_eval_results.csv"
STATE_OUTPUT_CSV = "300m_raw_ppl_eval_state.csv"
SUMMARY_JSON = "summary.json"
RAW_PPL_SUMMARY_RE = re.compile(r"/(?P<eval_key>rawppl300m_.+)-[0-9a-f]{6}/summary\.json$")
MAX_EVAL_LENGTH = 4096
MAX_DOC_BYTES = 32_768
MAX_DOCS_PER_DATASET = 512
CANARY_DOCS_PER_DATASET = 32
EXPECTED_CANDIDATE_COUNT = 262
ALLOWED_PANELS = {
    "signal_300m_6b",
    "fixed_seed_noise_300m_6b",
    "variable_subset_noise_300m_6b",
}
SIGNAL_SOURCE_EXPERIMENTS = {
    "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b",
    "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
}

PRIORITY_BUNDLE = "priority"
FINEWEB2_REPRESENTATIVE_BUNDLE = "fineweb2-representative"
FINEWEB2_FULL_BUNDLE = "fineweb2-full"
DEFAULT_BUNDLES = (PRIORITY_BUNDLE, FINEWEB2_REPRESENTATIVE_BUNDLE)
BUNDLE_CHOICES = (PRIORITY_BUNDLE, FINEWEB2_REPRESENTATIVE_BUNDLE, FINEWEB2_FULL_BUNDLE)
FINEWEB2_REPRESENTATIVE_CONFIGS = (
    "deu_Latn",
    "fra_Latn",
    "rus_Cyrl",
    "arb_Arab",
    "hin_Deva",
    "ben_Beng",
    "cmn_Hani",
    "jpn_Jpan",
    "kor_Hang",
    "vie_Latn",
    "tha_Thai",
    "tam_Taml",
    "heb_Hebr",
    "ukr_Cyrl",
    "ind_Latn",
)
DEFERRED_PRIORITY_DATASETS = {
    # The current archive materializer fetches the full TPTP tarball into
    # memory before applying the output byte cap, which OOMed the canary.
    "formal_methods/tptp",
    # NCBI no longer exposes the viral GFF path used by the shared RefSeq
    # materializer. Because FASTA and GFF are produced by one step, defer both
    # RefSeq slices until the source surface is redesigned.
    "bio_chem/refseq/refseq_viral_fasta",
    "bio_chem/refseq/refseq_viral_gff",
    # These current raw surfaces can legitimately produce no scored documents
    # for the bounded sample window, so requiring them blocks failure-only retry
    # skipping even when all emitted metrics are present.
    "gh_archive_structured_output/WorkflowRunEvent",
    "hardware_rtl/rtl_coder",
    "hardware_rtl/rtl_repo",
}

BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "eligible", "has_existing_results", "uses_east5_checkpoint"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest_step",
    "dataset_count",
}


@dataclass(frozen=True)
class RawPplCandidate:
    """One trained checkpoint candidate for raw-PPL scoring."""

    panel: str
    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int


@dataclass(frozen=True)
class RawPplEvalSpec:
    """One raw-PPL eval state row and potential launch unit."""

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
    bundle_key: str
    dataset_count: int
    dataset_names: str
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    eligible: bool
    launch_decision: str
    step_name: str
    result_path: str


@dataclass(frozen=True)
class CollectRawPplResultsConfig:
    """Config for collecting raw-PPL score outputs."""

    output_path: str
    state_rows_json: str
    results_by_eval_key: dict[str, InputName]


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _candidate_from_row(row: pd.Series, *, panel: str) -> RawPplCandidate:
    root = _string_value(row.get("checkpoint_root")).rstrip("/")
    if not root:
        raise ValueError(f"{panel} row is missing checkpoint_root:\n{row.to_string()}")
    expected_checkpoint_step = pd.to_numeric(row.get("expected_checkpoint_step", 22_887), errors="coerce")
    if pd.isna(expected_checkpoint_step):
        expected_checkpoint_step = pd.to_numeric(row.get("target_final_checkpoint_step", 22_887), errors="coerce")
    if pd.isna(expected_checkpoint_step):
        raise ValueError(f"{panel} row is missing expected checkpoint step:\n{row.to_string()}")
    return RawPplCandidate(
        panel=panel,
        run_name=_string_value(row.get("run_name")),
        registry_key=_string_value(row.get("registry_run_key", row.get("registry_key"))),
        source_experiment=_string_value(row.get("source_experiment")),
        cohort=_string_value(row.get("cohort")),
        checkpoint_root=root,
        expected_checkpoint_step=int(expected_checkpoint_step),
    )


def _matrix_candidates(path: Path, *, panel: str, expected_rows: int) -> list[RawPplCandidate]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw-PPL candidate matrix: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(frame)}")
    candidates = [_candidate_from_row(row, panel=panel) for _, row in frame.iterrows()]
    roots = [candidate.checkpoint_root for candidate in candidates]
    duplicates = sorted(root for root in set(roots) if roots.count(root) > 1)
    if duplicates:
        raise ValueError(f"Duplicate checkpoint roots in {path}: {duplicates[:10]}")
    return candidates


def _raw_ppl_candidate_records(requested_panels: set[str]) -> list[RawPplCandidate]:
    candidates: list[RawPplCandidate] = []
    if "signal_300m_6b" in requested_panels:
        candidates.extend(_matrix_candidates(SIGNAL_MATRIX_CSV, panel="signal_300m_6b", expected_rows=242))
    if "fixed_seed_noise_300m_6b" in requested_panels:
        candidates.extend(_matrix_candidates(FIXED_NOISE_MATRIX_CSV, panel="fixed_seed_noise_300m_6b", expected_rows=10))
    if "variable_subset_noise_300m_6b" in requested_panels:
        candidates.extend(
            _matrix_candidates(VARIABLE_NOISE_MATRIX_CSV, panel="variable_subset_noise_300m_6b", expected_rows=10)
        )
    candidates.extend(candidate for candidate in _extra_candidate_records() if candidate.panel in requested_panels)
    roots = [candidate.checkpoint_root for candidate in candidates]
    duplicates = sorted(root for root in set(roots) if roots.count(root) > 1)
    if duplicates:
        raise ValueError(f"Duplicate checkpoint roots across raw-PPL candidate panels: {duplicates[:10]}")
    return sorted(candidates, key=lambda row: (row.panel, row.run_name))


def _extra_candidate_records() -> list[RawPplCandidate]:
    paths = [path.strip() for path in os.environ.get(EXTRA_CANDIDATES_ENV, "").split(",") if path.strip()]
    candidates: list[RawPplCandidate] = []
    for path in paths:
        frame = _read_csv(path)
        if "panel" not in frame.columns:
            raise ValueError(f"Extra raw-PPL candidate CSV {path} is missing panel")
        candidates.extend(_candidate_from_row(row, panel=_string_value(row.get("panel"))) for _, row in frame.iterrows())
    return candidates


def _requested_panels() -> set[str]:
    raw = os.environ.get(PANEL_FILTER_ENV, "").strip()
    if not raw:
        return set(ALLOWED_PANELS)
    return {part.strip() for part in raw.split(",") if part.strip()}


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


def _require_fineweb2_configs(configs: tuple[str, ...]) -> None:
    available = set(FINEWEB2_MULTILINGUAL_EVAL_CONFIGS)
    missing = sorted(set(configs) - available)
    if missing:
        raise ValueError(f"FineWeb2 representative configs not registered: {missing}")


def _priority_raw_ppl_datasets() -> dict[str, RawTextEvaluationDataset]:
    datasets: dict[str, RawTextEvaluationDataset] = {}
    factories = (
        lm_eval_bridge_raw_validation_sets,
        raw_web_markup_raw_validation_sets,
        bio_chem_raw_validation_sets,
        formal_methods_hardware_rtl_raw_validation_sets,
        gh_archive_structured_output_raw_validation_sets,
        binary_network_security_raw_validation_sets,
        package_metadata_raw_validation_sets,
        game_music_raw_validation_sets,
        noisy_asr_ocr_raw_validation_sets,
        runnable_long_tail_raw_validation_sets,
    )
    for factory in factories:
        factory_datasets = factory()
        overlap = set(datasets).intersection(factory_datasets)
        if overlap:
            raise ValueError(f"Duplicate raw-PPL dataset keys from {factory.__name__}: {sorted(overlap)}")
        datasets.update(factory_datasets)
    return datasets


def build_raw_ppl_datasets(bundle_keys: tuple[str, ...]) -> dict[str, RawTextEvaluationDataset]:
    """Return deterministic raw-PPL datasets for the selected bundle keys."""
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for bundle_key in bundle_keys:
        if bundle_key == PRIORITY_BUNDLE:
            bundle_datasets = _priority_raw_ppl_datasets()
        elif bundle_key == FINEWEB2_REPRESENTATIVE_BUNDLE:
            _require_fineweb2_configs(FINEWEB2_REPRESENTATIVE_CONFIGS)
            bundle_datasets = fineweb2_multilingual_raw_validation_sets(configs=FINEWEB2_REPRESENTATIVE_CONFIGS)
        elif bundle_key == FINEWEB2_FULL_BUNDLE:
            bundle_datasets = fineweb2_multilingual_raw_validation_sets()
        else:
            raise ValueError(f"Unknown raw-PPL bundle: {bundle_key}")
        overlap = set(datasets).intersection(bundle_datasets)
        if overlap:
            raise ValueError(f"Duplicate raw-PPL dataset keys across selected bundles: {sorted(overlap)}")
        datasets.update(bundle_datasets)
    for dataset_name in DEFERRED_PRIORITY_DATASETS:
        datasets.pop(dataset_name, None)
    return dict(sorted(datasets.items()))


def _bundle_key(bundle_keys: tuple[str, ...]) -> str:
    return "+".join(bundle_keys)


def _existing_result_roots(path: Path, dataset_names: tuple[str, ...]) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "checkpoint_root" not in frame.columns:
        return set()
    required_columns = [f"raw_ppl/{dataset_name}/bpb" for dataset_name in dataset_names]
    if not all(column in frame.columns for column in required_columns):
        return set()
    complete = frame[required_columns].notna().all(axis=1)
    return {
        _string_value(row.get("checkpoint_root")).rstrip("/")
        for _, row in frame.loc[complete].iterrows()
        if _string_value(row.get("checkpoint_root")).rstrip("/")
    }


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
    bundle_keys: tuple[str, ...],
    dataset_names: tuple[str, ...],
    allow_partial: bool,
) -> list[RawPplEvalSpec]:
    """Build raw-PPL state rows for the 300M signal plus run_00097 noise panels."""
    existing_roots = _existing_result_roots(RESULTS_CSV_LOCAL, dataset_names)
    requested_panels = _requested_panels()
    candidates = [
        candidate
        for candidate in _raw_ppl_candidate_records(requested_panels)
        if candidate.panel in requested_panels
        and (candidate.panel != "signal_300m_6b" or candidate.source_experiment in SIGNAL_SOURCE_EXPERIMENTS)
    ]
    canonical_panel_request = requested_panels == ALLOWED_PANELS
    if len(candidates) != EXPECTED_CANDIDATE_COUNT and canonical_panel_request and not allow_partial:
        counts = pd.Series([candidate.panel for candidate in candidates]).value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} 300M signal/noise candidates, found {len(candidates)}; "
            f"panel counts={counts}. Pass --allow-partial only for debugging."
        )

    rows: list[RawPplEvalSpec] = []
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
        eval_key = f"rawppl300m_{idx:03d}_{_slug(candidate.panel)}_{_slug(candidate.run_name)}{suffix}"
        rows.append(
            RawPplEvalSpec(
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
                bundle_key=_bundle_key(bundle_keys),
                dataset_count=len(dataset_names),
                dataset_names=";".join(dataset_names),
                launch_tpu_type=default_tpu_type,
                launch_tpu_region=default_tpu_region,
                launch_tpu_zone=default_tpu_zone,
                eligible=eligible,
                launch_decision=launch_decision,
                step_name=f"analysis/model_perplexity_scores/raw_ppl/{eval_key}",
                result_path="",
            )
        )
    return rows


def _load_state_rows(path: str | Path) -> list[RawPplEvalSpec]:
    frame = _read_csv(path)
    expected_fields = {field.name for field in fields(RawPplEvalSpec)}
    missing = expected_fields - set(frame.columns)
    if missing:
        raise ValueError(f"State CSV {path} is missing columns: {sorted(missing)}")
    rows: list[RawPplEvalSpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(RawPplEvalSpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(RawPplEvalSpec(**kwargs))
    return rows


def _write_local_outputs(rows: list[RawPplEvalSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def _summary_rows(summary_path: str) -> list[dict[str, Any]]:
    with fsspec.open(summary_path, "rt") as handle:
        summary = json.load(handle)
    rows = summary.get("datasets", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected model-perplexity summary format in {summary_path}")
    return [row for row in rows if isinstance(row, dict)]


def _metric_name(dataset_name: str, leaf: str) -> str:
    return f"raw_ppl/{dataset_name}/{leaf}"


def _metrics_from_summary_path(
    summary_path: str,
    *,
    allowed_dataset_names: set[str] | None = None,
) -> tuple[dict[str, float], str]:
    try:
        rows = _summary_rows(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {}, str(exc)

    metrics: dict[str, float] = {}
    for row in rows:
        dataset_name = _string_value(row.get("name"))
        if not dataset_name:
            continue
        if allowed_dataset_names is not None and dataset_name not in allowed_dataset_names:
            continue
        for leaf in ("bpb", "documents", "bytes", "bits"):
            value = row.get(leaf)
            if isinstance(value, int | float):
                metrics[_metric_name(dataset_name, leaf)] = float(value)
        bits = row.get("bits")
        if isinstance(bits, int | float):
            metrics[_metric_name(dataset_name, "loss")] = float(bits) * math.log(2.0)
    return metrics, ""


def collect_raw_ppl_results(config: CollectRawPplResultsConfig) -> None:
    """Collect raw-PPL score outputs into one normalized CSV."""
    state_rows = [RawPplEvalSpec(**row) for row in json.loads(config.state_rows_json)]
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
        metrics, error = _metrics_from_summary_path(
            summary_path,
            allowed_dataset_names=set(row.dataset_names.split(";")),
        )
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = str(result_path)
        records.append(record)

    with fsspec.open(os.path.join(output_path, RESULTS_CSV), "wt") as handle:
        pd.DataFrame.from_records(records).to_csv(handle, index=False)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def collect_raw_ppl_results_from_prefixes(
    *,
    prefixes: list[str],
    state_rows: list[RawPplEvalSpec],
    output_csv: Path,
) -> None:
    """Collect raw-PPL score outputs from already-written executor prefixes."""
    summary_paths_by_eval_key: dict[str, str] = {}
    for prefix in prefixes:
        pattern = os.path.join(prefix.rstrip("/"), "analysis/model_perplexity_scores/raw_ppl", "*", SUMMARY_JSON)
        for summary_path in sorted(fsspec_glob(pattern)):
            match = RAW_PPL_SUMMARY_RE.search(summary_path)
            if match is None:
                continue
            summary_paths_by_eval_key[match.group("eval_key")] = summary_path

    records: list[dict[str, Any]] = []
    for row in state_rows:
        record = asdict(row)
        summary_path = summary_paths_by_eval_key.get(row.eval_key)
        if summary_path is None:
            record["collection_status"] = "missing_summary"
            record["collection_error"] = ""
            records.append(record)
            continue
        metrics, error = _metrics_from_summary_path(
            summary_path,
            allowed_dataset_names=set(row.dataset_names.split(";")),
        )
        record.update(metrics)
        record["collection_status"] = "collected" if metrics else "missing_metrics"
        record["collection_error"] = error
        record["result_path"] = os.path.dirname(summary_path)
        records.append(record)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_csv, index=False)
    logger.info(
        "Collected %d/%d raw-PPL rows from %d prefix(es) into %s",
        sum(record.get("collection_status") == "collected" for record in records),
        len(records),
        len(prefixes),
        output_csv,
    )


def build_eval_steps(
    *,
    name_prefix: str,
    state_rows: list[RawPplEvalSpec],
    datasets: dict[str, RawTextEvaluationDataset],
    max_docs_per_dataset: int | None,
    max_eval_length: int,
    max_doc_bytes: int,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build raw-PPL model-perplexity eval steps."""
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
            name=f"raw_ppl/{row.eval_key}",
            wandb_tags=["300m", "raw-ppl", "signal-noise", f"raw_ppl_bundle={row.bundle_key}"],
        )
        eval_steps.append(eval_step)
        results_by_eval_key[row.eval_key] = output_path_of(eval_step)
    return eval_steps, results_by_eval_key


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[RawPplEvalSpec],
    results_by_eval_key: dict[str, InputName],
) -> ExecutorStep:
    """Build the final raw-PPL collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_results",
        description=f"Collect raw-PPL results for {len(results_by_eval_key)} eval steps",
        fn=collect_raw_ppl_results,
        config=CollectRawPplResultsConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            results_by_eval_key=results_by_eval_key,
        ),
    )


def _parse_bundle_keys(bundle_args: list[str]) -> tuple[str, ...]:
    bundle_keys = tuple(bundle_args) if bundle_args else DEFAULT_BUNDLES
    if FINEWEB2_REPRESENTATIVE_BUNDLE in bundle_keys and FINEWEB2_FULL_BUNDLE in bundle_keys:
        raise ValueError("Choose either fineweb2-representative or fineweb2-full, not both.")
    return bundle_keys


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
    parser.add_argument("--bundle", action="append", choices=BUNDLE_CHOICES, default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--collect-from-prefix", action="append", default=[])
    parser.add_argument("--collect-output-csv", default=str(RESULTS_CSV_LOCAL))
    return parser.parse_known_args()


def _apply_requested_subset(
    rows: list[RawPplEvalSpec],
    *,
    include_run_names: list[str],
    canary: bool,
) -> list[RawPplEvalSpec]:
    if include_run_names:
        include = set(include_run_names)
        selected = [row for row in rows if row.run_name in include]
        missing = sorted(include - {row.run_name for row in selected})
        if missing:
            raise ValueError(f"Requested run names not present in raw-PPL state: {missing}")
        return selected
    if not canary:
        return rows

    selected_eval_keys: set[str] = set()
    launch_rows = [row for row in rows if row.launch_decision == "launch"]
    for panel in ("signal_300m_6b", "fixed_seed_noise_300m_6b", "variable_subset_noise_300m_6b"):
        for row in launch_rows:
            if row.panel == panel:
                selected_eval_keys.add(row.eval_key)
                break
    if len(selected_eval_keys) != 3:
        raise ValueError(f"Raw-PPL canary expected one launch row from each panel, selected {len(selected_eval_keys)}")
    return [row for row in rows if row.eval_key in selected_eval_keys]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    bundle_keys = _parse_bundle_keys(args.bundle)
    datasets = build_raw_ppl_datasets(bundle_keys)
    dataset_names = tuple(datasets)
    if args.state_csv is None:
        state_rows = build_state_rows(
            default_tpu_type=args.tpu_type,
            default_tpu_region=args.tpu_region,
            default_tpu_zone=args.tpu_zone,
            eval_key_suffix=args.eval_key_suffix,
            bundle_keys=bundle_keys,
            dataset_names=dataset_names,
            allow_partial=args.allow_partial,
        )
    else:
        state_rows = _load_state_rows(args.state_csv)
    state_rows = _apply_requested_subset(state_rows, include_run_names=args.include_run_name, canary=args.canary)
    if args.collect_from_prefix:
        collect_raw_ppl_results_from_prefixes(
            prefixes=args.collect_from_prefix,
            state_rows=state_rows,
            output_csv=Path(args.collect_output_csv),
        )
        return
    _write_local_outputs(state_rows)
    launch_count = sum(row.launch_decision == "launch" for row in state_rows)
    logger.info("Selected raw-PPL bundles: %s", _bundle_key(bundle_keys))
    logger.info("Selected %d raw-PPL datasets", len(dataset_names))
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Prepared %d raw-PPL eval steps over %d candidate checkpoints", launch_count, len(state_rows))
    if args.dry_run or os.getenv("CI") is not None:
        return

    max_docs_per_dataset = CANARY_DOCS_PER_DATASET if args.canary else args.max_docs_per_dataset
    eval_steps, results_by_eval_key = build_eval_steps(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        datasets=datasets,
        max_docs_per_dataset=max_docs_per_dataset,
        max_eval_length=args.max_eval_length,
        max_doc_bytes=args.max_doc_bytes,
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
        description=f"{args.name_prefix}: 300M raw-PPL evals",
    )


if __name__ == "__main__":
    main()
