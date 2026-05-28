# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Launch a proportional-checkpoint canary for MDE-style checkpoint features.

This canary scores the 300M/6B ``baseline_proportional`` checkpoint on bounded
smooth surfaces and stores aligned feature artifacts:

* raw-text per-byte losses in ``scored_documents.parquet``;
* teacher-forced GSM8K/HumanEval request loglikelihoods;
* MCQ smooth-proxy per-choice loglikelihoods.

The resulting artifacts are checkpoint-feature/log-mixture surrogate features,
not true MDE vertex-expert features.
"""

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
from marin.evaluation.perplexity_gap import GapFinderModelConfig, RawTextEvaluationDataset, model_perplexity_scores
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.execution.remote import remote

from experiments.domain_phase_mix.agentic_coding_eval_dataset import DEFAULT_OUTPUT_URI as AGENTIC_CODING_BUNDLE_URI
from experiments.domain_phase_mix.launch_300m_agentic_coding_bpb_evals import _agentic_datasets
from experiments.domain_phase_mix.launch_300m_generative_smooth_proxy_evals import (
    DEFAULT_REQUEST_CACHE_URI as DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI,
)
from experiments.domain_phase_mix.launch_300m_generative_smooth_proxy_evals import (
    REQUEST_FEATURES_PARQUET as TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
)
from experiments.domain_phase_mix.launch_300m_generative_smooth_proxy_evals import (
    SmoothProxyScoreConfig,
    score_teacher_forced_smooth_proxies,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    _bool_value,
    _exact_hf_checkpoint,
    _executor_prefix,
    _slug,
    _string_value,
)
from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import (
    DEFAULT_REQUEST_CACHE_URI as DEFAULT_MCQ_REQUEST_CACHE_URI,
)
from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import (
    REQUEST_FEATURES_PARQUET as MCQ_REQUEST_FEATURES_PARQUET,
)
from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import (
    McqSmoothProxyScoreConfig,
    score_mcq_smooth_proxies,
)
from experiments.domain_phase_mix.launch_300m_raw_ppl_evals import PRIORITY_BUNDLE, build_raw_ppl_datasets
from experiments.evals.exp1600_uncheatable_evals import uncheatable_eval_raw_validation_sets
from experiments.marin_models import marin_tokenizer
from experiments.paloma import paloma_raw_validation_sets

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry"
OUTPUT_DIR = METRIC_REGISTRY_DIR / "300m_checkpoint_features_canary"
STATE_CSV = OUTPUT_DIR / "300m_checkpoint_features_canary_state.csv"
LAUNCH_MANIFEST_CSV = OUTPUT_DIR / "300m_checkpoint_features_canary_launch_manifest.csv"
SIGNAL_MATRIX_CSV = METRIC_REGISTRY_DIR / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"

DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_300m_checkpoint_features_canary_20260528"
DEFAULT_RUN_NAME = "baseline_proportional"
DEFAULT_EXPECTED_STEP = 22_887
# The canary scope is intentionally fixed to this exact proportional checkpoint.
# For other checkpoints, pass --matrix-csv or explicit checkpoint metadata and bump the canary version.
DEFAULT_REGISTRY_KEY = "300m_6b:signal:pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b:baseline_proportional"
DEFAULT_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
DEFAULT_COHORT = "signal"
DEFAULT_CHECKPOINT_ROOT = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696"
)
SUMMARY_JSON = "checkpoint_feature_canary_summary.json"
STATE_OUTPUT_CSV = "300m_checkpoint_features_canary_state.csv"
TEXT_FEATURE_SURFACE = "raw_text_loss_features"
TEACHER_FORCED_SURFACE = "teacher_forced_request_features"
MCQ_SURFACE = "mcq_request_features"
TEXT_BUNDLE_CHOICES = ("paloma", "uncheatable", "raw_ppl_priority", "agentic_coding")
DEFAULT_TEXT_BUNDLES = ("paloma", "uncheatable", "raw_ppl_priority", "agentic_coding")
BOOL_STATE_FIELDS = {"has_exact_hf_checkpoint", "uses_east5_checkpoint", "eligible"}
INT_STATE_FIELDS = {
    "expected_checkpoint_step",
    "hf_checkpoint_latest_step",
    "text_dataset_count",
    "max_docs_per_dataset",
    "max_eval_instances",
}


@dataclass(frozen=True)
class CheckpointFeatureCanarySpec:
    """One proportional checkpoint-feature canary state row."""

    run_name: str
    registry_key: str
    source_experiment: str
    cohort: str
    checkpoint_root: str
    expected_checkpoint_step: int
    hf_checkpoint_latest: str
    hf_checkpoint_latest_step: int
    has_exact_hf_checkpoint: bool
    uses_east5_checkpoint: bool
    launch_tpu_type: str
    launch_tpu_region: str
    launch_tpu_zone: str
    text_bundle_key: str
    text_dataset_count: int
    text_dataset_names: str
    max_docs_per_dataset: int
    max_eval_instances: int
    eligible: bool
    launch_decision: str
    step_name: str


@dataclass(frozen=True)
class CollectCheckpointFeatureCanaryConfig:
    """Config for collecting checkpoint-feature canary output paths."""

    output_path: str
    state_rows_json: str
    surface_output_paths: dict[str, InputName]


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _baseline_proportional_row(matrix_csv: str | Path, run_name: str) -> pd.Series:
    frame = _read_csv(matrix_csv)
    matches = frame[frame["run_name"].eq(run_name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {run_name!r} row in {matrix_csv}, found {len(matches)}.")
    return matches.iloc[0]


def _expected_checkpoint_step(row: pd.Series) -> int:
    for column in ("expected_checkpoint_step", "target_final_checkpoint_step"):
        if column not in row:
            continue
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return int(value)
    return DEFAULT_EXPECTED_STEP


def _uses_east5_checkpoint(checkpoint_root: str) -> bool:
    return checkpoint_root.startswith("gs://marin-us-east5/")


def _require_nonempty_request_cache(uri: str) -> None:
    try:
        with fsspec.open(uri, "rb") as handle:
            first_byte = handle.read(1)
    except OSError as exc:
        raise FileNotFoundError(f"Checkpoint-feature canary request cache is missing: {uri}") from exc
    if not first_byte:
        raise ValueError(f"Checkpoint-feature canary request cache is empty: {uri}")


def _launch_decision(*, checkpoint_root: str, has_exact_hf_checkpoint: bool, uses_east5_checkpoint: bool) -> str:
    if not checkpoint_root:
        return "defer_missing_checkpoint"
    if not uses_east5_checkpoint:
        return "defer_checkpoint_not_east5"
    if not has_exact_hf_checkpoint:
        return "defer_missing_exact_hf_checkpoint"
    return "launch"


def build_text_feature_datasets(bundle_keys: tuple[str, ...]) -> dict[str, RawTextEvaluationDataset]:
    """Build deterministic raw-text surfaces for bounded checkpoint-feature scoring."""
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for bundle_key in bundle_keys:
        if bundle_key == "paloma":
            bundle_datasets = paloma_raw_validation_sets()
        elif bundle_key == "uncheatable":
            bundle_datasets = uncheatable_eval_raw_validation_sets()
        elif bundle_key == "raw_ppl_priority":
            bundle_datasets = build_raw_ppl_datasets((PRIORITY_BUNDLE,))
        elif bundle_key == "agentic_coding":
            bundle_datasets = _agentic_datasets(materializer_step=None, bundle_uri=AGENTIC_CODING_BUNDLE_URI)
        else:
            raise ValueError(f"Unknown text feature bundle: {bundle_key}")
        overlap = sorted(set(datasets).intersection(bundle_datasets))
        if overlap:
            raise ValueError(f"Duplicate checkpoint-feature dataset names from {bundle_key}: {overlap}")
        datasets.update(bundle_datasets)
    return dict(sorted(datasets.items()))


def _build_state_row_from_values(
    *,
    run_name: str,
    registry_key: str,
    source_experiment: str,
    cohort: str,
    checkpoint_root: str,
    expected_checkpoint_step: int,
    text_bundle_keys: tuple[str, ...],
    text_dataset_names: tuple[str, ...],
    max_docs_per_dataset: int,
    max_eval_instances: int,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
) -> CheckpointFeatureCanarySpec:
    """Build a self-contained checkpoint-feature canary row."""
    checkpoint_root = checkpoint_root.rstrip("/")
    hf_checkpoint = _exact_hf_checkpoint(checkpoint_root, expected_checkpoint_step)
    has_exact_hf_checkpoint = bool(hf_checkpoint)
    uses_east5_checkpoint = _uses_east5_checkpoint(checkpoint_root)
    launch_decision = _launch_decision(
        checkpoint_root=checkpoint_root,
        has_exact_hf_checkpoint=has_exact_hf_checkpoint,
        uses_east5_checkpoint=uses_east5_checkpoint,
    )
    return CheckpointFeatureCanarySpec(
        run_name=run_name,
        registry_key=registry_key,
        source_experiment=source_experiment,
        cohort=cohort,
        checkpoint_root=checkpoint_root,
        expected_checkpoint_step=expected_checkpoint_step,
        hf_checkpoint_latest=hf_checkpoint,
        hf_checkpoint_latest_step=expected_checkpoint_step if has_exact_hf_checkpoint else -1,
        has_exact_hf_checkpoint=has_exact_hf_checkpoint,
        uses_east5_checkpoint=uses_east5_checkpoint,
        launch_tpu_type=default_tpu_type,
        launch_tpu_region=default_tpu_region,
        launch_tpu_zone=default_tpu_zone,
        text_bundle_key="+".join(text_bundle_keys),
        text_dataset_count=len(text_dataset_names),
        text_dataset_names=";".join(text_dataset_names),
        max_docs_per_dataset=max_docs_per_dataset,
        max_eval_instances=max_eval_instances,
        eligible=launch_decision == "launch",
        launch_decision=launch_decision,
        step_name=f"checkpoint_feature_canary/{_slug(run_name)}",
    )


def build_state_row(
    *,
    matrix_csv: str | Path,
    run_name: str,
    text_bundle_keys: tuple[str, ...],
    text_dataset_names: tuple[str, ...],
    max_docs_per_dataset: int,
    max_eval_instances: int,
    default_tpu_type: str,
    default_tpu_region: str,
    default_tpu_zone: str,
) -> CheckpointFeatureCanarySpec:
    """Build the canary row from a local or GCS matrix."""
    row = _baseline_proportional_row(matrix_csv, run_name)
    return _build_state_row_from_values(
        run_name=_string_value(row.get("run_name")),
        registry_key=_string_value(row.get("registry_run_key", row.get("registry_key"))),
        source_experiment=_string_value(row.get("source_experiment")),
        cohort=_string_value(row.get("cohort")),
        checkpoint_root=_string_value(row.get("checkpoint_root")),
        expected_checkpoint_step=_expected_checkpoint_step(row),
        text_bundle_keys=text_bundle_keys,
        text_dataset_names=text_dataset_names,
        max_docs_per_dataset=max_docs_per_dataset,
        max_eval_instances=max_eval_instances,
        default_tpu_type=default_tpu_type,
        default_tpu_region=default_tpu_region,
        default_tpu_zone=default_tpu_zone,
    )


def _load_state_rows(path: str | Path) -> list[CheckpointFeatureCanarySpec]:
    frame = _read_csv(path)
    missing = {field.name for field in fields(CheckpointFeatureCanarySpec)} - set(frame.columns)
    if missing:
        raise ValueError(f"State CSV {path} is missing columns: {sorted(missing)}")
    rows: list[CheckpointFeatureCanarySpec] = []
    for _, row in frame.iterrows():
        kwargs: dict[str, Any] = {}
        for field in fields(CheckpointFeatureCanarySpec):
            value = row[field.name]
            if field.name in BOOL_STATE_FIELDS:
                kwargs[field.name] = _bool_value(value)
            elif field.name in INT_STATE_FIELDS:
                kwargs[field.name] = int(value)
            else:
                kwargs[field.name] = _string_value(value)
        rows.append(CheckpointFeatureCanarySpec(**kwargs))
    return rows


def _write_local_outputs(rows: list[CheckpointFeatureCanarySpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records([asdict(row) for row in rows])
    frame.to_csv(STATE_CSV, index=False)
    frame[frame["launch_decision"].eq("launch")].to_csv(LAUNCH_MANIFEST_CSV, index=False)


def build_feature_steps(
    *,
    name_prefix: str,
    spec: CheckpointFeatureCanarySpec,
    datasets: dict[str, RawTextEvaluationDataset],
    teacher_forced_request_cache_uri: str,
    mcq_request_cache_uri: str,
) -> tuple[list[ExecutorStep], dict[str, InputName]]:
    """Build the three bounded checkpoint-feature canary steps."""
    if spec.launch_decision != "launch":
        return [], {}

    resource_config = ResourceConfig.with_tpu(
        spec.launch_tpu_type,
        regions=[spec.launch_tpu_region],
        zone=spec.launch_tpu_zone,
    )
    text_step = model_perplexity_scores(
        model=GapFinderModelConfig(
            checkpoint_path=spec.hf_checkpoint_latest,
            checkpoint_is_hf=True,
            tokenizer=marin_tokenizer,
        ),
        datasets=datasets,
        resource_config=resource_config,
        per_device_batch_size=4,
        max_eval_length=4096,
        max_docs_per_dataset=spec.max_docs_per_dataset,
        max_doc_bytes=32_768,
        name=f"{name_prefix}/{TEXT_FEATURE_SURFACE}/{_slug(spec.run_name)}",
        wandb_tags=["300m", "checkpoint-features", "mde-canary", "raw-text"],
    )
    teacher_step = ExecutorStep(
        name=f"{name_prefix}/{TEACHER_FORCED_SURFACE}/{_slug(spec.run_name)}",
        description=f"Score teacher-forced request features for {spec.run_name}",
        fn=remote(
            score_teacher_forced_smooth_proxies,
            resources=resource_config,
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=SmoothProxyScoreConfig(
            eval_key=f"featurecanary_teacher_{_slug(spec.run_name)}",
            checkpoint_root=spec.hf_checkpoint_latest,
            output_path=this_output_path(),
            request_cache_uri=teacher_forced_request_cache_uri,
            max_eval_instances=spec.max_eval_instances,
        ),
    )
    mcq_step = ExecutorStep(
        name=f"{name_prefix}/{MCQ_SURFACE}/{_slug(spec.run_name)}",
        description=f"Score MCQ request features for {spec.run_name}",
        fn=remote(
            score_mcq_smooth_proxies,
            resources=resource_config,
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=McqSmoothProxyScoreConfig(
            eval_key=f"featurecanary_mcq_{_slug(spec.run_name)}",
            checkpoint_root=spec.hf_checkpoint_latest,
            output_path=this_output_path(),
            request_cache_uri=mcq_request_cache_uri,
            request_cache_dependency=mcq_request_cache_uri,
            max_eval_instances=spec.max_eval_instances,
        ),
    )
    return [text_step, teacher_step, mcq_step], {
        TEXT_FEATURE_SURFACE: output_path_of(text_step),
        TEACHER_FORCED_SURFACE: output_path_of(teacher_step),
        MCQ_SURFACE: output_path_of(mcq_step),
    }


def collect_checkpoint_feature_canary(config: CollectCheckpointFeatureCanaryConfig) -> None:
    """Collect canary output locations and lightweight artifact expectations."""
    state_rows = [CheckpointFeatureCanarySpec(**row) for row in json.loads(config.state_rows_json)]
    output_path = config.output_path.rstrip("/")
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    surface_rows = []
    for surface, path in sorted(config.surface_output_paths.items()):
        artifact = ""
        if surface == TEXT_FEATURE_SURFACE:
            artifact = "scored_documents.parquet"
        elif surface == TEACHER_FORCED_SURFACE:
            artifact = TEACHER_FORCED_REQUEST_FEATURES_PARQUET
        elif surface == MCQ_SURFACE:
            artifact = MCQ_REQUEST_FEATURES_PARQUET
        surface_rows.append({"surface": surface, "output_path": str(path), "primary_artifact": artifact})
    summary = {
        "state_rows": [asdict(row) for row in state_rows],
        "surfaces": surface_rows,
        "semantics": (
            "checkpoint-feature/log-mixture surrogate canary; proportional checkpoint only; "
            "not true single-domain MDE vertex-expert features"
        ),
    }
    with fsspec.open(os.path.join(output_path, SUMMARY_JSON), "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with fsspec.open(os.path.join(output_path, STATE_OUTPUT_CSV), "wt") as handle:
        pd.DataFrame.from_records([asdict(row) for row in state_rows]).to_csv(handle, index=False)


def build_collect_step(
    *,
    name_prefix: str,
    state_rows: list[CheckpointFeatureCanarySpec],
    surface_output_paths: dict[str, InputName],
) -> ExecutorStep:
    """Build the final checkpoint-feature canary collection step."""
    return ExecutorStep(
        name=f"{name_prefix}/collect_checkpoint_feature_canary",
        description="Collect proportional checkpoint-feature canary outputs",
        fn=collect_checkpoint_feature_canary,
        config=CollectCheckpointFeatureCanaryConfig(
            output_path=this_output_path(),
            state_rows_json=json.dumps([asdict(row) for row in state_rows], sort_keys=True),
            surface_output_paths=surface_output_paths,
        ),
    )


def _parse_text_bundles(bundle_args: list[str]) -> tuple[str, ...]:
    bundle_keys = tuple(bundle_args) if bundle_args else DEFAULT_TEXT_BUNDLES
    unknown = sorted(set(bundle_keys) - set(TEXT_BUNDLE_CHOICES))
    if unknown:
        raise ValueError(f"Unknown text bundle(s): {unknown}")
    return bundle_keys


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--matrix-csv",
        help="Optional matrix CSV to resolve the run row. Defaults to self-contained proportional metadata.",
    )
    parser.add_argument("--registry-key", default=DEFAULT_REGISTRY_KEY)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--expected-checkpoint-step", type=int, default=DEFAULT_EXPECTED_STEP)
    parser.add_argument("--text-bundle", action="append", choices=TEXT_BUNDLE_CHOICES, default=[])
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-docs-per-dataset", type=int, default=32)
    parser.add_argument("--max-eval-instances", type=int, default=32)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--state-csv")
    parser.add_argument("--teacher-forced-request-cache-uri", default=DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI)
    parser.add_argument("--mcq-request-cache-uri", default=DEFAULT_MCQ_REQUEST_CACHE_URI)
    parser.add_argument(
        "--skip-request-cache-check",
        action="store_true",
        help="Skip pre-submit existence checks for smooth-proxy request-cache JSONL files.",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    text_bundle_keys = _parse_text_bundles(args.text_bundle)
    if not args.skip_request_cache_check:
        _require_nonempty_request_cache(args.teacher_forced_request_cache_uri)
        _require_nonempty_request_cache(args.mcq_request_cache_uri)
    datasets = build_text_feature_datasets(text_bundle_keys)
    dataset_names = tuple(datasets)
    if args.state_csv is None:
        if args.matrix_csv:
            state_row = build_state_row(
                matrix_csv=args.matrix_csv,
                run_name=args.run_name,
                text_bundle_keys=text_bundle_keys,
                text_dataset_names=dataset_names,
                max_docs_per_dataset=args.max_docs_per_dataset,
                max_eval_instances=args.max_eval_instances,
                default_tpu_type=args.tpu_type,
                default_tpu_region=args.tpu_region,
                default_tpu_zone=args.tpu_zone,
            )
        else:
            state_row = _build_state_row_from_values(
                run_name=args.run_name,
                registry_key=args.registry_key,
                source_experiment=args.source_experiment,
                cohort=args.cohort,
                checkpoint_root=args.checkpoint_root,
                expected_checkpoint_step=args.expected_checkpoint_step,
                text_bundle_keys=text_bundle_keys,
                text_dataset_names=dataset_names,
                max_docs_per_dataset=args.max_docs_per_dataset,
                max_eval_instances=args.max_eval_instances,
                default_tpu_type=args.tpu_type,
                default_tpu_region=args.tpu_region,
                default_tpu_zone=args.tpu_zone,
            )
        state_rows = [state_row]
    else:
        state_rows = _load_state_rows(args.state_csv)
        if len(state_rows) != 1:
            raise ValueError(f"Checkpoint-feature canary expects exactly one state row, got {len(state_rows)}.")
        state_row = state_rows[0]

    _write_local_outputs(state_rows)
    logger.info("Selected text feature bundles: %s", "+".join(text_bundle_keys))
    logger.info("Selected %d text feature datasets", len(dataset_names))
    logger.info("Wrote state to %s", STATE_CSV)
    logger.info("Wrote launch manifest to %s", LAUNCH_MANIFEST_CSV)
    logger.info("Canary launch decision for %s: %s", state_row.run_name, state_row.launch_decision)
    if args.dry_run or os.getenv("CI") is not None:
        return

    feature_steps, surface_output_paths = build_feature_steps(
        name_prefix=args.name_prefix,
        spec=state_row,
        datasets=datasets,
        teacher_forced_request_cache_uri=args.teacher_forced_request_cache_uri,
        mcq_request_cache_uri=args.mcq_request_cache_uri,
    )
    collect_step = build_collect_step(
        name_prefix=args.name_prefix,
        state_rows=state_rows,
        surface_output_paths=surface_output_paths,
    )
    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*feature_steps, collect_step],
        description=f"{args.name_prefix}: 300M proportional checkpoint-feature canary",
    )


if __name__ == "__main__":
    main()
