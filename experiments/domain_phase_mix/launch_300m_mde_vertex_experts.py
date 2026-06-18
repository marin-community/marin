# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch 300M single-domain MDE vertex experts and feature extraction.

This graph trains one no-simulated-epoching expert per 39-domain qsplit240
top-level domain, adds two full-6B small-domain controls, scores all experts on
the current MDE feature surfaces, extracts a bounded raw-text token sketch, and
compacts the surfaces into dense matrices for local modeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, InputName, executor_main, output_path_of, this_output_path
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.compact_mde_vertex_expert_features_300m import (
    CompactMdeVertexFeaturesConfig,
    compact_mde_vertex_features,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.extract_mde_uncheatable_token_features_300m import (
    CollectShardedFeaturesConfig,
    ExtractRunFeaturesConfig,
    SelectTokenSketchConfig,
    collect_sharded_features,
    extract_run_features,
    select_token_sketch,
)
from experiments.domain_phase_mix.launch_300m_checkpoint_features_canary import (
    DEFAULT_MCQ_REQUEST_CACHE_URI,
    DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI,
    DEFAULT_TEXT_BUNDLES,
    MCQ_REQUEST_FEATURES_PARQUET,
    MCQ_SURFACE,
    TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
    TEACHER_FORCED_SURFACE,
    TEXT_FEATURE_SURFACE,
    TEXT_BUNDLE_CHOICES,
    CheckpointFeatureCanarySpec,
    _parse_text_bundles,
    _require_nonempty_request_cache,
    build_feature_steps,
    build_text_feature_datasets,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import _executor_prefix
from experiments.domain_phase_mix.proxy_sweep import get_num_train_steps, regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR, skip_eval_harness_for_training_step
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    PHASE_NAMES,
    PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
    SEQ_LEN,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "mde_vertex_experts_300m_20260531"
)
DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_mde_vertex_experts_300m_20260531"
DEFAULT_MAX_CONCURRENT = 8
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_WORKER_CPU = 1.0
DEFAULT_WORKER_RAM = "12g"
DEFAULT_WORKER_DISK = "64g"
EXPERIMENT_BUDGET = 6_000_000_000
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
RUN_ID_BASE = 900_000
CONTROL_DOMAINS = ("dolma3_wikipedia", "dolmino_stem_heavy_crawl")
DEFAULT_TOKEN_REFERENCE_DOMAIN = "dolmino_common_crawl_hq"
FEATURE_INDEX_FILE = "feature_surface_index.csv"
TRAINING_MANIFEST_CSV = "mde_vertex_expert_training_manifest.csv"
EPOCH_SUMMARY_CSV = "mde_vertex_expert_epoch_summary.csv"
RUN_SPECS_JSON = "mde_vertex_expert_run_specs.json"
SUMMARY_JSON = "summary.json"
SURFACE_ARTIFACTS = {
    TEXT_FEATURE_SURFACE: "scored_documents.parquet",
    TEACHER_FORCED_SURFACE: TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
    MCQ_SURFACE: MCQ_REQUEST_FEATURES_PARQUET,
}


def _read_executor_status(cache_path: str) -> str | None:
    """Return the final non-empty executor status line for a prebuilt cache."""
    status_uri = cache_path.rstrip("/") + "/.executor_status"
    fs, _, paths = fsspec.get_fs_token_paths(status_uri)
    if len(paths) != 1:
        raise ValueError(f"Expected one status path for {cache_path!r}, got {paths}")
    try:
        with fs.open(paths[0], "rt") as handle:
            last_line = ""
            for line in handle:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
    except FileNotFoundError:
        return None
    return last_line or None


def _status_is_success(status: str | None) -> bool:
    if status is None:
        return False
    if status == "SUCCESS":
        return True
    try:
        payload = json.loads(status)
    except json.JSONDecodeError:
        return '"SUCCESS"' in status
    return payload.get("status") == "SUCCESS"


def validate_prebuilt_runtime_caches(
    runtime_cache_region: str,
    *,
    status_reader: Callable[[str], str | None] = _read_executor_status,
) -> None:
    """Fail fast when a configured prebuilt runtime cache is not complete."""
    cache_paths = PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION.get(runtime_cache_region, {})
    incomplete = {
        domain_name: cache_path
        for domain_name, cache_path in cache_paths.items()
        if not _status_is_success(status_reader(cache_path))
    }
    if incomplete:
        formatted = "\n".join(f"- {domain_name}: {cache_path}" for domain_name, cache_path in sorted(incomplete.items()))
        raise ValueError(
            f"{len(incomplete)} prebuilt runtime cache(s) for {runtime_cache_region} are incomplete:\n{formatted}"
        )


@dataclass(frozen=True)
class MdeVertexRunSpec:
    """One MDE vertex expert or control training run."""

    run_order: int
    run_id: int
    run_name: str
    domain_name: str
    domain_tokens: int
    train_tokens: int
    realized_train_tokens: int
    num_train_steps: int
    expected_checkpoint_step: int
    materialized_epochs: float
    is_control: bool
    target_budget: None
    data_seed: int
    trainer_seed: int | None
    simulated_epoch_subset_seed: int | None
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class CollectMdeVertexFeatureIndexConfig:
    """Config for collecting vertex-expert feature artifact paths."""

    output_path: str
    run_specs_json: str
    surface_output_paths: dict[str, InputName]
    checkpoint_paths: dict[str, InputName]


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved vertex-expert executor graph."""

    run_specs: list[MdeVertexRunSpec]
    training_steps: list[ExecutorStep]
    feature_steps: list[ExecutorStep]
    feature_index_step: ExecutorStep | None
    token_steps: list[ExecutorStep]
    token_collect_step: ExecutorStep | None
    dense_step: ExecutorStep | None

    @property
    def steps(self) -> list[ExecutorStep]:
        steps: list[ExecutorStep] = [*self.training_steps, *self.feature_steps]
        if self.feature_index_step is not None:
            steps.append(self.feature_index_step)
        steps.extend(self.token_steps)
        if self.token_collect_step is not None:
            steps.append(self.token_collect_step)
        if self.dense_step is not None:
            steps.append(self.dense_step)
        return steps


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError(f"Cannot create slug from {value!r}")
    return normalized


def _one_hot_phase_weights(domain_name: str) -> dict[str, dict[str, float]]:
    if domain_name not in DOMAIN_NAMES:
        raise ValueError(f"Unknown MDE vertex domain {domain_name!r}")
    weights = {candidate: 1.0 if candidate == domain_name else 0.0 for candidate in DOMAIN_NAMES}
    return {phase_name: dict(weights) for phase_name in PHASE_NAMES}


def _training_tokens(domain_name: str, *, full_budget: bool) -> int:
    domain_tokens = int(TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name])
    return EXPERIMENT_BUDGET if full_budget else min(EXPERIMENT_BUDGET, domain_tokens)


def _run_spec(*, run_order: int, run_id: int, domain_name: str, is_control: bool) -> MdeVertexRunSpec:
    domain_tokens = int(TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name])
    train_tokens = _training_tokens(domain_name, full_budget=is_control)
    num_train_steps = get_num_train_steps(train_tokens, BATCH_SIZE, SEQ_LEN)
    if num_train_steps <= 0:
        raise ValueError(f"Domain {domain_name} has too few train tokens for one batch: {train_tokens}")
    realized_train_tokens = num_train_steps * TOKENS_PER_STEP
    prefix = "mde_vertex_full6b" if is_control else "mde_vertex_cap1"
    run_name = f"{prefix}_{_slug(domain_name)}"
    return MdeVertexRunSpec(
        run_order=run_order,
        run_id=run_id,
        run_name=run_name,
        domain_name=domain_name,
        domain_tokens=domain_tokens,
        train_tokens=train_tokens,
        realized_train_tokens=realized_train_tokens,
        num_train_steps=num_train_steps,
        expected_checkpoint_step=num_train_steps - 1,
        materialized_epochs=realized_train_tokens / domain_tokens,
        is_control=is_control,
        target_budget=None,
        data_seed=run_id,
        trainer_seed=None,
        simulated_epoch_subset_seed=None,
        phase_weights=_one_hot_phase_weights(domain_name),
    )


def build_run_specs(*, include_domains: tuple[str, ...] = (), skip_controls: bool = False) -> list[MdeVertexRunSpec]:
    """Build stable 39-domain cap-1 experts plus two full-6B controls."""
    domain_filter = set(include_domains)
    unknown = sorted(domain_filter - set(DOMAIN_NAMES))
    if unknown:
        raise ValueError(f"Unknown MDE vertex domains: {unknown}")
    selected_domains = [domain_name for domain_name in DOMAIN_NAMES if not domain_filter or domain_name in domain_filter]
    run_specs: list[MdeVertexRunSpec] = []
    for domain_name in selected_domains:
        run_specs.append(
            _run_spec(
                run_order=len(run_specs),
                run_id=RUN_ID_BASE + len(run_specs),
                domain_name=domain_name,
                is_control=False,
            )
        )
    if not skip_controls:
        for domain_name in CONTROL_DOMAINS:
            if domain_filter and domain_name not in domain_filter:
                continue
            run_specs.append(
                _run_spec(
                    run_order=len(run_specs),
                    run_id=RUN_ID_BASE + len(run_specs),
                    domain_name=domain_name,
                    is_control=True,
                )
            )
    run_names = [run_spec.run_name for run_spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate MDE vertex run names: {run_names}")
    return run_specs


def _configure_training_step(training_step: ExecutorStep, *, tpu_region: str) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, env_vars=cast(dict[str, str], env_vars)))


def _create_training_step(
    *,
    name_prefix: str,
    run_spec: MdeVertexRunSpec,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> ExecutorStep:
    resources = ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=run_spec.train_tokens,
        target_budget=None,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        resources=resources,
        eval_harness_tasks=(),
        eval_datasets_cache_path=None,
        runtime_cache_region=tpu_region,
    )
    if experiment.num_train_steps != run_spec.num_train_steps:
        raise ValueError(
            f"{run_spec.run_name} expected {run_spec.num_train_steps} train steps, got {experiment.num_train_steps}"
        )
    if experiment.target_budget is not None:
        raise ValueError(f"{run_spec.run_name} unexpectedly uses simulated epoching")
    training_step = experiment.create_training_step(
        weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
        name_prefix=name_prefix,
        run_name=run_spec.run_name,
        data_seed=run_spec.data_seed,
        trainer_seed=run_spec.trainer_seed,
        simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        steps_per_hf_export=run_spec.num_train_steps,
    )
    return _configure_training_step(skip_eval_harness_for_training_step(training_step), tpu_region=tpu_region)


def _checkpoint_path(training_step: ExecutorStep, run_spec: MdeVertexRunSpec) -> InputName:
    return output_path_of(training_step, f"hf/step-{run_spec.expected_checkpoint_step}")


def _checkpoint_feature_spec(
    *,
    run_spec: MdeVertexRunSpec,
    checkpoint_path: InputName,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    text_bundle_keys: tuple[str, ...],
    text_dataset_names: tuple[str, ...],
    max_docs_per_dataset: int | None,
    max_eval_instances: int | None,
) -> CheckpointFeatureCanarySpec:
    return CheckpointFeatureCanarySpec(
        run_name=run_spec.run_name,
        registry_key=f"300m_6b:mde_vertex:{run_spec.domain_name}:{run_spec.run_name}",
        source_experiment=DEFAULT_NAME_PREFIX,
        cohort="mde_vertex_experts_300m",
        checkpoint_root=str(checkpoint_path),
        expected_checkpoint_step=run_spec.expected_checkpoint_step,
        hf_checkpoint_latest=checkpoint_path,  # type: ignore[arg-type]
        hf_checkpoint_latest_step=run_spec.expected_checkpoint_step,
        has_exact_hf_checkpoint=True,
        uses_east5_checkpoint=True,
        launch_tpu_type=tpu_type,
        launch_tpu_region=tpu_region,
        launch_tpu_zone=tpu_zone,
        text_bundle_key="+".join(text_bundle_keys),
        text_dataset_count=len(text_dataset_names),
        text_dataset_names=";".join(text_dataset_names),
        max_docs_per_dataset=max_docs_per_dataset,
        max_eval_instances=max_eval_instances,
        eligible=True,
        launch_decision="launch",
        step_name=f"mde_vertex_feature/{_slug(run_spec.run_name)}",
    )


def _surface_key(run_name: str, surface: str) -> str:
    return f"{_slug(run_name)}::{surface}"


def _artifact_size_bytes(uri: str) -> int | None:
    """Return artifact size when the backing filesystem exposes it."""
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    if len(paths) != 1:
        raise ValueError(f"Expected one artifact path, got {paths}")
    try:
        return int(fs.size(paths[0]))
    except FileNotFoundError:
        return None


def collect_mde_vertex_feature_index(config: CollectMdeVertexFeatureIndexConfig) -> None:
    """Collect feature surface artifact URIs for downstream dense compaction."""
    run_specs = [MdeVertexRunSpec(**row) for row in json.loads(config.run_specs_json)]
    run_by_slug = {_slug(run_spec.run_name): run_spec for run_spec in run_specs}
    rows: list[dict[str, object]] = []
    for key, output_path in sorted(config.surface_output_paths.items()):
        run_slug, surface = key.split("::", maxsplit=1)
        run_spec = run_by_slug[run_slug]
        artifact_name = SURFACE_ARTIFACTS[surface]
        output_path_str = str(output_path).rstrip("/")
        artifact_uri = f"{output_path_str}/{artifact_name}"
        checkpoint_path = str(config.checkpoint_paths[run_slug])
        rows.append(
            {
                "run_order": run_spec.run_order,
                "run_id": run_spec.run_id,
                "run_name": run_spec.run_name,
                "domain_name": run_spec.domain_name,
                "is_control": run_spec.is_control,
                "domain_tokens": run_spec.domain_tokens,
                "train_tokens": run_spec.train_tokens,
                "realized_train_tokens": run_spec.realized_train_tokens,
                "num_train_steps": run_spec.num_train_steps,
                "expected_checkpoint_step": run_spec.expected_checkpoint_step,
                "materialized_epochs": run_spec.materialized_epochs,
                "surface": surface,
                "output_path": output_path_str,
                "artifact_name": artifact_name,
                "artifact_uri": artifact_uri,
                "size_bytes": _artifact_size_bytes(artifact_uri),
                "checkpoint_path": checkpoint_path,
            }
        )

    output_path = config.output_path.rstrip("/")
    with open_or_fsspec(f"{output_path}/{FEATURE_INDEX_FILE}", "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open_or_fsspec(f"{output_path}/{SUMMARY_JSON}", "wt") as handle:
        json.dump(
            {
                "format": "mde_vertex_feature_index_v1",
                "feature_index": f"{output_path}/{FEATURE_INDEX_FILE}",
                "num_runs": len(run_specs),
                "num_surface_rows": len(rows),
                "surfaces": sorted(SURFACE_ARTIFACTS),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def open_or_fsspec(path: str, mode: str):
    """Open a local path or fsspec URI."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one path, got {paths}")
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    return fs.open(paths[0], mode)


def _worker_resources(*, cpu: float, ram: str, disk: str, region: str, zone: str) -> ResourceConfig:
    return ResourceConfig.with_cpu(cpu=cpu, ram=ram, disk=disk, preemptible=False, regions=[region], zone=zone)


def _token_reference_run(run_specs: list[MdeVertexRunSpec], requested_domain: str) -> str:
    preferred = f"mde_vertex_cap1_{_slug(requested_domain)}"
    run_names = {run_spec.run_name for run_spec in run_specs}
    if preferred in run_names:
        return preferred
    if len(run_specs) == 41:
        raise ValueError(f"Default token reference run {preferred!r} is missing from the full MDE vertex run set")
    fallback = run_specs[0].run_name
    if fallback not in run_names:
        raise ValueError(f"Fallback token reference run {fallback!r} is missing")
    return fallback


def _build_token_steps(
    *,
    name_prefix: str,
    feature_index_path: InputName,
    run_specs: list[MdeVertexRunSpec],
    reference_domain: str,
    sample_tokens_per_dataset: int,
    batch_size: int,
    progress_every_batches: int,
    worker_resources: ResourceConfig,
) -> tuple[list[ExecutorStep], ExecutorStep]:
    reference_run = _token_reference_run(run_specs, reference_domain)
    select_step = ExecutorStep(
        name=f"{name_prefix}/select_raw_text_token_sketch",
        description="Select deterministic raw-text token sketch for MDE vertex experts",
        fn=select_token_sketch,
        config=SelectTokenSketchConfig(
            output_path=this_output_path(),
            feature_index=feature_index_path,  # type: ignore[arg-type]
            reference_run=reference_run,
            sample_tokens_per_dataset=sample_tokens_per_dataset,
            dataset_prefix="",
            batch_size=batch_size,
            allow_gcs_read=True,
        ),
        resources=worker_resources,
    )
    run_steps: list[ExecutorStep] = []
    run_output_paths: dict[str, InputName] = {}
    for run_spec in run_specs:
        step = ExecutorStep(
            name=f"{name_prefix}/extract_raw_text_token_features/{_slug(run_spec.run_name)}",
            description=f"Extract raw-text token/document shards for {run_spec.run_name}",
            fn=extract_run_features,
            config=ExtractRunFeaturesConfig(
                output_path=this_output_path(),
                feature_index=feature_index_path,  # type: ignore[arg-type]
                selected_tokens_path=output_path_of(select_step, "selected_tokens.parquet"),
                run_name=run_spec.run_name,
                dataset_prefix="",
                batch_size=batch_size,
                allow_gcs_read=True,
                progress_every_batches=progress_every_batches,
            ),
            resources=worker_resources,
        )
        run_steps.append(step)
        run_output_paths[run_spec.run_name] = output_path_of(step)
    collect_step = ExecutorStep(
        name=f"{name_prefix}/collect_raw_text_token_features",
        description=f"Collect raw-text token feature shards for {len(run_specs)} MDE vertex experts",
        fn=collect_sharded_features,
        config=CollectShardedFeaturesConfig(
            output_path=this_output_path(),
            feature_index=feature_index_path,  # type: ignore[arg-type]
            selected_tokens_path=output_path_of(select_step, "selected_tokens.parquet"),
            run_output_paths=run_output_paths,
            reference_run=reference_run,
            sample_tokens_per_dataset=sample_tokens_per_dataset,
            dataset_prefix="",
        ),
        resources=worker_resources,
    )
    return [select_step, *run_steps], collect_step


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    include_domains: tuple[str, ...],
    skip_controls: bool,
    include_features: bool,
    include_token_features: bool,
    include_dense_compaction: bool,
    text_bundle_keys: tuple[str, ...],
    max_docs_per_dataset: int | None,
    max_eval_instances: int | None,
    teacher_forced_request_cache_uri: str,
    mcq_request_cache_uri: str,
    sample_tokens_per_dataset: int,
    token_batch_size: int,
    token_progress_every_batches: int,
    dense_dtype: str,
    worker_cpu: float,
    worker_ram: str,
    worker_disk: str,
) -> LaunchArtifacts:
    """Build the full MDE vertex-expert launch graph without submitting."""
    if tpu_region != DEFAULT_TPU_REGION or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"MDE vertex experts are pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    if "us-central" in teacher_forced_request_cache_uri or "us-central" in mcq_request_cache_uri:
        raise ValueError("MDE vertex feature request caches must be east5-local")
    run_specs = build_run_specs(include_domains=include_domains, skip_controls=skip_controls)
    training_steps = [
        _create_training_step(
            name_prefix=name_prefix,
            run_spec=run_spec,
            tpu_type=tpu_type,
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
        )
        for run_spec in run_specs
    ]
    feature_steps: list[ExecutorStep] = []
    surface_output_paths: dict[str, InputName] = {}
    feature_index_step: ExecutorStep | None = None
    token_steps: list[ExecutorStep] = []
    token_collect_step: ExecutorStep | None = None
    dense_step: ExecutorStep | None = None

    if include_features:
        datasets = build_text_feature_datasets(text_bundle_keys)
        text_dataset_names = tuple(datasets)
        for run_spec, training_step in zip(run_specs, training_steps, strict=True):
            spec = _checkpoint_feature_spec(
                run_spec=run_spec,
                checkpoint_path=_checkpoint_path(training_step, run_spec),
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                text_bundle_keys=text_bundle_keys,
                text_dataset_names=text_dataset_names,
                max_docs_per_dataset=max_docs_per_dataset,
                max_eval_instances=max_eval_instances,
            )
            row_steps, row_outputs = build_feature_steps(
                name_prefix=name_prefix,
                spec=spec,
                datasets=datasets,
                teacher_forced_request_cache_uri=teacher_forced_request_cache_uri,
                mcq_request_cache_uri=mcq_request_cache_uri,
            )
            feature_steps.extend(row_steps)
            for surface, output_path in row_outputs.items():
                surface_output_paths[_surface_key(run_spec.run_name, surface)] = output_path

        feature_index_step = ExecutorStep(
            name=f"{name_prefix}/collect_mde_vertex_feature_index",
            description=f"Collect MDE vertex feature index for {len(run_specs)} experts",
            fn=collect_mde_vertex_feature_index,
            config=CollectMdeVertexFeatureIndexConfig(
                output_path=this_output_path(),
                run_specs_json=json.dumps([asdict(run_spec) for run_spec in run_specs], sort_keys=True),
                surface_output_paths=surface_output_paths,
                checkpoint_paths={
                    _slug(run_spec.run_name): _checkpoint_path(training_step, run_spec)
                    for run_spec, training_step in zip(run_specs, training_steps, strict=True)
                },
            ),
        )

        feature_index_path = output_path_of(feature_index_step, FEATURE_INDEX_FILE)
        if include_token_features:
            worker_resources = _worker_resources(
                cpu=worker_cpu,
                ram=worker_ram,
                disk=worker_disk,
                region=tpu_region,
                zone=tpu_zone,
            )
            token_steps, token_collect_step = _build_token_steps(
                name_prefix=name_prefix,
                feature_index_path=feature_index_path,
                run_specs=run_specs,
                reference_domain=DEFAULT_TOKEN_REFERENCE_DOMAIN,
                sample_tokens_per_dataset=sample_tokens_per_dataset,
                batch_size=token_batch_size,
                progress_every_batches=token_progress_every_batches,
                worker_resources=worker_resources,
            )
        if include_dense_compaction:
            if include_token_features and token_collect_step is None:
                raise ValueError("Dense compaction requested token features but token collect step is missing")
            dense_step = ExecutorStep(
                name=f"{name_prefix}/compact_mde_vertex_dense_features",
                description="Compact MDE vertex feature surfaces into dense matrices",
                fn=compact_mde_vertex_features,
                config=CompactMdeVertexFeaturesConfig(
                    output_path=this_output_path(),
                    feature_index=feature_index_path,  # type: ignore[arg-type]
                    token_feature_dir=output_path_of(token_collect_step) if token_collect_step is not None else None,
                    dtype=dense_dtype,
                ),
                resources=_worker_resources(
                    cpu=worker_cpu,
                    ram=worker_ram,
                    disk=worker_disk,
                    region=tpu_region,
                    zone=tpu_zone,
                ),
            )
    elif include_token_features or include_dense_compaction:
        raise ValueError("Token features and dense compaction require --include-features")

    artifacts = LaunchArtifacts(
        run_specs=run_specs,
        training_steps=training_steps,
        feature_steps=feature_steps,
        feature_index_step=feature_index_step,
        token_steps=token_steps,
        token_collect_step=token_collect_step,
        dense_step=dense_step,
    )
    validate_launch_artifacts(
        artifacts,
        include_features=include_features,
        include_token_features=include_token_features,
        include_dense_compaction=include_dense_compaction,
    )
    return artifacts


def validate_launch_artifacts(
    artifacts: LaunchArtifacts,
    *,
    include_features: bool,
    include_token_features: bool,
    include_dense_compaction: bool,
) -> None:
    """Validate graph invariants before launch."""
    expected_runs = len(artifacts.run_specs)
    if expected_runs <= 0:
        raise ValueError("No MDE vertex expert runs selected")
    if len(artifacts.training_steps) != expected_runs:
        raise ValueError("Training step count does not match run spec count")
    run_names = [run_spec.run_name for run_spec in artifacts.run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate MDE vertex run names")
    cap1 = [run_spec for run_spec in artifacts.run_specs if not run_spec.is_control]
    controls = [run_spec for run_spec in artifacts.run_specs if run_spec.is_control]
    if not cap1:
        raise ValueError("Expected at least one cap-1 MDE vertex expert")
    if len(artifacts.run_specs) == 41 and (len(cap1) != 39 or len(controls) != 2):
        raise ValueError(f"Expected 39 cap-1 and 2 control runs, got {len(cap1)} and {len(controls)}")
    for run_spec in cap1:
        if run_spec.materialized_epochs > 1.0 + 1e-12:
            raise ValueError(f"{run_spec.run_name} cap-1 expert exceeds one epoch")
    for run_spec in controls:
        if run_spec.domain_name not in CONTROL_DOMAINS:
            raise ValueError(f"Unexpected MDE vertex control domain {run_spec.domain_name}")
        if run_spec.materialized_epochs <= 1.0:
            raise ValueError(f"{run_spec.run_name} control does not exceed one epoch")
    for run_spec in artifacts.run_specs:
        if set(run_spec.phase_weights) != set(PHASE_NAMES):
            raise ValueError(f"{run_spec.run_name} phase names do not match topology")
        for phase_name, weights in run_spec.phase_weights.items():
            if set(weights) != set(DOMAIN_NAMES):
                raise ValueError(f"{run_spec.run_name}/{phase_name} does not enumerate all domains")
            if sum(value > 0 for value in weights.values()) != 1:
                raise ValueError(f"{run_spec.run_name}/{phase_name} is not one-hot")
            if weights[run_spec.domain_name] != 1.0:
                raise ValueError(f"{run_spec.run_name}/{phase_name} active domain is not 1.0")
        if run_spec.target_budget is not None:
            raise ValueError(f"{run_spec.run_name} unexpectedly has target_budget={run_spec.target_budget}")

    for run_spec, training_step in zip(artifacts.run_specs, artifacts.training_steps, strict=True):
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(DEFAULT_TPU_REGION):
            raise ValueError(f"{training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        if env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) != "1":
            raise ValueError(f"{training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")
        if int(config.train_config.trainer.num_train_steps) != run_spec.num_train_steps:
            raise ValueError(f"{training_step.name} has wrong num_train_steps")
        if int(config.train_config.hf_save_steps) != run_spec.num_train_steps:
            raise ValueError(f"{training_step.name} does not export final HF checkpoint")
        if config.train_config.eval_harness is not None:
            raise ValueError(f"{training_step.name} unexpectedly has eval harness configured")

    if include_features:
        if artifacts.feature_index_step is None:
            raise ValueError("Feature index step missing")
        if len(artifacts.feature_steps) != expected_runs * 3:
            raise ValueError(f"Expected {expected_runs * 3} feature steps, got {len(artifacts.feature_steps)}")
    elif artifacts.feature_steps or artifacts.feature_index_step is not None:
        raise ValueError("Feature steps present despite include_features=False")
    if include_token_features:
        if artifacts.token_collect_step is None:
            raise ValueError("Token collect step missing")
        if len(artifacts.token_steps) != expected_runs + 1:
            raise ValueError(f"Expected select step plus {expected_runs} token steps")
    if include_dense_compaction and artifacts.dense_step is None:
        raise ValueError("Dense compaction step missing")


def _flat_run_spec(run_spec: MdeVertexRunSpec) -> dict[str, Any]:
    row = asdict(run_spec)
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[f"{phase_name}_{domain_name}"] = value
    return row


def write_local_manifests(artifacts: LaunchArtifacts, output_dir: Path) -> None:
    """Write local audit artifacts for the MDE vertex launch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flat_run_spec(run_spec) for run_spec in artifacts.run_specs]
    with (output_dir / TRAINING_MANIFEST_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    epoch_rows = [
        {
            "run_name": run_spec.run_name,
            "domain_name": run_spec.domain_name,
            "domain_tokens": run_spec.domain_tokens,
            "train_tokens": run_spec.train_tokens,
            "realized_train_tokens": run_spec.realized_train_tokens,
            "materialized_epochs": run_spec.materialized_epochs,
            "num_train_steps": run_spec.num_train_steps,
            "is_control": run_spec.is_control,
        }
        for run_spec in artifacts.run_specs
    ]
    with (output_dir / EPOCH_SUMMARY_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    (output_dir / RUN_SPECS_JSON).write_text(
        json.dumps([asdict(run_spec) for run_spec in artifacts.run_specs], indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "name_prefix": DEFAULT_NAME_PREFIX,
        "num_runs": len(artifacts.run_specs),
        "num_cap1": sum(not run_spec.is_control for run_spec in artifacts.run_specs),
        "num_controls": sum(run_spec.is_control for run_spec in artifacts.run_specs),
        "feature_steps": len(artifacts.feature_steps),
        "token_steps": len(artifacts.token_steps),
        "has_dense_compaction": artifacts.dense_step is not None,
        "cap1_epoch_median": sorted(
            run_spec.materialized_epochs for run_spec in artifacts.run_specs if not run_spec.is_control
        )[len([run_spec for run_spec in artifacts.run_specs if not run_spec.is_control]) // 2],
        "semantics": "true 300M no-simulated-epoching single-domain MDE vertex experts plus two full-6B controls",
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-domain", action="append", default=[])
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-token-features", action="store_true")
    parser.add_argument("--skip-dense-compaction", action="store_true")
    parser.add_argument("--text-bundle", action="append", choices=TEXT_BUNDLE_CHOICES, default=[])
    parser.add_argument("--max-docs-per-dataset", type=int, default=512)
    parser.add_argument("--max-eval-instances", type=int)
    parser.add_argument("--teacher-forced-request-cache-uri", default=DEFAULT_TEACHER_FORCED_REQUEST_CACHE_URI)
    parser.add_argument("--mcq-request-cache-uri", default=DEFAULT_MCQ_REQUEST_CACHE_URI)
    parser.add_argument("--skip-request-cache-check", action="store_true")
    parser.add_argument("--skip-runtime-cache-completeness-check", action="store_true")
    parser.add_argument("--sample-tokens-per-dataset", type=int, default=4096)
    parser.add_argument("--token-batch-size", type=int, default=1)
    parser.add_argument("--token-progress-every-batches", type=int, default=250)
    parser.add_argument("--dense-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--worker-cpu", type=float, default=DEFAULT_WORKER_CPU)
    parser.add_argument("--worker-ram", default=DEFAULT_WORKER_RAM)
    parser.add_argument("--worker-disk", default=DEFAULT_WORKER_DISK)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--local-artifact-dir", default=str(LOCAL_ARTIFACT_DIR))
    return parser.parse_known_args()


def main() -> None:
    """Build and optionally submit the vertex-expert graph."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    text_bundle_keys = _parse_text_bundles(args.text_bundle)
    include_features = not args.skip_features
    include_token_features = include_features and not args.skip_token_features
    include_dense_compaction = include_features and not args.skip_dense_compaction
    if include_features and not args.skip_request_cache_check:
        _require_nonempty_request_cache(args.teacher_forced_request_cache_uri)
        _require_nonempty_request_cache(args.mcq_request_cache_uri)
    if os.getenv("CI") is None and not args.skip_runtime_cache_completeness_check:
        validate_prebuilt_runtime_caches(args.tpu_region)
    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        include_domains=tuple(args.include_domain),
        skip_controls=args.skip_controls,
        include_features=include_features,
        include_token_features=include_token_features,
        include_dense_compaction=include_dense_compaction,
        text_bundle_keys=text_bundle_keys,
        max_docs_per_dataset=args.max_docs_per_dataset,
        max_eval_instances=args.max_eval_instances,
        teacher_forced_request_cache_uri=args.teacher_forced_request_cache_uri,
        mcq_request_cache_uri=args.mcq_request_cache_uri,
        sample_tokens_per_dataset=args.sample_tokens_per_dataset,
        token_batch_size=args.token_batch_size,
        token_progress_every_batches=args.token_progress_every_batches,
        dense_dtype=args.dense_dtype,
        worker_cpu=args.worker_cpu,
        worker_ram=args.worker_ram,
        worker_disk=args.worker_disk,
    )
    write_local_manifests(artifacts, Path(args.local_artifact_dir))
    launch_summary = {
        "name_prefix": args.name_prefix,
        "run_count": len(artifacts.run_specs),
        "training_steps": len(artifacts.training_steps),
        "feature_steps": len(artifacts.feature_steps),
        "token_steps": len(artifacts.token_steps),
        "has_dense_compaction": artifacts.dense_step is not None,
        "max_concurrent": args.max_concurrent,
        "tpu_type": args.tpu_type,
        "tpu_region": args.tpu_region,
        "tpu_zone": args.tpu_zone,
        "local_artifact_dir": args.local_artifact_dir,
    }
    logger.info("Prepared 300M MDE vertex experts: %s", json.dumps(launch_summary, sort_keys=True))
    print(json.dumps(launch_summary, indent=2, sort_keys=True))
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=f"{args.name_prefix}: 300M MDE vertex experts, feature surfaces, and dense matrices",
    )


if __name__ == "__main__":
    main()
