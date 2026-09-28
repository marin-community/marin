# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch dense WSD80 surfaces across token horizon and StarCoder support."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from fray.types import ResourceConfig
from levanter.data.mixture import MixtureDataset
from levanter.main.train_lm import TrainLmConfig
from levanter.store.cache import CACHE_LAYOUT_CONSOLIDATED, CacheLedger
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.experiment.data import mixture
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem import prefix_join

from experiments.datasets.dolma import DOLMA_LLAMA3_OVERRIDES, dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_surfaces_20260808"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_dense_support_surfaces_20260808"
PANEL_TAG = "wsd80_dense_support_surfaces"
DESIGN_VERSION = "2026-08-08-v5"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_dense_support_surface_design_20260808.json")
EXPECTED_DESIGN_SHA256 = "d4ffb9079f969af808230c623555315262cb314434a21db6d36e9651b747cd48"
EXPECTED_RUN_COUNT = 3_668
EXPECTED_CELL_COUNT = 4
EXPECTED_SUPPORT_IDS = frozenset({"full", "m0125", "m025", "m050", "m100", "m200", "m400"})
EXPECTED_COORDINATES_PER_CELL = 125
EXPECTED_UNIQUE_SEQUENCE_IDENTITIES = 596
EXPECTED_ALIAS_ROWS = 504
EXPECTED_REPEAT_ROWS_PER_BLOCK = 8 * 3
EXPECTED_STARCODER_SOURCE_TOKENS = 216_567_300_822
EXPECTED_STARCODER_SOURCE_TOKEN_PROVENANCE = (
    "experiments/domain_phase_mix/domains.py:DOLMA_TOKENS; central1 tokenized cache queried 2025-01-28"
)
EXPECTED_STARCODER_CACHE_DOCUMENTS = 206_640_114
EXPECTED_STARCODER_CACHE_SHARDS = 49
EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA = {
    "append_bos": False,
    "append_eos": True,
    "max_length": 131_072,
    "padding": False,
    "return_attention_mask": False,
    "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
    "vocab_size": 128_256,
}
EXPECTED_RUNTIME_CACHE_CONTRACT = {
    "relative_path": DOLMA_LLAMA3_OVERRIDES["starcoder"],
    "document_count": EXPECTED_STARCODER_CACHE_DOCUMENTS,
    "shard_count": EXPECTED_STARCODER_CACHE_SHARDS,
    "layout": CACHE_LAYOUT_CONSOLIDATED,
    "tokenizer_metadata": EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA,
    "legacy_token_count_policy": (
        "the pinned cache predates train/.stats.json; validate exact path, completion, document count, shards, "
        "layout, and tokenizer metadata at parent startup; retain the frozen token count from the domain registry "
        "as provenance rather than reconstructing it from document count"
    ),
}
EXPECTED_DESIGN_ENVIRONMENT = {
    "jax_version": "0.11.0",
    "numpy_version": "2.3.5",
    "jax_default_prng_impl": "threefry2x32",
    "jax_enable_x64": False,
    "uv_lock_sha256": "d6a6a17fda4dd7d6c3733efcbee87151bb74971219e7a17c6ed05ba7a788086d",
}
EXPECTED_TRAINING_ENVIRONMENT = {
    "jax_version": "0.10.1",
    "numpy_version": "2.3.5",
    "jax_default_prng_impl": "threefry2x32",
    "jax_enable_x64": False,
}
SUPPORT_ORDER = {
    support_id: index for index, support_id in enumerate(("full", "m0125", "m025", "m050", "m100", "m200", "m400"))
}
DEFAULT_MAX_CONCURRENT = 128
COVERAGE_GATE_WORKERS = 64


@dataclass(frozen=True)
class Cell:
    """One fixed-model token-horizon cell."""

    cell_id: str
    cell_slug: str
    rung: int
    hidden_size: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int


@dataclass(frozen=True)
class SurfaceRun:
    """One policy coordinate, support regime, and training seed."""

    run_name: str
    cell_id: str
    cell_slug: str
    rung: int
    hidden_size: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int
    support_id: str
    support_role: str
    epoch_multiplier: float | None
    starcoder_support_batches: int | None
    starcoder_realized_support_tokens: int
    starcoder_support_fraction: float
    coordinate_id: str
    policy_role: str
    coordinate_sources: list[str]
    phase_0_starcoder: float
    phase_1_starcoder: float
    aggregate_starcoder: float
    phase_contrast: float
    starcoder_phase_0_sequences: int
    starcoder_phase_1_sequences: int
    starcoder_total_sequences: int
    starcoder_phase_0_epochs: float
    starcoder_phase_1_epochs: float
    starcoder_support_wraps: bool
    nemotron_max_total_epochs: float
    data_seed: int
    replicate_kind: str


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_starcoder_source(marin_prefix: str, requests: tuple[SurfaceRun, ...]) -> str:
    """Validate the pinned legacy cache without requiring a modern stats sidecar."""
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    expected_cache_dir = prefix_join(marin_prefix, DOLMA_LLAMA3_OVERRIDES["starcoder"])
    observed_cache_dir = starcoder.path(marin_prefix)
    if observed_cache_dir != expected_cache_dir:
        raise ValueError(f"StarCoder cache identity drifted: {observed_cache_dir!r} != {expected_cache_dir!r}")

    ledger = CacheLedger.load(prefix_join(observed_cache_dir, "train"))
    expected_shards = {f"{index:02d}_json_gz" for index in range(EXPECTED_STARCODER_CACHE_SHARDS)}
    if not ledger.is_finished:
        raise ValueError(f"StarCoder cache is incomplete: {observed_cache_dir}")
    if ledger.layout != CACHE_LAYOUT_CONSOLIDATED:
        raise ValueError(f"StarCoder cache layout drifted: {ledger.layout!r}")
    if ledger.total_num_rows != EXPECTED_STARCODER_CACHE_DOCUMENTS:
        raise ValueError(
            f"StarCoder cache document count drifted: {ledger.total_num_rows} != {EXPECTED_STARCODER_CACHE_DOCUMENTS}"
        )
    if set(ledger.finished_shards) != expected_shards or set(ledger.shard_rows) != expected_shards:
        raise ValueError("StarCoder cache shard identities drifted from the frozen 49-shard source")
    if ledger.metadata.preprocessor_metadata != EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA:
        raise ValueError("StarCoder cache tokenizer metadata drifted from the frozen source")

    if ledger.field_counts:
        observed_tokens = ledger.field_counts.get("input_ids")
        if observed_tokens != EXPECTED_STARCODER_SOURCE_TOKENS:
            logger.warning(
                "Pinned StarCoder cache reports input_ids=%r rather than frozen registry provenance %d; "
                "using the registry value for the intervention design",
                observed_tokens,
                EXPECTED_STARCODER_SOURCE_TOKENS,
            )
    else:
        logger.warning(
            "Pinned StarCoder cache predates token-count ledgers; retaining frozen registry provenance for %d tokens",
            EXPECTED_STARCODER_SOURCE_TOKENS,
        )

    finite_support_tokens = [
        request.starcoder_realized_support_tokens
        for request in requests
        if request.starcoder_support_batches is not None
    ]
    if finite_support_tokens and max(finite_support_tokens) >= EXPECTED_STARCODER_SOURCE_TOKENS:
        raise ValueError("A finite StarCoder support cap no longer binds below the frozen physical source")
    logger.info(
        "Validated pinned StarCoder cache %s (%d documents, %d shards); largest finite support %.3fB tokens",
        observed_cache_dir,
        ledger.total_num_rows,
        len(ledger.finished_shards),
        max(finite_support_tokens, default=0) / 1e9,
    )
    return observed_cache_dir


def _request_order(request: SurfaceRun) -> tuple[int, int, int, str, int]:
    replicate_order = {"coverage": 0, "calibration_repeat": 1}
    return (
        replicate_order[request.replicate_kind],
        request.rung,
        SUPPORT_ORDER[request.support_id],
        request.coordinate_id,
        request.data_seed,
    )


def _load_payload() -> dict[str, Any]:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version in {DESIGN_PATH}")
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = _canonical_sha256(payload)
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design self-hash mismatch: {observed_hash} != {claimed_hash}")
    if payload.get("design_environment") != EXPECTED_DESIGN_ENVIRONMENT:
        raise ValueError("Frozen design environment metadata drifted")
    if payload.get("training_environment") != EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError("Frozen training environment metadata drifted")
    if payload.get("runtime_cache_contract") != EXPECTED_RUNTIME_CACHE_CONTRACT:
        raise ValueError("Frozen runtime cache contract drifted")
    runtime_uv_lock_sha256 = _file_sha256(Path(__file__).resolve().parents[2] / "uv.lock")
    if runtime_uv_lock_sha256 != EXPECTED_DESIGN_ENVIRONMENT["uv_lock_sha256"]:
        logger.warning(
            "uv.lock drifted from frozen design provenance: %s != %s",
            runtime_uv_lock_sha256,
            EXPECTED_DESIGN_ENVIRONMENT["uv_lock_sha256"],
        )
    payload["design_sha256"] = claimed_hash
    return payload


def _runtime_scientific_environment() -> dict[str, str | bool]:
    return {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _validate_runtime_scientific_environment() -> None:
    observed = _runtime_scientific_environment()
    if observed != EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError(f"Runtime scientific environment drifted from frozen training environment: {observed}")
    logger.info("Validated frozen training environment: %s", observed)


def audit_runtime_sequence_counts() -> int:
    """Reproduce every frozen cell-coordinate-seed sequence count."""
    payload = _load_payload()
    complete_rows = payload["runs"] + payload["deterministic_aliases"]
    rows_by_identity: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in complete_rows:
        identity = (row["cell_id"], row["coordinate_id"], row["data_seed"])
        rows_by_identity.setdefault(identity, row)
    if len(rows_by_identity) != EXPECTED_UNIQUE_SEQUENCE_IDENTITIES:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_SEQUENCE_IDENTITIES} sequence identities, got {len(rows_by_identity)}"
        )

    component_names = [f"nemotron_{index}" for index in range(len(base.NEMOTRON_TOKEN_COUNTS))]
    component_names.append("starcoder")
    datasets = {name: object() for name in component_names}
    total_nemotron_tokens = sum(base.NEMOTRON_TOKEN_COUNTS.values())

    for row in rows_by_identity.values():
        phase_weights = []
        for starcoder_weight in (row["phase_0_starcoder"], row["phase_1_starcoder"]):
            broad_weight = 1.0 - starcoder_weight
            broad_weights = [
                broad_weight * token_count / total_nemotron_tokens for token_count in base.NEMOTRON_TOKEN_COUNTS.values()
            ]
            phase_weights.append((*broad_weights, starcoder_weight))
        weights = [
            (0, dict(zip(component_names, phase_weights[0], strict=True))),
            (
                row["boundary_step"] * base.BATCH_SIZE,
                dict(zip(component_names, phase_weights[1], strict=True)),
            ),
        ]
        mix_key, _ = jax.random.split(jax.random.PRNGKey(row["data_seed"]))
        mixture_dataset = MixtureDataset(
            datasets,
            weights,
            block_size=base.MIXTURE_BLOCK_SIZE,
            key=mix_key,
        )
        starcoder_index = (
            mixture_dataset.dataset_index.get_index("starcoder")
            if "starcoder" in mixture_dataset.dataset_index
            else None
        )
        boundary_sequences = row["boundary_step"] * base.BATCH_SIZE
        total_sequences = row["total_steps"] * base.BATCH_SIZE
        observed = [0, 0]
        for phase, (start, stop) in enumerate(((0, boundary_sequences), (boundary_sequences, total_sequences))):
            first_block, first_offset = divmod(start, base.MIXTURE_BLOCK_SIZE)
            if first_offset != 0:
                raise ValueError(f"{row['run_name']}: phase boundary is not mixture-block aligned")
            full_blocks, remainder = divmod(stop - start, base.MIXTURE_BLOCK_SIZE)
            if starcoder_index is None:
                continue
            observed[phase] += full_blocks * int(mixture_dataset._counts_per_block_per_stage[phase][starcoder_index])
            if remainder:
                block = mixture_dataset._get_block(first_block + full_blocks)
                observed[phase] += int(np.count_nonzero((block[:remainder] >> 16) == starcoder_index))
        expected = [row["starcoder_phase_0_sequences"], row["starcoder_phase_1_sequences"]]
        if observed != expected:
            raise ValueError(f"{row['run_name']}: runtime sequence counts {observed} != frozen counts {expected}")

    logger.info(
        "Reproduced all %d frozen cell-coordinate-seed sequence identities",
        len(rows_by_identity),
    )
    return len(rows_by_identity)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_design(
    *,
    selected_cells: frozenset[str] | None = None,
    selected_supports: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
    selected_replicate_kind: str | None = None,
) -> tuple[dict[str, Cell], tuple[SurfaceRun, ...]]:
    """Load and structurally audit the immutable launch manifest."""
    payload = _load_payload()
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected dense-support run count")
    if payload.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ValueError("Unexpected dense-support cell count")
    if payload.get("support_count") != len(EXPECTED_SUPPORT_IDS):
        raise ValueError("Unexpected dense-support regime count")
    if payload.get("coordinates_per_cell") != EXPECTED_COORDINATES_PER_CELL:
        raise ValueError("Unexpected coordinate count")
    if payload.get("starcoder_source_tokens") != EXPECTED_STARCODER_SOURCE_TOKENS:
        raise ValueError("Unexpected frozen StarCoder source-token count")
    if payload.get("starcoder_source_token_provenance") != EXPECTED_STARCODER_SOURCE_TOKEN_PROVENANCE:
        raise ValueError("Unexpected frozen StarCoder source-token provenance")

    cells = {row["cell_id"]: Cell(**row) for row in payload["cells"]}
    requests = tuple(SurfaceRun(**row) for row in payload["runs"])
    run_fields = SurfaceRun.__dataclass_fields__.keys()
    aliases = tuple(
        SurfaceRun(**{field: row[field] for field in run_fields}) for row in payload["deterministic_aliases"]
    )
    if len(cells) != EXPECTED_CELL_COUNT or len(requests) != EXPECTED_RUN_COUNT:
        raise ValueError("Manifest rows do not match declared counts")
    if len(aliases) != EXPECTED_ALIAS_ROWS:
        raise ValueError("Manifest alias rows do not match declared counts")
    if len({request.run_name for request in requests}) != len(requests):
        raise ValueError("Dense-support run names are not unique")
    if {request.run_name for request in requests} & {alias.run_name for alias in aliases}:
        raise ValueError("Launched and aliased run names overlap")
    if requests != tuple(sorted(requests, key=_request_order)):
        raise ValueError("Coverage requests must precede all calibration repeats")

    support_ids = EXPECTED_SUPPORT_IDS
    coordinate_count = EXPECTED_COORDINATES_PER_CELL
    repeat_count = EXPECTED_REPEAT_ROWS_PER_BLOCK
    complete_rows = requests + aliases
    reference_coordinates: dict[str, tuple[float, float]] | None = None
    for cell_id, cell in cells.items():
        rows = tuple(request for request in complete_rows if request.cell_id == cell_id)
        if {request.support_id for request in rows} != support_ids:
            raise ValueError(f"{cell_id}: support regimes are incomplete")
        for support_id in support_ids:
            support_rows = tuple(request for request in rows if request.support_id == support_id)
            coverage = tuple(request for request in support_rows if request.replicate_kind == "coverage")
            repeats = tuple(request for request in support_rows if request.replicate_kind == "calibration_repeat")
            if len(coverage) != coordinate_count or len(repeats) != repeat_count:
                raise ValueError(f"{cell_id}, {support_id}: incomplete coverage or repeat block")
            if len({request.coordinate_id for request in coverage}) != coordinate_count:
                raise ValueError(f"{cell_id}, {support_id}: duplicate coverage coordinate")
            coordinates = {
                request.coordinate_id: (request.phase_0_starcoder, request.phase_1_starcoder) for request in coverage
            }
            if reference_coordinates is None:
                reference_coordinates = coordinates
            elif coordinates != reference_coordinates:
                raise ValueError(f"{cell_id}, {support_id}: coordinate matrix differs from the frozen common grid")
        for request in rows:
            if request.total_steps != cell.total_steps or request.materialized_tokens != cell.materialized_tokens:
                raise ValueError(f"{request.run_name}: run/cell schedule mismatch")
            if request.boundary_step != cell.boundary_step:
                raise ValueError(f"{request.run_name}: phase-boundary mismatch")
            if request.total_steps * base.BATCH_SIZE * base.SEQ_LEN != request.materialized_tokens:
                raise ValueError(f"{request.run_name}: token accounting drifted")
            if request.boundary_step * 5 != request.total_steps * 4:
                raise ValueError(f"{request.run_name}: phase boundary is not exactly 80/20")
            if not 0.0 <= request.phase_0_starcoder <= 1.0:
                raise ValueError(f"{request.run_name}: phase-0 weight is infeasible")
            if not 0.0 <= request.phase_1_starcoder <= 1.0:
                raise ValueError(f"{request.run_name}: phase-1 weight is infeasible")
            if request.support_id == "full" and request.starcoder_support_batches is not None:
                raise ValueError(f"{request.run_name}: physical-full support is unexpectedly capped")
            if request.support_id != "full" and request.starcoder_support_batches is None:
                raise ValueError(f"{request.run_name}: finite support is missing its cap")
    launched_names = {request.run_name for request in requests}
    for row in payload["deterministic_aliases"]:
        alias = SurfaceRun(**{field: row[field] for field in run_fields})
        if row["alias_of_run_name"] not in launched_names:
            raise ValueError(f"{alias.run_name}: alias source is not launched")
        if alias.support_id == "full":
            raise ValueError(f"{alias.run_name}: full-pool row cannot be a support alias")
        support_sequences = alias.starcoder_realized_support_tokens // base.SEQ_LEN
        if alias.starcoder_support_wraps or alias.starcoder_total_sequences > support_sequences:
            raise ValueError(f"{alias.run_name}: aliased finite support would wrap")

    if selected_cells is not None:
        unknown = selected_cells - set(cells)
        if unknown:
            raise ValueError(f"Unknown cells: {sorted(unknown)}")
        cells = {cell_id: cell for cell_id, cell in cells.items() if cell_id in selected_cells}
        requests = tuple(request for request in requests if request.cell_id in selected_cells)
    if selected_supports is not None:
        unknown = selected_supports - support_ids
        if unknown:
            raise ValueError(f"Unknown supports: {sorted(unknown)}")
        requests = tuple(request for request in requests if request.support_id in selected_supports)
    if selected_runs is not None:
        available = {request.run_name for request in requests}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs after cell/support filtering: {sorted(unknown)}")
        requests = tuple(request for request in requests if request.run_name in selected_runs)
    if selected_replicate_kind is not None:
        if selected_replicate_kind not in {"coverage", "calibration_repeat"}:
            raise ValueError(f"Unknown replicate kind: {selected_replicate_kind}")
        requests = tuple(request for request in requests if request.replicate_kind == selected_replicate_kind)
    if not requests:
        raise ValueError("Launch filters selected no runs")
    requests = tuple(sorted(requests, key=_request_order))
    return cells, requests


def _with_source_support(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    train_datasets: dict[ArtifactStep[TokenizedCache], float],
    validation_datasets: tuple[ArtifactStep[TokenizedCache], ...],
    phase_weights: list[tuple[int, dict[str, float]]],
    starcoder: ArtifactStep[TokenizedCache],
    starcoder_name: str,
    request: SurfaceRun,
) -> ArtifactStep[LevanterCheckpoint]:
    """Install a StarCoder-only support cap without global simulated slicing."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        data_config = mixture(ctx, train_datasets, validation=validation_datasets)
        if starcoder_name not in data_config.components:
            raise ValueError(f"StarCoder support cap key is absent from mixture components: {starcoder_name}")
        max_train_batches = (
            None if request.starcoder_support_batches is None else {starcoder_name: request.starcoder_support_batches}
        )
        data_config = replace(
            data_config,
            train_weights=phase_weights,
            mixture_block_size=base.MIXTURE_BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches=max_train_batches,
        )
        trainer = replace(
            pod_config.train_config.trainer,
            seed=request.data_seed,
            checkpointer=replace(pod_config.train_config.trainer.checkpointer, keep=None),
        )
        train_config = replace(
            pod_config.train_config,
            data=data_config,
            data_seed=request.data_seed,
            trainer=trainer,
        )
        return replace(pod_config, train_config=train_config)

    return replace(training, build_config=build_config)


def _validate_runtime_model(cells: dict[str, Cell]) -> Any:
    heuristic = CompletedAdamHHeuristic()
    model = heuristic._build_model_config(640, seq_len=base.SEQ_LEN)
    total_parameters = model.total_trainable_params(llama3_tokenizer_vocab_size)
    non_embedding_parameters = model.total_trainable_params(0)
    for cell in cells.values():
        if cell.hidden_size != 640:
            raise ValueError(f"{cell.cell_id}: hidden size drifted")
        if cell.total_parameters != total_parameters or cell.non_embedding_parameters != non_embedding_parameters:
            raise ValueError(f"{cell.cell_id}: model parameter count drifted")
        schedule = base._schedule_summary(cell.materialized_tokens)
        if schedule["total_steps"] != cell.total_steps:
            raise ValueError(f"{cell.cell_id}: runtime schedule drifted")
    return model


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_cells: frozenset[str] | None = None,
    selected_supports: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
    selected_replicate_kind: str | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for the selected frozen rows."""
    cells, requests = load_design(
        selected_cells=selected_cells,
        selected_supports=selected_supports,
        selected_runs=selected_runs,
        selected_replicate_kind=selected_replicate_kind,
    )
    model = _validate_runtime_model(cells)
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    expected_names = (*tuple(nemotron[split].name for split in base.NEMOTRON_TOKEN_COUNTS), starcoder.name)
    observed_names = tuple(handle.name for handle in training_handles)
    if observed_names != expected_names or len(set(observed_names)) != len(observed_names):
        raise ValueError(f"Training-cache ordering drifted: {observed_names}")
    validation_handles = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for request in requests:
        phase_0_weights = base._phase_leaf_weights(
            request.phase_0_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        phase_1_weights = base._phase_leaf_weights(
            request.phase_1_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        if set(phase_0_weights) != set(expected_names) or set(phase_1_weights) != set(expected_names):
            raise ValueError(f"{request.run_name}: phase-weight keys drifted from the frozen data sources")
        training = train_lm(
            name=f"checkpoints/{name_prefix}/{request.run_name}",
            version=base.VERSION,
            model=model,
            optimizer=base._optimizer(request.materialized_tokens),
            datasets=static_weights,
            validation=validation_handles,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=request.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=name_prefix,
            run_id=request.run_name,
            tags=(
                WANDB_EXPERIMENT_TAG,
                PANEL_TAG,
                request.cell_slug,
                request.support_id,
                request.coordinate_id,
                request.replicate_kind,
                "starcoder",
                "wsd80_20",
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            _with_source_support(
                training,
                train_datasets=static_weights,
                validation_datasets=validation_handles,
                phase_weights=[(0, phase_0_weights), (request.boundary_step, phase_1_weights)],
                starcoder=starcoder,
                starcoder_name=starcoder.name,
                request=request,
            )
        )
    return tuple(steps)


def audit_materialized_runtime_configs(
    requests: tuple[SurfaceRun, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
) -> int:
    """Materialize one real run config from every selected cell-support block."""
    if len(requests) != len(steps):
        raise ValueError(f"Request/step cardinality mismatch: {len(requests)} != {len(steps)}")

    audited_blocks: set[tuple[str, str]] = set()
    for request, step in zip(requests, steps, strict=True):
        block = (request.cell_id, request.support_id)
        if block in audited_blocks:
            continue
        config = materialized_config(step, marin_prefix)
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"{request.run_name}: unexpected materialized config {type(config)}")
        train_config = cast(TrainLmConfig, config.train_config)
        expected_cap = (
            None if request.starcoder_support_batches is None else {"dolma/starcoder": request.starcoder_support_batches}
        )
        if train_config.data.max_train_batches != expected_cap:
            raise ValueError(
                f"{request.run_name}: materialized StarCoder support cap drifted: "
                f"{train_config.data.max_train_batches} != {expected_cap}"
            )
        if train_config.data_seed != request.data_seed or train_config.trainer.seed != request.data_seed:
            raise ValueError(f"{request.run_name}: materialized data/trainer seed drifted")
        if train_config.data.experiment_budget is not None or train_config.data.target_budget is not None:
            raise ValueError(f"{request.run_name}: global simulated-epoch budget leaked into the intervention")
        if train_config.data.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{request.run_name}: global simulated subset seed leaked into the intervention")
        phase_weights = train_config.data.train_weights
        if not isinstance(phase_weights, list) or [boundary for boundary, _ in phase_weights] != [
            0,
            request.boundary_step,
        ]:
            raise ValueError(f"{request.run_name}: materialized 80/20 phase boundary drifted")
        audited_blocks.add(block)

    expected_blocks = {(request.cell_id, request.support_id) for request in requests}
    if audited_blocks != expected_blocks:
        raise ValueError(f"Materialized runtime audit missed blocks: {sorted(expected_blocks - audited_blocks)}")
    logger.info("Materialized and audited %d cell-support runtime configs", len(audited_blocks))
    return len(audited_blocks)


def _require_complete_coverage(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> None:
    """Refuse calibration until every frozen coverage artifact is complete."""
    coverage_steps = build_training_steps(
        name_prefix=name_prefix,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        selected_replicate_kind="coverage",
    )

    def artifact_succeeded(step: ArtifactStep[LevanterCheckpoint]) -> bool:
        return StatusFile(step.path(), worker_id="dense-support-coverage-gate").status == STATUS_SUCCESS

    with ThreadPoolExecutor(max_workers=COVERAGE_GATE_WORKERS) as executor:
        built = tuple(executor.map(artifact_succeeded, coverage_steps))
    incomplete = [step.path() for step, is_built in zip(coverage_steps, built, strict=True) if not is_built]
    if incomplete:
        examples = ", ".join(incomplete[:3])
        raise RuntimeError(
            f"Calibration is gated on all {len(coverage_steps)} coverage artifacts; "
            f"{len(incomplete)} remain incomplete. Examples: {examples}"
        )


def _parse_set(value: str | None, option: str) -> frozenset[str] | None:
    if value is None:
        return None
    items = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError(f"{option} must contain at least one value")
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--cells", help="Comma-separated cell IDs for one parent or retry")
    parser.add_argument("--supports", help="Comma-separated support IDs for a partial retry")
    parser.add_argument("--runs", help="Comma-separated exact run names for a partial retry")
    parser.add_argument("--replicate-kind", choices=("coverage", "calibration_repeat"))
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--audit-runtime-identities", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping dense WSD80 support surfaces in CI")
        return
    if args.name_prefix != NAME:
        raise ValueError(f"Checkpoint identity is frozen: {args.name_prefix!r} != {NAME!r}")
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child placement must remain central1-local: "
            f"region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    _validate_runtime_scientific_environment()

    selected_cells = _parse_set(args.cells, "--cells")
    selected_supports = _parse_set(args.supports, "--supports")
    selected_runs = _parse_set(args.runs, "--runs")
    if args.replicate_kind is None and not (args.audit_manifest or args.audit_runtime_identities):
        raise ValueError("--replicate-kind is required for every launch or dry-run")
    cells, requests = load_design(
        selected_cells=selected_cells,
        selected_supports=selected_supports,
        selected_runs=selected_runs,
        selected_replicate_kind=args.replicate_kind,
    )
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    max_concurrent = min(args.max_concurrent, len(requests))
    logger.info(
        "Prepared %d dense support-surface runs across cells=%s and supports=%s",
        len(requests),
        sorted(cells),
        sorted({request.support_id for request in requests}),
    )
    if args.audit_runtime_identities:
        pass
    elif args.audit_manifest:
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    _validate_starcoder_source(args.marin_prefix, requests)
    audit_runtime_sequence_counts()
    if args.audit_runtime_identities:
        coverage_cells, coverage_requests = load_design(
            selected_cells=selected_cells,
            selected_supports=selected_supports,
            selected_runs=selected_runs,
            selected_replicate_kind="coverage",
        )
        coverage_steps = build_training_steps(
            name_prefix=args.name_prefix,
            tpu_type=args.tpu_type,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
            selected_cells=frozenset(coverage_cells),
            selected_supports=selected_supports,
            selected_runs=selected_runs,
            selected_replicate_kind="coverage",
        )
        audit_materialized_runtime_configs(coverage_requests, coverage_steps, marin_prefix=args.marin_prefix)
        return
    if args.replicate_kind == "calibration_repeat":
        _require_complete_coverage(
            name_prefix=args.name_prefix,
            tpu_type=args.tpu_type,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
        )
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_cells=selected_cells,
        selected_supports=selected_supports,
        selected_runs=selected_runs,
        selected_replicate_kind=args.replicate_kind,
    )
    audit_materialized_runtime_configs(requests, steps, marin_prefix=args.marin_prefix)
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
