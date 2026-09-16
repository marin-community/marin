# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch paired fresh-seed confirmations of dense-support empirical optima."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
from fray.types import ResourceConfig
from levanter.data.mixture import MixtureDataset
from marin.execution.lazy import ArtifactStep, lower, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as source
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_empirical_optimum_confirmation_20260811"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_dense_support_opt_confirm_20260811"
PANEL_TAG = "dense_support_empirical_optimum_confirmation"
MAX_WANDB_TAG_LENGTH = 64
DESIGN_VERSION = "2026-08-11-v1"
DESIGN_PATH = Path(__file__).with_name(
    "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811.json.gz"
)
EXPECTED_DESIGN_SHA256 = "ea116688ba7b0fa38713b5e616fb560f7708e2d385e782b757fc745674beecec"
EXPECTED_RUN_COUNT = 280
EXPECTED_BLOCK_COUNT = 28
EXPECTED_SEED_COUNT = 5
POLICY_CLASSES = frozenset({"tied", "untied"})
DEFAULT_MAX_CONCURRENT = 128


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_payload() -> dict[str, Any]:
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version in {DESIGN_PATH}")
    claimed = payload.pop("design_sha256", None)
    observed = _canonical_sha256(payload)
    if claimed != EXPECTED_DESIGN_SHA256 or observed != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design self-hash mismatch: {observed} != {claimed}")
    payload["design_sha256"] = claimed
    return payload


def load_design(
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, source.Cell], tuple[source.SurfaceRun, ...]]:
    """Load and structurally audit the frozen paired-confirmation manifest."""
    payload = _load_payload()
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected confirmation run count")
    if payload.get("block_count") != EXPECTED_BLOCK_COUNT:
        raise ValueError("Unexpected confirmation block count")
    if payload.get("source_design_sha256") != source.EXPECTED_DESIGN_SHA256:
        raise ValueError("Confirmation source design differs from the dense coverage design")
    if payload.get("training_environment") != source.EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError("Confirmation training environment drifted")
    if payload.get("runtime_cache_contract") != source.EXPECTED_RUNTIME_CACHE_CONTRACT:
        raise ValueError("Confirmation cache contract drifted")

    cells = {row["cell_id"]: source.Cell(**row) for row in payload["cells"]}
    run_fields = source.SurfaceRun.__dataclass_fields__.keys()
    requests = tuple(source.SurfaceRun(**{field: row[field] for field in run_fields}) for row in payload["runs"])
    metadata = {row["run_name"]: row for row in payload["runs"]}
    if len(cells) != 4 or len(requests) != EXPECTED_RUN_COUNT:
        raise ValueError("Confirmation cells or runs are incomplete")
    if len(metadata) != len(requests):
        raise ValueError("Confirmation run names are not unique")

    fresh_seeds = {int(seed) for seed in payload["fresh_seeds"]}
    if len(fresh_seeds) != EXPECTED_SEED_COUNT or payload["discovery_seed"] in fresh_seeds:
        raise ValueError("Confirmation seed block is invalid")
    blocks = {(request.cell_id, request.support_id) for request in requests}
    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise ValueError("Confirmation blocks are incomplete")
    for block in blocks:
        rows = [request for request in requests if (request.cell_id, request.support_id) == block]
        if len(rows) != EXPECTED_SEED_COUNT * len(POLICY_CLASSES):
            raise ValueError(f"{block}: incomplete paired confirmation block")
        for seed in fresh_seeds:
            pair = [request for request in rows if request.data_seed == seed]
            classes = {str(metadata[request.run_name]["policy_class"]) for request in pair}
            if len(pair) != 2 or classes != POLICY_CLASSES:
                raise ValueError(f"{block}, seed {seed}: incomplete policy-class pair")
            if {int(metadata[request.run_name]["pair_seed"]) for request in pair} != {seed}:
                raise ValueError(f"{block}, seed {seed}: pair seed drifted")
        for request in rows:
            cell = cells[request.cell_id]
            row = metadata[request.run_name]
            policy_class = str(row["policy_class"])
            contrast = abs(request.phase_1_starcoder - request.phase_0_starcoder)
            if request.total_steps != cell.total_steps or request.boundary_step != cell.boundary_step:
                raise ValueError(f"{request.run_name}: schedule differs from its source cell")
            if policy_class == "tied" and contrast > 1e-12:
                raise ValueError(f"{request.run_name}: tied winner is not tied")
            if policy_class == "untied" and contrast < float(payload["minimum_untied_absolute_contrast"]) - 1e-12:
                raise ValueError(f"{request.run_name}: untied winner violates the contrast threshold")
            if request.support_id != "full" and not request.starcoder_support_wraps:
                raise ValueError(f"{request.run_name}: finite-support winner unexpectedly does not wrap")

    if selected_runs is not None:
        unknown = selected_runs - set(metadata)
        if unknown:
            raise ValueError(f"Unknown confirmation runs: {sorted(unknown)}")
        requests = tuple(request for request in requests if request.run_name in selected_runs)
        selected_cells = {request.cell_id for request in requests}
        cells = {cell_id: cell for cell_id, cell in cells.items() if cell_id in selected_cells}
    if not requests:
        raise ValueError("Launch filters selected no confirmation runs")
    return cells, requests


def _audit_runtime_sequence_counts(requests: tuple[source.SurfaceRun, ...]) -> int:
    """Recompute exact policy sequence counts in the runtime JAX environment."""
    component_names = [f"nemotron_{index}" for index in range(len(base.NEMOTRON_TOKEN_COUNTS))]
    component_names.append("starcoder")
    datasets = {name: object() for name in component_names}
    total_nemotron_tokens = sum(base.NEMOTRON_TOKEN_COUNTS.values())

    for request in requests:
        phase_weights = []
        for starcoder_weight in (request.phase_0_starcoder, request.phase_1_starcoder):
            broad_weight = 1.0 - starcoder_weight
            broad_weights = [
                broad_weight * token_count / total_nemotron_tokens for token_count in base.NEMOTRON_TOKEN_COUNTS.values()
            ]
            phase_weights.append((*broad_weights, starcoder_weight))
        weights = [
            (0, dict(zip(component_names, phase_weights[0], strict=True))),
            (
                request.boundary_step * base.BATCH_SIZE,
                dict(zip(component_names, phase_weights[1], strict=True)),
            ),
        ]
        mix_key, _ = jax.random.split(jax.random.PRNGKey(request.data_seed))
        mixture_dataset = MixtureDataset(
            datasets,
            weights,
            block_size=base.MIXTURE_BLOCK_SIZE,
            key=mix_key,
        )
        starcoder_index = mixture_dataset.dataset_index.get_index("starcoder")
        boundary_sequences = request.boundary_step * base.BATCH_SIZE
        total_sequences = request.total_steps * base.BATCH_SIZE
        observed = [0, 0]
        for phase, (start, stop) in enumerate(((0, boundary_sequences), (boundary_sequences, total_sequences))):
            first_block, first_offset = divmod(start, base.MIXTURE_BLOCK_SIZE)
            if first_offset != 0:
                raise ValueError(f"{request.run_name}: phase boundary is not mixture-block aligned")
            full_blocks, remainder = divmod(stop - start, base.MIXTURE_BLOCK_SIZE)
            observed[phase] += full_blocks * int(mixture_dataset._counts_per_block_per_stage[phase][starcoder_index])
            if remainder:
                block = mixture_dataset._get_block(first_block + full_blocks)
                observed[phase] += int(np.count_nonzero((block[:remainder] >> 16) == starcoder_index))
        expected = [request.starcoder_phase_0_sequences, request.starcoder_phase_1_sequences]
        if observed != expected:
            raise ValueError(f"{request.run_name}: runtime sequence counts {observed} != {expected}")
    logger.info("Reproduced all %d confirmation sequence identities", len(requests))
    return len(requests)


def _wandb_tags(request: source.SurfaceRun, policy_class: str) -> tuple[str, ...]:
    tags = (
        WANDB_EXPERIMENT_TAG,
        PANEL_TAG,
        request.cell_slug,
        request.support_id,
        policy_class,
        "empirical_optimum_confirmation",
        "starcoder",
        "wsd80_20",
    )
    invalid = [tag for tag in tags if not 1 <= len(tag) <= MAX_WANDB_TAG_LENGTH]
    if invalid:
        raise ValueError(f"W&B tags must contain 1-{MAX_WANDB_TAG_LENGTH} characters: {invalid}")
    return tags


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_runs: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for the selected confirmation rows."""
    cells, requests = load_design(selected_runs)
    model = source._validate_runtime_model(cells)
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles: tuple[ArtifactStep[TokenizedCache], ...] = (
        *tuple(nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS),
        starcoder,
    )
    validation_handles = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    metadata = {row["run_name"]: row for row in _load_payload()["runs"]}
    for request in requests:
        phase_0_weights = base._phase_leaf_weights(request.phase_0_starcoder, nemotron=nemotron, starcoder=starcoder)
        phase_1_weights = base._phase_leaf_weights(request.phase_1_starcoder, nemotron=nemotron, starcoder=starcoder)
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        policy_class = str(metadata[request.run_name]["policy_class"])
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
            tags=_wandb_tags(request, policy_class),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            source._with_source_support(
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


def _parse_runs(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    runs = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not runs:
        raise argparse.ArgumentTypeError("--runs must contain at least one exact run name")
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--runs", help="Comma-separated exact run names for an idempotent partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--audit-runtime-identities", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping dense-support empirical-optimum confirmation in CI")
        return
    if args.name_prefix != NAME:
        raise ValueError(f"Confirmation checkpoint identity is frozen: {args.name_prefix!r} != {NAME!r}")
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError("StarCoder child placement must remain us-central1-a")
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")

    source._validate_runtime_scientific_environment()
    selected_runs = _parse_runs(args.runs)
    cells, requests = load_design(selected_runs)
    max_concurrent = min(args.max_concurrent, len(requests))
    logger.info("Prepared %d confirmation arms across %d cells", len(requests), len(cells))
    if args.audit_manifest:
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    source._validate_starcoder_source(args.marin_prefix, requests)
    _audit_runtime_sequence_counts(requests)
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_runs=selected_runs,
    )
    source.audit_materialized_runtime_configs(requests, steps, marin_prefix=args.marin_prefix)
    if args.audit_runtime_identities:
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d confirmation handles", len(steps))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
