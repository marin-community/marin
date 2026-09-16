# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan or explicitly submit the matched-epoch StarCoder proxy experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import re
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import fsspec
import jax
import numpy as np
from fray.types import ResourceConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.artifact import FingerprintMismatchError, read_record
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.experiment.data import mixture
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import audit_starcoder_epoch_matching_indices as index_audit
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as historical
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix import starcoder_epoch_matching as design_module
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_epoch_matching_20260908"
VERSION = "2026.09.08"
TOTAL_PARAMETERS = 210_052_480
NON_EMBEDDING_PARAMETERS = 45_884_800


def _with_epoch_support(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    datasets: dict[ArtifactStep[TokenizedCache], float],
    validation: tuple[ArtifactStep[TokenizedCache], ...],
    weights: dict[str, float],
    request: design_module.RunSpec,
    component_shuffle_keys: dict[str, tuple[int, int]],
) -> ArtifactStep[LevanterCheckpoint]:
    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = training.build_config(ctx)
        data = mixture(ctx, datasets, validation=validation)
        if set(weights) != set(component_shuffle_keys) or not set(weights).issubset(data.components):
            raise ValueError(f"{request.run_name}: component shuffle-key coverage drifted")
        data = replace(
            data,
            train_weights=[(0, weights), (request.boundary_step, weights)],
            mixture_block_size=design_module.BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches={"dolma/starcoder": request.starcoder_support_batches},
            train_component_shuffle_keys=component_shuffle_keys,
        )
        trainer = replace(
            pod.train_config.trainer,
            seed=request.trainer_seed,
            checkpointer=replace(pod.train_config.trainer.checkpointer, keep=None),
        )
        return replace(
            pod,
            train_config=replace(pod.train_config, data=data, data_seed=request.data_seed, trainer=trainer),
        )

    return replace(training, build_config=build_config)


def build_training_steps(
    design: design_module.ExperimentDesign, stage: str = "pilot"
) -> tuple[tuple[design_module.RunSpec, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build immutable training identities without reading caches or submitting jobs."""
    requests = design_module.select_runs(design, stage)
    model = CompletedAdamHHeuristic()._build_model_config(640, seq_len=design_module.SEQ_LEN)
    if (
        model.total_trainable_params(llama3_tokenizer_vocab_size) != TOTAL_PARAMETERS
        or model.total_trainable_params(0) != NON_EMBEDDING_PARAMETERS
    ):
        raise ValueError("Historical StarCoder model geometry drifted")
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    handles = (*tuple(nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS), starcoder)
    if tuple(handle.name for handle in handles) != design.component_order:
        raise ValueError("Training component order differs from the frozen shuffle-key contract")
    validation = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(
        base.DEFAULT_TPU_TYPE, regions=(base.DEFAULT_TPU_REGION,), zone=base.DEFAULT_TPU_ZONE
    )
    steps = []
    for request in requests:
        weights = base._phase_leaf_weights(request.starcoder_weight, nemotron=nemotron, starcoder=starcoder)
        datasets = {handle: weights[handle.name] for handle in handles}
        schedule = base._schedule_summary(request.materialized_tokens)
        if schedule["total_steps"] != request.total_steps or schedule["boundary_step"] != request.boundary_step:
            raise ValueError(f"{request.run_name}: optimizer schedule differs from the frozen design")
        training = train_lm(
            name=f"checkpoints/{NAME}/{request.run_name}",
            version=VERSION,
            model=model,
            optimizer=base._optimizer(request.materialized_tokens),
            datasets=datasets,
            validation=validation,
            batch_size=design_module.BATCH_SIZE,
            seq_len=design_module.SEQ_LEN,
            num_train_steps=request.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=NAME,
            run_id=request.run_name,
            tags=("starcoder_epoch_matching_20260908", "starcoder", "wsd80_20", request.arm, request.coordinate_id),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        training = _with_epoch_support(
            training,
            datasets=datasets,
            validation=validation,
            weights=weights,
            request=request,
            component_shuffle_keys=design.component_shuffle_keys,
        )
        steps.append(replace(training, expected_fingerprint=training.fingerprint()))
    return requests, tuple(steps)


def pending_training_steps(
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...], *, marin_prefix: str
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Reuse only successful artifacts with matching recorded config fingerprints."""
    pending = []
    for step in steps:
        path = step.path(marin_prefix)
        status = StatusFile(path, worker_id="epoch-matching-resume-check").status
        record = read_record(path)
        if record is not None and record.fingerprint != step.fingerprint():
            raise FingerprintMismatchError(f"{path}: recorded training config differs from the requested experiment")
        if status == STATUS_SUCCESS:
            if record is None:
                raise RuntimeError(f"{path}: successful output has no artifact fingerprint; cannot safely reuse it")
            continue
        pending.append(step)
    return tuple(pending)


def require_previous_stage_complete(design: design_module.ExperimentDesign, stage: str, *, marin_prefix: str) -> None:
    previous_stage = {"refinement": "pilot", "primary": "pilot", "replicated": "primary"}.get(stage)
    if previous_stage is None:
        return
    _, previous_steps = build_training_steps(design, previous_stage)
    incomplete = pending_training_steps(previous_steps, marin_prefix=marin_prefix)
    if incomplete:
        raise RuntimeError(
            f"Cannot submit {stage}: {len(incomplete)} of {len(previous_steps)} "
            f"{previous_stage} artifacts remain incomplete"
        )


def audit_runtime_configs(
    design: design_module.ExperimentDesign,
    requests: tuple[design_module.RunSpec, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
) -> None:
    """Check resolved paths, support, shuffle keys, and independent trainer seeds."""
    artifact_cache: dict[int, Any] = {}
    for request, step in zip(requests, steps, strict=True):
        pod = materialized_config(step, marin_prefix, artifact_cache=artifact_cache)
        if not isinstance(pod, TrainLmOnPodConfig) or not isinstance(pod.train_config, TrainLmConfig):
            raise TypeError(f"{request.run_name}: unexpected training config")
        config = pod.train_config
        data = config.data
        if data.max_train_batches != {"dolma/starcoder": request.starcoder_support_batches}:
            raise ValueError(f"{request.run_name}: source support cap drifted")
        if data.train_component_shuffle_keys != design.component_shuffle_keys:
            raise ValueError(f"{request.run_name}: frozen component permutations drifted")
        if data.max_train_batches_subset_seed is not None:
            raise ValueError(f"{request.run_name}: an independent subset permutation is active")
        if data.shuffle != BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"):
            raise ValueError(f"{request.run_name}: legacy block-shuffle geometry drifted")
        if config.data_seed != request.data_seed or config.trainer.seed != request.trainer_seed:
            raise ValueError(f"{request.run_name}: data/trainer seeds drifted")
        if any(
            value is not None for value in (data.experiment_budget, data.target_budget, data.simulated_epoch_subset_seed)
        ):
            raise ValueError(f"{request.run_name}: global simulated-epoch slicing is active")
        if not isinstance(data.train_weights, list) or [boundary for boundary, _ in data.train_weights] != [
            0,
            request.boundary_step,
        ]:
            raise ValueError(f"{request.run_name}: mixture schedule drifted")
        for _boundary, weights in data.train_weights:
            if weights["dolma/starcoder"] != request.starcoder_weight:
                raise ValueError(f"{request.run_name}: StarCoder weight drifted")
        for component in data.components.values():
            cache_dir = component.cache_dir
            if cache_dir is not None and not cache_dir.startswith(f"{base.DEFAULT_MARIN_PREFIX}/"):
                raise ValueError(f"{request.run_name}: non-central1 training cache {cache_dir}")
        if config.optimizer.decay != request.total_steps - request.boundary_step:
            raise ValueError(f"{request.run_name}: WSD decay interval drifted")


def observed_runtime_environment() -> dict[str, str | bool]:
    """Describe the process performing the launch audit, without identifying a child runtime."""
    return {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def intended_child_runtime(design: design_module.ExperimentDesign) -> dict[str, Any]:
    """Bind intended child dependencies to the repository lock, not the historical version."""
    lock_path = design_module.REPO_ROOT / "uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    versions = {}
    for name in ("jax", "jaxlib", "numpy"):
        candidates = {package["version"] for package in lock["package"] if package["name"] == name}
        if len(candidates) != 1:
            raise ValueError(f"Expected one {name} version in the child runtime lock")
        versions[name] = candidates.pop()
    for name in ("jax", "numpy"):
        if design.training_environment[f"{name}_version"] != versions[name]:
            raise ValueError(f"Intended child {name} version disagrees with the frozen design")
    return {
        "environment": design.training_environment,
        "locked_packages": versions,
        "uv_lock_sha256": design_module.file_sha256(lock_path),
        "source": "repository uv.lock bundled with the child job",
        "observed_on_child": False,
    }


def validate_runtime_indices(design: design_module.ExperimentDesign, receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute every parent and matched sequence index and compare with the reviewed receipt."""
    if receipt.get("design_sha256") != design.design_sha256 or receipt.get("status") != "passed":
        raise ValueError("Runtime index verification requires a passed receipt for this design")
    if any(receipt.get(field) is not True for field in ("legacy_parent_match", "nested_subset_match")):
        raise ValueError("Runtime index verification requires both reviewed source-identity checks")
    count = receipt["packed_starcoder_sequence_count"]
    with tempfile.TemporaryDirectory(prefix="starcoder-runtime-indices-") as directory:
        result = asyncio.run(index_audit.audit_indices(count, Path(directory), design=design))
    for field in (
        "design_sha256",
        "packed_starcoder_sequence_count",
        "parent_indices_sha256",
        "matched_indices_sha256",
        "legacy_parent_match",
        "nested_subset_match",
    ):
        if result[field] != receipt[field]:
            raise ValueError(f"Runtime source-index audit differs from the reviewed receipt: {field}")
    return {**result, "status": "passed", "verified_process": "parent", "child_process_verified": False}


def validate_reuse_audit(design: design_module.ExperimentDesign, receipt_path: Path | None) -> dict[str, Any]:
    """Require reviewed historical-config and sequence-identity evidence before training."""
    if receipt_path is None:
        raise ValueError("--submit requires --reuse-audit with a passed historical-target reuse receipt")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("Reuse audit receipt must be a JSON object")
    if receipt.get("design_sha256") != design.design_sha256:
        raise ValueError("Reuse audit receipt is bound to a different experiment design")
    if receipt.get("status") != "passed":
        raise ValueError("Historical-target reuse audit has not passed")
    for field in ("legacy_parent_match", "nested_subset_match"):
        if receipt.get(field) is not True:
            raise ValueError(f"Reuse audit does not establish {field}")
    uri = receipt.get("historical_config_uri")
    prefix = f"{base.DEFAULT_MARIN_PREFIX}/"
    if not isinstance(uri, str) or not uri.startswith(prefix) or not uri.removeprefix(prefix).strip():
        raise ValueError("Reuse audit needs a historical config URI in the pinned central1 bucket")
    for field in ("historical_config_sha256", "parent_indices_sha256", "matched_indices_sha256"):
        digest = receipt.get(field)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Reuse audit needs a SHA-256 digest for {field}")
    sequence_count = receipt.get("packed_starcoder_sequence_count")
    if type(sequence_count) is not int or sequence_count < design_module.PARENT_BATCHES * design_module.BATCH_SIZE:
        raise ValueError("Reuse audit packed StarCoder sequence count cannot cover the frozen parent")
    return receipt


def collect_results(
    design: design_module.ExperimentDesign,
    requests: tuple[design_module.RunSpec, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
    output_path: Path,
) -> int:
    """Export verified final-checkpoint metrics from durable per-evaluation records."""
    rows = []
    for request, step in zip(requests, steps, strict=True):
        if pending_training_steps((step,), marin_prefix=marin_prefix):
            raise RuntimeError(f"{request.run_name}: training has not completed successfully")
        artifact_path = step.path(marin_prefix)
        metrics_path = f"{LevanterCheckpoint(path=artifact_path).checkpoint_dir}/eval_metrics.jsonl"
        expected_step = request.total_steps - 1
        values = []
        with fsspec.open(metrics_path, "rt") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("step") != expected_step or design.primary_metric not in record:
                    continue
                value = float(record[design.primary_metric])
                if not math.isfinite(value):
                    raise ValueError(f"{request.run_name}: nonfinite final evaluation")
                values.append(value)
        if not values:
            raise ValueError(f"{request.run_name}: no {design.primary_metric} at final step {expected_step}")
        if len(set(values)) != 1:
            raise ValueError(f"{request.run_name}: conflicting evaluations at final step {expected_step}")
        rows.append(
            {
                "run_name": request.run_name,
                "step": expected_step,
                "metric": design.primary_metric,
                "value": values[0],
                "design_sha256": design.design_sha256,
                "config_fingerprint": step.fingerprint(),
                "status": "succeeded",
            }
        )
    if not rows:
        raise ValueError("No selected training runs to collect")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)
    return len(rows)


def persist_submission_plan(plan: dict[str, Any], path: str) -> None:
    """Create the complete reviewed submission plan once, refusing changed contents."""
    encoded = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(plain_path, "rb") as handle:
            if handle.read() != encoded:
                raise ValueError(f"Existing submission plan differs; refusing to overwrite {path}") from error
    with fs.open(plain_path, "rb") as handle:
        if handle.read() != encoded:
            raise RuntimeError(f"Submission plan did not persist exactly: {path}")


def launch_plan(
    design: design_module.ExperimentDesign,
    stage: str,
    requests: tuple[design_module.RunSpec, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
) -> dict[str, Any]:
    plan = {
        "design_sha256": design.design_sha256,
        "stage": stage,
        "submission_plan_uri": (
            f"{base.DEFAULT_MARIN_PREFIX}/experiments/starcoder_epoch_matching_20260908/"
            f"{design.design_sha256}/{stage}/launch_plan.json"
        ),
        "metric": design.primary_metric,
        "marin_prefix": base.DEFAULT_MARIN_PREFIX,
        "tpu_region": base.DEFAULT_TPU_REGION,
        "tpu_zone": base.DEFAULT_TPU_ZONE,
        "observed_parent_runtime": observed_runtime_environment(),
        "intended_child_runtime": intended_child_runtime(design),
        "new_training_runs": len(requests),
        "runs": [
            {**asdict(request), "output_path": step.path(base.DEFAULT_MARIN_PREFIX), "fingerprint": step.fingerprint()}
            for request, step in zip(requests, steps, strict=True)
        ],
        "target_observations": [asdict(observation) for observation in design.target_observations],
    }

    if stage == "refinement":
        plan["adaptive_release"] = {
            "based_on_stage": "pilot",
            "additional_starcoder_weights": list(design_module.REFINEMENT_WEIGHTS),
            "trainer_seed": design_module.REFERENCE_SEED,
        }
    return plan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=design_module.DESIGN_PATH)
    parser.add_argument("--stage", choices=design_module.STAGES, default="pilot")
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--audit-runtime", action="store_true")
    parser.add_argument(
        "--reuse-audit", type=Path, help="Passed historical-target reuse receipt required for submission"
    )
    parser.add_argument("--collect-results", type=Path, metavar="CSV", help="Export verified durable final evaluations")
    parser.add_argument("--submit", action="store_true", help="Submit training; otherwise only emit the plan")
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if (args.marin_prefix, args.tpu_region, args.tpu_zone) != (
        base.DEFAULT_MARIN_PREFIX,
        base.DEFAULT_TPU_REGION,
        base.DEFAULT_TPU_ZONE,
    ):
        raise ValueError("This historical StarCoder experiment is frozen to us-central1/us-central1-a")
    configured_prefix = os.environ.get("MARIN_PREFIX")
    if args.submit and configured_prefix not in (None, args.marin_prefix):
        raise ValueError(f"MARIN_PREFIX conflicts with the central1 experiment: {configured_prefix!r}")
    design = design_module.load_design(args.design)
    receipt = validate_reuse_audit(design, args.reuse_audit) if args.submit or args.reuse_audit is not None else None
    requests, steps = build_training_steps(design, args.stage)
    for step in steps:
        lower(step)
    plan = launch_plan(design, args.stage, requests, steps)
    if receipt is not None:
        plan["reuse_audit"] = receipt
        plan["historical_training_environment"] = receipt.get("audit_runtime")
    if args.audit_runtime or args.submit:
        if receipt is not None:
            plan["runtime_index_audit"] = validate_runtime_indices(design, receipt)
        if args.submit:
            require_previous_stage_complete(design, args.stage, marin_prefix=args.marin_prefix)
        historical._validate_starcoder_source(args.marin_prefix, ())
        audit_runtime_configs(design, requests, steps, marin_prefix=args.marin_prefix)
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.plan_path is not None:
        args.plan_path.parent.mkdir(parents=True, exist_ok=True)
        args.plan_path.write_text(encoded)
    else:
        print(encoded, end="")
    if args.submit:
        persist_submission_plan(plan, plan["submission_plan_uri"])
        os.environ["MARIN_PREFIX"] = args.marin_prefix
        pending = pending_training_steps(steps, marin_prefix=args.marin_prefix)
        logger.info("Selected %d training runs; submitting %d incomplete artifacts", len(steps), len(pending))
        if pending:
            run(*pending, max_concurrent=len(pending), force_run_failed=True)
        incomplete = pending_training_steps(steps, marin_prefix=args.marin_prefix)
        if incomplete:
            raise RuntimeError(f"{len(incomplete)} selected training artifacts did not complete successfully")
        logger.info("All %d selected training artifacts are complete", len(steps))
    if args.collect_results is not None:
        count = collect_results(
            design, requests, steps, marin_prefix=args.marin_prefix, output_path=args.collect_results
        )
        logger.info("Collected %d verified final evaluations into %s", count, args.collect_results)


if __name__ == "__main__":
    main()
