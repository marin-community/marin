# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch and summarize paired MRCR checkpoint evaluations.

The checked-in launch surface describes the complete scientific matrix, but
does not submit it.  Use ``SMOKE_EVALUATION_KEYS`` and ``SMOKE_CONTEXT_CAPS``
for the bounded engineering smoke before selecting the full matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from experiments.datasets.mrcr import (
    MRCR_BOOTSTRAP_SAMPLES,
    MRCR_CONTEXT_CAPS,
    MrcrDatasetBundle,
    MrcrPromptVariant,
    mrcr_datasets,
)
from experiments.grug.moe.evaluate import (
    GrugCheckpointEvalConfig,
    GrugCheckpointEvalRuntimeConfig,
    dispatch_grug_checkpoint_eval,
)
from experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon import (
    step as QK175_TRAINING_STEP,
)
from experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk157 import (
    step as QK157_TRAINING_STEP,
)

SOURCE_STEP = 156_000
FINAL_STEP = 157_000
PRE_EXTENSION_STEP = 141_000
TRAINING_OFFSETS = (250, 500, 750, 1_000)
SENSITIVITY_CONTEXT_CAPS = (8_192, 32_768)
AGGREGATE_CONTEXT_CAP = 262_144
CONTEXT_PARALLEL_TPU_SHAPES = ("v4-32-cp4-fsdp", "v4-64-cp2", "v4-64-cp8-ep4", "v4-128-cp4")


@dataclass(frozen=True)
class MrcrCheckpointPackage:
    """A checkpoint and its inference configuration in the evaluation matrix."""

    name: str
    qk_mult: float
    checkpoint_step: int
    training_offset: int
    baseline_package_name: str | None


SOURCE_QK157 = MrcrCheckpointPackage("step-156000-source-qk157", 1.57, SOURCE_STEP, 0, None)
SOURCE_QK175 = MrcrCheckpointPackage("step-156000-source-qk175", 1.75, SOURCE_STEP, 0, None)
PRE_EXTENSION_QK157 = MrcrCheckpointPackage(
    "step-141000-pre-extension-qk157",
    1.57,
    PRE_EXTENSION_STEP,
    PRE_EXTENSION_STEP - SOURCE_STEP,
    None,
)

PRIMARY_CHECKPOINT_PACKAGES: tuple[MrcrCheckpointPackage, ...] = (
    SOURCE_QK157,
    SOURCE_QK175,
    *(
        MrcrCheckpointPackage(
            f"qk157-step{SOURCE_STEP + offset}",
            1.57,
            SOURCE_STEP + offset,
            offset,
            SOURCE_QK157.name,
        )
        for offset in TRAINING_OFFSETS
    ),
    *(
        MrcrCheckpointPackage(
            f"qk175-step{SOURCE_STEP + offset}",
            1.75,
            SOURCE_STEP + offset,
            offset,
            SOURCE_QK175.name,
        )
        for offset in TRAINING_OFFSETS
    ),
)
PACKAGE_BY_NAME = {package.name: package for package in (*PRIMARY_CHECKPOINT_PACKAGES, PRE_EXTENSION_QK157)}
FINAL_PACKAGE_NAMES = ("qk157-step157000", "qk175-step157000")
SENSITIVITY_PACKAGE_NAMES = (SOURCE_QK157.name, SOURCE_QK175.name, *FINAL_PACKAGE_NAMES)


@dataclass(frozen=True)
class MrcrEvaluationKey:
    package_name: str
    prompt_variant: MrcrPromptVariant


@dataclass(frozen=True)
class MrcrEvaluationArtifact:
    package_name: str
    prompt_variant: MrcrPromptVariant
    context_cap: int
    qk_mult: float
    training_offset: int
    baseline_package_name: str | None
    example_losses_path: str


PRIMARY_EVALUATION_KEYS = tuple(
    MrcrEvaluationKey(package.name, MrcrPromptVariant.TWO_SHOT) for package in PRIMARY_CHECKPOINT_PACKAGES
)
SMOKE_EVALUATION_KEYS = (
    MrcrEvaluationKey(SOURCE_QK157.name, MrcrPromptVariant.TWO_SHOT),
    MrcrEvaluationKey("qk175-step157000", MrcrPromptVariant.TWO_SHOT),
)
SENSITIVITY_EVALUATION_KEYS = tuple(
    MrcrEvaluationKey(package_name, variant)
    for package_name in SENSITIVITY_PACKAGE_NAMES
    for variant in MrcrPromptVariant
)
SMOKE_CONTEXT_CAPS = SENSITIVITY_CONTEXT_CAPS
DEFAULT_TPU_VARIANT = "v4-2048"
CANONICAL_CONTEXT_MODEL = QK175_TRAINING_STEP.config.model.value
SOURCE_CHECKPOINT = InputName.hardcoded(
    "grug/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k-a30ef8/checkpoints/step-156000"
)
PRE_EXTENSION_CHECKPOINT = InputName.hardcoded(
    "grug/moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T-9fcc1f/checkpoints/step-141000"
)


@dataclass(frozen=True)
class MrcrEvaluationCell:
    """One independently dispatchable checkpoint/cap/prompt evaluation."""

    package: MrcrCheckpointPackage
    context_cap: int
    prompt_variant: MrcrPromptVariant

    @property
    def run_id(self) -> str:
        qk_name = str(self.package.qk_mult).replace(".", "")
        return f"mrcr-67b-step{self.package.checkpoint_step}-qk{qk_name}-cap{self.context_cap}-{self.prompt_variant}"


def primary_evaluation_cells() -> tuple[MrcrEvaluationCell, ...]:
    """Return the complete 10-package by 6-cap primary matrix."""

    return tuple(
        MrcrEvaluationCell(package, cap, MrcrPromptVariant.TWO_SHOT)
        for package in PRIMARY_CHECKPOINT_PACKAGES
        for cap in MRCR_CONTEXT_CAPS
    )


def sensitivity_evaluation_cells() -> tuple[MrcrEvaluationCell, ...]:
    """Return the 16 additional one-shot and no-prefix sensitivity cells."""

    return tuple(
        MrcrEvaluationCell(PACKAGE_BY_NAME[package_name], cap, variant)
        for package_name in SENSITIVITY_PACKAGE_NAMES
        for cap in SENSITIVITY_CONTEXT_CAPS
        for variant in (MrcrPromptVariant.ONE_SHOT, MrcrPromptVariant.TWO_SHOT_NO_PREFIX)
    )


def smoke_evaluation_cells() -> tuple[MrcrEvaluationCell, ...]:
    """Return the bounded four-cell source/final engineering smoke."""

    return tuple(
        MrcrEvaluationCell(PACKAGE_BY_NAME[key.package_name], cap, key.prompt_variant)
        for key in SMOKE_EVALUATION_KEYS
        for cap in SMOKE_CONTEXT_CAPS
    )


def aggregate_262k_evaluation_cells() -> tuple[MrcrEvaluationCell, ...]:
    """Return the matched-qk pre-extension, 64K, and 262K training trajectory."""

    packages = (PRE_EXTENSION_QK157, SOURCE_QK157, PACKAGE_BY_NAME["qk157-step157000"])
    return tuple(MrcrEvaluationCell(package, AGGREGATE_CONTEXT_CAP, MrcrPromptVariant.TWO_SHOT) for package in packages)


def evaluation_cells(selection: str) -> tuple[MrcrEvaluationCell, ...]:
    """Select an explicit launch matrix without silently expanding the smoke."""

    if selection == "smoke":
        return smoke_evaluation_cells()
    if selection == "aggregate_262k":
        return aggregate_262k_evaluation_cells()
    if selection == "aggregate_262k_probe":
        return aggregate_262k_evaluation_cells()[:1]
    if selection == "aggregate_262k_extension":
        return aggregate_262k_evaluation_cells()[1:]
    if selection == "aggregate_262k_deployable_final":
        package = PACKAGE_BY_NAME["qk175-step157000"]
        return (MrcrEvaluationCell(package, AGGREGATE_CONTEXT_CAP, MrcrPromptVariant.TWO_SHOT),)
    if selection == "primary":
        return primary_evaluation_cells()
    if selection == "sensitivity":
        return sensitivity_evaluation_cells()
    if selection == "complete":
        return (*primary_evaluation_cells(), *sensitivity_evaluation_cells())
    raise ValueError(f"Unknown MRCR evaluation selection: {selection}")


@dataclass(frozen=True)
class _MrcrDispatchStepConfig:
    evaluation: GrugCheckpointEvalConfig
    resources: ResourceConfig


@dataclass(frozen=True)
class _MrcrEvaluationShape:
    tpu_variant: str
    output_suffix: str
    eval_batch_size: int
    data_axis_size: int
    context_axis_size: int
    preemptible: bool
    expert_axis_size: int = 1
    expert_weight_hidden_axis: str = "data"
    ram: str = "4g"
    parameter_sharding_axes: tuple[str, ...] = ("data",)

    @property
    def resources(self) -> ResourceConfig:
        return ResourceConfig.with_tpu(self.tpu_variant, preemptible=self.preemptible, ram=self.ram)


def _evaluation_shape(tpu_variant: str) -> _MrcrEvaluationShape:
    if tpu_variant == DEFAULT_TPU_VARIANT:
        return _MrcrEvaluationShape(
            tpu_variant,
            output_suffix="",
            eval_batch_size=256,
            data_axis_size=256,
            context_axis_size=4,
            preemptible=False,
        )
    if tpu_variant == "v4-64":
        return _MrcrEvaluationShape(
            tpu_variant,
            output_suffix="v464",
            eval_batch_size=32,
            data_axis_size=32,
            context_axis_size=1,
            preemptible=True,
        )
    if tpu_variant == "v4-32":
        return _MrcrEvaluationShape(
            tpu_variant,
            output_suffix="v432",
            eval_batch_size=16,
            data_axis_size=16,
            context_axis_size=1,
            preemptible=True,
        )
    if tpu_variant == "v4-64-cp2":
        return _MrcrEvaluationShape(
            "v4-64",
            output_suffix="v464cp2",
            eval_batch_size=16,
            data_axis_size=16,
            context_axis_size=2,
            preemptible=True,
        )
    if tpu_variant == "v4-64-cp8-ep4":
        return _MrcrEvaluationShape(
            "v4-64",
            output_suffix="v464cp8ep4",
            eval_batch_size=4,
            data_axis_size=1,
            context_axis_size=8,
            expert_axis_size=4,
            expert_weight_hidden_axis="context",
            preemptible=True,
            ram="350g",
        )
    if tpu_variant == "v4-32-cp4-fsdp":
        return _MrcrEvaluationShape(
            "v4-32",
            output_suffix="v432cp4fsdp",
            eval_batch_size=4,
            data_axis_size=4,
            context_axis_size=4,
            preemptible=True,
            parameter_sharding_axes=("data", "context"),
        )
    if tpu_variant == "v4-128-cp4":
        return _MrcrEvaluationShape(
            "v4-128",
            output_suffix="v4128cp4",
            eval_batch_size=16,
            data_axis_size=16,
            context_axis_size=4,
            preemptible=True,
        )
    raise ValueError(f"Unsupported MRCR evaluation TPU shape: {tpu_variant}")


def _dispatch_evaluation(config: _MrcrDispatchStepConfig) -> None:
    dispatch_grug_checkpoint_eval(config.evaluation, resources=config.resources)


def default_checkpoint_paths() -> dict[str, InputName]:
    """Return source and nonblocking extension checkpoint dependencies."""

    paths = {
        PRE_EXTENSION_QK157.name: PRE_EXTENSION_CHECKPOINT,
        SOURCE_QK157.name: SOURCE_CHECKPOINT,
        SOURCE_QK175.name: SOURCE_CHECKPOINT,
    }
    for package in PRIMARY_CHECKPOINT_PACKAGES:
        if package.training_offset == 0:
            continue
        training_step = QK157_TRAINING_STEP if package.qk_mult == 1.57 else QK175_TRAINING_STEP
        paths[package.name] = training_step.cd(f"checkpoints/step-{package.checkpoint_step}").nonblocking()
    return paths


def _cell_data(bundle: MrcrDatasetBundle, cell: MrcrEvaluationCell):
    prefix = f"{cell.prompt_variant}/cap_{cell.context_cap}/"
    components = {key: step for key, step in bundle.datasets.items() if key.startswith(prefix)}
    if not components:
        raise ValueError(f"MRCR bundle has no datasets for {prefix}")
    return lm_mixture_data_config(
        components,
        {},
        shuffle=False,
        include_raw_paths=False,
        block_cross_document_attention=False,
    )


def build_evaluation_steps(
    cells: tuple[MrcrEvaluationCell, ...],
    *,
    bundle: MrcrDatasetBundle,
    checkpoint_paths: dict[str, InputName],
    tpu_variant: str = DEFAULT_TPU_VARIANT,
    bootstrap_samples: int = MRCR_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
) -> tuple[ExecutorStep[_MrcrDispatchStepConfig], ...]:
    """Materialize one current-executor step for every requested matrix cell."""

    shape = _evaluation_shape(tpu_variant)
    steps: list[ExecutorStep[_MrcrDispatchStepConfig]] = []
    for cell in cells:
        checkpoint_path = checkpoint_paths.get(cell.package.name)
        if checkpoint_path is None:
            raise ValueError(f"Missing checkpoint path for {cell.package.name}")
        prefix = f"{cell.prompt_variant}/cap_{cell.context_cap}/"
        manifests = {key: path for key, path in bundle.manifests.items() if key.startswith(prefix)}
        if not manifests:
            raise ValueError(f"MRCR bundle has no manifests for {prefix}")
        model = replace(
            CANONICAL_CONTEXT_MODEL,
            max_seq_len=cell.context_cap,
            qk_mult=cell.package.qk_mult,
            expert_weight_hidden_axis=shape.expert_weight_hidden_axis,
        )
        resource_suffix = f"-{shape.output_suffix}" if shape.output_suffix else ""
        run_id = f"{cell.run_id}{resource_suffix}"
        evaluation = GrugCheckpointEvalConfig(
            run_id=run_id,
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            context_cap=cell.context_cap,
            prompt_variant=cell.prompt_variant,
            qk_mult=cell.package.qk_mult,
            model=versioned(model),  # type: ignore[arg-type]
            data=_cell_data(bundle, cell),
            dataset_stats_path=bundle.stats,  # type: ignore[arg-type]
            dataset_manifest_paths=manifests,  # type: ignore[arg-type]
            runtime=versioned(
                GrugCheckpointEvalRuntimeConfig(
                    mp="params=float32,compute=bfloat16,output=bfloat16",
                    tracker=WandbConfig(
                        project="marin_moe",
                        tags=["mrcr", "base_eval", "paired_likelihood", tpu_variant.replace("-", "_")],
                        group="mrcr-67b-base-context-eval",
                        name=run_id,
                    ),
                    eval_batch_size=shape.eval_batch_size,
                    data_axis_size=shape.data_axis_size,
                    context_axis_size=shape.context_axis_size,
                    expert_axis_size=shape.expert_axis_size,
                    parameter_sharding_axes=shape.parameter_sharding_axes,
                )
            ),  # type: ignore[arg-type]
            output_path=this_output_path(),  # type: ignore[arg-type]
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        steps.append(
            ExecutorStep(
                name=(f"eval/mrcr/{cell.package.name}/{cell.prompt_variant}/cap-{cell.context_cap}{resource_suffix}"),
                fn=_dispatch_evaluation,
                config=_MrcrDispatchStepConfig(evaluation=evaluation, resources=versioned(shape.resources)),
            )
        )
    return tuple(steps)


def build_default_steps(
    selection: str, *, tpu_variant: str = DEFAULT_TPU_VARIANT
) -> tuple[ExecutorStep[_MrcrDispatchStepConfig], ...]:
    """Build the selected checked-in matrix without submitting any other cells."""

    cells = evaluation_cells(selection)
    bounded_262k_selections = (
        "aggregate_262k",
        "aggregate_262k_probe",
        "aggregate_262k_extension",
        "aggregate_262k_deployable_final",
    )
    if selection in bounded_262k_selections and tpu_variant not in CONTEXT_PARALLEL_TPU_SHAPES:
        raise ValueError(f"{selection} requires a context-parallel TPU shape")
    if tpu_variant != DEFAULT_TPU_VARIANT and selection not in ("smoke", *bounded_262k_selections):
        raise ValueError("Noncanonical TPU shapes are limited to the bounded smoke selection")
    variants = tuple(dict.fromkeys(cell.prompt_variant for cell in cells))
    bundle = mrcr_datasets(prompt_variants=variants)
    return build_evaluation_steps(
        cells,
        bundle=bundle,
        checkpoint_paths=default_checkpoint_paths(),
        tpu_variant=tpu_variant,
    )


def expected_evaluations_for_stage(summary_stage: str) -> tuple[MrcrEvaluationKey, ...]:
    """Return the preregistered package set for a matrix-summary stage."""

    if summary_stage == "smoke":
        return SMOKE_EVALUATION_KEYS
    if summary_stage == "source_qk":
        return tuple(key for key in PRIMARY_EVALUATION_KEYS if key.package_name.startswith("step-156000-source"))
    if summary_stage == "prompt_sensitivity":
        return SENSITIVITY_EVALUATION_KEYS
    if summary_stage == "complete":
        return PRIMARY_EVALUATION_KEYS
    if summary_stage.startswith("offset_"):
        offset = int(summary_stage.removeprefix("offset_"))
        if offset not in TRAINING_OFFSETS:
            raise ValueError(f"Unknown MRCR summary stage: {summary_stage}")
        names = {
            SOURCE_QK157.name,
            SOURCE_QK175.name,
            f"qk157-step{SOURCE_STEP + offset}",
            f"qk175-step{SOURCE_STEP + offset}",
        }
        return tuple(key for key in PRIMARY_EVALUATION_KEYS if key.package_name in names)
    raise ValueError(f"Unknown MRCR summary stage: {summary_stage}")


@dataclass(frozen=True)
class _Comparison:
    kind: str
    terms: tuple[tuple[MrcrEvaluationKey, int], ...]
    packages: tuple[str, ...]
    baselines: tuple[str, ...]
    left_prompt_variant: str
    right_prompt_variant: str


def _evaluation_key(artifact: MrcrEvaluationArtifact) -> MrcrEvaluationKey:
    return MrcrEvaluationKey(artifact.package_name, artifact.prompt_variant)


def _read_example_losses(path: str) -> dict[tuple[int, str], dict[str, dict[str, Any]]]:
    slices: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    with fsspec.open(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            source_id = row["source_id"]
            slice_key = (int(row["n_needles"]), str(row["distance_band"]))
            if source_id in slices[slice_key]:
                raise ValueError(f"Duplicate source_id {source_id} in {path}:{line_number}")
            scored_tokens = int(row["scored_tokens"])
            if scored_tokens <= 0:
                raise ValueError(f"Non-positive scored_tokens for {source_id} in {path}")
            slices[slice_key][source_id] = row
    return dict(slices)


def _comparison_specs(expected: set[MrcrEvaluationKey]) -> tuple[_Comparison, ...]:
    comparisons: list[_Comparison] = []
    variants = sorted({key.prompt_variant for key in expected}, key=str)
    for variant in variants:
        variant_keys = {key.package_name: key for key in expected if key.prompt_variant == variant}
        source_157 = variant_keys.get(SOURCE_QK157.name)
        source_175 = variant_keys.get(SOURCE_QK175.name)
        if source_157 is not None and source_175 is not None:
            comparisons.append(
                _Comparison(
                    "source_inference_qk",
                    ((source_157, -1), (source_175, 1)),
                    (SOURCE_QK157.name, SOURCE_QK175.name),
                    (),
                    str(variant),
                    str(variant),
                )
            )
        for package in PRIMARY_CHECKPOINT_PACKAGES:
            if package.baseline_package_name is None or package.name not in variant_keys:
                continue
            baseline = variant_keys.get(package.baseline_package_name)
            if baseline is None:
                continue
            extension = variant_keys[package.name]
            comparisons.append(
                _Comparison(
                    "adaptation",
                    ((baseline, -1), (extension, 1)),
                    (package.name,),
                    (package.baseline_package_name,),
                    str(variant),
                    str(variant),
                )
            )
        for offset in TRAINING_OFFSETS:
            q157 = variant_keys.get(f"qk157-step{SOURCE_STEP + offset}")
            q175 = variant_keys.get(f"qk175-step{SOURCE_STEP + offset}")
            if q157 is None or q175 is None:
                continue
            comparisons.append(
                _Comparison(
                    "deployable_arm",
                    ((q157, -1), (q175, 1)),
                    (q157.package_name, q175.package_name),
                    (),
                    str(variant),
                    str(variant),
                )
            )
            if source_157 is not None and source_175 is not None:
                comparisons.append(
                    _Comparison(
                        "difference_in_differences",
                        ((source_175, -1), (q175, 1), (source_157, 1), (q157, -1)),
                        (q157.package_name, q175.package_name),
                        (SOURCE_QK157.name, SOURCE_QK175.name),
                        str(variant),
                        str(variant),
                    )
                )

    for package_name in sorted({key.package_name for key in expected}):
        primary = MrcrEvaluationKey(package_name, MrcrPromptVariant.TWO_SHOT)
        for kind, variant in (
            ("shot_sensitivity", MrcrPromptVariant.ONE_SHOT),
            ("prefix_sensitivity", MrcrPromptVariant.TWO_SHOT_NO_PREFIX),
        ):
            sensitivity = MrcrEvaluationKey(package_name, variant)
            if primary in expected and sensitivity in expected:
                comparisons.append(
                    _Comparison(
                        kind,
                        ((primary, -1), (sensitivity, 1)),
                        (package_name,),
                        (),
                        str(primary.prompt_variant),
                        str(variant),
                    )
                )
    return tuple(comparisons)


def _metric_values(
    comparison: _Comparison,
    rows: dict[MrcrEvaluationKey, dict[str, dict[str, Any]]],
) -> tuple[float, float, np.ndarray, np.ndarray, tuple[str, ...]]:
    source_ids = tuple(sorted(next(iter(rows.values()))))
    for key, key_rows in rows.items():
        if tuple(sorted(key_rows)) != source_ids:
            raise ValueError(f"Mismatched source IDs in comparison {comparison.kind}: {key}")
    token_counts = np.asarray([int(next(iter(rows.values()))[source_id]["scored_tokens"]) for source_id in source_ids])
    numerator = np.zeros(len(source_ids), dtype=np.float64)
    macro_gain = np.zeros(len(source_ids), dtype=np.float64)
    for key, coefficient in comparison.terms:
        key_rows = rows[key]
        key_tokens = np.asarray([int(key_rows[source_id]["scored_tokens"]) for source_id in source_ids])
        if not np.array_equal(key_tokens, token_counts):
            raise ValueError(f"Mismatched scored-token counts in comparison {comparison.kind}: {key}")
        condition_delta = np.asarray(
            [
                float(key_rows[source_id]["query_only_loss_sum"]) - float(key_rows[source_id]["full_context_loss_sum"])
                for source_id in source_ids
            ]
        )
        numerator += coefficient * condition_delta
        macro_gain += coefficient * condition_delta / token_counts
    return (
        float(numerator.sum() / token_counts.sum()),
        float(macro_gain.mean()),
        numerator,
        token_counts,
        source_ids,
    )


def _bootstrap_interval(
    numerator: np.ndarray,
    token_counts: np.ndarray,
    *,
    macro_values: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
    metric_prefix: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    seed_offset = int.from_bytes(hashlib.sha256(metric_prefix.encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64((bootstrap_seed + seed_offset) % 2**64))
    micro_samples = np.empty(bootstrap_samples, dtype=np.float64)
    macro_samples = np.empty(bootstrap_samples, dtype=np.float64)
    for sample in range(bootstrap_samples):
        indices = rng.integers(0, len(numerator), size=len(numerator))
        micro_samples[sample] = numerator[indices].sum() / token_counts[indices].sum()
        macro_samples[sample] = macro_values[indices].mean()
    micro_percentiles = np.percentile(micro_samples, (2.5, 97.5))
    macro_percentiles = np.percentile(macro_samples, (2.5, 97.5))
    micro_interval = (float(micro_percentiles[0]), float(micro_percentiles[1]))
    macro_interval = (float(macro_percentiles[0]), float(macro_percentiles[1]))
    return micro_interval, macro_interval


def _comparison_rows(
    comparison: _Comparison,
    artifact_rows: dict[MrcrEvaluationKey, dict[tuple[int, str], dict[str, dict[str, Any]]]],
    *,
    context_cap: int,
    claim_gain_floor: float,
    claim_min_examples: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    required = {key for key, _ in comparison.terms}
    common_slices = set.intersection(*(set(artifact_rows[key]) for key in required))
    all_slices = set.union(*(set(artifact_rows[key]) for key in required))
    if common_slices != all_slices:
        raise ValueError(f"Mismatched needle/distance slices in comparison {comparison.kind}")
    output: list[dict[str, Any]] = []
    for n_needles, distance_band in sorted(common_slices):
        rows = {key: artifact_rows[key][(n_needles, distance_band)] for key in required}
        micro, macro, numerator, token_counts, source_ids = _metric_values(comparison, rows)
        macro_values = numerator / token_counts
        prefix = "/".join(
            (
                comparison.kind,
                str(context_cap),
                str(n_needles),
                distance_band,
                *comparison.packages,
                *comparison.baselines,
                comparison.left_prompt_variant,
                comparison.right_prompt_variant,
            )
        )
        micro_ci, macro_ci = _bootstrap_interval(
            numerator,
            token_counts,
            macro_values=macro_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            metric_prefix=prefix,
        )
        output.append(
            {
                "kind": comparison.kind,
                "packages": list(comparison.packages),
                "baselines": list(comparison.baselines),
                "left_prompt_variant": comparison.left_prompt_variant,
                "right_prompt_variant": comparison.right_prompt_variant,
                "context_cap": context_cap,
                "n_needles": n_needles,
                "distance_band": distance_band,
                "micro_difference": micro,
                "micro_difference_ci95_low": micro_ci[0],
                "micro_difference_ci95_high": micro_ci[1],
                "macro_difference": macro,
                "macro_difference_ci95_low": macro_ci[0],
                "macro_difference_ci95_high": macro_ci[1],
                "examples": len(source_ids),
                "claim_eligible": len(source_ids) >= claim_min_examples and micro_ci[0] > claim_gain_floor,
            }
        )
    return output


def _serialized_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _persist_idempotently(path: str, content: bytes) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    if fs.exists(fs_path):
        with fs.open(fs_path, "rb") as stream:
            if stream.read() == content:
                return
        raise ValueError(f"Conflicting MRCR summary output already exists: {path}")
    parent = str(Path(fs_path).parent)
    fs.makedirs(parent, exist_ok=True)
    temporary_path = f"{fs_path}.tmp-{uuid.uuid4().hex}"
    try:
        with fs.open(temporary_path, "wb") as stream:
            stream.write(content)
        fs.mv(temporary_path, fs_path)
    finally:
        if fs.exists(temporary_path):
            fs.rm(temporary_path)


def summarize_mrcr_matrix(
    evaluations: tuple[MrcrEvaluationArtifact, ...],
    *,
    summary_stage: str,
    expected_evaluations: tuple[MrcrEvaluationKey, ...],
    output_path: str,
    claim_gain_floor: float,
    claim_min_examples: int,
    bootstrap_samples: int = MRCR_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
) -> None:
    """Persist available paired comparisons and explicit partial-stage state."""

    if not math.isfinite(claim_gain_floor):
        raise ValueError("claim_gain_floor must be finite")
    if claim_min_examples <= 0:
        raise ValueError("claim_min_examples must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    expected = set(expected_evaluations)
    if len(expected) != len(expected_evaluations):
        raise ValueError("expected_evaluations contains duplicates")
    available: dict[MrcrEvaluationKey, MrcrEvaluationArtifact] = {}
    for artifact in evaluations:
        key = _evaluation_key(artifact)
        if key not in expected:
            raise ValueError(f"Unexpected MRCR evaluation: {key}")
        if key in available:
            raise ValueError(f"Duplicate MRCR evaluation: {key}")
        package = PACKAGE_BY_NAME.get(artifact.package_name)
        if package is None:
            raise ValueError(f"Unknown checkpoint package: {artifact.package_name}")
        if (
            artifact.qk_mult != package.qk_mult
            or artifact.training_offset != package.training_offset
            or artifact.baseline_package_name != package.baseline_package_name
        ):
            raise ValueError(f"Checkpoint package metadata mismatch: {artifact.package_name}")
        available[key] = artifact
    context_caps = {artifact.context_cap for artifact in evaluations}
    if len(context_caps) > 1:
        raise ValueError("A matrix summary must contain exactly one context cap")
    context_cap = next(iter(context_caps), 0)

    artifact_rows = {key: _read_example_losses(artifact.example_losses_path) for key, artifact in available.items()}
    comparison_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for comparison in _comparison_specs(expected):
        required = {key for key, _ in comparison.terms}
        missing = sorted(required - set(available), key=lambda key: (key.package_name, str(key.prompt_variant)))
        if missing:
            blocked.append(
                {
                    "kind": comparison.kind,
                    "packages": list(comparison.packages),
                    "baselines": list(comparison.baselines),
                    "left_prompt_variant": comparison.left_prompt_variant,
                    "right_prompt_variant": comparison.right_prompt_variant,
                    "missing": [
                        {"package_name": key.package_name, "prompt_variant": str(key.prompt_variant)} for key in missing
                    ],
                }
            )
            continue
        comparison_rows.extend(
            _comparison_rows(
                comparison,
                artifact_rows,
                context_cap=context_cap,
                claim_gain_floor=claim_gain_floor,
                claim_min_examples=claim_min_examples,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )

    def key_dict(key: MrcrEvaluationKey) -> dict[str, str]:
        return {"package_name": key.package_name, "prompt_variant": str(key.prompt_variant)}

    ordered_expected = sorted(expected, key=lambda key: (key.package_name, str(key.prompt_variant)))
    ordered_available = sorted(available, key=lambda key: (key.package_name, str(key.prompt_variant)))
    missing = sorted(expected - set(available), key=lambda key: (key.package_name, str(key.prompt_variant)))
    summary = {
        "summary_stage": summary_stage,
        "context_cap": context_cap,
        "expected_evaluations": [key_dict(key) for key in ordered_expected],
        "available_evaluations": [key_dict(key) for key in ordered_available],
        "missing_evaluations": [key_dict(key) for key in missing],
        "blocked_comparisons": blocked,
        "claim_gain_floor": claim_gain_floor,
        "claim_min_examples": claim_min_examples,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "complete": not missing,
    }
    comparison_content = b"".join(_serialized_json(row) for row in comparison_rows)
    summary_content = _serialized_json(summary)
    _persist_idempotently(os.path.join(output_path, "mrcr_matrix_comparisons.jsonl"), comparison_content)
    _persist_idempotently(os.path.join(output_path, "summary.json"), summary_content)


def matrix_cell_count() -> tuple[int, int]:
    """Return primary and additional prompt-sensitivity job counts."""

    primary = len(PRIMARY_CHECKPOINT_PACKAGES) * len(MRCR_CONTEXT_CAPS)
    sensitivity = len(SENSITIVITY_PACKAGE_NAMES) * len(SENSITIVITY_CONTEXT_CAPS) * 2
    return primary, sensitivity


if __name__ == "__main__":
    selection = os.environ.get("MRCR_MATRIX_SELECTION")
    if selection is None:
        raise ValueError(
            "Set MRCR_MATRIX_SELECTION explicitly to smoke, aggregate_262k_probe, aggregate_262k_extension, "
            "aggregate_262k_deployable_final, aggregate_262k, primary, "
            "sensitivity, or complete; "
            "the launcher does not default to an expensive matrix"
        )
    tpu_variant = os.environ.get("MRCR_EVAL_TPU", DEFAULT_TPU_VARIANT)
    selected_steps = build_default_steps(selection, tpu_variant=tpu_variant)
    executor_main(
        steps=list(selected_steps),
        description=(
            f"Paired MRCR base-checkpoint likelihood evaluation "
            f"({selection}, {len(selected_steps)} cells, {tpu_variant})."
        ),
    )
