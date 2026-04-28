# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical bundle/model registry for raw perplexity scoring and gap diffs.

This is the reusable layer behind the checkpoint-confidence workflow in #5005:

- bundle definitions say which raw eval datasets belong together
- model definitions say how each checkpoint should be scored
- the coverage plan expands to one score step per model/bundle plus selected
  pairwise diffs derived from those cached scores
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fray.v2.types import ResourceConfig

from experiments.defaults import default_raw_validation_sets
from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets
from experiments.evals.long_tail_ppl_runnable import runnable_long_tail_raw_validation_sets
from experiments.marin_models import marin_tokenizer
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import ExecutorStep

DEFAULT_MAX_EVAL_LENGTH = 4096
DEFAULT_MAX_DOCS_PER_DATASET = 256
DEFAULT_MAX_DOC_BYTES = 32_768


@dataclass(frozen=True)
class PerplexityGapBundle:
    key: str
    description: str
    datasets_factory: Callable[[], dict[str, RawTextEvaluationDataset]]
    max_eval_length: int = DEFAULT_MAX_EVAL_LENGTH
    max_docs_per_dataset: int | None = DEFAULT_MAX_DOCS_PER_DATASET
    max_doc_bytes: int | None = DEFAULT_MAX_DOC_BYTES

    def datasets(self) -> dict[str, RawTextEvaluationDataset]:
        return self.datasets_factory()


@dataclass(frozen=True)
class PerplexityGapModel:
    key: str
    label: str
    config: GapFinderModelConfig
    per_device_batch_size: int


@dataclass(frozen=True)
class PerplexityGapPair:
    left_model_key: str
    right_model_key: str


@dataclass(frozen=True)
class PerplexityGapCoveragePlan:
    score_steps: dict[tuple[str, str], ExecutorStep]
    pairwise_gap_steps: dict[tuple[str, str, str], ExecutorStep]


def base_raw_bundle() -> PerplexityGapBundle:
    return PerplexityGapBundle(
        key="base_raw",
        description="Paloma + uncheatable raw validation sets.",
        datasets_factory=default_raw_validation_sets,
    )


def multilingual_bundle() -> PerplexityGapBundle:
    return PerplexityGapBundle(
        key="multilingual_raw",
        description="Base raw validation sets plus FineWeb2 multilingual.",
        datasets_factory=lambda: {**default_raw_validation_sets(), **fineweb2_multilingual_raw_validation_sets()},
    )


def runnable_long_tail_bundle() -> PerplexityGapBundle:
    return PerplexityGapBundle(
        key="runnable_long_tail",
        description="HF-backed long-tail slices that are directly runnable today.",
        datasets_factory=runnable_long_tail_raw_validation_sets,
    )


def registered_perplexity_gap_bundles() -> tuple[PerplexityGapBundle, ...]:
    return (
        base_raw_bundle(),
        multilingual_bundle(),
        runnable_long_tail_bundle(),
    )


def registered_perplexity_gap_models() -> tuple[PerplexityGapModel, ...]:
    return (
        PerplexityGapModel(
            key="marin_8b",
            label="marin-community/marin-8b-base",
            config=GapFinderModelConfig(
                checkpoint_path="marin-community/marin-8b-base",
                checkpoint_is_hf=True,
                tokenizer=marin_tokenizer,
            ),
            per_device_batch_size=4,
        ),
        PerplexityGapModel(
            key="llama3_1_8b",
            label="meta-llama/Llama-3.1-8B",
            config=GapFinderModelConfig(
                checkpoint_path="meta-llama/Llama-3.1-8B",
                checkpoint_is_hf=True,
                tokenizer="meta-llama/Llama-3.1-8B",
            ),
            per_device_batch_size=4,
        ),
        PerplexityGapModel(
            key="qwen3_8b",
            label="Qwen/Qwen3-8B-Base",
            config=GapFinderModelConfig(
                checkpoint_path="Qwen/Qwen3-8B-Base",
                checkpoint_is_hf=True,
                tokenizer="Qwen/Qwen3-8B",
            ),
            per_device_batch_size=4,
        ),
        PerplexityGapModel(
            key="marin_32b",
            label="marin-community/marin-32b-base",
            config=GapFinderModelConfig(
                checkpoint_path="marin-community/marin-32b-base",
                checkpoint_is_hf=True,
                tokenizer=marin_tokenizer,
            ),
            per_device_batch_size=1,
        ),
        PerplexityGapModel(
            key="qwen3_32b",
            label="Qwen/Qwen3-32B",
            config=GapFinderModelConfig(
                checkpoint_path="Qwen/Qwen3-32B",
                checkpoint_is_hf=True,
                tokenizer="Qwen/Qwen3-32B",
            ),
            per_device_batch_size=1,
        ),
    )


def registered_perplexity_gap_pairs() -> tuple[PerplexityGapPair, ...]:
    return (
        PerplexityGapPair("marin_8b", "llama3_1_8b"),
        PerplexityGapPair("marin_8b", "qwen3_8b"),
        PerplexityGapPair("marin_32b", "qwen3_32b"),
    )


def build_registered_perplexity_gap_coverage_plan(
    *,
    resource_config: ResourceConfig,
    bundles: Iterable[PerplexityGapBundle] | None = None,
    models: Iterable[PerplexityGapModel] | None = None,
    comparison_pairs: Iterable[PerplexityGapPair] | None = None,
) -> PerplexityGapCoveragePlan:
    resolved_bundles = tuple(registered_perplexity_gap_bundles() if bundles is None else bundles)
    resolved_models = tuple(registered_perplexity_gap_models() if models is None else models)
    resolved_pairs = tuple(registered_perplexity_gap_pairs() if comparison_pairs is None else comparison_pairs)

    models_by_key = {model.key: model for model in resolved_models}
    score_steps: dict[tuple[str, str], ExecutorStep] = {}

    for bundle in resolved_bundles:
        datasets = bundle.datasets()
        for model in resolved_models:
            score_steps[(bundle.key, model.key)] = model_perplexity_scores(
                name=f"{bundle.key}/{model.key}",
                model=model.config,
                datasets=datasets,
                resource_config=resource_config,
                per_device_batch_size=model.per_device_batch_size,
                max_eval_length=bundle.max_eval_length,
                max_docs_per_dataset=bundle.max_docs_per_dataset,
                max_doc_bytes=bundle.max_doc_bytes,
                wandb_tags=[
                    "eval=model-perplexity",
                    f"dataset_bundle={bundle.key}",
                    f"model={model.label}",
                ],
            )

    pairwise_gap_steps: dict[tuple[str, str, str], ExecutorStep] = {}
    for bundle in resolved_bundles:
        for pair in resolved_pairs:
            left_model = models_by_key[pair.left_model_key]
            right_model = models_by_key[pair.right_model_key]
            left_score = score_steps[(bundle.key, left_model.key)]
            right_score = score_steps[(bundle.key, right_model.key)]
            pairwise_gap_steps[(bundle.key, left_model.key, right_model.key)] = model_perplexity_gap_from_scores(
                name=f"{bundle.key}/{left_model.key}-vs-{right_model.key}",
                model_a_name=left_model.label,
                model_b_name=right_model.label,
                model_a_scores_path=left_score.as_input_name(),
                model_b_scores_path=right_score.as_input_name(),
                wandb_tags=[
                    "eval=perplexity-gap",
                    f"dataset_bundle={bundle.key}",
                    f"model_a={left_model.label}",
                    f"model_b={right_model.label}",
                ],
            )

    return PerplexityGapCoveragePlan(score_steps=score_steps, pairwise_gap_steps=pairwise_gap_steps)
