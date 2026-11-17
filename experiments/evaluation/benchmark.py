# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Supported evaluation configurations and their corpus builders."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

from evalrecords.records import PromptFormat
from taskcompendium.evaluation.spec import OperationKind

from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.source import RlDataSource


@dataclass(frozen=True)
class ProtocolVariant:
    protocol: str
    num_fewshot: int | None
    prompt_format: PromptFormat | None


@dataclass(frozen=True)
class EvaluationBenchmark:
    """Supported evaluation configurations and their preparation builder."""

    name: str
    variants: tuple[ProtocolVariant, ...]
    build: Callable[[ProtocolVariant], tuple[RlDataSource[EvaluationCorpusRecipe], ...]]

    def resolve(
        self,
        protocol: str | None = None,
        num_fewshot: int | None = None,
        prompt_format: PromptFormat | None = None,
        *,
        model_format: PromptFormat | None = None,
    ) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
        protocol = protocol or self.variants[0].protocol
        variants = [variant for variant in self.variants if variant.protocol == protocol]
        if not variants:
            raise ValueError(f"Unknown {self.name} protocol: {protocol}")
        shots = num_fewshot if num_fewshot is not None else variants[0].num_fewshot
        variants = [variant for variant in variants if variant.num_fewshot == shots]
        if prompt_format is not None:
            variants = [variant for variant in variants if variant.prompt_format == prompt_format]
        elif model_format is not None:
            matching = [variant for variant in variants if variant.prompt_format == model_format]
            variants = matching or variants
        if not variants:
            raise ValueError(f"Unsupported {self.name} protocol: {protocol}, shots={shots}, format={prompt_format}")
        return self.build(variants[0])


def _compiled_variant(
    variants: dict[ProtocolVariant, tuple[RlDataSource[EvaluationCorpusRecipe], ...]],
    variant: ProtocolVariant,
) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    return variants[variant]


def fixed_benchmarks(
    sources: list[RlDataSource[EvaluationCorpusRecipe]],
    protocol: str | None = None,
    names: dict[str, str] | None = None,
) -> list[EvaluationBenchmark]:
    """Expose already-fixed protocols, retaining dataset-owned partition membership."""
    groups = defaultdict(lambda: defaultdict(list))
    for source in sources:
        dataset = source.config
        assert dataset is not None
        name = (names or {}).get(dataset.protocol.benchmark, dataset.protocol.benchmark.lower().replace("_", "-"))
        form = {
            OperationKind.CHAT_GENERATION: PromptFormat.CHAT,
            OperationKind.TEXT_GENERATION: PromptFormat.COMPLETION,
        }.get(dataset.protocol.operation)
        protocol_name = protocol or dataset.protocol_name
        variant = ProtocolVariant(protocol_name, dataset.protocol.num_fewshot, form)
        groups[name][variant].append(
            replace(
                source,
                config=replace(
                    dataset,
                    protocol=dataset.protocol.model_copy(update={"benchmark": name}),
                    protocol_name=protocol_name,
                ),
            )
        )
    return [
        EvaluationBenchmark(
            name, tuple(variants), partial(_compiled_variant, {key: tuple(value) for key, value in variants.items()})
        )
        for name, variants in groups.items()
    ]
