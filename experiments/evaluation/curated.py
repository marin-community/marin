# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve benchmark selections to immutable corpus dependencies and run settings."""

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from huggingface_hub import HfApi
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import (
    DEFAULT_EVALUATION_CONCURRENCY,
    CuratedRef,
    CuratedSettings,
    EvalRef,
    EvaluationSelection,
    PromptFormat,
    ResolvedEvaluationSelection,
)
from marin.execution.lazy import ArtifactStep
from rigging.secrets import is_secret_reference
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.preparation import prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions


def pinned_model(model: ModelConfig) -> ModelConfig:
    """Freeze mutable Hub refs before they reach serving, tokenization or run identity."""
    api = HfApi()
    revision = model.revision
    if "://" not in model.location and not Path(model.location).is_dir():
        revision = api.model_info(model.location, revision=revision).sha
    tokenizer = model.tokenizer or model.location
    tokenizer_revision = (
        model.effective_tokenizer_revision or model.identity or tokenizer
        if "://" in tokenizer or Path(tokenizer).is_dir()
        else api.model_info(tokenizer, revision=model.effective_tokenizer_revision).sha
    )
    if revision is None and model.identity is None and "://" not in model.location:
        raise ValueError(f"No immutable model identity for {model.location}")
    return replace(model, revision=revision, tokenizer=tokenizer, tokenizer_revision=tokenizer_revision)


def load_evaluations(path: Path) -> tuple[EvaluationSelection, ...]:
    document = yaml.safe_load(path.read_text())
    entries = document["evaluations"]
    return tuple(EvaluationSelection.model_validate({"label": name, **entry}) for name, entry in entries.items())


def resolve_selection(selection: EvaluationSelection, model_format: PromptFormat | None) -> ResolvedEvaluationSelection:
    catalog = evaluation_benchmarks()
    if selection.benchmark not in catalog:
        raise ValueError(f"Unknown evaluation benchmark: {selection.benchmark}")
    sources = catalog[selection.benchmark].resolve(
        selection.protocol, selection.num_fewshot, selection.prompt_format, model_format=model_format
    )
    dataset = sources[0].config
    assert dataset is not None
    form = {"chat_generation": PromptFormat.CHAT, "text_generation": PromptFormat.COMPLETION}.get(
        dataset.protocol.operation
    )
    return ResolvedEvaluationSelection(
        benchmark=selection.benchmark,
        protocol=dataset.protocol_name,
        num_fewshot=dataset.protocol.num_fewshot,
        prompt_format=form,
        partitions=selection.partitions,
    )


def corpus_steps(selection: ResolvedEvaluationSelection, runtime: CampaignRuntime) -> tuple[ArtifactStep, ...]:
    benchmark = evaluation_benchmarks()[selection.benchmark]
    sources = benchmark.resolve(
        selection.protocol,
        selection.num_fewshot,
        PromptFormat(selection.prompt_format) if selection.prompt_format is not None else None,
    )
    if selection.partitions:
        available = {source.config.partition for source in sources if source.config is not None}
        unknown = set(selection.partitions) - available
        if unknown:
            raise ValueError(f"Unknown {selection.benchmark} partitions: {sorted(unknown)}")
        sources = tuple(
            source for source in sources if source.config is not None and source.config.partition in selection.partitions
        )
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime)
    return tuple(prepare_evaluation(source, options) for source in sources)


def requires_judge(selection: EvaluationSelection) -> bool:
    return any(
        source.config is not None and source.config.protocol.requires_judge
        for source in evaluation_benchmarks()[selection.benchmark].resolve(selection.protocol, selection.num_fewshot)
    )


def corpus_key(step: ArtifactStep) -> str:
    """Identify the publication recipe and version independently of its storage region."""
    return f"{step.name}@{step.version}"


def settings_for(
    selection: EvaluationSelection,
    model: ModelConfig,
    limit: int | None,
    seed: int | None,
) -> CuratedSettings:
    if model.effective_tokenizer_revision is None:
        raise ValueError("Curated evaluation requires an explicit tokenizer revision")
    run = selection.run
    if run.judge is not None:
        url = urlsplit(run.judge_endpoint or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("judge_endpoint must be an HTTP URL without credentials")
        if not run.judge_api_key or any(not is_secret_reference(ref) for ref in run.judge_api_key):
            raise ValueError("A judge requires secret references in judge_api_key")
    return CuratedSettings(
        concurrency=model.evaluation_num_concurrent or DEFAULT_EVALUATION_CONCURRENCY,
        max_tokens=run.max_tokens,
        source_limit=limit if limit is not None else run.limit,
        seed=seed if seed is not None else run.seed,
        judge=run.judge,
    )


def record_ref_for(
    selection: EvaluationSelection, protocol: ResolvedEvaluationSelection, settings: CuratedSettings
) -> EvalRef:
    return EvalRef(
        name=selection.name,
        mechanism="curated",
        family=selection.benchmark,
        source_digest="sha256:"
        + canonical_sha256(
            {
                "selection": selection.model_dump(mode="json"),
                "settings": settings.model_dump(mode="json", exclude={"concurrency"}),
            }
        ),
        curated=CuratedRef(
            datasets=tuple(corpus_key(step) for step in corpus_steps(protocol, CampaignRuntime())),
            settings=settings,
            selection=protocol,
        ),
    )
