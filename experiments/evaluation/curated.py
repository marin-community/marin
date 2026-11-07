# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind evaluation menu entries to the canonical task-curation artifact graph."""

from dataclasses import replace
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from huggingface_hub import HfApi
from marin.evaluation.dataset_evaluator import DatasetEvaluator
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import DEFAULT_EVALUATION_CONCURRENCY, CuratedRef, CuratedSettings, EvalRef, JudgeIdentity
from marin.execution.lazy import ArtifactStep
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.secrets import is_secret_reference
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.evaluation import curate
from experiments.post_training.task_curation.sources import evaluation_sources


@cache
def _evaluation_sources():
    return evaluation_sources()


def _source_for_shots(name: str, num_fewshot: int) -> str:
    catalog = _evaluation_sources()
    dataset = catalog[name].config
    assert dataset is not None
    if dataset.protocol.num_fewshot == num_fewshot:
        return name
    if dataset.fewshot_group is None:
        raise ValueError(f"Source {name} has no curated {num_fewshot}-shot variant")
    variants = [
        source.name
        for source in catalog.values()
        if source.config is not None
        and source.config.fewshot_group == dataset.fewshot_group
        and source.config.protocol.num_fewshot == num_fewshot
    ]
    if len(variants) != 1:
        raise ValueError(f"Source {name} requires exactly one curated {num_fewshot}-shot variant, found {variants}")
    return variants[0]


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


class CuratedDefinition(BaseModel):
    """A menu selection binds curated corpora to run settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: tuple[str, ...] = Field(min_length=1)
    num_fewshot: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    family: str | None = None
    seed: int | None = None
    judge: JudgeIdentity | None = None
    judge_endpoint: str | None = None
    judge_api_key: tuple[str, ...] = ()
    chat_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_judge(self) -> "CuratedDefinition":
        if self.judge is not None:
            url = urlsplit(self.judge_endpoint or "")
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("judge_endpoint must be an HTTP URL without credentials")
            if not self.judge_api_key or any(not is_secret_reference(ref) for ref in self.judge_api_key):
                raise ValueError("A judge requires secret references in judge_api_key")
        return self

    @property
    def requires_judge(self) -> bool:
        for name in self.sources:
            dataset = _evaluation_sources()[name].config
            assert dataset is not None
            if dataset.protocol.requires_judge:
                return True
        return False

    @classmethod
    def from_file(cls, path: Path) -> "CuratedDefinition":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def executor_for(self, model: ModelConfig, limit: int | None, seed: int | None) -> DatasetEvaluator:
        if model.effective_tokenizer_revision is None:
            raise ValueError("Curated evaluation requires an explicit tokenizer revision")
        settings = CuratedSettings(
            concurrency=model.evaluation_num_concurrent or DEFAULT_EVALUATION_CONCURRENCY,
            max_tokens=self.max_tokens,
            source_limit=limit if limit is not None else self.limit,
            seed=seed if seed is not None else self.seed,
            judge=self.judge,
        )
        # Corpus paths are bound in the cluster before inference starts.
        sources = self.chat_sources if model.apply_chat_template and self.chat_sources else self.sources
        if self.num_fewshot is not None:
            sources = tuple(_source_for_shots(name, self.num_fewshot) for name in sources)
        return DatasetEvaluator(sources, model, settings, self.judge_endpoint)

    def record_ref_for(self, executor: DatasetEvaluator) -> EvalRef:
        return EvalRef(
            name=self.name,
            mechanism="curated",
            family=self.family,
            source_digest="sha256:"
            + canonical_sha256(
                {
                    "sources": executor.datasets,
                    "limit": self.limit,
                    "max_tokens": self.max_tokens,
                    "seed": self.seed,
                    "judge": self.judge.model_dump(mode="json") if self.judge is not None else None,
                    "corpora": {
                        step.name: step.fingerprint() for step in corpus_steps(executor.datasets, CampaignRuntime())
                    },
                }
            ),
            curated=CuratedRef(datasets=executor.datasets, settings=executor.settings),
        )


def corpus_steps(names: tuple[str, ...], runtime: CampaignRuntime) -> tuple[ArtifactStep, ...]:
    catalog = _evaluation_sources()
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f"Unknown curated evaluation sources: {sorted(unknown)}")
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime)
    return tuple(curate(catalog[name], options) for name in dict.fromkeys(names))
