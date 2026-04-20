# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import Any

from fray.v2 import current_client
from fray.v2.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from levanter.data.text import DatasetComponent, HfDatasetSourceConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.main.perplexity_gap import (
    GapFinderConfig as LevanterGapFinderConfig,
    GapFinderModelConfig as LevanterGapFinderModelConfig,
)
from levanter.models.lm_model import LmConfig
from levanter.tokenizers import TokenizerBackend
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, InputName, VersionedValue, this_output_path, versioned
from marin.processing.tokenize import HfDatasetSpec
from marin.utilities.executor_utils import ckpt_path_to_step_name


@dataclass(frozen=True)
class GapFinderModelConfig:
    checkpoint_path: str | InputName
    model: LmConfig | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str | None = None
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    trust_remote_code: bool = False


@dataclass(frozen=True)
class RawTextEvaluationDataset:
    input_path: str | InputName | ExecutorStep | None = None
    hf_dataset_id: str | None = None
    hf_dataset_name: str | None = None
    text_key: str = "text"
    tags: tuple[str, ...] = ()


@dataclass
class ModelPerplexityGapConfig:
    name: str | None
    model_a: GapFinderModelConfig
    model_b: GapFinderModelConfig
    datasets: dict[str, RawTextEvaluationDataset]
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768
    wandb_tags: list[str] | None = None
    cache_key: dict[str, Any] | VersionedValue[dict[str, Any]] = field(default_factory=dict, repr=False)


def raw_text_dataset(
    source: str | InputName | ExecutorStep | HfDatasetSpec,
    *,
    text_key: str = "text",
    tags: tuple[str, ...] = (),
) -> RawTextEvaluationDataset:
    if isinstance(source, HfDatasetSpec):
        return RawTextEvaluationDataset(
            hf_dataset_id=source.id,
            hf_dataset_name=source.name,
            text_key=text_key,
            tags=tags,
        )
    return RawTextEvaluationDataset(input_path=source, text_key=text_key, tags=tags)


def default_model_perplexity_gap(
    *,
    model_a: GapFinderModelConfig,
    model_b: GapFinderModelConfig,
    datasets: dict[str, RawTextEvaluationDataset],
    resource_config: ResourceConfig,
    per_device_batch_size: int = 4,
    max_eval_length: int = 4096,
    max_docs_per_dataset: int | None = 256,
    max_doc_bytes: int | None = 32_768,
    name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> ExecutorStep:
    if name is None:
        name = _default_step_name(model_a, model_b)

    return ExecutorStep(
        name=f"analysis/perplexity_gap/{name}",
        fn=find_model_perplexity_gap,
        config=ModelPerplexityGapConfig(
            name=name,
            model_a=model_a,
            model_b=model_b,
            datasets=datasets,
            resource_config=resource_config,
            per_device_batch_size=per_device_batch_size,
            max_eval_length=max_eval_length,
            max_docs_per_dataset=max_docs_per_dataset,
            max_doc_bytes=max_doc_bytes,
            wandb_tags=wandb_tags,
            cache_key=versioned(
                {
                    "name": name,
                    "model_a": _cache_key_for_model(model_a),
                    "model_b": _cache_key_for_model(model_b),
                    "datasets": {dataset_name: _cache_key_for_dataset(ds) for dataset_name, ds in datasets.items()},
                    "resource_config": resource_config,
                    "per_device_batch_size": per_device_batch_size,
                    "max_eval_length": max_eval_length,
                    "max_docs_per_dataset": max_docs_per_dataset,
                    "max_doc_bytes": max_doc_bytes,
                    "wandb_tags": wandb_tags,
                }
            ),
        ),
    )


def do_find_perplexity_gap(config: LevanterGapFinderConfig) -> None:
    from levanter.main.perplexity_gap import main as gap_main

    gap_main(config)


def find_model_perplexity_gap(config: ModelPerplexityGapConfig) -> None:
    datasets = {name: _to_dataset_component(dataset) for name, dataset in config.datasets.items()}

    if config.name is None:
        run_name = os.path.basename(config.output_path.rstrip("/"))
    else:
        run_name = config.name.replace("/", "-")

    wandb_tags = ["perplexity_gap", *(config.wandb_tags or [])]
    levanter_config = LevanterGapFinderConfig(
        model_a=_to_levanter_model_config(config.model_a),
        model_b=_to_levanter_model_config(config.model_b),
        datasets=datasets,
        trainer=TrainerConfig(
            tracker=(
                WandbConfig(project="marin", tags=wandb_tags, name=run_name),
                JsonFileTrackerConfig(output_path=config.output_path),
            ),
            per_device_eval_parallelism=config.per_device_batch_size,
        ),
        output_path=config.output_path,
        max_eval_length=config.max_eval_length,
        max_docs_per_dataset=config.max_docs_per_dataset,
        max_doc_bytes=config.max_doc_bytes,
    )

    assert isinstance(config.resource_config.device, TpuConfig), "find_model_perplexity_gap requires TPU resources"

    client = current_client()
    job_request = JobRequest(
        name=f"perplexity-gap-{run_name}",
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_find_perplexity_gap, args=[levanter_config]),
        environment=create_environment(extras=["tpu"]),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def _to_levanter_model_config(config: GapFinderModelConfig) -> LevanterGapFinderModelConfig:
    return LevanterGapFinderModelConfig(
        checkpoint_path=config.checkpoint_path,  # type: ignore[arg-type]
        model=config.model,
        checkpoint_is_hf=config.checkpoint_is_hf,
        tokenizer=config.tokenizer,
        tokenizer_backend=config.tokenizer_backend,
        trust_remote_code=config.trust_remote_code,
    )


def _to_dataset_component(config: RawTextEvaluationDataset) -> DatasetComponent:
    dataset_format = TextLmDatasetFormat(text_key=config.text_key)
    if config.hf_dataset_id is not None:
        source = HfDatasetSourceConfig(
            id=config.hf_dataset_id,
            name=config.hf_dataset_name,
            format=dataset_format,
        )
    else:
        if config.input_path is None:
            raise ValueError("RawTextEvaluationDataset requires either input_path or hf_dataset_id.")
        input_path = config.input_path
        if isinstance(input_path, ExecutorStep):
            input_path = input_path.as_input_name()
        source = UrlDatasetSourceConfig(
            train_urls=[],
            validation_urls=[input_path],  # type: ignore[list-item]
            format=dataset_format,
        )
    return DatasetComponent(source=source, format=dataset_format, tags=list(config.tags))


def _default_step_name(model_a: GapFinderModelConfig, model_b: GapFinderModelConfig) -> str:
    left = ckpt_path_to_step_name(model_a.checkpoint_path)
    right = ckpt_path_to_step_name(model_b.checkpoint_path)
    return f"{left}-vs-{right}"


def _cache_key_for_model(config: GapFinderModelConfig) -> dict[str, Any]:
    checkpoint_path: str | None
    if isinstance(config.checkpoint_path, InputName):
        checkpoint_path = None
    else:
        checkpoint_path = config.checkpoint_path

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_is_hf": config.checkpoint_is_hf,
        "model": config.model,
        "tokenizer": config.tokenizer,
        "tokenizer_backend": config.tokenizer_backend.value,
        "trust_remote_code": config.trust_remote_code,
    }


def _cache_key_for_dataset(dataset: RawTextEvaluationDataset) -> dict[str, Any]:
    input_path: str | None
    if isinstance(dataset.input_path, (InputName, ExecutorStep)) or dataset.input_path is None:
        input_path = None
    else:
        input_path = dataset.input_path

    return {
        "input_path": input_path,
        "hf_dataset_id": dataset.hf_dataset_id,
        "hf_dataset_name": dataset.hf_dataset_name,
        "text_key": dataset.text_key,
        "tags": dataset.tags,
    }
