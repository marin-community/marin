# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from fray import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from levanter.analysis.model_perplexity import compare_scored_outputs
from levanter.analysis.perplexity_gap import write_report_files
from levanter.data.text import DatasetComponent, HfDatasetSourceConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.main.perplexity_gap import (
    GapFinderModelConfig as LevanterGapFinderModelConfig,
    ModelPerplexityConfig as LevanterModelPerplexityConfig,
    score_main,
)
from levanter.models.lm_model import LmConfig
from levanter.tokenizers import TokenizerBackend
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, InputName, VersionedValue, this_output_path, versioned
from marin.processing.tokenize import HfDatasetSpec
from marin.utilities.executor_utils import ckpt_path_to_step_name
from marin.utilities.wandb_utils import init_wandb


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
    split: str = "validation"
    tags: tuple[str, ...] = ()


@dataclass
class ModelPerplexityScoreConfig:
    name: str | None
    model: GapFinderModelConfig
    datasets: dict[str, RawTextEvaluationDataset]
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768
    wandb_tags: list[str] | None = None
    cache_key: dict[str, Any] | VersionedValue[dict[str, Any]] = field(default_factory=dict, repr=False)


@dataclass
class ModelPerplexityGapConfig:
    name: str | None
    model_a_name: str
    model_b_name: str
    model_a_scores_path: str | InputName | ExecutorStep
    model_b_scores_path: str | InputName | ExecutorStep
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    wandb_tags: list[str] | None = None
    cache_key: dict[str, Any] | VersionedValue[dict[str, Any]] = field(default_factory=dict, repr=False)


def raw_text_dataset(
    source: str | InputName | ExecutorStep | HfDatasetSpec,
    *,
    text_key: str = "text",
    split: str = "validation",
    tags: tuple[str, ...] = (),
) -> RawTextEvaluationDataset:
    if isinstance(source, HfDatasetSpec):
        return RawTextEvaluationDataset(
            hf_dataset_id=source.id,
            hf_dataset_name=source.name,
            text_key=text_key,
            split=split,
            tags=tags,
        )
    if split != "validation":
        raise ValueError("split is only supported for Hugging Face dataset sources; file paths use validation.")
    return RawTextEvaluationDataset(input_path=source, text_key=text_key, split=split, tags=tags)


def default_model_perplexity_scores(
    *,
    model: GapFinderModelConfig,
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
        name = _default_score_step_name(model)

    return ExecutorStep(
        name=f"analysis/model_perplexity_scores/{name}",
        fn=find_model_perplexity_scores,
        config=ModelPerplexityScoreConfig(
            name=name,
            model=model,
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
                    "model": _cache_key_for_model(model),
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
        name = _default_gap_step_name(model_a, model_b)

    score_step_a = default_model_perplexity_scores(
        model=model_a,
        datasets=datasets,
        resource_config=resource_config,
        per_device_batch_size=per_device_batch_size,
        max_eval_length=max_eval_length,
        max_docs_per_dataset=max_docs_per_dataset,
        max_doc_bytes=max_doc_bytes,
        wandb_tags=_default_score_wandb_tags(model_a),
    )
    score_step_b = default_model_perplexity_scores(
        model=model_b,
        datasets=datasets,
        resource_config=resource_config,
        per_device_batch_size=per_device_batch_size,
        max_eval_length=max_eval_length,
        max_docs_per_dataset=max_docs_per_dataset,
        max_doc_bytes=max_doc_bytes,
        wandb_tags=_default_score_wandb_tags(model_b),
    )

    return default_model_perplexity_gap_from_scores(
        model_a_name=_model_label(model_a),
        model_b_name=_model_label(model_b),
        model_a_scores_path=score_step_a.as_input_name(),
        model_b_scores_path=score_step_b.as_input_name(),
        name=name,
        wandb_tags=wandb_tags,
    )


def default_model_perplexity_gap_from_scores(
    *,
    model_a_name: str,
    model_b_name: str,
    model_a_scores_path: str | InputName | ExecutorStep,
    model_b_scores_path: str | InputName | ExecutorStep,
    name: str,
    wandb_tags: list[str] | None = None,
) -> ExecutorStep:
    return ExecutorStep(
        name=f"analysis/perplexity_gap/{name}",
        fn=find_model_perplexity_gap,
        config=ModelPerplexityGapConfig(
            name=name,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            model_a_scores_path=model_a_scores_path,
            model_b_scores_path=model_b_scores_path,
            wandb_tags=wandb_tags,
            cache_key=versioned(
                {
                    "name": name,
                    "model_a_name": model_a_name,
                    "model_b_name": model_b_name,
                    "wandb_tags": wandb_tags,
                }
            ),
        ),
    )


def do_find_model_perplexity_scores(config: LevanterModelPerplexityConfig) -> None:
    score_main(config)


def find_model_perplexity_scores(config: ModelPerplexityScoreConfig) -> None:
    datasets = {name: _to_dataset_component(dataset) for name, dataset in config.datasets.items()}

    if config.name is None:
        run_name = os.path.basename(config.output_path.rstrip("/"))
    else:
        run_name = config.name.replace("/", "-")

    levanter_config = LevanterModelPerplexityConfig(
        model=_to_levanter_model_config(config.model),
        datasets=datasets,
        trainer=TrainerConfig(
            tracker=(),
            per_device_eval_parallelism=config.per_device_batch_size,
        ),
        output_path=config.output_path,
        max_eval_length=config.max_eval_length,
        max_docs_per_dataset=config.max_docs_per_dataset,
        max_doc_bytes=config.max_doc_bytes,
    )

    assert isinstance(config.resource_config.device, TpuConfig), "find_model_perplexity_scores requires TPU resources"

    client = current_client()
    job_request = JobRequest(
        name=f"model-perplexity-scores-{run_name}",
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_find_model_perplexity_scores, args=[levanter_config]),
        environment=create_environment(extras=["tpu"]),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def find_model_perplexity_gap(config: ModelPerplexityGapConfig) -> None:
    summary = compare_scored_outputs(
        model_a_name=config.model_a_name,
        model_b_name=config.model_b_name,
        model_a_output_path=_resolve_path(config.model_a_scores_path),
        model_b_output_path=_resolve_path(config.model_b_scores_path),
        output_path=config.output_path,
    )
    _log_gap_report_to_wandb(config=config, summary=summary)


def _log_gap_report_to_wandb(*, config: ModelPerplexityGapConfig, summary: dict[str, Any]) -> None:
    run_name = (
        config.name.replace("/", "-") if config.name is not None else os.path.basename(config.output_path.rstrip("/"))
    )
    tags = ["perplexity_gap", *(config.wandb_tags or [])]
    run = init_wandb(
        run_name=run_name,
        tags=tags,
        config={"model_a": config.model_a_name, "model_b": config.model_b_name},
    )
    if run is None:
        return

    run.log(_summary_scalars(summary))

    with tempfile.TemporaryDirectory(prefix="perplexity-gap-report-") as tmpdir:
        write_report_files(tmpdir, summary)
        import wandb

        artifact = wandb.Artifact(name="perplexity_gap_report", type="perplexity_gap_report")
        artifact.add_dir(tmpdir)
        run.log_artifact(artifact)

    run.finish()


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
            splits=[config.split],
        )
    else:
        if config.input_path is None:
            raise ValueError("RawTextEvaluationDataset requires either input_path or hf_dataset_id.")
        if config.split != "validation":
            raise ValueError("RawTextEvaluationDataset split is only supported for Hugging Face dataset sources.")
        input_path = config.input_path
        if isinstance(input_path, ExecutorStep):
            input_path = input_path.as_input_name()
        source = UrlDatasetSourceConfig(
            train_urls=[],
            validation_urls=[input_path],  # type: ignore[list-item]
            format=dataset_format,
        )
    return DatasetComponent(source=source, format=dataset_format, tags=list(config.tags), split=config.split)


def _default_gap_step_name(model_a: GapFinderModelConfig, model_b: GapFinderModelConfig) -> str:
    left = ckpt_path_to_step_name(model_a.checkpoint_path)
    right = ckpt_path_to_step_name(model_b.checkpoint_path)
    return f"{left}-vs-{right}"


def _default_score_step_name(model: GapFinderModelConfig) -> str:
    return ckpt_path_to_step_name(model.checkpoint_path)


def _model_label(config: GapFinderModelConfig) -> str:
    if isinstance(config.checkpoint_path, InputName):
        return config.tokenizer or "model"
    return config.checkpoint_path


def _default_score_wandb_tags(model: GapFinderModelConfig) -> list[str]:
    return [f"model={_model_label(model)}"]


def _summary_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for row in summary["datasets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/datasets/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_a_bpb"] = float(row["model_a_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_b_bpb"] = float(row["model_b_bpb"])
    for row in summary["dataset_groups"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/groups/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
    for row in summary["pattern_buckets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/patterns/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
    return scalars


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
        "split": dataset.split,
        "tags": dataset.tags,
    }


def _resolve_path(path: str | InputName | ExecutorStep) -> str:
    if isinstance(path, ExecutorStep):
        raise TypeError("ExecutorStep dependencies should be resolved to InputName before execution.")
    if isinstance(path, InputName):
        raise TypeError("InputName should be resolved to a concrete path before execution.")
    return path
