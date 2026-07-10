# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoring model perplexity and comparing score outputs across two models."""

import tempfile
from dataclasses import dataclass
from typing import Any

import wandb
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from levanter.analysis.model_perplexity import add_prefixed_runtime_metric_scalars, compare_scored_outputs
from levanter.analysis.perplexity_gap import write_report_files
from levanter.data.text.datasets import DatasetComponent, HfDatasetSourceConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import SupervisedLmDatasetFormat, TextLmDatasetFormat
from levanter.main.perplexity_gap import (
    GapFinderModelConfig as LevanterGapFinderModelConfig,
)
from levanter.main.perplexity_gap import (
    ModelPerplexityConfig as LevanterModelPerplexityConfig,
)
from levanter.main.perplexity_gap import (
    score_main,
)
from levanter.models.lm_model import LmConfig
from levanter.tokenizers import TokenizerBackend
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.processing.tokenize import HfDatasetSpec
from marin.utilities.wandb_utils import init_wandb

WANDB_PROJECT = "marin-eval"


@dataclass(frozen=True)
class GapFinderModelConfig:
    checkpoint_path: str
    model: LmConfig | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str | None = None
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    trust_remote_code: bool = False


@dataclass(frozen=True)
class RawTextEvaluationDataset:
    input_path: str | None = None
    hf_dataset_id: str | None = None
    hf_dataset_name: str | None = None
    hf_dataset_revision: str | None = None
    text_key: str = "text"
    input_key: str | None = None
    target_key: str | None = None
    split: str = "validation"
    tags: tuple[str, ...] = ()


@dataclass
class ModelPerplexityScoreConfig:
    name: str
    model: GapFinderModelConfig
    datasets: dict[str, RawTextEvaluationDataset]
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    output_path: str = ""
    max_eval_length: int = 4096
    max_docs_per_dataset: int | None = 256
    max_doc_bytes: int | None = 32_768
    wandb_tags: list[str] | None = None


@dataclass
class ModelPerplexityGapConfig:
    name: str
    model_a_name: str
    model_b_name: str
    model_a_scores_path: str
    model_b_scores_path: str
    output_path: str = ""
    wandb_tags: list[str] | None = None


def raw_text_dataset(
    source: str | HfDatasetSpec,
    *,
    text_key: str = "text",
    split: str = "validation",
    tags: tuple[str, ...] = (),
) -> RawTextEvaluationDataset:
    if isinstance(source, HfDatasetSpec):
        return RawTextEvaluationDataset(
            hf_dataset_id=source.id,
            hf_dataset_name=source.name,
            hf_dataset_revision=source.revision,
            text_key=text_key,
            split=split,
            tags=tags,
        )
    if split != "validation":
        raise ValueError("split is only supported for Hugging Face dataset sources; file paths use validation.")
    return RawTextEvaluationDataset(input_path=source, text_key=text_key, split=split, tags=tags)


def supervised_text_dataset(
    source: str | HfDatasetSpec,
    *,
    input_key: str = "input",
    target_key: str = "target",
    split: str = "validation",
    tags: tuple[str, ...] = (),
) -> RawTextEvaluationDataset:
    if isinstance(source, HfDatasetSpec):
        return RawTextEvaluationDataset(
            hf_dataset_id=source.id,
            hf_dataset_name=source.name,
            hf_dataset_revision=source.revision,
            input_key=input_key,
            target_key=target_key,
            split=split,
            tags=tags,
        )
    if split != "validation":
        raise ValueError("split is only supported for Hugging Face dataset sources; file paths use validation.")
    return RawTextEvaluationDataset(
        input_path=source,
        input_key=input_key,
        target_key=target_key,
        split=split,
        tags=tags,
    )


def find_model_perplexity_scores(config: ModelPerplexityScoreConfig) -> None:
    datasets = {name: _to_dataset_component(dataset) for name, dataset in config.datasets.items()}

    run_name = config.name.replace("/", "-")
    tags = ["model_perplexity_scores", *(config.wandb_tags or [])]

    levanter_config = LevanterModelPerplexityConfig(
        model=_to_levanter_model_config(config.model),
        datasets=datasets,
        trainer=TrainerConfig(
            tracker=WandbConfig(project=WANDB_PROJECT, tags=tags, name=run_name),
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
        entrypoint=Entrypoint.from_callable(score_main, args=[levanter_config]),
        environment=create_environment(extras=["tpu"]),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def find_model_perplexity_gap(config: ModelPerplexityGapConfig) -> None:
    summary = compare_scored_outputs(
        model_a_name=config.model_a_name,
        model_b_name=config.model_b_name,
        model_a_output_path=config.model_a_scores_path,
        model_b_output_path=config.model_b_scores_path,
        output_path=config.output_path,
    )
    _log_gap_report_to_wandb(config=config, summary=summary)


def _log_gap_report_to_wandb(*, config: ModelPerplexityGapConfig, summary: dict[str, Any]) -> None:
    run_name = config.name.replace("/", "-")
    tags = ["perplexity_gap", *(config.wandb_tags or [])]
    run = init_wandb(
        run_name=run_name,
        tags=tags,
        config={"model_a": config.model_a_name, "model_b": config.model_b_name},
        project=WANDB_PROJECT,
    )
    if run is None:
        return

    run.log(_summary_scalars(summary))

    with tempfile.TemporaryDirectory(prefix="perplexity-gap-report-") as tmpdir:
        write_report_files(tmpdir, summary)
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
    if config.input_key is None and config.target_key is None:
        dataset_format = TextLmDatasetFormat(text_key=config.text_key)
    elif config.input_key is not None and config.target_key is not None:
        dataset_format = SupervisedLmDatasetFormat(input_key=config.input_key, target_key=config.target_key)
    else:
        raise ValueError("RawTextEvaluationDataset must set both input_key and target_key for supervised data.")
    if config.hf_dataset_id is not None:
        source = HfDatasetSourceConfig(
            id=config.hf_dataset_id,
            name=config.hf_dataset_name,
            revision=config.hf_dataset_revision,
            format=dataset_format,
            splits=[config.split],
        )
    else:
        if config.input_path is None:
            raise ValueError("RawTextEvaluationDataset requires either input_path or hf_dataset_id.")
        if config.split != "validation":
            raise ValueError("RawTextEvaluationDataset split is only supported for Hugging Face dataset sources.")
        source = UrlDatasetSourceConfig(
            train_urls=[],
            validation_urls=[config.input_path],
            format=dataset_format,
        )
    return DatasetComponent(source=source, format=dataset_format, tags=list(config.tags), split=config.split)


def _summary_scalars(summary: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for row in summary["datasets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/datasets/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_a_bpb"] = float(row["model_a_bpb"])
        scalars[f"gap/datasets/{row['name']}/model_b_bpb"] = float(row["model_b_bpb"])
        add_prefixed_runtime_metric_scalars(
            scalars,
            key_prefix=f"gap/datasets/{row['name']}",
            row=row,
            prefix="model_a",
        )
        add_prefixed_runtime_metric_scalars(
            scalars,
            key_prefix=f"gap/datasets/{row['name']}",
            row=row,
            prefix="model_b",
        )
    for row in summary["dataset_groups"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/groups/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
        add_prefixed_runtime_metric_scalars(
            scalars,
            key_prefix=f"gap/groups/{row['name']}",
            row=row,
            prefix="model_a",
        )
        add_prefixed_runtime_metric_scalars(
            scalars,
            key_prefix=f"gap/groups/{row['name']}",
            row=row,
            prefix="model_b",
        )
    for row in summary["pattern_buckets"]:
        if row["gap_bpb"] is None:
            continue
        scalars[f"gap/patterns/{row['name']}/bpb_gap"] = float(row["gap_bpb"])
    return scalars
