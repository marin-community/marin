# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primitives for tokenization and dataset download."""

import os
from collections.abc import Sequence
from typing import Any

from fray import ResourceConfig
from levanter.data.text import (
    LmDatasetFormatBase,
    TextLmDatasetFormat,
)
from levanter.utils import fsspec_utils

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, VersionedValue, ensure_versioned, this_output_path
from marin.processing.tokenize import (
    HfDatasetSpec,
    TokenizeConfig,
    tokenize,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig

HF_BUCKET_URI_PREFIX = "hf://buckets/"
HF_BUCKET_PATH_PREFIX = "buckets/"


def _is_hf_bucket_path(path: str) -> bool:
    return path.startswith(HF_BUCKET_URI_PREFIX) or path.startswith(HF_BUCKET_PATH_PREFIX)


def _normalize_hf_bucket_path(path: str) -> str:
    if path.startswith(HF_BUCKET_URI_PREFIX):
        return path.removeprefix("hf://")
    return path


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str | None = None,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> InputName:
    """
    Download a HuggingFace dataset and upload it to a specified path with default configuration.

    Args:
        name: The name of the Download step. It forms the basis of the output path
            unless override_output_path is explicitly specified.
        hf_dataset_id: Hugging Face source. Either `$ORG/$DATASET` on HF Hub or `hf://buckets/...`.
        revision: The revision of the dataset to download for Hub datasets.
            Optional for bucket paths.
        override_output_path: Optional. The output path for the dataset.
        **kwargs: Additional keyword arguments that are passed to the download config.

    The final output data will reside in '{output_path}/{revision}'.
    """

    download_kwargs = dict(kwargs)
    hf_repo_type_prefix = download_kwargs.pop("hf_repo_type_prefix", None)
    if _is_hf_bucket_path(hf_dataset_id):
        normalized_dataset_id = _normalize_hf_bucket_path(hf_dataset_id)
        description = f"Download {hf_dataset_id}"
        resolved_hf_repo_type_prefix = "" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = "main" if revision is None else revision
    else:
        if revision is None:
            raise ValueError("revision is required for non-bucket Hugging Face dataset downloads.")
        normalized_dataset_id = hf_dataset_id
        description = f"Download {hf_dataset_id} revision {revision}"
        resolved_hf_repo_type_prefix = "datasets" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = revision

    step = ExecutorStep(
        name=name,
        description=description,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=normalized_dataset_id,
            revision=resolved_revision,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_repo_type_prefix=resolved_hf_repo_type_prefix,
            **download_kwargs,
        ),
        override_output_path=override_output_path,
    )

    return step.as_input_name()


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
    levanter_batch_size: int | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset:  The dataset to tokenize. This can be an InputName, ExecutorStep, a string as a
            path to the dataset or a HuggingFace dataset ID, or ``HfDatasetSpec`` to specify a
            dataset with a particular subset name.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        format: The format of the dataset. This is used to determine how to tokenize the data.

            See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/)
            for more details.
        sample_count: Optional limit on the number of samples to tokenize per shard. If ``None``, tokenize everything.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
        tags: Tags to attach to the Levanter dataset source for tagged evaluation.
    Returns:
        An ExecutorStep that represents the tokenized dataset.
    """

    # Common kwargs for config constructors
    extra_kwargs: dict = {}
    if worker_resources is not None:
        extra_kwargs["worker_resources"] = worker_resources

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    elif (
        isinstance(dataset, str)
        and not _is_hf_bucket_path(dataset)
        and dataset.count("/") == 1
        and not fsspec_utils.exists(dataset)
    ):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=remote(
            tokenize,
            resources=resources or ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
            pip_dependency_groups=["cpu"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
            },
        ),
        config=config,
    )
