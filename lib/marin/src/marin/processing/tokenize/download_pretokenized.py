# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Uses download_hf to download a pretokenized dataset cache from Hugging Face
and prepares it as a tokenized dataset source for Levanter.
"""

import dataclasses
import logging
import os

from levanter.data.text import (
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.store.cache import CacheOptions

from marin.download.huggingface.download_hf import (
    DownloadConfig as HfDownloadConfig,
    download_hf as hf_download_logic,
)
from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, InputName, ensure_versioned
from marin.processing.tokenize.tokenize import TokenizeConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PretokenizedCacheDownloadConfig(TokenizeConfigBase):
    """Configuration for downloading a pre-existing Levanter cache from Hugging Face."""

    # Fields from TokenizeConfigBase
    cache_path: str  # Resolved by Executor: where the cache will be downloaded to
    tokenizer: str  # Tokenizer name/path associated with this cache

    # Fields specific to downloading from HF
    hf_repo_id: str  # Hugging Face repository ID (e.g., "username/my_cache_repo")
    hf_revision: str | None = None  # Revision, branch, or tag on Hugging Face
    hf_repo_type_prefix: str = "datasets"  # Typically "datasets" or "models"
    hf_token: str | None = None  # Hugging Face API token for private repositories

    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    cache_options: CacheOptions | None = None  # For TokenizeConfigBase interface, not used during download

    tags: list[str] = dataclasses.field(default_factory=list)  # Tags for Levanter's dataset source config

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
    ) -> LmDatasetSourceConfigBase:
        """
        Returns a Levanter dataset source config that points to the downloaded cache.
        Since the cache is already tokenized and in Levanter format, train_urls and validation_urls
        are empty. Levanter will load directly from the cache_dir.
        """
        if actual_output_path is None:
            raise ValueError("actual_output_path must be provided for a downloaded cache.")

        return UrlDatasetSourceConfig(
            tags=self.tags,
            train_urls=[],
            validation_urls=[],  # No raw validation URLs; cache is already built
            cache_dir=actual_output_path,
            format=self.format,  # Retain format info if needed downstream
        )


def download_pretokenized_cache(
    output_cache_path_name: str,  # Name for the ExecutorStep, forms part of the output path
    hf_repo_id: str,
    tokenizer: str,  # The tokenizer this cache was built with
    hf_revision: str | None = None,
    hf_token: str | None = None,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa: A002
) -> ExecutorStep[PretokenizedCacheDownloadConfig]:
    """
    Creates an ExecutorStep to download a pre-tokenized Levanter cache from Hugging Face.

    Args:
        output_cache_path_name: The logical name for this download step. The Executor will use this
                                to construct the actual output directory for the cache.
                                "tokenized/subcache" will be prepended to this name.
        hf_repo_id: The Hugging Face repository ID (e.g., "username/my_cache_repo").
        tokenizer: The name or path of the tokenizer associated with this cache.
        hf_revision: The specific revision, branch, or tag of the repository to download.
        hf_token: An optional Hugging Face API token for accessing private repositories.
        tags: Optional list of tags for the Levanter dataset source config.
        format: The format of the dataset (default is TextLmDatasetFormat).

    Returns:
        An ExecutorStep that, when run, will download the cache and output a
        PretokenizedCacheDownloadConfig pointing to the downloaded data.
    """
    config = PretokenizedCacheDownloadConfig(
        cache_path=THIS_OUTPUT_PATH,  # ExecutorStep will resolve this to the actual output path
        tokenizer=ensure_versioned(tokenizer),
        hf_repo_id=ensure_versioned(hf_repo_id),  # type: ignore[call-arg]
        hf_revision=ensure_versioned(hf_revision),  # type: ignore[call-arg]
        hf_repo_type_prefix="datasets",  # Default for Hugging Face datasets
        hf_token=hf_token,
        tags=tags or [],
        format=format,
    )

    return ExecutorStep(
        name=os.path.join("tokenized", "subcache", output_cache_path_name),
        fn=_actually_download_pretokenized_cache,
        config=config,
    )


def _actually_download_pretokenized_cache(
    cfg: PretokenizedCacheDownloadConfig,
) -> PretokenizedCacheDownloadConfig:
    """
    The function executed by the ExecutorStep to download the cache.
    It uses the hf_download logic from operations.download.huggingface.download.
    """
    logger.info(
        f"Starting download of pretokenized cache '{cfg.hf_repo_id}' (revision: {cfg.hf_revision}) "
        f"to '{cfg.cache_path}'."
    )

    # The hf_download_logic uses HF_TOKEN from environment variables.
    # Temporarily set it if provided in the config.

    try:
        # Map our config to the HfDownloadConfig required by hf_download_logic
        download_op_config = HfDownloadConfig(
            hf_dataset_id=cfg.hf_repo_id,
            revision=cfg.hf_revision,
            hf_repo_type_prefix=cfg.hf_repo_type_prefix,
            gcs_output_path=cfg.cache_path,
        )

        # Execute the download
        hf_download_logic(download_op_config)

        logger.info(f"Successfully downloaded pretokenized cache to '{cfg.cache_path}'.")

    except Exception:
        logger.exception(f"Failed to download pretokenized cache '{cfg.hf_repo_id}' to '{cfg.cache_path}'.")
        raise  # Re-raise the exception to mark the step as failed

    # The ExecutorStep's output is the config object itself.
    # After this function, cfg.cache_path (resolved by the Executor)
    # now contains the downloaded cache.
    return cfg
