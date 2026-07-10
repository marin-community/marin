# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Uses download_hf to download a pretokenized dataset cache from Hugging Face
and prepares it as a tokenized dataset source for Levanter.
"""

import dataclasses
import logging

from levanter.data.text.datasets import LmDatasetSourceConfigBase, UrlDatasetSourceConfig
from levanter.data.text.formats import LmDatasetFormatBase, TextLmDatasetFormat
from levanter.store.cache import CacheOptions

from marin.datakit.download.huggingface import (
    DownloadConfig as HfDownloadConfig,
)
from marin.datakit.download.huggingface import (
    download_hf as hf_download_logic,
)
from marin.processing.tokenize.tokenize import TokenizeConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PretokenizedCacheDownloadConfig(TokenizeConfigBase):
    """Configuration for downloading a pre-existing Levanter cache from Hugging Face."""

    # Fields from TokenizeConfigBase
    cache_path: str  # Where the cache will be downloaded to
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
        self, actual_output_path: str | None, *, include_raw_paths=True
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


def fetch_pretokenized_cache(
    cfg: PretokenizedCacheDownloadConfig,
) -> PretokenizedCacheDownloadConfig:
    """Download the Levanter cache described by ``cfg`` from Hugging Face into
    ``cfg.cache_path`` and return ``cfg``."""
    logger.info(
        f"Starting download of pretokenized cache '{cfg.hf_repo_id}' (revision: {cfg.hf_revision}) "
        f"to '{cfg.cache_path}'."
    )

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
        raise

    return cfg
