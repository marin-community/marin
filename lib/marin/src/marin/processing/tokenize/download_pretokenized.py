# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Library code for downloading pretokenized dataset caches from Hugging Face.

This module contains pure processing functions that work with concrete paths.
For step wrappers that handle dependencies, see experiments/steps/download_pretokenized.py
"""

import dataclasses
import logging

from levanter.data.text import (
    LmDatasetFormatBase,
    LMDatasetSourceConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.store.cache import CacheOptions

from marin.download.huggingface.download_hf import (
    DownloadConfig as HfDownloadConfig,
    download_hf as hf_download_logic,
)
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

    tags: list[str] = dataclasses.field(default_factory=list)  # Tags for Levanter's LMDatasetSourceConfig

    def as_lm_dataset_source_config(
        self, actual_output_path: str | None, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        """
        Returns a Levanter LMDatasetSourceConfig that points to the downloaded cache.
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
    cfg: PretokenizedCacheDownloadConfig,
) -> PretokenizedCacheDownloadConfig:
    """
    Downloads a pre-tokenized Levanter cache from Hugging Face.

    This is the library function that performs the actual download.
    It uses the hf_download logic from marin.download.huggingface.download_hf.

    Args:
        cfg: Configuration specifying the cache to download, including the HF repo ID,
             revision, and destination path.

    Returns:
        The same config object, with the cache now downloaded to cfg.cache_path.
    """
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
        raise  # Re-raise the exception to mark the step as failed

    # After this function, cfg.cache_path now contains the downloaded cache.
    return cfg
