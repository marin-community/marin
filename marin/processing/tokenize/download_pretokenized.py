"""
Uses download_hf to download a pretokenized dataset cache from Hugging Face
and prepares it as a tokenized dataset source for Levanter.
"""
import dataclasses
import logging
import os

from levanter.data.text import (
    LMDatasetSourceConfig,
    LmDatasetFormatBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.store.cache import CacheOptions

from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, InputName
from marin.processing.tokenize.tokenize import TokenizeConfigBase
from operations.download.huggingface.download import DownloadConfig as HfDownloadConfig
from operations.download.huggingface.download_hf import (
    download_hf as hf_download_logic,
)

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

    format: LmDatasetFormatBase = TextLmDatasetFormat()
    cache_options: CacheOptions | None = None  # For TokenizeConfigBase interface, not used during download

    tags: list[str] = dataclasses.field(default_factory=list)  # Tags for Levanter's LMDatasetSourceConfig

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths=True
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
            train_urls=[],  # No raw train URLs; cache is already built
            validation_urls=[],  # No raw validation URLs; cache is already built
            cache_dir=str(actual_output_path),  # This is where the cache was downloaded
            format=self.format,  # Retain format info if needed downstream
        )


def download_pretokenized_cache(
    output_cache_path_name: str,  # Name for the ExecutorStep, forms part of the output path
    hf_repo_id: str,
    tokenizer_name: str,  # The tokenizer this cache was built with
    hf_revision: str | None = None,
    hf_repo_type_prefix: str = "datasets",
    hf_token: str | None = None,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),
) -> ExecutorStep[PretokenizedCacheDownloadConfig]:
    """
    Creates an ExecutorStep to download a pre-tokenized Levanter cache from Hugging Face.

    Args:
        output_cache_path_name: The logical name for this download step. The Executor will use this
                                to construct the actual output directory for the cache.
        hf_repo_id: The Hugging Face repository ID (e.g., "username/my_cache_repo").
        tokenizer_name: The name or path of the tokenizer associated with this cache.
        hf_revision: The specific revision, branch, or tag of the repository to download.
        hf_repo_type_prefix: The type of Hugging Face repository (e.g., "datasets", "models").
        hf_token: An optional Hugging Face API token for accessing private repositories.
        tags: Optional list of tags for the Levanter LMDatasetSourceConfig.

    Returns:
        An ExecutorStep that, when run, will download the cache and output a
        PretokenizedCacheDownloadConfig pointing to the downloaded data.
    """
    config = PretokenizedCacheDownloadConfig(
        cache_path=THIS_OUTPUT_PATH,  # ExecutorStep will resolve this to the actual output path
        tokenizer=tokenizer_name,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_repo_type_prefix=hf_repo_type_prefix,
        hf_token=hf_token,
        tags=tags or [],
        format=format,
    )

    return ExecutorStep(
        name=output_cache_path_name,
        fn=_actually_download_pretokenized_cache,
        config=config,
        # TODO: consider resource requests if download is heavy
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
    original_hf_token_env = os.environ.get("HF_TOKEN")
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token
        logger.debug("Temporarily set HF_TOKEN from config for download.")

    try:
        # Map our config to the HfDownloadConfig required by hf_download_logic
        download_op_config = HfDownloadConfig(
            hf_dataset_id=cfg.hf_repo_id,
            revision=cfg.hf_revision,
            hf_repo_type_prefix=cfg.hf_repo_type_prefix,
            gcs_output_path=cfg.cache_path,  # Download directly into the step's output path
            # hf_urls_glob is intentionally None to download all files in the repo path,
            # which is usually desired for a complete Levanter cache.
        )

        # Execute the download
        hf_download_logic(download_op_config)

        logger.info(f"Successfully downloaded pretokenized cache to '{cfg.cache_path}'.")

    except Exception:
        logger.exception(
            f"Failed to download pretokenized cache '{cfg.hf_repo_id}' to '{cfg.cache_path}'."
        )
        raise  # Re-raise the exception to mark the step as failed
    finally:
        # Restore original HF_TOKEN environment state
        if cfg.hf_token:  # If we set it from config
            if original_hf_token_env is not None:
                os.environ["HF_TOKEN"] = original_hf_token_env
                logger.debug("Restored original HF_TOKEN.")
            else:
                del os.environ["HF_TOKEN"]
                logger.debug("Cleared temporarily set HF_TOKEN.")

    # The ExecutorStep's output is the config object itself.
    # After this function, cfg.cache_path (resolved by the Executor)
    # now contains the downloaded cache.
    return cfg
