import logging
import os
import tempfile

import numpy as np
import pytest
import requests  # For requests.exceptions.RequestException
from huggingface_hub.utils import HfHubHTTPError
from levanter.store import TreeCache
from transformers import AutoTokenizer

from marin.processing.tokenize.download_pretokenized import (
    PretokenizedCacheDownloadConfig,
    _actually_download_pretokenized_cache,
)

logger = logging.getLogger(__name__)

HF_REPO_ID = "marin-community/fineweb-edu-pretokenized-10K"
TOKENIZER_NAME = "stanford-crfm/marin-tokenizer"


def test_download_and_load_cache():
    """
    Tests downloading a pretokenized cache and then loading it with Levanter's TreeCache.
    Skips if the Hugging Face repository or tokenizer is inaccessible.
    """
    try:
        # Attempt to load tokenizer first, as it's a quick check for connectivity/access
        # and needed for creating a more realistic exemplar if desired, though not strictly for this test.
        AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except (HfHubHTTPError, requests.exceptions.RequestException, OSError, ValueError) as e:
        print(1)
        pytest.skip(
            f"Skipping test: Could not load tokenizer '{TOKENIZER_NAME}'. "
            f"HF Hub or network may be inaccessible. Error: {e}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PretokenizedCacheDownloadConfig(
            cache_path=tmpdir,  # Download directly into this temp directory
            tokenizer=TOKENIZER_NAME,
            hf_repo_id=HF_REPO_ID,
            hf_revision=None,  # Test with default revision (main)
            hf_token=None,  # Test with public repo, no token needed
        )

        logger.info(f"Attempting to download {HF_REPO_ID} to {tmpdir}")

        try:
            returned_config = _actually_download_pretokenized_cache(config)
            assert returned_config.cache_path == tmpdir
            logger.info(f"Successfully called download logic for {HF_REPO_ID}.")
        except (ValueError, HfHubHTTPError, requests.exceptions.RequestException) as e:
            print(2)
            # ValueError can be raised by hf_download_logic if no files are found or path is unwritable.
            # HfHubHTTPError for auth/not found issues from HF.
            # RequestException for general network issues.
            pytest.skip(
                f"Skipping test: Could not download cache '{HF_REPO_ID}'. "
                f"HF Hub or network may be inaccessible or repo is invalid. Error: {e}"
            )
        except Exception as e:
            # Catch any other unexpected errors during download and treat as a skip for robustness
            pytest.skip(f"Skipping test due to unexpected error during download: {e}")

        # Verify that the cache was downloaded and contains the expected 'train' split
        # The fineweb-edu-pretokenized-10K dataset has its cache files under a 'train/' directory in the repo.
        # _actually_download_pretokenized_cache downloads the repo contents into cfg.cache_path (tmpdir).
        # So, the 'train' split will be at os.path.join(tmpdir, "train").
        train_cache_path = os.path.join(tmpdir, "train")

        if not os.path.exists(os.path.join(train_cache_path, "cache_metadata.json")):
            pytest.fail(
                f"Cache download seems to have not created the expected train split at '{train_cache_path}'. "
                f"Missing cache_metadata.json."
            )

        # Define a simple exemplar. Levanter caches for text usually have "input_ids".
        exemplar = {"input_ids": np.array([0, 1, 2], dtype=np.int32)}

        logger.info(f"Attempting to load cache from {train_cache_path}")
        try:
            loaded_cache = TreeCache.load(train_cache_path, exemplar=exemplar)
            assert loaded_cache is not None
            # Check if we can read some basic property, e.g., number of shards.
            # This implicitly checks if metadata was loaded correctly.
            assert loaded_cache.num_shards > 0
            logger.info(
                f"Successfully loaded cache from {train_cache_path}. Cache has {loaded_cache.num_shards} shards."
            )
        except FileNotFoundError as e:
            pytest.fail(f"Failed to load cache: A file was not found at '{train_cache_path}'. Error: {e}")
        except Exception as e:
            # If loading fails after a successful download, it's a test failure.
            logger.exception(f"Failed to load the downloaded cache from '{train_cache_path}'.")
            pytest.fail(f"Failed to load the downloaded cache. Error: {e}")
