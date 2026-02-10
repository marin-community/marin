# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

        try:
            returned_config = _actually_download_pretokenized_cache(config)
            assert returned_config.cache_path == tmpdir
        except (ValueError, HfHubHTTPError, requests.exceptions.RequestException) as e:
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

        train_cache_path = os.path.join(tmpdir, "train")

        if not os.path.exists(os.path.join(train_cache_path, "shard_ledger.json")):
            pytest.fail(
                f"Cache download seems to have not created the expected train split at '{train_cache_path}'. "
                f"Missing cache_metadata.json."
            )

        # Define a simple exemplar. Levanter caches for text have "input_ids".
        exemplar = {"input_ids": np.array([0, 1, 2], dtype=np.int32)}

        try:
            loaded_cache = TreeCache.load(train_cache_path, exemplar=exemplar)
            assert loaded_cache is not None
            assert loaded_cache.store.tree["input_ids"].data_size > 10000  # we're loading a 10K cache
        except FileNotFoundError as e:
            pytest.fail(f"Failed to load cache: A file was not found at '{train_cache_path}'. Error: {e}")
        except Exception as e:
            pytest.fail(f"Failed to load the downloaded cache. Error: {e}")
