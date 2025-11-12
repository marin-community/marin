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

import asyncio

import jax
import numpy as np
import pytest
from levanter.data.mixture import MixtureDataset
from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, TreeCache
from marin.execution import InputName
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, tokenize

# Dummy values for other required TokenizeConfig fields
DUMMY_CACHE_PATH = "/dummy/cache"
DUMMY_TOKENIZER = "dummy_tokenizer"
DUMMY_VALIDATION_PATHS = []


@pytest.mark.parametrize(
    "train_paths, should_error, expected_error_path",
    [
        (["gs://bucket/data/train/file.jsonl"], False, None),
        (["gs://bucket/data/test/file.jsonl"], True, "gs://bucket/data/test/file.jsonl"),
        (["gs://bucket/data/validation/file.jsonl"], True, "gs://bucket/data/validation/file.jsonl"),
        (["gs://bucket/data/latest_updates/file.jsonl"], False, None),
        (
            [
                "gs://bucket/data/train/file1.jsonl",
                "gs://bucket/data/test/file2.jsonl",
                "gs://bucket/data/train/file3.jsonl",
            ],
            True,
            "gs://bucket/data/test/file2.jsonl",
        ),
        ([], False, None),
    ],
)
def test_train_paths_variants(train_paths, should_error, expected_error_path):
    if should_error:
        with pytest.raises(ValueError) as excinfo:
            TokenizeConfig(
                train_paths=train_paths,
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
        if expected_error_path:
            assert expected_error_path in str(excinfo.value)
    else:
        try:
            TokenizeConfig(
                train_paths=train_paths,
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        except ValueError as e:
            if "contains a forbidden pattern" in str(e):
                pytest.fail("Unexpected ValueError for valid path")


@pytest.mark.parametrize(
    "input_name, should_error",
    [
        (InputName.hardcoded("gs://bucket/data/train/file.jsonl"), False),
        (InputName.hardcoded("gs://bucket/data/test/file.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/validation/file.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/latest_updates/file.jsonl"), False),
        (InputName.hardcoded("gs://bucket/data/train/file_test.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/train/file_validation.jsonl"), True),
    ],
)
def test_inputname_variants(input_name, should_error):
    if should_error:
        with pytest.raises(ValueError) as excinfo:
            TokenizeConfig(
                train_paths=[input_name],
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
        assert input_name.name in str(excinfo.value)
    else:
        try:
            TokenizeConfig(
                train_paths=[input_name],
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        except ValueError as e:
            if "contains a forbidden pattern" in str(e):
                pytest.fail("Unexpected ValueError for valid InputName")


def test_mixed_paths_one_invalid_inputname():
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[
                "gs://bucket/data/train/file1.jsonl",
                InputName.hardcoded("gs://bucket/data/test/file2.jsonl"),
                "gs://bucket/data/train/file3.jsonl",
            ],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file2.jsonl" in str(excinfo.value)


@pytest.mark.slow
def test_tokenize_full_pipeline_integration(tmp_path):
    """Integration test for the full tokenization pipeline."""
    config = HfTokenizeConfig(
        id="dlwh/wikitext_2_detokenized",
        cache_path=str(tmp_path / "cache"),
        tokenizer="gpt2",  # Simple, widely available tokenizer
        sample_count=100,  # Keep test fast by limiting samples
        format=TextLmDatasetFormat(),
    )

    tokenize(config)

    train_cache_dir = tmp_path / "cache" / "train"
    train_ledger_path = train_cache_dir / "shard_ledger.json"
    assert train_ledger_path.exists(), f"Ledger not found at {train_ledger_path}"

    ledger = CacheLedger.load(str(train_cache_dir))
    assert ledger.is_finished, "Ledger should be marked as finished"
    assert ledger.total_num_rows > 0, f"Cache should have non-zero rows, got {ledger.total_num_rows}"

    print("\nLedger info:")
    print(f"  total_num_rows: {ledger.total_num_rows}")
    print(f"  shard_rows: {ledger.shard_rows}")
    print(f"  finished_shards: {ledger.finished_shards}")

    # The exemplar should match the output structure of tokenization
    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    cache = TreeCache.load(str(train_cache_dir), exemplar=exemplar)

    cache_len = len(cache)
    assert cache_len == ledger.total_num_rows, f"Cache length {cache_len} != ledger rows {ledger.total_num_rows}"

    first_example = cache[0]
    assert "input_ids" in first_example, "Example should have input_ids field"

    print("\nFirst 5 examples:")
    for i in range(min(5, cache_len)):
        example = cache[i]
        print(f"  Example {i}: input_ids length = {len(example['input_ids'])}")
        assert len(example["input_ids"]) > 0, f"Example {i} has empty input_ids"

    # 8. Test that the cache can be used in a mixture without ZeroDivisionError

    mixture = MixtureDataset(
        datasets={"test": cache},
        weights={"test": 1.0},
        block_size=128,
        key=jax.random.PRNGKey(0),
    )

    # This should not raise ZeroDivisionError
    mixture_example = asyncio.run(mixture.getitem_async(0))
    assert mixture_example is not None
    assert "input_ids" in mixture_example
    print("\nSuccessfully created mixture and sampled example!")
