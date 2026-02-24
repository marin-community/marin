# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import jax
import numpy as np
import pytest
from levanter.data.mixture import MixtureDataset
from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, TreeCache
from marin.execution import InputName
from marin.processing.tokenize.tokenize import (
    MIN_GROUP_BYTES,
    HfTokenizeConfig,
    TokenizeConfig,
    _bundle_files_by_size,
    compute_target_group_bytes,
    tokenize,
)

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


@pytest.mark.parametrize(
    "total_bytes, max_workers, expected",
    [
        # Normal: 100 GB across 100 workers → 1 GB per group
        (100_000_000_000, 100, 1_000_000_000),
        # Floor kicks in: 1 GB across 100 workers → would be 10 MB, but MIN_GROUP_BYTES = 100 MB
        (1_000_000_000, 100, MIN_GROUP_BYTES),
        # Single worker: entire dataset in one group
        (50_000_000_000, 1, 50_000_000_000),
        # Tiny dataset: floor still applies
        (10_000_000, 4096, MIN_GROUP_BYTES),
        # Exact division
        (4_000_000_000, 4, 1_000_000_000),
    ],
)
def test_compute_target_group_bytes(total_bytes, max_workers, expected):
    assert compute_target_group_bytes(total_bytes, max_workers) == expected


def test_bundle_files_produces_expected_groups():
    """Auto-computed grouping should produce approximately max_workers groups."""
    file_infos = [{"filename": f"file_{i}.jsonl", "size": 500_000_000} for i in range(20)]
    total_bytes = sum(f["size"] for f in file_infos)  # 10 GB total
    max_workers = 4
    target = compute_target_group_bytes(total_bytes, max_workers)  # 2.5 GB per group

    groups = list(_bundle_files_by_size(file_infos, target))
    # _bundle_files_by_size yields a group when adding the next file would reach
    # the target (uses >=). With target=2.5 GB and 500 MB files, each group fits
    # 4 files (2 GB < 2.5 GB), yielding 5 groups.
    assert len(groups) == 5
    for group in groups:
        assert len(group) == 4


def test_bundle_files_single_large_file():
    """A single file larger than target_group_bytes gets its own group."""
    file_infos = [
        {"filename": "big.jsonl", "size": 5_000_000_000},
        {"filename": "small1.jsonl", "size": 100_000_000},
        {"filename": "small2.jsonl", "size": 100_000_000},
    ]
    target = 1_000_000_000  # 1 GB
    groups = list(_bundle_files_by_size(file_infos, target))
    assert groups[0] == ["big.jsonl"]
    assert groups[1] == ["small1.jsonl", "small2.jsonl"]


@pytest.mark.slow
def test_tokenize_full_pipeline_integration(tmp_path):
    """Integration test for the full tokenization pipeline."""
    config = HfTokenizeConfig(
        id="dlwh/wikitext_103_detokenized",
        cache_path=str(tmp_path / "cache"),
        tokenizer="gpt2",
        sample_count=100,
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
