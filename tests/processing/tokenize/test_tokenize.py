# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger
from marin.processing.tokenize.tokenize import (
    MIN_GROUP_BYTES,
    TokenizeConfig,
    bundle_files_by_size,
    compute_target_group_bytes,
    tokenize,
)
from zephyr.dataset import FileEntry
from zephyr.readers import InputFileSpec

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
        # 'test'/'validation' as a filename substring (underscore boundary) is still forbidden.
        (["gs://bucket/data/train/file_test.jsonl"], True, "gs://bucket/data/train/file_test.jsonl"),
        (["gs://bucket/data/train/file_validation.jsonl"], True, "gs://bucket/data/train/file_validation.jsonl"),
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


def _fe(path: str, size: int) -> FileEntry:
    return FileEntry(spec=InputFileSpec(path=path), size=size)


def test_bundle_files_produces_expected_groups():
    """Auto-computed grouping should produce approximately max_workers groups."""
    files = [_fe(f"file_{i}.jsonl", 500_000_000) for i in range(20)]
    total_bytes = sum(f.size for f in files)  # 10 GB total
    max_workers = 4
    target = compute_target_group_bytes(total_bytes, max_workers)  # 2.5 GB per group

    groups = list(bundle_files_by_size(files, target))
    # bundle_files_by_size yields a group when adding the next file would reach
    # the target (uses >=). With target=2.5 GB and 500 MB files, each group fits
    # 4 files (2 GB < 2.5 GB), yielding 5 groups.
    assert len(groups) == 5
    for group in groups:
        assert len(group) == 4


def test_bundle_files_single_large_file():
    """A single file larger than target_group_bytes gets its own group."""
    files = [
        _fe("big.jsonl", 5_000_000_000),
        _fe("small1.jsonl", 100_000_000),
        _fe("small2.jsonl", 100_000_000),
    ]
    target = 1_000_000_000  # 1 GB
    groups = list(bundle_files_by_size(files, target))
    assert groups[0] == ["big.jsonl"]
    assert groups[1] == ["small1.jsonl", "small2.jsonl"]


def test_tokenize_skips_empty_leading_data(tmp_path):
    """Empty shards and documents must not determine the cache exemplar."""
    # train_paths must not contain "test"; pytest's tmp_path always does.
    with tempfile.TemporaryDirectory(prefix="sparse_") as raw_dir:
        data_dir = Path(raw_dir)
        pq.write_table(
            pa.table({"text": pa.array([], type=pa.string())}),
            str(data_dir / "data-00000.parquet"),
        )
        pq.write_table(
            pa.table({"text": ["", "hello world"]}),
            str(data_dir / "data-00001.parquet"),
        )

        config = TokenizeConfig(
            train_paths=[f"{data_dir}/*.parquet"],
            validation_paths=[],
            cache_path=str(tmp_path / "cache"),
            tokenizer="gpt2",
            format=TextLmDatasetFormat(text_key="text"),
        )
        tokenize(config)

    ledger = CacheLedger.load(str(tmp_path / "cache" / "train"))
    assert ledger.is_finished
    assert ledger.total_num_rows == 1
