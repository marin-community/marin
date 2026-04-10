# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.count_tokens import _is_matching_output_dir, _matching_output_dirs


def test_is_matching_output_dir_requires_basename_boundary():
    assert _is_matching_output_dir("bucket/tokenized/foo-123", "foo")
    assert _is_matching_output_dir("bucket/tokenized/foo_legacy", "foo")
    assert not _is_matching_output_dir("bucket/tokenized/foobar-123", "foo")
    assert not _is_matching_output_dir("bucket/tokenized/foo", "foo")


def test_matching_output_dirs_filters_to_expected_entries():
    entries = [
        "bucket/tokenized/foo-123",
        "bucket/tokenized/foo_legacy",
        "bucket/tokenized/foobar-123",
        "bucket/tokenized/bar-123",
    ]

    assert _matching_output_dirs(entries, "foo") == [
        "bucket/tokenized/foo-123",
        "bucket/tokenized/foo_legacy",
    ]
