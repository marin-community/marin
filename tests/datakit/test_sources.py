# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline sanity checks for the Datakit source registry."""

import re

import pytest

from marin.datakit.sources import DatakitSource, all_sources, pinned_sources

_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

_ALL = all_sources()
_PINNED = pinned_sources()


def test_sources_count_matches_token_count_viewer():
    """The token-count-viewer Space backs 102 datasets; the registry must mirror."""
    assert len(_ALL) == 102


def test_sources_dict_is_cached():
    assert all_sources() is _ALL


def test_dict_keys_match_source_names():
    for name, src in _ALL.items():
        assert name == src.name


@pytest.mark.parametrize("src", list(_ALL.values()), ids=lambda s: s.name)
def test_source_fields(src: DatakitSource):
    assert src.name, "name must be non-empty"
    if src.hf_dataset_id:
        assert "/" in src.hf_dataset_id, f"{src.name}: hf_dataset_id must look like org/repo"
    if src.revision is not None:
        assert _HF_COMMIT_RE.match(src.revision), f"{src.name}: revision {src.revision!r} is not a hex SHA"
    assert src.id_field, f"{src.name}: id_field must be non-empty (silent source_id drop risk)"
    assert src.text_field, f"{src.name}: text_field must be non-empty"
    assert src.file_extensions, f"{src.name}: file_extensions must be non-empty"
    if src.rough_token_count_b is not None:
        assert src.rough_token_count_b > 0, f"{src.name}: rough_token_count_b must be positive if set"
    if src.hf_urls_glob is not None:
        assert isinstance(src.hf_urls_glob, tuple) and all(src.hf_urls_glob)


def test_pinned_sources_only_have_revision_and_repo():
    for src in _PINNED.values():
        assert src.revision is not None
        assert src.hf_dataset_id != ""
    # Entries in the full registry but not pinned must be missing one or the other.
    for name, src in _ALL.items():
        if name not in _PINNED:
            assert src.revision is None or src.hf_dataset_id == ""


def test_pinned_sources_is_majority():
    assert len(_PINNED) >= len(_ALL) * 0.8


def test_pinned_sources_is_cached():
    assert pinned_sources() is _PINNED


def test_lookup_by_name():
    for name in _ALL:
        assert _ALL[name].name == name
    with pytest.raises(KeyError):
        _ = _ALL["does-not-exist"]
