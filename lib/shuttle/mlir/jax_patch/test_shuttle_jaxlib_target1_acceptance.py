# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static installed-wheel gates that do not pretend to run the unbuilt wheel."""

import hashlib
import json
from pathlib import Path

import pytest
from shuttle_jaxlib_target1_acceptance import (
    PIPELINE_ABI_VERSION,
    acceptance_tuning,
    attributed_new_cache_entry,
    cache_snapshot,
    contract_identities,
    expected_identity,
)

from shuttle import Numerics, compiler_options


def test_driver_has_exact_twelve_cell_identity_matrix() -> None:
    identities = contract_identities()
    assert len(identities) == len(set(identities)) == 12
    assert set(identities) == {
        (shape_id, boundary, policy)
        for shape_id in ("44d152ecc3e9ff18", "81928ab3539c0f03")
        for boundary in ("forward", "backward", "composed")
        for policy in ("source_ordered", "fast")
    }


def test_driver_pins_abi_and_policy_in_public_compiler_options() -> None:
    assert PIPELINE_ABI_VERSION == 9
    for numerics in Numerics:
        canonical = compiler_options(numerics=numerics, tuning=acceptance_tuning())["xla_shuttle_options"]
        payload = json.loads(canonical)
        identity = expected_identity(numerics)
        assert payload["pipeline_abi_version"] == 9
        assert payload["numerics"] == numerics.value
        assert identity.policy == numerics.value
        assert identity.canonical_options == canonical


def test_cache_entry_attribution_rejects_zero_or_multiple_deltas() -> None:
    assert attributed_new_cache_entry(frozenset({"old-cache"}), frozenset({"old-cache", "new-cache"}), "cell") == (
        "new-cache"
    )
    for after in (frozenset({"old-cache"}), frozenset({"old-cache", "one-cache", "two-cache"})):
        try:
            attributed_new_cache_entry(frozenset({"old-cache"}), after, "cell")
        except AssertionError as error:
            assert "exactly one cache entry" in str(error)
        else:
            raise AssertionError("invalid cache delta was accepted")


def test_cache_snapshot_rejects_unknown_nested_and_link_entries(tmp_path: Path) -> None:
    valid = tmp_path / ("jit_forward-" + "0" * 64 + "-cache")
    valid.write_bytes(b"cache payload")
    expected_digest = hashlib.sha256(b"cache payload").hexdigest()
    assert cache_snapshot(tmp_path) == {valid.name: (13, expected_digest)}

    unknown = tmp_path / "ignored-atime"
    unknown.write_bytes(b"unknown")
    with pytest.raises(AssertionError, match="unknown entry"):
        cache_snapshot(tmp_path)
    unknown.unlink()

    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(AssertionError, match="non-regular"):
        cache_snapshot(tmp_path)
    nested.rmdir()

    link = tmp_path / ("jit_backward-" + "1" * 64 + "-cache")
    link.symlink_to(valid)
    with pytest.raises(AssertionError, match="non-regular"):
        cache_snapshot(tmp_path)


def test_cache_snapshot_detects_byte_mutation(tmp_path: Path) -> None:
    cache = tmp_path / ("jit_composed-" + "2" * 64 + "-cache")
    cache.write_bytes(b"before")
    before = cache_snapshot(tmp_path)
    cache.write_bytes(b"after!")
    assert cache_snapshot(tmp_path) != before
