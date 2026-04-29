# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``infra/configure_main_buckets.py``.

The script is a shell-out wrapper around `gcloud`; only its pure helpers
(rule merging, soft-delete detection, rule shape) are unit-tested here.
The shell-out branches are exercised by the canary smoke workflow.
"""

import importlib.util
from pathlib import Path

import pytest

from rigging.filesystem import ALLOWED_TTL_DAYS, REGION_TO_DATA_BUCKET, TEMP_PATH_PREFIX

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "infra" / "configure_main_buckets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("configure_main_buckets", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cmb():
    return _load_module()


def test_buckets_match_runtime_data_bucket_map(cmb):
    """The script's BUCKETS map must mirror the rigging REGION_TO_DATA_BUCKET map."""
    assert cmb.BUCKETS == {bucket: region for region, bucket in REGION_TO_DATA_BUCKET.items()}


def test_build_ttl_rules_emits_one_per_allowed_ttl(cmb):
    rules = cmb.build_ttl_rules()
    assert len(rules) == len(ALLOWED_TTL_DAYS)
    for rule, n in zip(rules, ALLOWED_TTL_DAYS, strict=True):
        assert rule == {
            "action": {"type": "Delete"},
            "condition": {"age": n, "matchesPrefix": [f"{TEMP_PATH_PREFIX}/ttl={n}d/"]},
        }


def test_is_marin_ttl_rule_recognizes_owned_shape(cmb):
    for n in ALLOWED_TTL_DAYS:
        assert cmb._is_marin_ttl_rule(
            {
                "action": {"type": "Delete"},
                "condition": {"age": n, "matchesPrefix": [f"{TEMP_PATH_PREFIX}/ttl={n}d/"]},
            }
        )


@pytest.mark.parametrize(
    "rule",
    [
        # Wrong action.
        {"action": {"type": "SetStorageClass"}, "condition": {"age": 1, "matchesPrefix": ["tmp/ttl=1d/"]}},
        # Wrong (extra) condition keys.
        {
            "action": {"type": "Delete"},
            "condition": {"age": 1, "matchesPrefix": ["tmp/ttl=1d/"], "isLive": True},
        },
        # Different prefix entirely.
        {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["scratch/"]}},
        # Multiple prefixes — not our shape.
        {
            "action": {"type": "Delete"},
            "condition": {"age": 1, "matchesPrefix": ["tmp/ttl=1d/", "tmp/ttl=2d/"]},
        },
        # TTL value outside the allowed set.
        {"action": {"type": "Delete"}, "condition": {"age": 9, "matchesPrefix": ["tmp/ttl=9d/"]}},
    ],
)
def test_is_marin_ttl_rule_rejects_foreign_rules(cmb, rule):
    assert not cmb._is_marin_ttl_rule(rule)


def test_merge_preserves_foreign_rules_and_replaces_owned(cmb):
    foreign_keep = {"action": {"type": "Delete"}, "condition": {"age": 90, "matchesPrefix": ["archive/"]}}
    foreign_setclass = {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 365},
    }
    stale_owned = {
        "action": {"type": "Delete"},
        "condition": {"age": 1, "matchesPrefix": [f"{TEMP_PATH_PREFIX}/ttl=1d/"]},
    }

    owned = cmb.build_ttl_rules()
    merged = cmb.merge_lifecycle_rules([foreign_keep, stale_owned, foreign_setclass], owned)

    # Foreign rules survive in original order.
    assert merged[0] == foreign_keep
    assert merged[1] == foreign_setclass
    # Then exactly one copy of every owned rule.
    assert merged[2:] == owned


def test_merge_on_empty_existing_returns_owned(cmb):
    owned = cmb.build_ttl_rules()
    assert cmb.merge_lifecycle_rules([], owned) == owned


@pytest.mark.parametrize(
    "info, expected",
    [
        ({}, False),
        ({"soft_delete_policy": {}}, False),
        ({"soft_delete_policy": {"retentionDurationSeconds": "0"}}, False),
        ({"soft_delete_policy": {"retentionDurationSeconds": "604800"}}, True),
        ({"softDeletePolicy": {"retention_duration_seconds": 86400}}, True),
    ],
)
def test_soft_delete_enabled(cmb, info, expected):
    assert cmb.soft_delete_enabled(info) is expected
