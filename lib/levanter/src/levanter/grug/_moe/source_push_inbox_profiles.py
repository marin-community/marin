# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Named source-push inbox benchmark profile defaults."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


SOURCE_PUSH_PROFILE_NONE = "none"
SOURCE_PUSH_PROFILE_STABLE_216 = "hopper_source_push_inbox_rough_balanced_216"
SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072 = "blackwell_source_push_inbox_65k_d3072_i3072"

SOURCE_PUSH_PROFILES = (
    SOURCE_PUSH_PROFILE_NONE,
    SOURCE_PUSH_PROFILE_STABLE_216,
    SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
)

_COMMON_HOPPER_QUEUE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "tokens_per_rank": 32768,
        "hidden_dim": 2560,
        "intermediate_dim": 1280,
        "experts_per_rank": 32,
        "topk": 4,
        "entries_per_rank": 288,
        "inbox_slots": 12,
        "send_worker_programs_per_peer": 2,
        "worker_programs_per_peer": 32,
        "block_m": 64,
        "block_k": 128,
        "block_n": 128,
        "n_group": 1,
        "n_groups_per_job": 2,
        "warmup": 2,
        "steps": 7,
        "repeat_runs": 48,
        "check": False,
        "separate_compile": True,
        "progress_events": True,
    }
)


def _profile_defaults(**overrides: Any) -> Mapping[str, Any]:
    return MappingProxyType({**_COMMON_HOPPER_QUEUE_DEFAULTS, **overrides})


_SOURCE_PUSH_PROFILE_DEFAULTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        SOURCE_PUSH_PROFILE_NONE: MappingProxyType({}),
        SOURCE_PUSH_PROFILE_STABLE_216: _profile_defaults(
            routing="roughly_balanced",
            send_pipeline_depth=1,
        ),
        SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072: _profile_defaults(
            tokens_per_rank=65536,
            hidden_dim=3072,
            intermediate_dim=3072,
            entries_per_rank=576,
            inbox_slots=32,
            routing="roughly_balanced",
            block_k=256,
            block_n=256,
            n_groups_per_job=1,
            send_worker_programs_per_peer=4,
            send_pipeline_depth=1,
        ),
    }
)


def source_push_profile_defaults(profile: str) -> dict[str, Any]:
    """Return a copy of the named source-push profile defaults."""
    if profile not in _SOURCE_PUSH_PROFILE_DEFAULTS:
        raise ValueError(f"unknown source-push profile {profile!r}; expected one of {SOURCE_PUSH_PROFILES}")
    return dict(_SOURCE_PUSH_PROFILE_DEFAULTS[profile])
