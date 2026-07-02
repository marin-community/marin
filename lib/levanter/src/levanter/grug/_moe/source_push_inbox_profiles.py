# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile defaults for the Hopper source-push inbox MGPU MoE prototype.

This module is the package-private boundary for the current source-push inbox
candidate configurations. The source-push prototype kernel and CLI harness live
next to this module in `source_push_inbox`; benchmark scripts should stay thin
wrappers around those package-private entrypoints.
"""

from __future__ import annotations

from typing import Any


SOURCE_PUSH_PROFILES = (
    "none",
    "hopper_queue_ngroups2_210",
    "hopper_queue_roughly_balanced_ngroups2_210",
)

_SOURCE_PUSH_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "none": {},
    # Uniform-routing Hopper profile used as a deterministic all-to-all control.
    "hopper_queue_ngroups2_210": {
        "implementation": "m_n_slots",
        "queue_mode": "routing",
        "routing": "uniform",
        "traffic_pattern": "all_to_all",
        "peer_loop": "grid_switch",
        "lowering_semantics": "lane",
        "metadata_mode": "static_recv",
        "output_mode": "perf",
        "hidden_output_mode": "queue",
        "tokens_per_rank": 32768,
        "hidden_dim": 2560,
        "intermediate_dim": 1280,
        "experts_per_rank": 32,
        "topk": 4,
        "entries_per_rank": 288,
        "inbox_slots": 12,
        "num_send_sms": 2,
        "num_sms": 32,
        "block_m": 64,
        "block_k": 128,
        "block_n": 128,
        "n_group": 1,
        "n_groups_per_job": 2,
        "receiver_schedule": "fixed_wait",
        "send_pipeline_depth": 2,
        "slot_order": "current",
        "warmup": 2,
        "steps": 7,
        "repeat_runs": 48,
        "check": False,
        "separate_compile": True,
        "progress_events": True,
    },
    # Target-like deterministic Hopper profile with realistic route imbalance.
    "hopper_queue_roughly_balanced_ngroups2_210": {
        "implementation": "m_n_slots",
        "queue_mode": "routing",
        "routing": "roughly_balanced",
        "traffic_pattern": "all_to_all",
        "peer_loop": "grid_switch",
        "lowering_semantics": "lane",
        "metadata_mode": "static_recv",
        "output_mode": "perf",
        "hidden_output_mode": "queue",
        "tokens_per_rank": 32768,
        "hidden_dim": 2560,
        "intermediate_dim": 1280,
        "experts_per_rank": 32,
        "topk": 4,
        "entries_per_rank": 288,
        "inbox_slots": 12,
        "num_send_sms": 2,
        "num_sms": 32,
        "block_m": 64,
        "block_k": 128,
        "block_n": 128,
        "n_group": 1,
        "n_groups_per_job": 2,
        "receiver_schedule": "fixed_wait",
        "send_pipeline_depth": 1,
        "slot_order": "current",
        "warmup": 2,
        "steps": 7,
        "repeat_runs": 48,
        "check": False,
        "separate_compile": True,
        "progress_events": True,
    },
}


def source_push_profile_defaults(profile: str) -> dict[str, Any]:
    """Return a copy of the named source-push profile defaults."""
    if profile not in _SOURCE_PUSH_PROFILE_DEFAULTS:
        raise ValueError(f"unknown source-push profile {profile!r}; expected one of {SOURCE_PUSH_PROFILES}")
    return dict(_SOURCE_PUSH_PROFILE_DEFAULTS[profile])
