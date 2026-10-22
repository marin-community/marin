# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path

import draccus
import pytest
from fray.types import GpuConfig, ResourceConfig

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft import BaseReference, SftConfig, sft_manifest
from experiments.grug.moe_hero_ep.sft_campaign import (
    RecoveryParent,
    SftCampaign,
    latest_completed_checkpoint,
    read_campaign,
)


def campaign_at(root: Path, steps: int = 8) -> SftCampaign:
    return SftCampaign(
        name="test-campaign",
        version="test",
        source_revision="a" * 40,
        reference=BaseReference("prepared-C", "b" * 64, "source-PT", 100, "B", "tokenizer", "c" * 40),
        model=GrugModelConfig(vocab_size=128032, max_seq_len=16384),
        sft=SftConfig(16, 16.0, (128006, 128007, 128002, 128003, 128009), 1e-4, 1e-5, steps, 2, 7),
        data_root="prepared-data",
        data_manifest="data-manifest.json",
        data_manifest_digest="d" * 64,
        acquisition_panel="panel.json",
        acquisition_panel_digest="e" * 64,
        cache_prefix="cache",
        output_root=str(root),
        resources=ResourceConfig(cpu=1.0, device=GpuConfig(variant="GB200", count=4), replicas=16),
        batch_size=64,
        expert_axis_size=64,
        context_axis_size=1,
        replica_axis_size=1,
        max_processed_tokens=steps * 64 * 16384,
        maximum_metrics=(("loss", 20.0), ("gradient_norm", 1000.0)),
    )


def save_checkpoint(
    campaign: SftCampaign, uid: str, step: int, *, temporary: bool = False, processed: int | None = None
):
    root = Path(campaign.output_root)
    destination = root / "attempts" / uid
    records = root / "attempt-records"
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{uid}.json").write_text(
        json.dumps(
            {
                "request_digest": digest(draccus.encode(campaign)),
                "leader_attempt_uid": uid,
                "destination": str(destination),
            }
        )
    )
    checkpoint = destination / "adapters" / f"step-{step}"
    checkpoint.mkdir(parents=True)
    metadata = {
        "is_temporary": temporary,
        "step": step,
        "hero_sft": json.loads(json.dumps(sft_manifest(campaign.reference, campaign.sft))),
        "hero_sft_counters": {
            "sft_step": step,
            "processed_tokens": processed or step * 1000,
            "supervised_tokens": step * 700,
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    return {
        "path": str(checkpoint),
        "step": step,
        "metadata_digest": digest(metadata),
        "counters": metadata["hero_sft_counters"],
        "sft_config": draccus.encode(campaign.sft),
    }


def continuation(tmp_path: Path):
    previous = campaign_at(tmp_path / "smoke")
    selected = save_checkpoint(previous, "first", 8)
    request = tmp_path / "smoke.json"
    raw = (json.dumps(draccus.encode(previous), indent=2) + "\n").encode()
    request.write_bytes(raw)
    parent = RecoveryParent(
        str(request),
        hashlib.sha256(raw).hexdigest(),
        digest(draccus.encode(previous)),
        **{key: selected[key] for key in ("path", "step", "metadata_digest", "counters")},
    )
    current = dataclasses.replace(campaign_at(tmp_path / "long", 64), recovery_parent=parent)
    return previous, current, selected, request


def test_continuation_retains_exact_parent_state_and_prefers_own_completed_checkpoint(tmp_path):
    _, current, parent, request = continuation(tmp_path)
    assert latest_completed_checkpoint(current) == parent
    assert parent["sft_config"]["num_steps"] == 8
    assert current.sft.num_steps == 64
    save_checkpoint(current, "interrupted", 10, temporary=True)
    assert latest_completed_checkpoint(current) == parent
    own = save_checkpoint(current, "resumed", 9)
    assert latest_completed_checkpoint(current) == own
    # Parent discovery is a fallback only: completed own state remains usable
    # even when the original parent request cannot be accessed during a retry.
    request.unlink()
    assert latest_completed_checkpoint(current) == own
    assert own["counters"]["sft_step"] == 9
    assert own["sft_config"]["num_steps"] == 64


def test_parent_request_and_latest_boundary_are_both_hash_bound(tmp_path):
    previous, current, _, request = continuation(tmp_path)
    raw = request.read_bytes()
    request.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="parent request bytes changed"):
        latest_completed_checkpoint(current)
    request.write_bytes(raw)
    # A newer parent checkpoint is never silently substituted for the reviewed
    # full-Adam boundary, even when it is otherwise valid.
    save_checkpoint(previous, "later", 8, processed=8500)
    with pytest.raises(ValueError, match="data cursor"):
        latest_completed_checkpoint(current)


def test_parent_ancestry_rejects_changed_data_and_schedule_parameters(tmp_path):
    _, current, _, _ = continuation(tmp_path)
    changed = dataclasses.replace(current, data_manifest_digest="f" * 64)
    with pytest.raises(ValueError, match="changed weights, data"):
        latest_completed_checkpoint(changed)
    changed = dataclasses.replace(current, sft=dataclasses.replace(current.sft, seed=8))
    with pytest.raises(ValueError, match="changed weights, data"):
        latest_completed_checkpoint(changed)


def test_request_roundtrip_preserves_checkpoint_manifest_identity(tmp_path):
    _, campaign, _, _ = continuation(tmp_path)
    request = tmp_path / "long.json"
    request.write_text(json.dumps(draccus.encode(campaign)))
    restored, _ = read_campaign(str(request))
    assert digest(draccus.encode(restored)) == digest(draccus.encode(campaign))
    assert latest_completed_checkpoint(restored) == latest_completed_checkpoint(campaign)
