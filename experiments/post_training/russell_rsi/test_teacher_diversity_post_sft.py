# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.external_dependencies import MARIN_SKYRL

from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import (
    LAUNCH_PROTOCOL,
    completed_durable_producer,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_sft import SFT_VERSION


def pinned(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_completed_training_is_bound_to_actual_record_source_and_success(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    source_head = "f" * 40
    source_files = {"lib/levanter/src/levanter/tracker/json_logger.py": "a" * 64}
    expected = ArtifactStep.adopt("checkpoints/durable-training", SFT_VERSION, str(root))
    study_pin = pinned(tmp_path / "study.json", {"version": "2026.10.06.15"})
    review_pin = pinned(
        tmp_path / "review.json",
        {"status": "approved", "source_head": source_head, "runtime_commit": MARIN_SKYRL.commit},
    )
    request_pin = pinned(
        tmp_path / "request.json",
        {
            "stage": "sft",
            "version": SFT_VERSION,
            "source_head": source_head,
            "runtime_commit": MARIN_SKYRL.commit,
            "config_uri": study_pin["uri"],
            "config_sha256": study_pin["sha256"],
        },
    )
    preflight_pin = pinned(
        tmp_path / "preflight.json",
        {"exit_code": 0, "identity": {"source_head": source_head, "request_sha256": request_pin["sha256"]}},
    )
    actual_config = {"train_config": {"trainer": {"id": "durable-run", "tracker": [{"metric_destination": str(root)}]}}}
    launch = {
        "protocol": LAUNCH_PROTOCOL,
        "source_head": source_head,
        "runtime_commit": MARIN_SKYRL.commit,
        "source_files": source_files,
        "config": study_pin,
        "source_review": review_pin,
        "request": request_pin,
        "preflight": preflight_pin,
        "producer_identity": artifact_identity(expected),
        "output_path": str(root),
        "bound_config": actual_config,
        "rl_authorized": False,
        "signal_gate_passed": None,
    }
    record = {
        "name": expected.name,
        "version": expected.version,
        "fingerprint": expected.fingerprint(),
        "output_path": str(root),
        "config": actual_config,
        "provenance": {"base_commit": source_head, "dirty": False},
    }
    config = {
        "sft_launch_proof": pinned(tmp_path / "launch.json", launch),
        "sft_source_review": review_pin,
        "sft_config_uri": study_pin["uri"],
        "sft_config_sha256": study_pin["sha256"],
        "sft_producer": pinned(root / ".artifact.json", record),
    }
    amendment = {
        "source": {"head": source_head, "files": source_files},
        "sft": {"identity": artifact_identity(expected), "output_path": str(root)},
    }
    status = root / ".executor_status"
    status.write_text("SUCCESS")
    assert completed_durable_producer(config, "sft", expected, amendment) == record

    status.write_text("RUNNING")
    with pytest.raises(ValueError, match="exact successful producer"):
        completed_durable_producer(config, "sft", expected, amendment)
    status.write_text("SUCCESS")
    record = {**record, "config": {"train_config": {"trainer": {"id": "another-run"}}}}
    config["sft_producer"] = pinned(root / ".artifact.json", record)
    with pytest.raises(ValueError, match="exact successful producer"):
        completed_durable_producer(config, "sft", expected, amendment)
    record = {**record, "config": actual_config, "provenance": {"base_commit": "a" * 40, "dirty": False}}
    config["sft_producer"] = pinned(root / ".artifact.json", record)
    with pytest.raises(ValueError, match="exact successful producer"):
        completed_durable_producer(config, "sft", expected, amendment)
