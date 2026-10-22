# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import sys
from unittest.mock import patch

import draccus
import pytest
from fray.types import GpuConfig, ResourceConfig

from experiments.grug.moe_hero_ep.format_job import LaunchConfig, main, prepare_launch, submit_launch


def launch_config(tmp_path):
    request = tmp_path / "request.json"
    raw = json.dumps({"source_revision": "a" * 40, "output_root": str(tmp_path / "output")}).encode()
    request.write_bytes(raw)
    return LaunchConfig(
        name="/test/format-campaign",
        source_revision="a" * 40,
        controller_cluster="test-controller",
        resources=ResourceConfig(cpu=1.0, device=GpuConfig(variant="GB200", count=4), replicas=16),
        processes_per_task=4,
        command=("python", "-m", "experiments.grug.moe_hero_ep.sft_campaign", "request.json", "--worker"),
        request_path=str(request),
        request_sha256=hashlib.sha256(raw).hexdigest(),
        output_root=str(tmp_path / "output"),
        moe_implementation="ragged_all_to_all",
        remat_mode="offload_carry",
        cache_prefix=str(tmp_path / "cache"),
    )


def test_launch_cli_defaults_to_local_review_without_serializing_credentials(tmp_path, monkeypatch, capsys):
    config = launch_config(tmp_path)
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(draccus.encode(config)))
    monkeypatch.setenv("CW_KEY_SECRET", "test-secret-not-for-reporting")
    monkeypatch.setattr(sys, "argv", ["format_job", str(launch)])
    with patch("experiments.grug.moe_hero_ep.format_job.connect_controller", side_effect=AssertionError("No network")):
        main()
    output = capsys.readouterr().out
    report = json.loads(output)
    assert not report["submission_requested"]
    assert report["request_sha256"] == hashlib.sha256((tmp_path / "request.json").read_bytes()).hexdigest()
    assert "test-secret-not-for-reporting" not in output
    assert not (tmp_path / "output").exists()


def test_frozen_request_tampering_and_ambiguous_submission_block_network(tmp_path):
    config = launch_config(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"submission_pending"}\n')
    with patch("experiments.grug.moe_hero_ep.format_job.connect_controller", side_effect=AssertionError("No network")):
        with pytest.raises(FileExistsError, match="Existing receipt/output"):
            submit_launch(config, receipt, tmp_path)
    original = receipt.read_bytes()
    (tmp_path / "request.json").write_text('{"kind":"changed"}\n')
    with pytest.raises(ValueError, match="request bytes changed"):
        prepare_launch(config)
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("worker", ("sft", "rollout"))
def test_launch_rejects_same_bytes_with_mismatched_source_or_guard_root(tmp_path, worker):
    config = launch_config(tmp_path)
    if worker == "rollout":
        raw = json.dumps(
            {"native_request": {"panel": {"source_revision": config.source_revision}}, "destination": config.output_root}
        ).encode()
        (tmp_path / "request.json").write_bytes(raw)
        config = dataclasses.replace(
            config,
            command=("python", "-m", "experiments.grug.moe_hero_ep.rollout_sanity", "run", "request.json"),
            request_sha256=hashlib.sha256(raw).hexdigest(),
        )
    summary, frozen = prepare_launch(config)
    assert summary["request_sha256"] == hashlib.sha256(frozen).hexdigest()
    # The bytes and their SHA remain identical. An old embedded execution pin
    # cannot be made truthful merely by selecting a newer source checkout.
    with pytest.raises(ValueError, match="Embedded worker source"):
        prepare_launch(dataclasses.replace(config, source_revision="b" * 40))
    with pytest.raises(ValueError, match="Embedded worker destination"):
        prepare_launch(dataclasses.replace(config, output_root=str(tmp_path / "unguarded-output")))
