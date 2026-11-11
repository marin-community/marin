# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.mochi_one_rack.launch import build_smoke
from experiments.post_training.mochi_one_rack.runtime import vllm_backend


def test_smoke_restores_the_first_stage_and_keeps_both_context_caps() -> None:
    smoke = build_smoke("2026.11.11.9261")
    launches = []
    for step in (smoke.four_k, smoke.thirty_two_k):
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        launches.append(yaml.safe_load(config.launch_config_yaml))
    first, resumed = launches
    assert first["iris"]["allocation"]["num_nodes"] == resumed["iris"]["allocation"]["num_nodes"] == 16
    assert first["skyrl"]["context_budget"]["request_window_tokens"] == 4096
    assert resumed["skyrl"]["context_budget"]["request_window_tokens"] == 32768
    assert first["skyrl"]["trainer"]["max_steps"] == 5
    assert resumed["skyrl"]["trainer"]["max_steps"] == 10
    assert resumed["skyrl"]["trainer"]["resume_mode"] == "latest"
    assert "mochi-one-rack/4k@2026.11.11.9261" in resumed["artifacts"]["checkpoint_root"]
    assert smoke.four_k in smoke.thirty_two_k.deps
    assert not first["run"]["export_hf"] and not resumed["run"]["export_hf"]


def test_experimental_marin_launcher_uses_its_own_wheel_and_torch_pin(monkeypatch) -> None:
    monkeypatch.setattr("marin.inference.vllm_release.platform.machine", lambda: "aarch64")
    backend = vllm_backend()
    command = backend.command()
    assert f"torch=={backend.gpu_release.torch_version}" in command
    requirement = command[command.index("--from") + 1]
    assert backend.gpu_release.wheels[0].url in requirement
    assert backend.gpu_release.wheels[0].sha256 in requirement
    # The startup verifier receives this exact experimental source identity.
    provenance = next(json.loads(value) for value in command if value.startswith('{"release_tag"'))
    assert provenance["source_commit"] == backend.gpu_release.source_commit
