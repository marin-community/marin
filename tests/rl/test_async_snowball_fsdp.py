# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The optional FSDP2 benchmark must preserve Snowball's workload and default launch."""

import json
from dataclasses import asdict

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext

from experiments.post_training import async_snowball as snowball


@pytest.mark.parametrize("runner", list(snowball.Runner))
def test_fsdp2_cli_and_builder_keep_the_comparison_workload_and_backend_identity(runner):
    arguments = [
        "--version",
        "2026.09.06.77",
        "--runner",
        runner.value,
        "--scale",
        "qualification",
        "--completion",
        "metrics",
        "--response-tokens",
        "4096",
        "--context-tokens",
        "8192",
    ]
    previews = []
    for flags in ([], ["--backend", "megatron"], ["--backend", "fsdp2"]):
        result = CliRunner().invoke(snowball.main, arguments + flags)
        assert result.exit_code == 0, result.output
        previews.append(json.loads(result.output))
    baseline, explicit, fsdp = [preview["request"] for preview in previews]
    assert baseline == explicit
    assert fsdp["run_id"] != baseline["run_id"]

    built = snowball.build_experiment(
        version="2026.09.06.77",
        runner=runner,
        scale=snowball.Scale.QUALIFICATION,
        completion="metrics",
        timeout_seconds=3600,
        response_tokens=4096,
        context_tokens=8192,
        backend=snowball.Backend.FSDP2,
    )
    request = built.build_config(StepContext.for_fingerprint(built.runtime_args, built.deps)).request
    assert fsdp == json.loads(json.dumps(asdict(request)))
    assert fsdp["runtime"]["profile"] == "fsdp"
    assert baseline["runtime"]["profile"] == "megatron"
    for field in ("model", "train_data", "validation_data", "topology", "seed", "completion_mode"):
        assert fsdp[field] == baseline[field]
    assert built.runtime_args["skyrl_execution"].cluster == "cw-us-east-02a"

    # The actual launch request must carry undeclared keys with Hydra's ++ prefix.
    assert fsdp["overrides"] == [
        *baseline["overrides"][:1],
        "++trainer.policy.fsdp_config.mixed_precision.reduce_dtype=bf16",
        "++trainer.policy.optimizer_config.bf16_update_mode=stochastic",
        *baseline["overrides"][1:],
    ]
    before, after = [yaml.safe_load(request["config_yaml"]) for request in (baseline, fsdp)]
    trainer = after["trainer"]
    assert trainer["strategy"] == "fsdp2"
    assert trainer["flash_attn"] is True
    assert trainer.pop("gradient_checkpointing_use_reentrant") is False
    assert trainer["gradient_checkpointing"] is True
    assert trainer["use_sample_packing"] is False
    assert trainer["micro_train_batch_size_per_gpu"] == trainer["micro_forward_batch_size_per_gpu"] == 1
    assert trainer["train_batch_size"] == trainer["policy_mini_batch_size"] == 32
    assert after["generator"]["n_samples_per_prompt"] == 4
    assert trainer["policy"].pop("fsdp_config") == {
        "cpu_offload": False,
        "reshard_after_forward": True,
        "use_grouped_mm": True,
        "expert_model_parallel_size": 1,
    }
    assert trainer["policy"]["optimizer_config"].pop("optimizer") == "AdamW"
    assert "megatron_config" not in trainer["policy"]
    assert "ref" not in trainer
    trainer["strategy"] = "megatron"
    trainer["flash_attn"] = False
    trainer["policy"]["megatron_config"] = before["trainer"]["policy"]["megatron_config"]
    trainer["ref"] = before["trainer"]["ref"]
    assert after == before


@pytest.mark.parametrize("backend", list(snowball.Backend))
def test_regular_no_tis_is_a_distinct_snowball_objective_at_either_backend(backend):
    requests = []
    for correction in snowball.Correction:
        step = snowball.build_experiment(
            version="2026.09.06.77",
            scale=snowball.Scale.GATE,
            timeout_seconds=3600,
            completion="metrics",
            backend=backend,
            correction=correction,
        )
        requests.append(step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request)
    assert len({request.run_id for request in requests}) == len(snowball.Correction)
    objectives = {
        correction: yaml.safe_load(request.config_yaml)["trainer"]["algorithm"]
        for correction, request in zip(snowball.Correction, requests, strict=True)
    }
    no_tis = objectives[snowball.Correction.REGULAR_NO_TIS]
    assert no_tis["policy_loss_type"] == "regular"
    assert no_tis["use_tis"] is False
    assert no_tis["require_rollout_logprobs"] is True
    assert objectives[snowball.Correction.BEHAVIOR_CLIP]["policy_loss_type"] == "behavior_clip"
    assert objectives[snowball.Correction.REGULAR_TIS]["use_tis"] is True


def test_snowball_rejects_an_unknown_backend_before_building_an_experiment():
    with pytest.raises(ValueError, match="not a valid Backend"):
        snowball.build_experiment(
            version="2026.09.06.77",
            scale=snowball.Scale.GATE,
            timeout_seconds=3600,
            backend="fsdp",
        )
