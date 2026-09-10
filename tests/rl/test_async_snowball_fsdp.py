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


@pytest.mark.parametrize("backend", list(snowball.Backend))
@pytest.mark.parametrize("policy_nodes,inference_replicas", [(2, 3), (4, 1)])
def test_snowball_placement_cli_preserves_workload_and_resolves_requested_roles(
    backend, policy_nodes, inference_replicas
):
    args = [
        "--version",
        "2026.09.10.77",
        "--completion",
        "metrics",
        "--backend",
        backend.value,
        "--scale",
        "cadence-gate",
        "--policy-nodes",
        str(policy_nodes),
        "--inference-replicas",
        str(inference_replicas),
    ]
    result = CliRunner().invoke(snowball.main, args)
    assert result.exit_code == 0, result.output
    request = json.loads(result.output)["request"]
    topology = request["topology"]
    assert topology["num_nodes"] == 5 and topology["gpus_per_node"] == 8
    assert topology["role_plan"]["policy_num_nodes"] == policy_nodes
    assert topology["role_plan"]["num_inference_engines"] == inference_replicas
    config = yaml.safe_load(request["config_yaml"])
    trainer = config["trainer"]
    assert trainer["train_batch_size"] == trainer["policy_mini_batch_size"] == 32
    assert config["generator"]["n_samples_per_prompt"] == 4
    if backend is snowball.Backend.MEGATRON:
        geometry = trainer["policy"]["megatron_config"]
        assert geometry["pipeline_model_parallel_size"] == 2
        assert geometry["expert_model_parallel_size"] == 8
    default = CliRunner().invoke(snowball.main, args[:-4])
    assert default.exit_code == 0, default.output
    baseline = json.loads(default.output)["request"]
    for field in ("model", "train_data", "validation_data", "seed", "runtime"):
        assert request[field] == baseline[field]
    assert (request == baseline) == (policy_nodes == 4)


@pytest.mark.parametrize("policy_nodes", [True, 0, 1, 3, 8])
def test_snowball_rejects_unsupported_policy_node_counts(policy_nodes):
    with pytest.raises(ValueError, match="policy_nodes must be 2 or 4"):
        snowball.training_config(snowball.Scale.GATE, policy_nodes=policy_nodes)


@pytest.mark.parametrize("scale", list(snowball.Scale))
def test_snowball_precision_presets_change_only_declared_optimizer_configuration(scale):
    baseline = yaml.safe_load(snowball.training_config(scale, policy_nodes=2, inference_replicas=3))
    configs = []
    for precision in snowball.OptimizerPrecision:
        config = yaml.safe_load(
            snowball.training_config(scale, policy_nodes=2, inference_replicas=3, optimizer_precision=precision)
        )
        configs.append(json.dumps(config, sort_keys=True))
        if precision is not snowball.OptimizerPrecision.NATIVE:
            assert config["trainer"].pop("optimizer_state_metrics") is True
            geometry = config["trainer"]["policy"]["megatron_config"]
            assert geometry.pop("ddp_config") == {
                "grad_reduce_in_fp32": precision is not snowball.OptimizerPrecision.BF16_GRAD_REDUCE
            }
            if precision is snowball.OptimizerPrecision.BF16_GRAD_REDUCE:
                assert "optimizer_config_kwargs" not in geometry
            else:
                optimizer = geometry.pop("optimizer_config_kwargs")
                assert optimizer["use_precision_aware_optimizer"] is True
                assert optimizer["store_param_remainders"] == (precision is snowball.OptimizerPrecision.FP32_REMAINDERS)
        assert config == baseline
    assert len(set(configs)) == len(snowball.OptimizerPrecision)


def test_snowball_rejects_megatron_precision_for_fsdp_before_submission():
    result = CliRunner().invoke(
        snowball.main, ["--version", "2026.09.10.77", "--backend", "fsdp2", "--optimizer-precision", "bf16_both"]
    )
    assert result.exit_code != 0
    assert "require the Megatron backend" in str(result.exception)


def test_snowball_cli_composes_p16_precision_request():
    result = CliRunner().invoke(
        snowball.main,
        [
            "--version",
            "2026.09.10.77",
            "--completion",
            "metrics",
            "--policy-nodes",
            "2",
            "--inference-replicas",
            "3",
            "--optimizer-precision",
            "bf16_both",
        ],
    )
    assert result.exit_code == 0, result.output
    request = json.loads(result.output)["request"]
    config = yaml.safe_load(request["config_yaml"])
    assert config["trainer"]["optimizer_state_metrics"] is True
    assert config["trainer"]["policy"]["megatron_config"]["optimizer_config_kwargs"]["exp_avg_sq_dtype"] == "bfloat16"
    assert request["topology"]["role_plan"]["policy_num_nodes"] == 2


@pytest.mark.parametrize("backend", list(snowball.Backend))
@pytest.mark.parametrize("correction", [snowball.Correction.REGULAR_MASK, snowball.Correction.BC_MASK])
def test_snowball_mask_presets_emit_declared_loss_and_only_mask_fields(backend, correction):
    baseline = yaml.safe_load(snowball.training_config(snowball.Scale.GATE, backend=backend))
    actual = yaml.safe_load(snowball.training_config(snowball.Scale.GATE, backend=backend, correction=correction))
    algorithm = actual["trainer"]["algorithm"]
    assert algorithm["use_tis"] is False
    assert algorithm.pop("require_rollout_logprobs") is True
    assert algorithm["policy_loss_type"] == (
        "regular" if correction == snowball.Correction.REGULAR_MASK else "behavior_clip"
    )
    algorithm["policy_loss_type"] = "behavior_clip"
    assert algorithm.pop("offpolicy_mask") == {
        "enabled": True,
        "ratio": "mismatch",
        "low": 0.5,
        "high": 5.0,
        "veto_ratio": 1e-5,
        "renormalize": False,
    }
    assert actual == baseline
