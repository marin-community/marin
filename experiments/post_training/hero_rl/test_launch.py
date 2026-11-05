# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.execution.lazy import StepContext
from marin.rl.skyrl import _launcher_command as launcher_command_for_test
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.post_training.hero_rl.launch import ExportCheckConfig, Hardware, build_qualification, check_export
from experiments.post_training.hero_rl.launch import main as launch_main

COMPOSE = """
import json
import sys
import io
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cloud.iris import iris_backend
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
from skyrl_train.callbacks.builtin import create_default_callbacks

result = []

class RecordingClient:
    def submit(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(job_id="/test/qualification")

for path in sys.argv[1:]:
    launch = load_launch_config(Path(path))
    args = iris_backend._iris_submission_state(Path(path), launch)
    args.no_wait = True
    client = RecordingClient()
    with redirect_stdout(io.StringIO()), patch(
        "iris.cli.connect.open_iris_client", return_value=nullcontext(client)
    ), patch("cloud.iris.iris_backend.wandb_launch_environment", return_value={}):
        iris_backend.launch(args, Path.cwd())
    result.append({
        "training_type": launch.runtime.training_type,
        "policy_gpus": (
            launch.skyrl.trainer.placement.policy_num_nodes * launch.skyrl.trainer.placement.policy_num_gpus_per_node
        ),
        "allocated_gpus": launch.iris.allocation.num_nodes * launch.iris.allocation.gpus_per_node,
        "request_window": launch.skyrl.generator.engine_init_kwargs.max_model_len,
        "response_cap": launch.skyrl.generator.sampling_params.max_generate_length,
        "inference_mp": launch.skyrl.generator.inference_engine_mp_backend,
        "probe_updates": (
            launch.skyrl.trainer.mismatch_probe.updates if launch.skyrl.trainer.mismatch_probe.enabled else None
        ),
        "max_steps": launch.skyrl.trainer.max_steps,
        "resume_mode": launch.skyrl.trainer.resume_mode,
        "export_hf": launch.run.export_hf,
        "cluster": launch.iris.cluster,
        "target_cluster": launch.iris.target_cluster,
        "priority": launch.iris.priority,
        "submission_retries": {name: client.request.get(name) for name in (
            "max_retries_failure", "max_retries_preemption", "max_task_failures",
        )},
        "memory": launch.iris.allocation.memory,
        "allocator_config": client.request["environment"].env_vars["PYTORCH_CUDA_ALLOC_CONF"],
        "geometry": {key: launch.skyrl.trainer.policy.megatron_config[key] for key in (
            "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
            "expert_model_parallel_size", "expert_tensor_parallel_size",
        )},
        "response_batch": launch.skyrl.trainer.train_batch_size * launch.skyrl.generator.n_samples_per_prompt,
        "checkpoint_cpu_offload": launch.skyrl.trainer.policy.megatron_config.transformer_config_kwargs.get(
            "grug_checkpoint_cpu_offload", False
        ),
        "compilation": (
            OmegaConf.to_container(launch.skyrl.generator.engine_init_kwargs.compilation_config)
            if "compilation_config" in launch.skyrl.generator.engine_init_kwargs else None
        ),
        "batch_invariance": launch.skyrl.trainer.algorithm.batch_invariant,
        "routed_experts": launch.skyrl.generator.engine_init_kwargs.enable_return_routed_experts,
        "cudagraph_metrics": launch.skyrl.generator.engine_init_kwargs.get("cudagraph_metrics", False),
        "callbacks": [type(callback).__name__ for callback in create_default_callbacks(launch.skyrl)],
    })
print(json.dumps(result))
"""


def test_hardware_cli_accepts_public_gb200_profile_value() -> None:
    # Click's Enum choices otherwise expect the Python member name GB200_48,
    # rejecting the documented gb200-48 command before the plan is built.
    result = CliRunner().invoke(
        launch_main,
        [
            "--version",
            "2026.11.03.48",
            "--hardware",
            "gb200-48",
            "--priority",
            "interactive",
            "--stage",
            "numerics",
            "--probe-prompts",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.timeout(180)  # Compose through the independently locked, CPU-only SkyRL executable.
def test_hero_qualification_composes_multinode_probe_and_updates(tmp_path: Path) -> None:
    workflow = build_qualification("2026.11.03.1")
    paths = []
    for name, step in (("numerics", workflow.numerics), ("updates", workflow.updates)):
        config = step.build_config(
            StepContext.for_run(
                output_path=f"s3://marin-us-east-02a/marin/users/test/hero-qualification/{name}",
                prefix="s3://marin-us-east-02a/marin",
                runtime_args=step.runtime_args,
                deps=step.deps,
            )
        )
        path = tmp_path / f"{name}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run([*bootstrap, "python", "-c", COMPOSE, *paths], capture_output=True, text=True, timeout=170)

    assert result.returncode == 0, result.stderr
    numerics, updates = json.loads(result.stdout)
    assert (numerics["training_type"], updates["training_type"]) == ("sync", "async")
    assert numerics["probe_updates"] == 0
    assert "MismatchProbeCallback" in numerics["callbacks"]
    assert updates["max_steps"] == 4
    for stage in (numerics, updates):
        assert stage["policy_gpus"] == 128
        assert stage["allocated_gpus"] == 152
        assert stage["inference_mp"] is True
        assert stage["request_window"] == 4096
        assert stage["response_cap"] == 3072
        assert stage["resume_mode"] == "none"
        assert stage["export_hf"] is False
        assert stage["allocator_config"] == "expandable_segments:True"
        assert "CheckpointCallback" not in stage["callbacks"]
        assert "HFModelSaveCallback" not in stage["callbacks"]


@pytest.mark.timeout(180)  # Compose the Blackwell trial without allocating accelerators.
def test_experimental_gb200_profile_composes_36_learners_and_12_servers(tmp_path: Path) -> None:
    workflow = build_qualification("2026.11.03.48", hardware=Hardware.GB200_48, priority="interactive")
    paths = []
    for name, step in (("numerics", workflow.numerics), ("updates", workflow.updates)):
        config = step.build_config(
            StepContext.for_run(
                output_path=f"s3://marin-us-east-08a/marin/users/test/hero-qualification/{name}",
                prefix="s3://marin-us-east-08a/marin",
                runtime_args=step.runtime_args,
                deps=step.deps,
            )
        )
        path = tmp_path / f"{name}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
        assert "/gb200-48/" in step.name
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run([*bootstrap, "python", "-c", COMPOSE, *paths], capture_output=True, text=True, timeout=170)

    assert result.returncode == 0, result.stderr
    numerics, updates = json.loads(result.stdout)
    assert (numerics["training_type"], updates["training_type"]) == ("sync", "async")
    assert numerics["probe_updates"] == 0
    assert updates["max_steps"] == 4
    for stage in (numerics, updates):
        assert (stage["policy_gpus"], stage["allocated_gpus"]) == (36, 48)
        assert stage["geometry"] == {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 12,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 3,
            "expert_tensor_parallel_size": 1,
        }
        assert stage["response_batch"] == 120
        assert stage["checkpoint_cpu_offload"] is True
        assert stage["compilation"] == {
            "mode": 3,
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        }
        assert stage["batch_invariance"] is True
        assert stage["routed_experts"] is True
        assert stage["cudagraph_metrics"] is True
        assert stage["cluster"] == stage["target_cluster"] == "cw-us-east-08a"
        assert stage["priority"] == "interactive"
        assert stage["memory"] == "940GiB"
        assert stage["allocator_config"] == "expandable_segments:True,pinned_max_cached_size_mb:1"
        assert stage["submission_retries"] == {
            "max_retries_failure": 0,
            "max_retries_preemption": 0,
            "max_task_failures": 0,
        }
        assert stage["request_window"] == 4096
        assert stage["export_hf"] is False
        assert "CheckpointCallback" not in stage["callbacks"]
        assert "HFModelSaveCallback" not in stage["callbacks"]
    default = build_qualification("2026.11.03.48")
    assert default.numerics.fingerprint() != workflow.numerics.fingerprint()
    assert default.updates.fingerprint() != workflow.updates.fingerprint()


@pytest.mark.timeout(180)  # Exercise staging through SkyRL's independently locked CPU runtime.
def test_export_check_stages_model_only_export_and_separate_tokenizer(tmp_path: Path) -> None:
    model = tmp_path / "completed-export"
    model.mkdir()
    manifest = b'{"step":144000,"complete":true}'
    (model / "export-manifest.json").write_bytes(manifest)
    (model / "config.json").write_text(
        json.dumps({"model_type": "grug_moe", "hidden_size": 6144, "num_hidden_layers": 48, "num_experts": 384})
    )
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
    (model / "shard.safetensors").write_bytes(b"model body must not be loaded by the metadata preflight")
    tokenizer = tmp_path / "separate-tokenizer"
    tokenizer.mkdir()
    Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(str(tokenizer / "tokenizer.json"))
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})
    )
    output = tmp_path / "check"
    output.mkdir()
    config = ExportCheckConfig(str(output), str(model), hashlib.sha256(manifest).hexdigest(), str(tokenizer), "a" * 40)

    # The model has no bundled tokenizer, and its shard is deliberately not a loadable model.
    check_export(config)
    summary = json.loads((output / "validation.json").read_text())
    assert summary["indexed_tensors"] == 1
    assert summary["model_uri"] == str(model)
    assert summary["runtime_staging"]["model_metadata_staged"] is True
    assert summary["runtime_staging"]["tokenizer_uri"] == str(tokenizer)
    assert summary["runtime_staging"]["tokenizer_staged"] is True
    (tokenizer / "tokenizer.json").write_text("not a valid tokenizer")
    invalid_output = tmp_path / "invalid-tokenizer"
    with pytest.raises(subprocess.CalledProcessError):
        check_export(replace(config, output_path=str(invalid_output)))
    assert not (invalid_output / "validation.json").exists()
    (model / "export-manifest.json").write_bytes(b'{"step":144001,"complete":false}')
    with pytest.raises(ValueError, match="Hero export manifest changed"):
        check_export(replace(config, output_path=str(tmp_path / "rejected")))
    assert not (tmp_path / "rejected" / "validation.json").exists()


def test_completed_export_identity_changes_qualification_identity() -> None:
    original = build_qualification("2026.11.03.1")
    changed = build_qualification(
        "2026.11.03.1",
        model_uri="s3://marin-us-east-02a/completed/another-export",
        model_version="2026.11.03.372000",
        export_manifest_sha256="a" * 64,
    )

    assert original.numerics.fingerprint() != changed.numerics.fingerprint()
    assert original.updates.fingerprint() != changed.updates.fingerprint()
    larger_probe = build_qualification("2026.11.03.1", probe_prompts=32)
    assert original.numerics.fingerprint() != larger_probe.numerics.fingerprint()
    assert original.updates.fingerprint() == larger_probe.updates.fingerprint()
    # Input adoption is a read-only dependency, with no producer/export job attached.
    for step in (original.numerics, original.updates):
        model = next(dep for dep in step.deps if dep.name == "models/hero-535b-rl-initializer")
        assert model.adopt_source is not None
        assert model.deps == ()
