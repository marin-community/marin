"""Behavior tests for config-native checkpoint export derivation."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np
from safetensors.numpy import save
from omegaconf import OmegaConf, open_dict

from cloud.iris import export_hf_checkpoint, launch
from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.export_hf_checkpoint import (
    ExportJobSpec,
    _run_export,
    checkpoint_export_launch_config,
)
from cloud.iris.iris_backend import IrisBackend
from marinskyrl.model_manifest import write_local_model_manifest
from cloud.iris.launch_config import derive_launch_topology, load_launch_config, validate_launch_config
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.role_plan import derive_num_nodes, derive_policy_export_role_plan, derive_role_plan
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT, RuntimeProfile
from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback, HFModelSaveCallback
from skyrl_train.config.utils import get_default_config
from skyrl_train.hf_export import read_hf_export_request, write_hf_export_request
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.hf_export_schema import HFExportRequest, HFExportStatus
from marinskyrl.checkpoint_paths import policy_export_path, selected_export_checkpoint
from skyrlrecipe import RecipePatch
from skyrlrecipe.sections import Trainer


def _training_config():
    return OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt", "mode": "train", "export_hf": True},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.main_base", "profile": "megatron"},
            "iris": {
                "job_name": "run",
                "allocation": {"num_nodes": 2, "gpus_per_node": 8, "gpu_variant": "H100"},
            },
            "artifacts": {
                "checkpoint_root": "s3://run/checkpoints",
                "export_root": "s3://run/exports",
                "attempts_root": "s3://run/attempts",
                "resolved_config_uri": "s3://run/resolved.yaml",
                "terminal_manifest_uri": "s3://run/terminal.json",
            },
            "skyrl": {
                "trainer": {
                    "strategy": "megatron",
                    "placement": {"policy_num_nodes": 2, "policy_num_gpus_per_node": 8},
                    "policy": {"model": {"path": "Qwen/Qwen3-8B"}},
                },
                "generator": {"inference_engine_tensor_parallel_size": 1},
                "model_num_attention_heads": 8,
            },
        }
    )


def _request() -> HFExportRequest:
    return HFExportRequest(
        step=7,
        checkpoint_base_path="s3://run/checkpoints",
        checkpoint_path="s3://run/checkpoints/global_step_7",
        export_path="s3://run/exports",
        model_path="Qwen/Qwen3-8B",
        num_nodes=1,
        gpus_per_node=4,
    )


def _spec(request: HFExportRequest) -> ExportJobSpec:
    return ExportJobSpec(
        request=request,
        cluster="cw-rno2a",
        priority="batch",
        gpu_variant="H100",
        job_name="run-export",
        timeout=7200,
        no_wait=False,
        launch_config_path="/tmp/export.yaml",
    )


@pytest.mark.parametrize(("transport", "private"), [("broadcast", True), ("expert_block", False)])
def test_checkpoint_export_config_carries_request_fields_as_data(tmp_path: Path, transport: str, private: bool) -> None:
    raw = raw_launch_config()
    raw["iris"]["allocation"]["gpus_per_node"] = 4
    raw["skyrl"]["trainer"]["placement"]["policy_num_gpus_per_node"] = 4
    raw["skyrl"]["trainer"]["policy"] = {
        "megatron_config": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 4,
        }
    }
    raw["skyrl"]["generator"].update(
        weight_sync_transport=transport,
        expert_block_sync={"verify": True},
        num_inference_engines=1,
        inference_engine_data_parallel_size=4,
        inference_engine_expert_parallel_size=4,
        engine_init_kwargs={"kernel_config": {"moe_backend": "triton"}},
    )
    raw["skyrl"]["trainer"].update(RecipePatch(trainer=Trainer(hf_hub_repo_id="org/model")).to_skyrl()["trainer"])
    path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create(raw), path)
    parser = export_hf_checkpoint.argument_parser()
    args = [
        "--launch-config", str(path), "--ckpt_path", "s3://run/checkpoints", "--step", "7",
        "--model_path", "Qwen/Qwen3-8B", "--num-nodes", "1", "--gpus-per-node", "4",
        "--gpu-variant", "H100", "--hf-hub-repo-id", "org/model", "--hf-hub-revision", "release",
    ]
    if private:
        args.append("--hf-hub-private")
    spec = export_hf_checkpoint.manual_spec(parser.parse_args(args), parser)
    request = spec.request
    config = load_launch_config(spec.launch_config_path)
    topology = validate_launch_config(config)
    assert (topology.num_nodes, topology.gpus_per_node) == (request.num_nodes, request.gpus_per_node)

    assert config.run.mode == "checkpoint_export"
    assert config.runtime.entrypoint == "skyrl_train.entrypoints.checkpoint_export"
    assert config.runtime.profile == "megatron-export"
    assert config.iris.timeout == 7200
    assert config.iris.allocation.gpu_variant == "H100"
    assert config.skyrl.trainer.placement.policy_num_nodes == 1
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4
    assert (request.hf_hub_repo_id, request.hf_hub_private, request.hf_hub_revision) == ("org/model", private, "release")
    assert OmegaConf.to_container(config.skyrl.checkpoint_export) == {
        "step": request.step,
        "checkpoint_path": request.checkpoint_path,
        "export_root": request.export_path,
    }


def test_checkpoint_export_config_preserves_saved_policy_geometry_on_whole_nodes() -> None:
    request = _request()
    spec = replace(_spec(request), allocation_gpus_per_node=8)

    config = checkpoint_export_launch_config(_training_config(), request, spec)

    assert config.iris.allocation.gpus_per_node == 8
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4
    assert config.runtime.entrypoint == CHECKPOINT_EXPORT_ENTRYPOINT
    assert config.runtime.profile == RuntimeProfile.MEGATRON_EXPORT.value


@pytest.mark.parametrize("policy_loss_type", ["dpo", "ftpo"])
def test_checkpoint_export_allocation_excludes_unequal_reference_pool(policy_loss_type: str) -> None:
    training = OmegaConf.merge(
        _training_config(),
        {
            "iris": {"allocation": {"cpu": 16, "memory": "128GB", "disk": "256GB"}},
            "skyrl": {
                "trainer": {
                    "placement": {
                        "colocate_all": False,
                        "colocate_policy_ref": False,
                        "policy_num_nodes": 15,
                        "ref_num_nodes": 8,
                        "ref_num_gpus_per_node": 8,
                    },
                    "algorithm": {"policy_loss_type": policy_loss_type},
                    "train_batch_size": 16,
                    "policy_mini_batch_size": 16,
                    "micro_train_batch_size_per_gpu": 2,
                },
                "generator": {
                    "backend": "vllm",
                    "run_engines_locally": True,
                    "num_inference_engines": 8,
                    "n_samples_per_prompt": 1,
                },
            },
        },
    )
    request = replace(_request(), num_nodes=15, gpus_per_node=8)
    export = checkpoint_export_launch_config(training, request, _spec(request))

    allocation = derive_launch_topology(OmegaConf.to_container(export, resolve=True))

    assert allocation.num_nodes * allocation.gpus_per_node == 120
    export_plan = derive_policy_export_role_plan(OmegaConf.to_container(export.skyrl, resolve=True))
    assert [claim.role_id for claim in export_plan.claims] == ["policy"]
    assert derive_num_nodes(export_plan) == 15
    training_plan = derive_role_plan(OmegaConf.to_container(training.skyrl, resolve=True))
    assert training_plan.claim("reference").num_nodes == 8
    assert derive_num_nodes(training_plan) == 24


def test_nested_export_keeps_parent_response_stream_clean(capfd) -> None:
    request = _request()
    spec = replace(_spec(request), no_wait=True)

    _run_export(spec, [sys.executable, "-c", "print('nested export response')"])

    captured = capfd.readouterr()
    assert "nested export response" not in captured.out
    assert "nested export response" in captured.err


def test_nested_export_reads_cluster_config_relative_to_parent_launch(tmp_path: Path, monkeypatch) -> None:
    cluster_config = tmp_path / "lib/iris/config/cw-rno2a.yaml"
    cluster_config.parent.mkdir(parents=True)
    cluster_config.write_text("platform:\n  coreweave:\n    kube_context: export-cluster\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TEST_KEY", "test-export-credential")
    document = raw_launch_config()
    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / "global_step_7"
    checkpoint.mkdir(parents=True)
    (checkpoints / CHECKPOINT_MARKER_FILENAME).write_text("7\n")
    document["iris"]["cluster_config"] = "lib/iris/config/cw-rno2a.yaml"
    document["skyrl"]["trainer"]["hf_export_checkpoint"] = "latest"
    for name, value in (("checkpoint_root", checkpoints), ("export_root", tmp_path / "exports")):
        document["artifacts"][name] = str(value)
    path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create(document), path)
    request = replace(
        _request(),
        checkpoint_base_path=str(checkpoints),
        checkpoint_path=str(checkpoint),
        export_path=str(tmp_path / "exports"),
    )
    write_hf_export_request(request)
    output = tmp_path / "resolved-cluster.json"

    def child_export(command, *, stdout):
        args = launch.create_parser().parse_args(command[3:])
        assert tuple(args.forward_env) == ("HF_TEST_KEY",)
        config = load_launch_config(args.config)
        cluster = OmegaConf.to_container(OmegaConf.load(config.iris.cluster_config), resolve=True)
        output.write_text(json.dumps(cluster))
        assert "test-export-credential" not in args.config.read_text()
        assert "test-export-credential" not in " ".join(command)
        export = Path(policy_export_path(config.artifacts.export_root, request.step))
        export.mkdir(parents=True)
        (export / "model.safetensors").write_bytes(save({"weight": np.ones(1, dtype=np.float32)}))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (export / name).write_text("{}")
        write_local_model_manifest(export)
        return 0

    monkeypatch.setattr(export_hf_checkpoint.subprocess, "call", child_export)
    IrisBackend(forward_env=("HF_TEST_KEY",)).export_terminal_policy(path)
    assert read_hf_export_request(str(checkpoint)).status is HFExportStatus.COMPLETE
    assert json.loads(output.read_text())["platform"]["coreweave"]["kube_context"] == "export-cluster"


def test_export_selection_distinguishes_best_saved_step_from_latest(tmp_path: Path) -> None:
    (tmp_path / "best_checkpoint.json").write_text(json.dumps({"step": 15, "model_format": "native"}))
    assert selected_export_checkpoint(str(tmp_path), "best", 20).step == 15
    assert selected_export_checkpoint(str(tmp_path), "latest", 20).step == 20


def test_export_selection_requires_the_best_initializer_uri(tmp_path: Path) -> None:
    (tmp_path / "best_checkpoint.json").write_text(json.dumps({"step": 0, "model_format": "hf"}))
    with pytest.raises(ValueError, match="initializer HF model URI"):
        selected_export_checkpoint(str(tmp_path), "best", 20)


@pytest.mark.parametrize("mode", ["export", "dry-run", "missing", "initializer"])
def test_recovery_exports_the_best_checkpoint_without_a_terminal_request(tmp_path, monkeypatch, capsys, mode):
    checkpoints, exports = tmp_path / "checkpoints", tmp_path / "exports"
    for step in (15, 20):
        checkpoint = checkpoints / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.pt").write_bytes(b"complete")
    (checkpoints / CHECKPOINT_MARKER_FILENAME).write_text("20")
    initial_model = "s3://models/sft/policy"
    if mode != "missing":
        selection = {"step": 15, "model_format": "native"}
        if mode == "initializer":
            selection = {"step": 0, "model_format": "hf", "model_uri": initial_model}
        (checkpoints / "best_checkpoint.json").write_text(json.dumps(selection))
    document = raw_launch_config()
    document["artifacts"].update(checkpoint_root=str(checkpoints), export_root=str(exports))
    path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create(document), path)

    def child_export(command, *, stdout):
        args = launch.create_parser().parse_args(command[3:])
        config = load_launch_config(args.config)
        policy = Path(policy_export_path(config.artifacts.export_root, config.skyrl.checkpoint_export.step))
        policy.mkdir(parents=True)
        (policy / "model.safetensors").write_bytes(save({"weight": np.ones(1, dtype=np.float32)}))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (policy / name).write_text("{}")
        write_local_model_manifest(policy)
        return 0

    monkeypatch.setattr(export_hf_checkpoint.subprocess, "call", child_export)
    argv = ["export_hf_checkpoint", "--launch-config", str(path), "--recover-best"]
    if mode == "dry-run":
        argv.append("--dry-run")
    monkeypatch.setattr("sys.argv", argv)
    if mode == "missing":
        with pytest.raises(ValueError, match="No best checkpoint record"):
            export_hf_checkpoint.main()
    else:
        export_hf_checkpoint.main()
    request = read_hf_export_request(str(checkpoints / "global_step_15"))
    if mode == "export":
        assert request.status is HFExportStatus.COMPLETE
        assert not (exports / "global_step_20").exists()
        policy = Path(policy_export_path(str(exports), 15))
        (policy / "model.safetensors").unlink()
        with pytest.raises(ValueError, match="completed request has no verified policy export"):
            export_hf_checkpoint.main()
        assert read_hf_export_request(str(checkpoints / "global_step_15")) == request
    else:
        assert not exports.exists()
        assert request is None
        if mode == "dry-run":
            assert "cloud.iris.launch" in capsys.readouterr().out
        elif mode == "initializer":
            assert initial_model in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selection", "expected_step", "private"),
    [("best", 15, False), ("latest", 20, True)],
)
def test_trainer_records_export_request_for_selected_checkpoint(tmp_path: Path, selection, expected_step, private) -> None:
    config = get_default_config()
    config.trainer.ckpt_path = str(tmp_path / "checkpoints")
    config.trainer.export_path = str(tmp_path / "exports")
    config.trainer.hf_export_checkpoint = selection
    config.trainer.export_hf_artifact = True
    trainer_recipe = Trainer(hf_hub_repo_id="org/model")
    if private:
        trainer_recipe = trainer_recipe.replace(hf_hub_private=True)
    with open_dict(config.trainer):
        config.trainer.merge_with(RecipePatch(trainer=trainer_recipe).to_skyrl()["trainer"])
    config.trainer.policy.model.path = "org/policy"
    root = Path(config.trainer.ckpt_path)
    for step in (15, 20):
        checkpoint = root / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.pt").write_bytes(b"complete")
    (root / "best_checkpoint.json").write_text(json.dumps({"step": 15, "model_format": "native"}))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.global_step = 20
    trainer.all_timings = {}
    trainer.handle_hf_export()
    selected = root / f"global_step_{expected_step}"
    request = read_hf_export_request(str(selected))
    assert request is not None
    assert request.step == expected_step
    assert request.checkpoint_path == str(selected)
    assert request.model_path == "org/policy"
    assert request.hf_hub_repo_id == "org/model"
    assert request.hf_hub_private is private
    other_step = 20 if expected_step == 15 else 15
    assert read_hf_export_request(str(root / f"global_step_{other_step}")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "best_metadata", "final_committed", "expected_step", "expected_error"),
    [
        ("best", None, False, None, None),
        ("best", {"step": 0, "model_format": "hf"}, False, None, ValueError),
        ("best", {"step": 15, "model_format": "native"}, False, 15, None),
        ("best_and_final", None, True, 20, None),
        ("best_and_final", {"step": 15, "model_format": "native"}, True, 15, None),
        ("best_and_final", {"step": 15, "model_format": "native"}, False, None, OSError),
    ],
)
async def test_best_export_waits_until_train_end(
    tmp_path: Path, selection, best_metadata, final_committed, expected_step, expected_error
) -> None:
    config = get_default_config()
    config.trainer.ckpt_path = str(tmp_path)
    config.trainer.export_path = str(tmp_path / "exports")
    config.trainer.hf_export_checkpoint = selection
    with open_dict(config.trainer):
        config.trainer.checkpoint_retention = selection
    config.trainer.policy.model.path = "org/policy"
    if best_metadata is not None:
        (tmp_path / "best_checkpoint.json").write_text(json.dumps(best_metadata))
        if best_metadata["model_format"] == "native":
            checkpoint = tmp_path / "global_step_15"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.pt").write_bytes(b"complete")
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.global_step = 10
    trainer._last_saved_step = 15
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.num_steps_per_epoch = 20
    trainer.total_training_steps = 20
    trainer.eval_dataset = None
    trainer.colocate_all = False
    trainer._control = TrainerControl()
    trainer.callback_handler = CallbackHandler([HFModelSaveCallback(save_steps=10)])
    state = TrainerState(global_step=10, epoch=0, total_steps=20, num_steps_per_epoch=20)

    await trainer._run_step_end_callbacks(state)

    assert not list(tmp_path.glob("global_step_*/hf_export_request.json"))
    if selection == "best_and_final":
        trainer.callback_handler = CallbackHandler([CheckpointCallback(save_steps=21), HFModelSaveCallback()])
        trainer._checkpoint_save_failures = 0

        async def rollout_state():
            return None

        async def upload(step):
            if not final_committed:
                raise OSError("fixture upload failure")
            checkpoint = tmp_path / f"global_step_{step}"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.pt").write_bytes(b"complete")
            trainer._last_saved_step = step
            return 0.0, 0.0

        trainer._rollout_state = rollout_state
        trainer._snapshot_checkpoint = lambda _state: trainer.global_step
        trainer._finish_checkpoint_upload = upload
    if expected_error is not None:
        with pytest.raises(expected_error, match="did not commit" if selection == "best_and_final" else None):
            await trainer._finalize_training(completed_step=20, epoch=0)
        assert not list(tmp_path.glob("global_step_*/hf_export_request.json"))
        return

    await trainer._finalize_training(completed_step=20, epoch=0)
    if expected_step is None:
        assert not list(tmp_path.glob("global_step_*/hf_export_request.json"))
        return
    checkpoint = tmp_path / f"global_step_{expected_step}"
    request = read_hf_export_request(str(checkpoint))
    assert request is not None
    assert request.step == expected_step
    assert request.checkpoint_path == str(checkpoint)
