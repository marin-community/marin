"""Behavior tests for config-native checkpoint export derivation."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cloud.iris.export_hf_checkpoint import (
    ExportJobSpec,
    _run_export,
    checkpoint_export_launch_config,
)
from cloud.iris.launch_config import validate_iris_allocation
from cloud.iris.role_plan import derive_num_nodes, derive_policy_export_role_plan, derive_role_plan
from skyrl_train.config.utils import get_default_config
from skyrl_train.hf_export import read_hf_export_request
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.hf_export_schema import HFExportRequest
from marinskyrl.checkpoint_paths import selected_export_step


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


def test_checkpoint_export_config_carries_request_fields_as_data() -> None:
    request = _request()
    config = checkpoint_export_launch_config(_training_config(), request, _spec(request))

    assert config.run.mode == "checkpoint_export"
    assert config.runtime.entrypoint == "skyrl_train.entrypoints.checkpoint_export"
    assert config.runtime.profile == "megatron-export"
    assert config.iris.timeout == 7200
    assert config.iris.allocation.gpu_variant == "H100"
    assert config.skyrl.trainer.placement.policy_num_nodes == 1
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4
    assert config.skyrl.checkpoint_export.checkpoint_path == request.checkpoint_path
    assert config.skyrl.checkpoint_export.export_root == request.export_path


def test_checkpoint_export_config_preserves_federated_routing() -> None:
    request = _request()
    spec = replace(
        _spec(request),
        target_cluster="cw-rno2a",
        parent_cluster_config="/tmp/marin.yaml",
    )

    config = checkpoint_export_launch_config(_training_config(), request, spec)

    assert config.iris.target_cluster == "cw-rno2a"
    assert config.iris.parent_cluster_config == "/tmp/marin.yaml"


def test_checkpoint_export_config_preserves_saved_policy_geometry_on_whole_nodes() -> None:
    request = _request()
    spec = replace(_spec(request), allocation_gpus_per_node=8)

    config = checkpoint_export_launch_config(_training_config(), request, spec)

    assert config.iris.allocation.gpus_per_node == 8
    assert config.skyrl.trainer.placement.policy_num_gpus_per_node == 4


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

    allocation = validate_iris_allocation(OmegaConf.to_container(export, resolve=True))

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
    output = tmp_path / "resolved-cluster.json"
    monkeypatch.chdir(tmp_path)
    spec = replace(_spec(_request()), no_wait=True)
    command = [
        sys.executable,
        "-c",
        "import json,sys,yaml; from pathlib import Path; "
        "config=yaml.safe_load(Path(sys.argv[1]).read_text()); "
        "Path(sys.argv[2]).write_text(json.dumps(config))",
        "lib/iris/config/cw-rno2a.yaml",
        str(output),
    ]

    _run_export(spec, command)

    assert json.loads(output.read_text())["platform"]["coreweave"]["kube_context"] == "export-cluster"


def test_export_selection_distinguishes_best_saved_step_from_latest(tmp_path: Path) -> None:
    (tmp_path / "best_checkpoint.json").write_text(json.dumps({"step": 15, "model_format": "native"}))
    assert selected_export_step(str(tmp_path), "best", 20) == 15
    assert selected_export_step(str(tmp_path), "latest", 20) == 20


def test_export_selection_does_not_substitute_training_for_best_initializer(tmp_path: Path) -> None:
    (tmp_path / "best_checkpoint.json").write_text(json.dumps({"step": 0, "model_format": "hf"}))
    with pytest.raises(ValueError, match="initial HF model"):
        selected_export_step(str(tmp_path), "best", 20)


@pytest.mark.parametrize(("selection", "expected_step"), [("best", 15), ("latest", 20)])
def test_trainer_records_export_request_for_selected_checkpoint(tmp_path: Path, selection, expected_step) -> None:
    config = get_default_config()
    config.trainer.ckpt_path = str(tmp_path / "checkpoints")
    config.trainer.export_path = str(tmp_path / "exports")
    config.trainer.hf_export_checkpoint = selection
    config.trainer.export_hf_artifact = True
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
    other_step = 20 if expected_step == 15 else 15
    assert read_hf_export_request(str(root / f"global_step_{other_step}")) is None
