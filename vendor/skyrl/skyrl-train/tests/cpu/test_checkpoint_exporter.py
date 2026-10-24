import os
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedTokenizerFast, Qwen2Config

from skyrl_train.checkpoint_exporter import CheckpointExportPlan, CheckpointExporter, RayPolicyExportWorkers
from skyrl_train.hf_export_schema import HFUploadMode
from skyrl_train.hf_model_io import local_hf_model_dir, verify_hf_model_export
from skyrl_train.hf_release_export import ReleaseExportOptions, finalize_hf_release
from marin_hf_release import PrecisionMode
from skyrl_train.hf_publisher import HuggingFacePublisher
from skyrl_train.hf_export import read_hf_export_request
from skyrl_train.io import io as checkpoint_exporter_io
from skyrl_train.trainer import RayPPOTrainer


class FakePolicyExportWorkers:
    def __init__(self):
        self.model_path = None
        self.checkpoint_path = None
        self.export_path = None
        self.tokenizer = None
        self.release_options = None
        self.closed = False

    def initialize(self, model_path: str) -> None:
        self.model_path = model_path

    def load_model_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def save_hf_model(
        self, export_path: str, tokenizer: PreTrainedTokenizerBase, release_options: ReleaseExportOptions
    ) -> None:
        self.export_path = export_path
        self.tokenizer = tokenizer
        self.release_options = release_options
        Path(export_path).mkdir(parents=True)
        Path(export_path, "model.safetensors").write_bytes(b"weights")
        Path(export_path, "hf_release_manifest.json").write_text("{}")
        Path(export_path, ".marinskyrl-model-manifest.json").write_text("{}")

    def close(self) -> None:
        self.closed = True


class MissingExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, _tokenizer: object, _release_options: ReleaseExportOptions) -> None:
        self.export_path = export_path


class MetadataOnlyExportWorkers(FakePolicyExportWorkers):
    def save_hf_model(self, export_path: str, _tokenizer: object, _release_options: ReleaseExportOptions) -> None:
        self.export_path = export_path
        Path(export_path).mkdir(parents=True)
        Path(export_path, "config.json").write_text("{}")
        Path(export_path, "model.safetensors.index.json").write_text(
            '{"weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"}}'
        )


class FakeRayActorGroup:
    def __init__(self):
        self.calls = []
        self.killed = False

    def async_run_ray_method(self, dispatch_type, method_name, *args, **kwargs):
        self.calls.append((dispatch_type, method_name, args, kwargs))
        return [f"{method_name}-ref"]

    def kill_actors(self):
        self.killed = True


class FakeHubApi:
    def __init__(self):
        self.repositories = []
        self.uploads = []

    def create_repo(self, **kwargs):
        self.repositories.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


class FakePublisher:
    def __init__(self):
        self.calls = []

    def publish(self, export_path: str, step: int) -> None:
        self.calls.append((export_path, step))


class TinyQwenExportWorkers:
    def __init__(self, config: Qwen2Config):
        self.config = config
        self.release_options = None

    def initialize(self, _model_path: str) -> None:
        pass

    def load_model_checkpoint(self, _checkpoint_path: str) -> None:
        pass

    def save_hf_model(self, export_path: str, tokenizer: object, release_options: ReleaseExportOptions) -> None:
        self.release_options = release_options
        model = AutoModelForCausalLM.from_config(self.config)
        with local_hf_model_dir(export_path) as directory:
            save_file(model.state_dict(), Path(directory) / "model.safetensors")
            finalize_hf_release(Path(directory), self.config, tokenizer, release_options)

    def close(self) -> None:
        pass


def _plan(tmp_path: Path, step: int = 12) -> CheckpointExportPlan:
    checkpoint_path = tmp_path / "checkpoints" / f"global_step_{step}"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "policy").mkdir()
    torch.save({"global_step": step}, checkpoint_path / "trainer_state.pt")
    source_model = tmp_path / "source-model"
    source_model.mkdir(exist_ok=True)
    (source_model / "generation_config.json").write_text('{"eos_token_id": 1}')
    return CheckpointExportPlan(
        step=step,
        checkpoint_path=str(checkpoint_path),
        export_root=str(tmp_path / "exports"),
        model_path=str(source_model),
        eos_token_ids=[1],
    )


def test_trainer_persists_resolved_omegaconf_export_metadata(tmp_path):
    checkpoint_path = tmp_path / "checkpoints" / "global_step_12"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "trainer_state.pt").write_bytes(b"completed")
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "ckpt_path": str(tmp_path / "checkpoints"),
                "export_path": str(tmp_path / "exports"),
                "placement": {"policy_num_nodes": 1, "policy_num_gpus_per_node": 1},
                "policy": {
                    "model": {
                        "path": "s3://test-bucket/source",
                        "source_uri": None,
                        "source_identity": None,
                    }
                },
                "hf_generation_eos_token_ids": [1, 2],
                "hf_generation_controls": {"temperature": 0.7, "nested": {"max_length": 32}},
            }
        }
    )
    trainer.global_step = 12

    trainer._handle_hf_export()
    trainer._handle_hf_export()

    request = read_hf_export_request(str(checkpoint_path))
    assert request is not None
    assert request.eos_token_ids == [1, 2]
    assert request.generation_controls == {"temperature": 0.7, "nested": {"max_length": 32}}


def test_checkpoint_exporter_converts_only_the_policy_model(tmp_path):
    plan = _plan(tmp_path)
    workers = FakePolicyExportWorkers()
    tokenizer = object()

    result = CheckpointExporter(plan, workers, tokenizer).run()

    assert result.step == 12
    assert result.export_path == str(tmp_path / "exports" / "global_step_12" / "policy")
    assert workers.model_path == str(tmp_path / "source-model")
    assert workers.checkpoint_path == str(tmp_path / "checkpoints" / "global_step_12" / "policy")
    assert workers.export_path == result.export_path
    assert workers.tokenizer is tokenizer
    assert workers.release_options.precision is PrecisionMode.PRESERVE
    assert workers.release_options.eos_token_ids == [1]
    assert workers.release_options.source_step == 12
    assert workers.closed


@pytest.mark.parametrize(("source_generation", "expected_eos"), [(True, [1, 2]), (False, [1])])
def test_checkpoint_exporter_releases_non_grug_s3_source_without_explicit_eos(
    tmp_path, monkeypatch, source_generation, expected_eos
):
    config = Qwen2Config(
        vocab_size=5,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    raw = Tokenizer(WordLevel({"[BOS]": 0, "[EOS]": 1, "[EOT]": 2, "hello": 3, "[UNK]": 4}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    plan = replace(_plan(tmp_path), model_path="s3://test-bucket/qwen-source", eos_token_ids=None)
    source_generation_path = "s3://test-bucket/qwen-source/generation_config.json"
    original_exists = checkpoint_exporter_io.exists
    original_open_file = checkpoint_exporter_io.open_file

    def source_exists(path):
        if path == source_generation_path:
            return source_generation
        return original_exists(path)

    monkeypatch.setattr(checkpoint_exporter_io, "exists", source_exists)
    monkeypatch.setattr(
        checkpoint_exporter_io,
        "open_file",
        lambda path, mode: StringIO('{"eos_token_id": [1, 2]}')
        if path == source_generation_path
        else original_open_file(path, mode),
    )
    workers = TinyQwenExportWorkers(config)

    result = CheckpointExporter(plan, workers, tokenizer).run()

    verify_hf_model_export(result.export_path, require_release_manifest=True)
    assert workers.release_options.eos_token_ids == ([1, 2] if source_generation else None)
    generation = json.loads((Path(result.export_path) / "generation_config.json").read_text())
    assert generation["eos_token_id"] == expected_eos


def test_checkpoint_exporter_rejects_a_mismatched_checkpoint_marker(tmp_path):
    plan = _plan(tmp_path)
    torch.save({"global_step": 11}, Path(plan.checkpoint_path) / "trainer_state.pt")
    workers = FakePolicyExportWorkers()

    with pytest.raises(ValueError, match="checkpoint step mismatch"):
        CheckpointExporter(plan, workers, object()).run()

    assert workers.closed


@pytest.mark.parametrize(
    ("worker_type", "error"),
    [
        (MissingExportWorkers, "no safetensors weights"),
        (MetadataOnlyExportWorkers, "missing 1 referenced safetensors shard"),
    ],
)
def test_checkpoint_exporter_rejects_incomplete_conversion_result(tmp_path, worker_type, error):
    workers = worker_type()
    publisher = FakePublisher()

    with pytest.raises(RuntimeError, match=error):
        CheckpointExporter(_plan(tmp_path), workers, object(), publisher).run()

    assert workers.closed
    assert publisher.calls == []


def test_unsharded_cloud_export_verification_does_not_head_optional_index(monkeypatch):
    export_path = "s3://bucket/exports/global_step_1/policy"
    monkeypatch.setattr(
        "skyrl_train.hf_model_io.io.list_dir",
        lambda path: [f"{path}/config.json", f"{path}/model.safetensors"],
    )

    def reject_head(path):
        raise AssertionError(f"verification issued an object HEAD request: {path}")

    monkeypatch.setattr("skyrl_train.hf_model_io.io.exists", reject_head)

    verify_hf_model_export(export_path)


def test_ray_policy_export_workers_loads_model_state_without_training_state():
    group = FakeRayActorGroup()
    resolved_refs = []
    workers = RayPolicyExportWorkers(group, resolve=lambda refs: resolved_refs.append(refs))

    workers.initialize("org/model")
    workers.load_model_checkpoint("/checkpoints/global_step_12/policy")
    options = ReleaseExportOptions(
        precision=PrecisionMode.PRESERVE,
        source_checkpoint="/checkpoints/global_step_12",
        source_step=12,
        checkpoint_identity="identity",
        source_model=None,
        source_model_commit=None,
        eos_token_ids=[1],
        generation_controls={},
        chat_template=None,
    )
    workers.save_hf_model("/exports/global_step_12/policy", tokenizer="tokenizer", release_options=options)
    workers.close()

    assert group.calls == [
        ("pass_through", "init_model_for_export", ("org/model",), {}),
        (
            "pass_through",
            "load_checkpoint",
            (),
            {
                "ckpt_dir": "/checkpoints/global_step_12/policy",
                "load_training_state": False,
            },
        ),
        (
            "pass_through",
            "save_hf_model",
            ("/exports/global_step_12/policy", "tokenizer", options),
            {},
        ),
    ]
    assert resolved_refs == [
        ["init_model_for_export-ref"],
        ["load_checkpoint-ref"],
        ["save_hf_model-ref"],
    ]
    assert group.killed


def test_hf_publisher_publishes_root_and_requested_archive(tmp_path, monkeypatch):
    export_path = tmp_path / "exports" / "global_step_12" / "policy"
    export_path.mkdir(parents=True)
    (export_path / "model.safetensors").write_bytes(b"weights")
    api = FakeHubApi()
    publisher = HuggingFacePublisher(
        repo_id="org/model",
        private=True,
        revision="main",
        upload_mode=HFUploadMode.ALL,
        api=api,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    publisher.publish(str(export_path), step=12)

    assert api.repositories == [{"repo_id": "org/model", "repo_type": "model", "private": True, "exist_ok": True}]
    assert [upload["path_in_repo"] for upload in api.uploads] == ["", "checkpoints/step_12"]
    assert all(upload["folder_path"] == str(export_path) for upload in api.uploads)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
