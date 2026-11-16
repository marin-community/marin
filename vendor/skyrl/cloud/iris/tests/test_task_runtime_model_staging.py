from argparse import Namespace
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from dataclasses import asdict
from pathlib import Path
import shutil

import huggingface_hub
import huggingface_hub.constants
import fsspec
import numpy as np
import pytest
from omegaconf import OmegaConf
from safetensors.numpy import save
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from cloud.iris import hf_model_cache, task_runtime
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, HuggingFaceSnapshotFile, publish_hugging_face_snapshot
from cloud.iris.task_runtime import (
    _write_final_config,
    policy_chat_template_model,
    prepare_draft_model,
    prepare_policy_model,
    prepare_policy_tokenizer,
)
from cloud.iris.tests.model_snapshot_fixtures import memory_model_snapshot
from marinskyrl.speculative_decoding import SpeculatorModelConfig
from marinskyrl.hf_model import HuggingFaceModelCache
from marinskyrl.resource_locator import RolloutModelLoading


@pytest.mark.parametrize(
    ("prestage_model", "model_local_path", "expected"),
    [
        ("", "/tmp/materialized-model", "/tmp/materialized-model"),
        ("org/model", "/tmp/materialized-model", "org/model"),
    ],
)
def test_policy_chat_template_selects_materialized_model(
    prestage_model: str, model_local_path: str, expected: str
) -> None:
    assert policy_chat_template_model(prestage_model, model_local_path) == expected


def test_manifest_policy_stages_metadata_without_materializing_weights(tmp_path, monkeypatch) -> None:
    policy_uri = str(tmp_path / "published-policy")
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"policy-source/{tmp_path.name}"), policy_uri, model_id="org/policy", revision="a" * 40
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    args = Namespace(
        model_source_uri=policy_uri,
        model_source_identity=manifest.identity,
        prestage_model="",
        model_revision="",
        runtime_profile="megatron",
        model_local_path="/tmp/materialized-model",
        rollout_model_loading=RolloutModelLoading.STREAM,
    )

    policy_model = prepare_policy_model(args)

    assert policy_model is not None
    staged = Path(policy_model.local_path)
    assert (staged / "config.json").read_text() == "{}"
    assert (staged / "tokenizer.json").is_file()
    assert not (staged / "model.safetensors").exists()


@pytest.mark.parametrize("manifest_identity", [True, False])
def test_local_rollout_staging_preserves_checkpoint_bytes_and_source(tmp_path, monkeypatch, manifest_identity) -> None:
    policy_uri = str(tmp_path / "published-policy")
    snapshot = memory_model_snapshot(f"local-policy-source/{tmp_path.name}")
    manifest = publish_hugging_face_snapshot(snapshot, policy_uri, model_id="org/policy", revision="a" * 40)
    identity = manifest.identity if manifest_identity else "policy-export/step-57"
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    prepared = prepare_policy_model(
        Namespace(
            model_source_uri=policy_uri,
            model_source_identity=identity,
            runtime_profile="megatron",
            rollout_model_loading=RolloutModelLoading.STAGE_LOCAL,
        )
    )
    assert prepared is not None
    assert prepared.source_uri == policy_uri
    assert prepared.source_identity == identity
    local = Path(prepared.local_path)
    assert (local / "model.safetensors").read_bytes() == Path(policy_uri, "model.safetensors").read_bytes()
    assert (local / "config.json").read_bytes() == Path(policy_uri, "config.json").read_bytes()
    # A repeated startup uses the completed artifact, including all weight shards.
    repeated = prepare_policy_model(
        Namespace(
            model_source_uri=policy_uri,
            model_source_identity=identity,
            runtime_profile="megatron",
            rollout_model_loading=RolloutModelLoading.STAGE_LOCAL,
        )
    )
    assert repeated == prepared
    assert (local / "model.safetensors").read_bytes() == Path(policy_uri, "model.safetensors").read_bytes()


def test_requested_local_policy_tokenizer_is_staged_independently(tmp_path, monkeypatch) -> None:
    tokenizer_source = tmp_path / "tokenizer-source"
    _write_tokenizer(tokenizer_source)
    destination = tmp_path / "prepared-tokenizer"
    monkeypatch.setattr(task_runtime, "_metadata_path", lambda _uri, _revision: str(destination))

    prepared = prepare_policy_tokenizer(
        Namespace(policy_tokenizer=str(tokenizer_source), policy_tokenizer_revision="tokenizer-commit")
    )

    assert prepared is not None
    assert prepared.local_path == str(destination)
    assert (destination / "tokenizer.json").read_bytes() == (tokenizer_source / "tokenizer.json").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(prepared.local_path, local_files_only=True)
    assert tokenizer.encode("hello world") == [1, 2]


def _write_tokenizer(destination: Path) -> None:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(destination)


@pytest.mark.parametrize("loading", [RolloutModelLoading.STREAM, RolloutModelLoading.STAGE_LOCAL])
@pytest.mark.parametrize("tokenizer_state", ["valid", "missing", "malformed"])
def test_model_only_artifact_uses_independently_validated_pinned_tokenizer(
    tmp_path, monkeypatch, loading, tokenizer_state
) -> None:
    source = tmp_path / "model-only"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "llama"}')
    weight_bytes = save({"weight": np.arange(4, dtype=np.float32)})
    (source / "model.safetensors").write_bytes(weight_bytes)
    tokenizer_source = tmp_path / "tokenizer-source"
    _write_tokenizer(tokenizer_source)
    if tokenizer_state == "missing":
        shutil.rmtree(tokenizer_source)
        tokenizer_source.mkdir()
    elif tokenizer_state == "malformed":
        (tokenizer_source / "tokenizer_config.json").write_text("not json")
    revision = "a" * 40

    def snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        assert repo_id == "org/separate-tokenizer"
        assert revision == "a" * 40
        shutil.copytree(tokenizer_source, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(hf_model_cache, "snapshot_download", snapshot_download)
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    args = Namespace(
        model_source_uri=source.as_uri(),
        model_source_identity="model-only@v1:completed",
        runtime_profile="megatron",
        rollout_model_loading=loading,
        policy_tokenizer="org/separate-tokenizer",
        policy_tokenizer_revision=revision,
    )
    model = prepare_policy_model(args)
    assert model is not None
    model_path = Path(model.local_path)
    assert (model_path / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert not (model_path / "tokenizer.json").exists()
    if loading is RolloutModelLoading.STAGE_LOCAL:
        assert (model_path / "model.safetensors").read_bytes() == weight_bytes
    else:
        assert not (model_path / "model.safetensors").exists()
    if tokenizer_state != "valid":
        with pytest.raises(ValueError):
            prepare_policy_tokenizer(args)
        return

    prepared = prepare_policy_tokenizer(args)
    assert prepared is not None
    assert prepared.local_path != model.local_path
    tokenizer = AutoTokenizer.from_pretrained(prepared.local_path, local_files_only=True)
    assert tokenizer.encode("hello world") == [1, 2]


def test_staged_models_are_written_as_structured_config(tmp_path, monkeypatch) -> None:
    identity = "sha256:" + "a" * 64
    policy = task_runtime.PreparedPolicyModel("s3://models/policy", identity, "/tmp/policy-metadata")
    tokenizer = task_runtime.PreparedPolicyTokenizer("/tmp/tokenizer-metadata")
    draft = SpeculatorModelConfig(
        source_uri="s3://models/tmp/ttl=14d/draft",
        source_identity=identity,
    )
    launch = OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt"},
            "inputs": {"model": {"uri": "s3://models/policy"}},
            "skyrl": {
                "trainer": {
                    "policy": {"model": {"path": "policy"}},
                    "ref": {"model": {"path": "policy"}},
                },
                "generator": {
                    "engine_init_kwargs": {"served_model_name": "policy"},
                    "speculative_decoding": {"model": {"source_uri": "old", "source_identity": "old"}},
                },
                "data": {"train_data": [], "val_data": [], "terminal_bench_data": []},
            },
        }
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path))

    path = _write_final_config(launch, policy_model=policy, policy_tokenizer=tokenizer, draft_model=draft)
    resolved = OmegaConf.load(path)

    assert resolved.skyrl.trainer.policy.model.path == "/tmp/policy-metadata"
    assert resolved.skyrl.trainer.ref.model.path == "/tmp/policy-metadata"
    assert resolved.skyrl.trainer.policy.model.source_uri == "s3://models/policy"
    assert resolved.skyrl.trainer.policy.model.tokenizer_path == "/tmp/tokenizer-metadata"
    assert resolved.skyrl.trainer.policy.model.tokenizer_revision is None
    assert resolved.skyrl.trainer.ref.model.tokenizer_path == "/tmp/tokenizer-metadata"
    assert resolved.skyrl.trainer.ref.model.tokenizer_revision is None
    assert resolved.skyrl.generator.speculative_decoding.model.source_uri == draft.source_uri


def test_staged_policy_model_supplies_its_embedded_tokenizer(tmp_path, monkeypatch) -> None:
    policy = task_runtime.PreparedPolicyModel("s3://models/policy", "step-630", "/tmp/policy-metadata")
    launch = OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt"},
            "inputs": {"model": {"uri": "s3://models/policy"}},
            "skyrl": {
                "trainer": {
                    "policy": {"model": {"path": "/tmp/stale-model", "tokenizer_path": "/tmp/stale-model"}},
                    "ref": {"model": {"path": "/tmp/stale-model", "tokenizer_path": "/tmp/stale-model"}},
                },
                "generator": {"engine_init_kwargs": {"served_model_name": "policy"}},
                "data": {"train_data": [], "val_data": [], "terminal_bench_data": []},
            },
        }
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path))

    path = _write_final_config(launch, policy_model=policy, policy_tokenizer=None, draft_model=None)
    resolved = OmegaConf.load(path)

    for role in ("policy", "ref"):
        model = resolved.skyrl.trainer[role].model
        assert model.path == "/tmp/policy-metadata"
        assert model.tokenizer_path == "/tmp/policy-metadata"
        assert model.tokenizer_revision is None


@pytest.mark.parametrize("requested_revision", ["a" * 40, "main", "refs/pr/7"])
def test_node_stages_bucket_mirror_with_hub_downloads_disabled(tmp_path, monkeypatch, requested_revision):
    mirror = tmp_path / "mirror"
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"staging-source/{tmp_path.name}"),
        str(mirror),
        model_id="org/policy",
        revision="a" * 40,
    )
    cache = tmp_path / "hf-cache"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    def reject_hub_download(*_args, **_kwargs):
        raise AssertionError("GPU node must not download from the Hub")

    monkeypatch.setattr(hf_model_cache, "snapshot_download", reject_hub_download)
    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", reject_hub_download)
    locator = HuggingFaceModelCache("org/policy", requested_revision, str(mirror), manifest.identity, manifest.revision)
    node_path = task_runtime.stage_model(locator)
    snapshot = huggingface_hub.snapshot_download(
        "org/policy",
        revision=requested_revision,
        cache_dir=cache,
        local_files_only=True,
    )
    assert Path(snapshot) == node_path
    assert (node_path / "model.safetensors").read_bytes() == (mirror / "model.safetensors").read_bytes()
    assert (node_path / "tokenizer.json").read_bytes() == (mirror / "tokenizer.json").read_bytes()


def test_export_stages_policy_without_requiring_old_teacher_mirrors(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"export-source/{tmp_path.name}"),
        str(mirror),
        model_id="org/policy",
        revision="a" * 40,
    )
    cache = tmp_path / "hf-cache"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    policy = HuggingFaceModelCache("org/policy", "a" * 40, str(mirror), manifest.identity, manifest.revision)
    # The derived export config can retain locators for training-only teachers whose TTL expired.
    teacher = HuggingFaceModelCache(
        "org/teacher", "b" * 40, "memory://expired-teacher-mirror", "sha256:" + "c" * 64, "b" * 40
    )
    config = OmegaConf.create(
        {
            "run": {"mode": "checkpoint_export"},
            "inputs": {"model": {"uri": "org/policy"}, "hugging_face_models": [asdict(policy), asdict(teacher)]},
            "skyrl": {"trainer": {"policy": {"model": {"revision": "a" * 40}}}},
        }
    )
    task_runtime.stage_hugging_face_models(config)
    path = Path(
        huggingface_hub.snapshot_download("org/policy", revision="a" * 40, cache_dir=cache, local_files_only=True)
    )
    assert (path / "model.safetensors").read_bytes() == (mirror / "model.safetensors").read_bytes()
    assert not (cache / "models--org--teacher").exists()


@pytest.mark.parametrize("requested_revision", ["a" * 40, "main", "refs/pr/7"])
@pytest.mark.parametrize("missing", ["model.safetensors", ".marinskyrl-model-manifest.json"])
def test_expired_node_mirror_falls_back_to_the_pinned_hub_commit(tmp_path, monkeypatch, missing, requested_revision):
    mirror = tmp_path / "mirror"
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"expired-source/{tmp_path.name}"),
        str(mirror),
        model_id="org/policy",
        revision="a" * 40,
    )
    (mirror / missing).unlink()
    cache = tmp_path / "hf-cache"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    locator = HuggingFaceModelCache("org/policy", requested_revision, str(mirror), manifest.identity, manifest.revision)
    downloads = []
    node_snapshot = cache / "models--org--policy" / "snapshots" / ("a" * 40)
    node_snapshot.mkdir(parents=True)
    weights = save({"weight": np.arange(4, dtype=np.float32)})

    def download(model_id, *, revision, allow_patterns):
        downloads.append((model_id, revision))
        # The Hub branch has advanced, but the CPU-selected commit remains available.
        selected = node_snapshot if revision == "a" * 40 else node_snapshot.parent / ("b" * 40)
        selected.mkdir(parents=True, exist_ok=True)
        (selected / "model.safetensors").write_bytes(weights)
        return selected

    monkeypatch.setattr(task_runtime, "download_hugging_face_snapshot", download)
    assert task_runtime.stage_model(locator) == node_snapshot
    assert (node_snapshot / "model.safetensors").read_bytes() == weights
    assert downloads == [("org/policy", "a" * 40)]
    assert (
        Path(
            huggingface_hub.snapshot_download(
                "org/policy", revision=requested_revision, cache_dir=cache, local_files_only=True
            )
        )
        == node_snapshot
    )


def test_concurrent_tasks_stage_the_same_node_snapshot(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"concurrent-staging/{tmp_path.name}"),
        str(mirror),
        model_id="org/policy",
        revision="a" * 40,
    )
    cache = tmp_path / "hf-cache"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    locator = HuggingFaceModelCache("org/policy", "main", str(mirror), manifest.identity, manifest.revision)
    ready = Barrier(8)

    def stage():
        ready.wait(timeout=10)
        return task_runtime.stage_model(locator)

    with ThreadPoolExecutor(max_workers=8) as workers:
        paths = list(workers.map(lambda _: stage(), range(8)))
    assert len(set(paths)) == 1
    assert (paths[0] / "model.safetensors").read_bytes() == (mirror / "model.safetensors").read_bytes()
    assert Path(huggingface_hub.snapshot_download("org/policy", revision="main", local_files_only=True)) == paths[0]


@pytest.mark.parametrize("missing", [None, "model.safetensors", ".marinskyrl-model-manifest.json"])
def test_cpu_draft_mirror_avoids_the_hub_until_any_mirror_file_expires(tmp_path, monkeypatch, missing):
    mirror = tmp_path / "draft"
    snapshot = memory_model_snapshot(f"draft-fallback/{tmp_path.name}")
    manifest = publish_hugging_face_snapshot(
        snapshot, str(mirror), model_id="org/draft", revision="d" * 40, tokenizer_mode="policy"
    )
    if missing:
        (mirror / missing).unlink()
    opened = []

    def open_snapshot(model_id, revision):
        opened.append((model_id, revision))
        return snapshot

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_, **__: str(mirror))
    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)
    cache = HuggingFaceModelCache("org/draft", "d" * 40, str(mirror), manifest.identity, manifest.revision)
    prepared = prepare_draft_model(
        SpeculatorModelConfig(str(mirror), manifest.identity),
        cache_ttl_days=14,
        cache_source_prefix="s3://region/run",
        model_caches=(cache,),
    )
    assert prepared == SpeculatorModelConfig(str(mirror), manifest.identity)
    assert (mirror / "model.safetensors").is_file()
    assert hf_model_cache.load_model_manifest(str(mirror)).identity == manifest.identity
    assert opened == ([("org/draft", "d" * 40)] if missing else [])


def test_shared_snapshot_keeps_tokenizers_and_templates_and_repairs_truncated_weights(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_tokenizer(source)
    (source / "config.json").write_text("{}")
    weights = save({"weight": np.arange(4, dtype=np.float32)})
    (source / "model.safetensors").write_bytes(weights)
    snapshot = HuggingFaceSnapshot(
        fsspec.filesystem("file"),
        str(source),
        tuple(
            HuggingFaceSnapshotFile(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in source.iterdir()
        ),
    )
    embedded = tmp_path / "embedded"
    embedded_manifest = publish_hugging_face_snapshot(snapshot, str(embedded), model_id="org/model", revision="a" * 40)
    model_only = tmp_path / "model-only"
    model_manifest = publish_hugging_face_snapshot(
        HuggingFaceSnapshot(
            snapshot.filesystem,
            snapshot.root,
            tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
        ),
        str(model_only),
        model_id="org/model",
        revision="a" * 40,
        tokenizer_mode="policy",
    )
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    full = HuggingFaceModelCache(
        "org/model", "a" * 40, str(embedded), embedded_manifest.identity, embedded_manifest.revision
    )
    partial = HuggingFaceModelCache(
        "org/model", "a" * 40, str(model_only), model_manifest.identity, model_manifest.revision
    )
    node = task_runtime.stage_model(full)
    shard = node / "model.safetensors"
    inode = shard.stat().st_ino
    template = tmp_path / "template.jinja"
    template.write_text("{{ messages[0]['content'] }}")
    task_runtime.apply_policy_chat_template(str(node), str(template))
    tokenizer_bytes = (node / "tokenizer.json").read_bytes()
    for locator in (partial, full):
        assert task_runtime.stage_model(locator) == node
        assert (node / "tokenizer.json").read_bytes() == tokenizer_bytes
        assert json.loads((node / "tokenizer_config.json").read_text())["chat_template"] == template.read_text()
        assert (node / "chat_template.jinja").read_text() == template.read_text()
        assert (node / "tokenizer_config.json.plainbak").is_file()
        assert shard.stat().st_ino == inode
    assert AutoTokenizer.from_pretrained(node, local_files_only=True).encode("hello world") == [1, 2]
    truncated = weights[:8]
    shard.write_bytes(truncated)
    original_get = snapshot.filesystem.get_file
    copies = []

    def copy_without_overwriting_live_file(remote, local, **kwargs):
        copies.append(remote)
        assert shard.read_bytes() == truncated
        original_get(remote, local, **kwargs)
        assert shard.read_bytes() == truncated

    monkeypatch.setattr(snapshot.filesystem, "get_file", copy_without_overwriting_live_file)
    assert task_runtime.stage_model(full) == node
    assert shard.read_bytes() == weights
    assert shard.stat().st_ino != inode
    assert copies == [str(embedded / "model.safetensors")]


def test_artifact_draft_staging_requires_matching_manifest_identity(tmp_path, monkeypatch) -> None:
    manifest = publish_hugging_face_snapshot(
        memory_model_snapshot(f"artifact-draft/{tmp_path.name}"),
        str(tmp_path / "draft"),
        model_id="org/draft",
        revision="a" * 40,
    )
    monkeypatch.setattr(task_runtime, "load_model_manifest", lambda _uri: manifest)
    artifact_uri = "s3://draft/artifact"
    artifact = prepare_draft_model(
        SpeculatorModelConfig(source_uri=artifact_uri, source_identity=manifest.identity),
        cache_ttl_days=14,
        cache_source_prefix="s3://region/run",
    )
    assert artifact.source_uri == artifact_uri and artifact.source_identity == manifest.identity
    mismatched = SpeculatorModelConfig(
        source_uri=artifact_uri, source_identity="sha256:" + ("1" if manifest.identity[-1] != "1" else "2") * 64
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_draft_model(
            mismatched,
            cache_ttl_days=14,
            cache_source_prefix="s3://region/run",
        )
