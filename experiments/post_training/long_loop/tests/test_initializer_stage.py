# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from haliax import Axis
from jax.sharding import AxisType, Mesh
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, NamedLmDataset
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.store.cache import SerialCacheWriter
from levanter.testing.devices import use_test_devices
from levanter.utils.thread_utils import blocking_wait
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.verified_copy import verified_copy_prefix
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.post_training.long_loop import exp3_initializer_smoke
from experiments.post_training.long_loop.exp3_initializer_smoke import initializer_recipe, run_initializer_smoke
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec, verified_initializer_model
from experiments.post_training.long_loop.sft import BASE_MODEL, BASE_REVISION
from experiments.post_training.long_loop.stage_guard import StageGuard


@pytest.fixture
def staged_initializer(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    weight = blobs / "weight"
    weight.write_bytes(b"unchanged pretrained tensor bytes")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(weight)
    destination = tmp_path / "regional"
    copied = verified_copy_prefix(str(snapshot), str(destination), workers=1)
    manifest_path = destination / ".verified-copy-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    receipt = {
        "repo": BASE_MODEL,
        "revision": BASE_REVISION,
        "model_ref": str(destination),
        "verified_copy_manifest": copied.manifest_url,
        "verified_copy_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "files": [
            {"path": "model.safetensors", "size": weight.stat().st_size, "sha256": digest, "lfs": {"sha256": digest}}
        ],
    }
    receipt_path = tmp_path / "source.json"
    raw = json.dumps(receipt).encode()
    receipt_path.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    return receipt_path, receipt_sha256, destination, weight


def test_verified_hub_snapshot_resolves_regional_model_and_rejects_changed_copy(staged_initializer):
    receipt_path, receipt_sha256, destination, weight = staged_initializer
    assert verified_initializer_model(str(receipt_path), receipt_sha256) == str(destination)
    assert (destination / "model.safetensors").read_bytes() == weight.read_bytes()
    manifest_path = destination / ".verified-copy-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion manifest SHA256 mismatch"):
        verified_initializer_model(str(receipt_path), receipt_sha256)


@pytest.fixture
def prepared_sources(tmp_path, staged_initializer):
    receipt_path, receipt_sha256, _, _ = staged_initializer
    tokens = {"<pad>": 0, **{f"doc{i}": i for i in range(1, 65)}, "<eos>": 65}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(tokens, unk_token="<pad>")), eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    cache = tmp_path / "cache"
    lengths = (3, 5, 12, 2, 11, 7, 4, 13)
    with SerialCacheWriter(
        str(cache), {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)}
    ) as writer:
        for index in range(1, 65):
            length = lengths[index % len(lengths)]
            weights = np.ones(length, np.float32)
            weights[-1] = 0
            writer.write_batch([{"input_ids": np.full(length, index, np.int32), "loss_weights": weights}])
    data = LmDataConfig(
        tokenizer=str(tmp_path / "tokenizer"),
        components={
            "source": DatasetComponent(
                cache_dir=str(cache),
                flat_cache=True,
                pack=True,
                format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
            )
        },
        train_weights={"source": 1.0},
        num_validation_sequences={"source": 8},
        shuffle_before_trainval_split=True,
        enforce_eos=False,
        auto_build_caches=False,
    )
    for filename in ("data.json", "c4-data.json"):
        (tmp_path / filename).write_text(json.dumps(draccus.encode(data)))
    (tmp_path / "source-loss.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "source": {
                        "loss_mode": "full_token",
                        "loss_weights_key": "loss_weights",
                        "loss_weights_indexing": "prediction",
                        "teacher_only": False,
                        "source_revision": "fixture-v1",
                    }
                },
            }
        )
    )
    (tmp_path / "initializer.json").write_text(
        json.dumps(
            {
                "initializer_receipt": str(receipt_path),
                "initializer_receipt_sha256": receipt_sha256,
            }
        )
    )
    return tmp_path


def test_exp3_reserved_validation_remains_disjoint_under_native_lookahead(prepared_sources):
    tmp_path = prepared_sources
    recipe = initializer_recipe(
        prepared=str(tmp_path), output=str(tmp_path / "output"), artifact_version="fixture", policy=StageGuard(1, 30, 2)
    )
    spec = exp3_sft_spec("fixture", "fixture", recipe)
    train = NamedLmDataset(
        spec.prepared_data.train_sets(Axis("position", 16), key=jrandom.PRNGKey(0))["source"], Axis("position", 16)
    )
    validation = spec.evaluation_data.validation_sets(Axis("position", 16))["source"]

    def document_ids(dataset):
        count = blocking_wait(dataset.async_len())
        rows = blocking_wait(dataset.get_batch(list(range(count))))
        return {int(token) for row in rows for token in np.asarray(row.tokens.array) if token not in (0, 65)}

    training_ids, validation_ids = document_ids(train), document_ids(validation)
    assert blocking_wait(validation.async_len()) == 8
    assert training_ids.isdisjoint(validation_ids)
    assert training_ids | validation_ids == set(range(1, 65))


def test_initializer_coordinator_validates_concrete_worker_config_before_dispatch(prepared_sources, monkeypatch):
    output = prepared_sources / "run"
    output.mkdir()
    submitted = []

    def complete_at_scheduler_boundary(request, policy, guard_output):
        submitted.append(request)
        return {"attempts": [], "h100_hours": 0}

    monkeypatch.setattr(exp3_initializer_smoke, "run_guarded_stage", complete_at_scheduler_boundary)
    run_initializer_smoke(
        output=str(output), prepared=str(prepared_sources), artifact_version="fixture", policy=StageGuard(1, 30, 2)
    )
    [request] = submitted
    assert request.entrypoint.callable_entrypoint is not None
    [pod] = request.entrypoint.callable_entrypoint.args
    config = pod.train_config
    assert isinstance(config, TrainLmConfig)
    assert config.hf_save_steps is None
    assert config.initialize_from_hf == str(prepared_sources / "regional")
    assert config.trainer.num_train_steps == 2
    assert isinstance(config.trainer.load_checkpoint_path, list)
    consumed = json.loads((output / "consumed-config.json").read_text())
    assert draccus.decode(TrainLmConfig, consumed["train_config"]) == config
    assert json.loads((StoragePath(str(output)) / "guard-result.json").read_text())["h100_hours"] == 0


@pytest.mark.multidevice
@pytest.mark.parametrize("context_parallelism", [2, 4])
def test_initializer_recipe_runs_forward_with_expert_and_context_parallelism(prepared_sources, context_parallelism):
    recipe = initializer_recipe(
        prepared=str(prepared_sources),
        output=str(prepared_sources / "output"),
        artifact_version="fixture",
        policy=StageGuard(1, 30, 2),
    )
    config = replace(
        draccus.decode(SnowballConfig, recipe["model"]),
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
    )
    # The full initializer previously loaded all weights before Sonic rejected EP8.
    with use_test_devices(8 * context_parallelism) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(1, 1, 8, context_parallelism, 1, 1),
            ("replica_dcn", "data", "expert", "context", "replica", "model"),
            axis_types=(AxisType.Explicit,) * 6,
        )
        with jax.set_mesh(mesh):
            model = SnowballLMHeadModel.init(Axis("vocab", 32), config, key=jrandom.PRNGKey(0))
            axes = (Axis("batch", 8), Axis("position", 16))
            tokens = hax.named(jnp.arange(128, dtype=jnp.int32).reshape(8, 16) % 32, axes)
            valid = hax.named(jnp.ones((8, 16), dtype=jnp.bool_), axes)
            hidden, _ = eqx.filter_jit(model.activations_with_stats)(tokens, valid)
            assert np.isfinite(np.asarray(hidden.array)).all()
