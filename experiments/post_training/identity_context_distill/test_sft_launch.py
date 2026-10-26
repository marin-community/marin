# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SFT job binds the frozen mixture, the measured recipe and the reviewed audits; it reads no real data here."""

import hashlib
import json
from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel, snowball_loss_with_dispatch_stats
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.post_training.identity_context_distill import lr0_probe, sft_launch
from experiments.post_training.identity_context_distill.test_lr0_probe import fixture
from experiments.post_training.identity_context_distill.test_runtime_recipe import inputs

RUN_ROOT = "s3://marin-rno2a/fixture/sft"
MIXTURE = "s3://marin-rno2a/fixture/mixture"
LR0_SOURCE = {"path": "s3://marin-rno2a/lr0/source.json", "sha256": "a" * 64}
SOURCE_OBJECTS = {"path": "s3://marin-rno2a/lr0/source-object-list.json", "sha256": "b" * 64, "objects": 45}


def test_checked_loss_matches_unmasked_loss_on_real_positions():
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
        moe_capacity_factor=4.0,
        router_combine_fp32=True,
        fp32_activations=True,
    )
    batch = hax.Axis("batch", 2 * jax.device_count())
    position = hax.Axis("position", 8)
    axes = (batch, position)
    tokens = hax.named(jnp.broadcast_to(jnp.array([1, 2, 3, 4, 5, 5, 5, 5]), (batch.size, 8)), axes)
    segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 0, 0, -1, -1, -1, -1]), tokens.array.shape), axes)
    example = hax.vmap(LmExample.causal, batch)(tokens, segment_ids=segments)
    example = eqx.tree_at(
        lambda value: value.loss_weight,
        example,
        hax.named(jnp.broadcast_to(jnp.array([0, 1, 1, 0, 0, 0, 0, 0], jnp.float32), tokens.array.shape), axes),
    )
    with (
        jax.set_mesh(compact_grug_mesh()),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(310))
        example = hax.auto_sharded(example)
        old = eqx.filter_jit(model.compute_next_token_loss)(example)
        checked, metrics = eqx.filter_jit(snowball_loss_with_dispatch_stats)(model, example)
        print(f"padded loss: old={float(old.array):.10g}, checked={float(checked.array):.10g}")
        np.testing.assert_array_equal(old.array, checked.array)
        np.testing.assert_array_equal(metrics["moe_dropped"], np.zeros(2))


def component(name: str) -> dict:
    chat = {
        "type": "chat",
        "chat_template": None,
        "chat_template_kwargs": "chat_template_kwargs",
        "mask_user_turns": True,
        "messages_field": "messages",
        "pack": False,
        "slice_strategy": "raise",
        "system_prompt": None,
    }
    return {
        "type": "cached",
        "cache_dir": f"{MIXTURE}/cache/{name}",
        "format": chat,
        "pack": False,
        "source": {"type": "url", "train_urls": [f"{MIXTURE}/inputs/{name}.jsonl"], "validation_urls": []},
    }


def data_config(**changes) -> bytes:
    """The frozen mixture's shape: two unpacked chat components with the ruled weights."""
    value = {
        "tokenizer": sft_launch.BASE_MODEL,
        "cache_dir": f"{MIXTURE}/cache",
        "auto_build_caches": False,
        "components": {"identity": component("identity"), "preservation": component("preservation")},
        "train_weights": {"identity": 264.0, "preservation": 547.0},
        "permutation_type": "feistel",
        "mixture_block_size": 2048,
        **changes,
    }
    return json.dumps(value).encode()


def frozen(monkeypatch, data_bytes: bytes) -> bytes:
    monkeypatch.setattr(sft_launch, "DATA_CONFIG_SHA256", hashlib.sha256(data_bytes).hexdigest())
    return data_bytes


def run_worker(monkeypatch, receipt_bytes, data_bytes):
    calls: dict = {"writer": 0}
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    monkeypatch.setattr(sft_launch, "RUNTIME_RECEIPT_SHA256S", {receipt_sha})

    def writer(action):
        calls["writer"] += 1
        return action()

    def run(config, **kwargs):
        calls["config"], calls["kwargs"] = config, kwargs
        return {"completed_updates": 16}

    monkeypatch.setattr(sft_launch, "StoragePath", lambda path: _Remote(str(path)))
    monkeypatch.setattr(sft_launch, "check_pod_storage", lambda: None)
    monkeypatch.setattr(sft_launch, "run_on_export_writer", writer)
    monkeypatch.setattr(
        sft_launch, "write_export_json", lambda path, value: calls.setdefault("result", (path.path, value))
    )
    monkeypatch.setattr(sft_launch, "run_identity_sft", run)
    audits = sft_launch.SftAudits(lambda it, config: (it, {}), lambda batch, update: {}, lambda state, path: None)
    sft_launch.sft_worker(
        RUN_ROOT, "t310-sft", receipt_bytes, receipt_sha, data_bytes, LR0_SOURCE, SOURCE_OBJECTS, lambda: audits
    )
    return calls, audits


_FILES: dict[str, bytes] = {}


class _Remote:
    def __init__(self, path):
        self.path = path

    def __truediv__(self, name):
        return _Remote(f"{self.path.rstrip('/')}/{name}")

    def read_bytes(self):
        return _FILES[self.path.rsplit("/", 1)[-1]]


def test_worker_binds_the_frozen_mixture_measured_recipe_and_audits(monkeypatch):
    receipt_bytes = fixture(
        monkeypatch,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        xla_precision="default",
    )
    _, args = inputs()
    _FILES.update(
        {
            "config.json": args["config_bytes"],
            "generation_config.json": args["generation_bytes"],
            "routing.json": args["rounding_bytes"],
        }
    )
    data_bytes = frozen(monkeypatch, data_config())
    calls, audits = run_worker(monkeypatch, receipt_bytes, data_bytes)
    config, kwargs = calls["config"], calls["kwargs"]
    assert config.data == sft_launch.frozen_data_config(data_bytes)
    assert config.data.train_weights == {"identity": 264.0, "preservation": 547.0}
    assert config.data_seed == config.trainer.seed == 310
    assert config.trainer.num_train_steps == 16 and config.trainer.load_checkpoint_path is None
    assert config.trainer.train_batch_size == 8 and config.trainer.per_device_parallelism == 1
    assert config.train_seq_len == 8192
    optimizer = config.optimizer
    assert (optimizer.learning_rate, optimizer.warmup, optimizer.weight_decay, optimizer.max_grad_norm) == (
        2e-6,
        2,
        0.0,
        0.5,
    )
    assert config.model.router_dot_fp32 is True
    assert config.model.moe_capacity_factor == 4.0
    assert config.model.router_combine_fp32 is True and config.model.fp32_activations is True
    assert config.model.attention_implementation == "torch_order"
    assert kwargs["receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert config.trainer.compute_dtype_overrides == {"router_bias": "float32"}
    assert config.adapter.frozen_field_names == ("router_bias",)
    assert isinstance(config.trainer.tracker, NoopConfig)
    assert config.hf_save_path == f"{RUN_ROOT}/hf"
    assert dict(config.trainer.mesh.dcn_axes) == {"replica_dcn": 1, "context": 4}
    assert kwargs["runtime_receipt"] == json.loads(receipt_bytes)
    assert kwargs["lr0_source"] == LR0_SOURCE and kwargs["source_objects"] == SOURCE_OBJECTS
    assert kwargs["run_root"] == RUN_ROOT and kwargs["load_audits"]() is audits
    # The result write is collective: the worker calls it on every rank, not behind a rank-0 guard.
    assert calls["writer"] == 1
    assert calls["result"] == (f"{RUN_ROOT}/sft-result.json", {"completed_updates": 16})


def test_frozen_data_config_refuses_other_bytes_and_packed_components(monkeypatch):
    frozen(monkeypatch, data_config())
    with pytest.raises(ValueError, match="frozen mixture"):
        sft_launch.frozen_data_config(data_config() + b" ")
    packed = component("identity")
    packed["format"] = {**packed["format"], "pack": True}
    with pytest.raises(ValueError, match="unpacked"):
        components = {"identity": packed, "preservation": component("preservation")}
        sft_launch.frozen_data_config(frozen(monkeypatch, data_config(components=components)))


class TinyModel(eqx.Module):
    weight: jax.Array


def test_each_update_checks_its_consumed_batch_before_the_update(monkeypatch):
    config = TrainerConfig(
        id="sft-test",
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
        train_batch_size=1,
        per_device_parallelism=1,
        load_checkpoint=False,
    )
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0, weight_decay=0.0).build(3)
    axes = (hax.Axis("batch", 1), hax.Axis("position", 2))

    def batch(value):
        return LmExample(hax.named(jnp.full((1, 2), value, dtype=jnp.int32), axes), hax.ones(axes))

    def loss(model, example, *, key=None):
        del key
        metrics = {name: jnp.zeros(26, jnp.int32) for name in sft_launch.DISPATCH_METRICS}
        return jnp.square(model.weight - jnp.mean(example.tokens.array)), metrics

    events = []

    def check(example, update):
        value = int(example.tokens.array[0, 0])
        events.append(f"check-{update}-{value}")
        if value == 9:
            raise ValueError(f"update {update}: sequence ids differ from the frozen order")
        return {"ids": [value]}

    def fresh(trainer):
        return trainer.initial_state(jax.random.PRNGKey(0), model=TinyModel(jnp.asarray(0.5, dtype=jnp.float32)))

    with Trainer(config, optimizer, loss, add_default_hooks=False) as trainer:
        train_step = trainer.train_step

        def counted_step(state, example):
            events.append(f"step-{int(example.tokens.array[0, 0])}")
            return train_step(state, example)

        monkeypatch.setattr(trainer, "train_step", counted_step)
        batches = iter([batch(1), batch(2), batch(3), batch(4)])
        final, updates = sft_launch.train_checked_updates(trainer, fresh(trainer), batches, check, 3)
        # Each batch is checked immediately before its own update; the fourth batch is not consumed.
        assert events == ["check-0-1", "step-1", "check-1-2", "step-2", "check-2-3", "step-3"]
        assert [u["update"] for u in updates] == [0, 1, 2] and [u["ids"] for u in updates] == [[1], [2], [3]]
        assert all(np.isfinite(u["loss"]) for u in updates) and int(final.step) == 3
        assert next(batches).tokens.array[0, 0] == 4

        events.clear()
        with pytest.raises(ValueError, match="update 1: sequence ids differ"):
            sft_launch.train_checked_updates(trainer, fresh(trainer), iter([batch(1), batch(9), batch(3)]), check, 3)
        # The mismatched batch never reached train_step.
        assert events == ["check-0-1", "step-1", "check-1-9"]


@pytest.mark.parametrize("missing", [False, True])
def test_update_refuses_dropped_or_unmeasured_dispatch_before_next_step(missing):
    config = TrainerConfig(
        id="dispatch-test",
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
        train_batch_size=1,
        per_device_parallelism=1,
        load_checkpoint=False,
    )
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0, weight_decay=0.0).build(2)
    axes = (hax.Axis("batch", 1), hax.Axis("position", 2))
    example = LmExample(hax.named(jnp.ones((1, 2), jnp.int32), axes), hax.ones(axes))

    def loss(model, example, *, key=None):
        metrics = (
            {}
            if missing
            else {
                "moe_dropped": jnp.ones(26, jnp.int32),
                "moe_sender_dropped": jnp.zeros(26, jnp.int32),
                "moe_receiver_dropped": jnp.ones(26, jnp.int32),
            }
        )
        return jnp.square(model.weight), metrics

    consumed = []

    def check(batch, update):
        consumed.append(update)
        return {}

    with Trainer(config, optimizer, loss, add_default_hooks=False) as trainer:
        state = trainer.initial_state(jax.random.PRNGKey(0), model=TinyModel(jnp.asarray(0.5)))
        with pytest.raises(ValueError, match="did not report" if missing else "dropped valid"):
            sft_launch.train_checked_updates(trainer, state, iter([example, example]), check, 2)
    assert consumed == [0]


def remote_store(monkeypatch, files: dict[str, dict]):
    store = {path: json.dumps(value).encode() for path, value in files.items()}
    monkeypatch.setattr(sft_launch, "read_remote", lambda path: store[path])
    return store


def passing_lr0(root: str, runtime: dict) -> dict[str, dict]:
    files = {
        f"{root}/precision-gate/gate.json": {"failures": []},
        f"{root}/lr0-result.json": {"completed_updates": 1, "runtime_recipe_receipt": runtime},
        f"{root}/audits/source-tensor-hashes.json": {"tensors": {f"t{i}": {} for i in range(502)}},
    }
    files.update({f"{root}/precision-gate/ranks/{r}.json": {"stage": "updated", "process": r} for r in range(4)})
    return files


def test_lr0_pass_needs_every_reading_rule_receipt(monkeypatch):
    root = "s3://marin-rno2a/fixture/lr0-r4"
    runtime = json.loads(fixture(monkeypatch, router_combine_fp32=True, fp32_activations=True))
    store = remote_store(monkeypatch, passing_lr0(root, runtime))
    record = sft_launch.lr0_pass(root, runtime)
    assert record["source_receipt"]["path"] == f"{root}/audits/source-tensor-hashes.json"
    gate_bytes = store[f"{root}/precision-gate/gate.json"]
    assert record["gate"]["sha256"] == hashlib.sha256(gate_bytes).hexdigest()

    for path, value, message in [
        (f"{root}/precision-gate/gate.json", {"failures": ["parity: x"]}, "gate did not pass"),
        (f"{root}/precision-gate/gate.json", {}, "gate did not pass"),
        (f"{root}/precision-gate/ranks/2.json", {"stage": "scored", "failures": None}, "rank 2"),
        (f"{root}/lr0-result.json", {"completed_updates": 0}, "one completed update"),
        (f"{root}/lr0-result.json", {"completed_updates": 1, "runtime_recipe_receipt": {}}, "different runtime"),
    ]:
        files = passing_lr0(root, runtime)
        files[path] = value
        remote_store(monkeypatch, files)
        with pytest.raises(ValueError, match=message):
            sft_launch.lr0_pass(root, runtime)
    files = passing_lr0(root, runtime)
    del files[f"{root}/lr0-result.json"]
    remote_store(monkeypatch, files)
    with pytest.raises(KeyError):
        sft_launch.lr0_pass(root, runtime)


def test_source_bind_matches_the_model_objects_and_the_lr0_receipt(monkeypatch):
    class Record:
        def __init__(self, shape, shard):
            self.shape, self.file_path = shape, f"{sft_launch.BASE_MODEL}{shard}"

    root = "bucket/model"
    model = {name: {"bytes": 10 + i, "etag": f'"e{i}"'} for i, name in enumerate(sft_launch.SOURCE_FILES)}
    model |= {"model-1.safetensors": {"bytes": 99, "etag": '"s1"'}, "model-2.safetensors": {"bytes": 98, "etag": '"s2"'}}
    remote = {**model, "README.md": {"bytes": 1, "etag": '"r"'}}

    class FS:
        def info(self, path):
            entry = remote[path.removeprefix(f"{root}/")]
            return {"size": entry["bytes"], "ETag": entry["etag"]}

    checked = []
    monkeypatch.setattr(sft_launch, "SOURCE_SHARDS", 2)
    monkeypatch.setattr(sft_launch, "url_to_fs", lambda url: (FS(), root + "/"))
    monkeypatch.setattr(sft_launch, "check_source_metadata", lambda root, hashes: checked.append((root, hashes)))
    monkeypatch.setattr(
        sft_launch,
        "checkpoint_records",
        lambda root: {"a": Record((2, 3), "model-1.safetensors"), "b": Record((4,), "model-2.safetensors")},
    )
    index_sha = sft_launch.SOURCE_FILE_HASHES[sft_launch.INDEX]
    listed = {
        "source_index_sha256": index_sha,
        "objects": {name: {**o, "last_modified": "t"} for name, o in model.items()},
    }
    receipt = {"tensors": {"a": {"shape": [2, 3]}, "b": {"shape": [4]}}}
    # Files outside the model (README) are neither listed nor compared.
    assert sft_launch.source_bind(receipt, listed) == {
        "metadata_sha256": sft_launch.SOURCE_FILE_HASHES,
        "objects": 8,
        "tensor_count": 2,
    }
    assert checked == [(sft_launch.BASE_MODEL, sft_launch.SOURCE_FILE_HASHES)]
    with pytest.raises(ValueError, match="names or shapes"):
        sft_launch.source_bind({"tensors": {"a": {"shape": [2, 3]}, "b": {"shape": [5]}}}, listed)
    remote["model-2.safetensors"] = {"bytes": 98, "etag": '"other"'}
    with pytest.raises(ValueError, match=r"Source objects differ.*model-2.safetensors"):
        sft_launch.source_bind(receipt, listed)
    missing = {**listed, "objects": {k: v for k, v in listed["objects"].items() if k != "model-1.safetensors"}}
    with pytest.raises(ValueError, match="exactly the metadata files and index shards"):
        sft_launch.source_bind(receipt, missing)
    with pytest.raises(ValueError, match="different index"):
        sft_launch.source_bind(receipt, {**listed, "source_index_sha256": "0" * 64})
    monkeypatch.setattr(sft_launch, "SOURCE_SHARDS", 39)
    with pytest.raises(ValueError, match="names 2 shards, not 39"):
        sft_launch.source_bind(receipt, listed)


def test_entrypoint_binds_the_frozen_dataset_and_export_audits(monkeypatch):
    calls = []
    monkeypatch.setattr(sft_launch, "sft_runtime_receipt", lambda *args: {"xla_precision": "default"})
    monkeypatch.setattr(sft_launch, "require_xla_precision", lambda expected: None)
    monkeypatch.setattr(sft_launch, "check_pod_storage", lambda: calls.append("storage"))
    monkeypatch.setattr(
        sft_launch, "load_dataset_audit_inputs", lambda root, inventory: calls.append(("frozen", root, inventory)) or "F"
    )
    monkeypatch.setattr(
        sft_launch, "audit_training_iterator", lambda batches, **kwargs: calls.append(("iterator", batches, kwargs))
    )
    monkeypatch.setattr(
        sft_launch, "checked_update", lambda batch, update, frozen: calls.append(("update", update, frozen))
    )
    monkeypatch.setattr(
        sft_launch, "audit_trained_export", lambda state, path, **kwargs: calls.append(("export", kwargs))
    )
    monkeypatch.setattr(sft_launch, "sft_worker", lambda *args: calls.append(("worker", args)))
    sft_launch.bound_sft_worker(RUN_ROOT, "job", b"receipt", "sha", b"data", LR0_SOURCE, SOURCE_OBJECTS)
    assert calls[0] == "storage"
    worker_args = calls[1][1]
    assert worker_args[:7] == (RUN_ROOT, "job", b"receipt", "sha", b"data", LR0_SOURCE, SOURCE_OBJECTS)
    audits = worker_args[7]()
    assert calls[2] == ("frozen", sft_launch.MIXTURE_ROOT, sft_launch.INVENTORY_PATH)
    audits.audit_iterator("B", "C")
    audits.check_batch("batch", 3)
    audits.save_audit("state", "export")
    assert calls[3:] == [
        (
            "iterator",
            "B",
            {
                "config": "C",
                "completed_updates": 0,
                "frozen": "F",
                "receipt_path": f"{RUN_ROOT}/audits/dataset-step-0.json",
            },
        ),
        ("update", 3, "F"),
        (
            "export",
            {
                "source_root": sft_launch.BASE_MODEL,
                "source_receipt_path": LR0_SOURCE["path"],
                "source_receipt_sha256": LR0_SOURCE["sha256"],
                "expected_hashes": sft_launch.SOURCE_FILE_HASHES,
            },
        ),
    ]


def test_plan_records_the_recipe_and_a_bounded_ceiling(monkeypatch):
    receipt_bytes = fixture(
        monkeypatch,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        xla_precision="default",
    )
    runtime = json.loads(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    _, args = inputs()
    data_bytes = frozen(monkeypatch, data_config())
    config = sft_launch.sft_config(RUN_ROOT, "job", args["config_bytes"], args["generation_bytes"], data_bytes, runtime)
    job_plan = sft_launch.plan("job", RUN_ROOT, 7200, "interactive", config, receipt_sha, runtime["xla_precision"])
    assert job_plan["runtime_receipt_sha256"] == receipt_sha
    assert job_plan["xla_precision"] == "default"
    assert job_plan["env_vars"] == {}
    assert job_plan["forward"]["attention_implementation"] == "torch_order"
    assert job_plan["stop_after_updates"] == 16
    assert job_plan["optimizer"] == {"learning_rate": 2e-6, "warmup": 2, "weight_decay": 0.0, "max_grad_norm": 0.5}
    assert job_plan["batch"] == {"train_batch_size": 8, "per_device_parallelism": 1, "seq_len": 8192}
    assert job_plan["mesh"]["dcn_axes"] == {"replica_dcn": 1, "context": 4}
    assert (job_plan["replicas"], job_plan["gpus"], job_plan["region"]) == (4, 32, "cw-rno2a")
    assert job_plan["retries_failure"] == job_plan["retries_preemption"] == 0
    assert job_plan["h100_hour_ceiling"] == 64


@pytest.mark.parametrize("entrypoint", ["sft_worker", "bound_sft_worker"])
def test_worker_rejects_precision_mismatch_before_storage_or_dataset_reads(monkeypatch, entrypoint):
    receipt_bytes = fixture(
        monkeypatch,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        xla_precision="default",
    )
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    monkeypatch.setattr(sft_launch, "RUNTIME_RECEIPT_SHA256S", {digest})
    monkeypatch.setenv("XLA_FLAGS", "--xla_allow_excess_precision=false")
    monkeypatch.setattr(
        "experiments.post_training.identity_context_distill.runtime_recipe.xla_bridge.backends_are_initialized",
        lambda: False,
    )
    reads = []
    monkeypatch.setattr(sft_launch, "check_pod_storage", lambda: reads.append("storage"))
    monkeypatch.setattr(sft_launch, "load_dataset_audit_inputs", lambda *args: reads.append("dataset"))
    arguments = [RUN_ROOT, "job", receipt_bytes, digest, b"data", LR0_SOURCE, SOURCE_OBJECTS]
    if entrypoint == "sft_worker":
        arguments.append(None)
    with pytest.raises(ValueError, match="XLA precision"):
        getattr(sft_launch, entrypoint)(*arguments)
    assert reads == []


def test_sft_refuses_unreviewed_or_changed_runtime_receipts(monkeypatch):
    receipt_bytes = fixture(
        monkeypatch, router_combine_fp32=True, fp32_activations=True, attention_implementation="torch_order"
    )
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    with pytest.raises(ValueError, match="reviewed schema-6"):
        sft_launch.sft_runtime_receipt(receipt_bytes, digest)
    monkeypatch.setattr(sft_launch, "RUNTIME_RECEIPT_SHA256S", {digest})
    with pytest.raises(ValueError, match="differs from"):
        sft_launch.sft_runtime_receipt(receipt_bytes + b" ", digest)
    assert sft_launch.sft_runtime_receipt(receipt_bytes, digest)["schema_version"] == 6


def test_submit_requires_the_precision_setting_before_any_job_access():
    missing_precision = partial(sft_launch.submit_entrypoint, "job", object(), 7200, "production")
    with pytest.raises(TypeError, match="xla_precision"):
        missing_precision()


@pytest.mark.parametrize(
    "receipt_sha",
    [
        "3f9cb13590b7c861af2b4eddb352ac8e0037eab775551aab140c2d29729e5fc2",
        "38e993dfca09124d2a94e0958e742faffc5fccc7ae1a0fcc2c6bef73834746dd",
    ],
)
def test_sft_refuses_other_reviewed_arm_receipts(receipt_sha):
    with pytest.raises(ValueError, match="reviewed schema-6"):
        sft_launch.sft_runtime_receipt(b"not read for an excluded receipt", receipt_sha)


@pytest.mark.parametrize("attention", ["torch_order", "gpu_fa4_cute"])
def test_sft_refuses_other_arm_settings_even_with_an_allowed_digest(monkeypatch, attention):
    receipt_bytes = fixture(
        monkeypatch,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation=attention,
        xla_precision="no_excess_precision",
    )
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    monkeypatch.setattr(sft_launch, "RUNTIME_RECEIPT_SHA256S", {digest})
    with pytest.raises(ValueError, match=r"must match arm \(a\)"):
        sft_launch.sft_runtime_receipt(receipt_bytes, digest)


def test_lr0_submit_keeps_route_policy_and_precision_through_shared_helper(monkeypatch):
    calls = {}
    entrypoint = object()

    def build(*args):
        calls["entrypoint"] = args
        return entrypoint

    def submit(*args):
        calls["submit"] = args
        return "job-id"

    monkeypatch.setattr(lr0_probe.Entrypoint, "from_callable", build)
    monkeypatch.setattr(lr0_probe, "submit_entrypoint", submit)
    result = lr0_probe.submit(
        "job", RUN_ROOT, "run", b"receipt", "digest", "report_g1b_l0", 7200, "production", "default"
    )
    assert result == "job-id"
    assert calls["entrypoint"] == (lr0_probe.lr0_worker, RUN_ROOT, "run", b"receipt", "digest", "report_g1b_l0")
    assert calls["submit"] == ("job", entrypoint, 7200, "production", "default")
