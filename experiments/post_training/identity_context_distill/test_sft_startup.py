# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the bound SFT startup through its first training boundary.

Seams: local synthetic HF/data, r3-result and authorization receipts and their hash pins; a tiny 26-layer
Snowball checkpoint; an eight-row ListAsyncDataset in place of cached chats;
the factory's CPU/data options; CPU slice metadata standing in for four hosts;
an external Iris endpoint-registry/client fake; a local alias for routing evidence; and
Trainer.train_step replaced only to stop before its computation. The worker functions, precision checks,
backend-state queries, frozen-input loader, audits, HF loader, distributed initialization and gathers are not
patched on the valid path. Iris task environment variables drive Levanter's normal initializer. The sensitivity
arm deliberately inserts a real gather before that initializer. Four processes expose eight CPU devices each;
no optimizer update runs.
"""

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from iris.client.client import IrisContext, iris_ctx_scope
from jax._src import xla_bridge
from jax.sharding import Mesh
from levanter import trainer as trainer_module
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.trainer import Trainer, TrainerConfig
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.interp.snowball_identity.multihost import gather_host_scalar
from experiments.post_training.identity_context_distill import dataset_audit, runtime_recipe, sft, sft_launch
from experiments.post_training.identity_context_distill.precision_source_audit import INDEX, SOURCE_FILES


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")
    return digest(path.read_bytes())


def example_arrays():
    tokens = np.zeros((8, 8192), dtype=np.int32)
    tokens[:, 0] = np.arange(8)
    weights = np.zeros_like(tokens)
    weights[:, 2] = 1
    segments = np.full_like(tokens, -1)
    segments[:, :4] = 0
    return tokens, weights, segments


@dataclass(frozen=True)
class StartupDataConfig(LmDataConfig):
    def train_set(self, Pos, batch_schedule, *, key):
        tokens, weights, segments = example_arrays()
        examples = [
            LmExample(
                hax.named(tokens[index], Pos),
                hax.named(weights[index].astype(np.float32), Pos),
                AttentionMask.causal().with_segment_ids(hax.named(segments[index], Pos)),
            )
            for index in range(8)
        ]
        return ListAsyncDataset(examples * 64)


def write_fixture(root):
    source, freeze = root / "source", root / "freeze"
    source.mkdir()
    freeze.mkdir()
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=26,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8192,
        sliding_window=4,
        qk_mult=1.75,
    )
    with jax.set_mesh(compact_grug_mesh()):
        shapes = jax.eval_shape(
            lambda: SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.key(310)).to_state_dict()
        )
    tensors = {
        name: np.full(shape.shape, 0.125, dtype=np.float32 if name.endswith(".mlp.router.bias") else jnp.bfloat16)
        for name, shape in shapes.items()
    }
    assert len(tensors) == 502
    save_file(tensors, str(source / "weights.safetensors"))
    write_json(
        source / INDEX,
        {
            "weight_map": {name: "weights.safetensors" for name in tensors},
            "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        },
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{index}": index for index in range(32)}, unk_token="t0")),
        unk_token="t0",
        bos_token="t1",
        eos_token="t2",
        pad_token="t2",
    )
    tokenizer.save_pretrained(source)
    template = json.loads((Path(__file__).parent / "testdata" / "tokenizer-export-r7.json").read_bytes())
    (source / "chat_template.jinja").write_bytes(base64.b64decode(template["source-chat_template.jinja"]))
    hf = json.loads(config.to_hf_config(32).to_json_string())
    hf.update(bos_token_id=128000, eos_token_id=128001, pad_token_id=128001)
    write_json(source / "config.json", hf)
    write_json(source / "generation_config.json", {"bos_token_id": 128000, "eos_token_id": [128001, 128009]})
    hashes = {name: digest((source / name).read_bytes()) for name in SOURCE_FILES}
    source_receipt = {
        "tensors": {
            name: {"shape": list(value.shape), "master_sha256": digest(value.astype(np.float32).tobytes())}
            for name, value in tensors.items()
        }
    }
    source_sha = write_json(root / "source-receipt.json", source_receipt)
    objects = {
        "source_index_sha256": hashes[INDEX],
        "objects": {
            name: {"bytes": (source / name).stat().st_size, "etag": None}
            for name in (*SOURCE_FILES, "weights.safetensors")
        },
    }
    objects_sha = write_json(root / "objects.json", objects)
    components = {}
    for name in ("identity", "preservation"):
        components[name] = {
            "source": {"type": "url", "train_urls": [str(root / "unused.jsonl")]},
            "cache_dir": str(root / name),
            "pack": False,
            "format": {"type": "chat", "pack": False, "slice_strategy": "raise", "mask_user_turns": True},
        }
    data = {
        "tokenizer": str(source),
        "cache_dir": str(root / "cache"),
        "components": components,
        "auto_build_caches": False,
        "train_weights": {"identity": 264.0, "preservation": 547.0},
    }
    data = draccus.encode(draccus.decode(LmDataConfig, data))
    data_sha = write_json(freeze / "data-config.json", data)
    tokens, weights, _ = example_arrays()
    native, inventory = {}, []
    for index in range(767):
        identifier = f"row-{index}"
        token_sha = digest(tokens[index].tobytes()) if index < 8 else digest(identifier.encode())
        mask_sha = digest(weights[0].tobytes())
        native[identifier] = {"tokens_sha256": token_sha, "target_mask_sha256": mask_sha, "class_tokens": {"replay": 1}}
        inventory.append(
            {
                "id": identifier,
                "accounting": {
                    "identity": {
                        "occurrence_id": identifier,
                        "sample_id": identifier,
                        "prompt_id": identifier,
                        "final_label": "replay",
                        "history_label": "replay",
                    },
                    "rendered_tokens": 8192,
                    "template_sha256": digest((source / "chat_template.jinja").read_bytes()),
                    "input_ids_sha256": token_sha,
                    "target_mask_sha256": mask_sha,
                    "turns": [{"message_index": 1, "label": "replay", "supervised_tokens": 1}],
                },
            }
        )
    inventory_path = root / "inventory.jsonl"
    inventory_path.write_text("\n".join(json.dumps(row) for row in inventory) + "\n")
    inventory_sha = digest(inventory_path.read_bytes())
    manifest = {"source_files": {"inventory": inventory_sha}}
    manifest_sha = write_json(freeze / "original-local-manifest.json", manifest)
    ordered = [{"id": f"row-{index % 8}", "supervised_tokens": 1} for index in range(512)]
    order_path = freeze / "occurrences-512.jsonl"
    order_path.write_text("\n".join(json.dumps(row) for row in ordered) + "\n")
    order_sha = digest(order_path.read_bytes())
    write_json(freeze / "native-row-fingerprints.json", native)

    def summary(size):
        return {
            "occurrences": size,
            "unique_ids": 8,
            "family_rows": {"replay": size},
            "family_tokens": {"replay": size},
            "class_tokens": {"replay": size},
            "supervised_tokens": size,
            "identity_tokens": 0,
            "identity_fraction": {"numerator": 0, "denominator": size, "value": 0.0},
        }

    frozen = {
        "files": {
            name: {"sha256": digest((freeze / name).read_bytes())}
            for name in (
                "data-config.json",
                "original-local-manifest.json",
                "occurrences-512.jsonl",
                "native-row-fingerprints.json",
            )
        },
        "original_ordered512_sha256": order_sha,
        "original_local_manifest_sha256": manifest_sha,
        "updates": [summary(8)] * 64,
        "realized_prefixes": {
            str(size): (
                summary(size)
                | {"mean_update_identity_ce_coefficient_mass": {"numerator": 0, "denominator": 1, "value": 0.0}}
            )
            for size in (128, 512)
        },
    }
    freeze_sha = write_json(freeze / "freeze.json", frozen)
    reference_updates = [
        {
            **summary(8),
            "update": update,
            "ids": [f"row-{index}" for index in range(8)],
            "coefficient_mass_not_gradient_share": True,
        }
        for update in range(64)
    ]
    write_json(
        root / "reference.json",
        {
            "completed_updates": 16,
            "dataset_audit": {"updates": reference_updates},
            "updates": [{**entry, "loss": 0.25} for entry in reference_updates[:16]],
        },
    )
    rounding = {
        "positions_sha256": runtime_recipe.POSITIONS_SHA256,
        "sequences": 465,
        "top_k": 4,
        "totals": {
            name: {"positions": 805974, "positions_changed": int(name != "bf16_bias")}
            for name in ("bf16_bias", "bf16_logits", "bf16_logits_bias")
        },
    }
    rounding_sha = write_json(root / "routing.json", rounding)
    write_json(
        root / "pins.json",
        {
            "source_hashes": hashes,
            "source_receipt_sha": source_sha,
            "objects_sha": objects_sha,
            "data_sha": data_sha,
            "freeze_sha": freeze_sha,
            "inventory_sha": inventory_sha,
            "order_sha": order_sha,
            "manifest_sha": manifest_sha,
            "rounding_sha": rounding_sha,
        },
    )


def cpu_mesh(*, ici_axes, dcn_axes, axis_types=None, **kwargs):
    names = list(ici_axes) + [name for name in dcn_axes if name not in ici_axes]
    dcn = [dcn_axes.get(name, 1) for name in names]
    ici = [ici_axes.get(name, 1) for name in names]
    devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    grid = np.array(devices, dtype=object).reshape(*dcn, *ici)
    grid = grid.transpose(
        [index for pair in zip(range(len(names)), range(len(names), 2 * len(names)), strict=True) for index in pair]
    )
    return Mesh(
        grid.reshape([outer * inner for outer, inner in zip(dcn, ici, strict=True)]), tuple(names), axis_types=axis_types
    )


def child(root, rank, port, mode):
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    assert not jax.distributed.is_initialized() and not xla_bridge.backends_are_initialized()
    pins = json.loads((root / "pins.json").read_bytes())
    source = str(root / "source")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sft, "BASE_MODEL", source)
        patch.setattr(sft_launch, "BASE_MODEL", source)
        patch.setattr(sft_launch, "SOURCE_FILE_HASHES", pins["source_hashes"])
        patch.setattr(sft_launch, "SOURCE_SHARDS", 1)
        patch.setattr(sft_launch, "DATA_CONFIG_SHA256", pins["data_sha"])
        patch.setattr(sft_launch, "MIXTURE_ROOT", str(root / "freeze"))
        patch.setattr(sft_launch, "INVENTORY_PATH", str(root / "inventory.jsonl"))
        for name, key in (
            ("FREEZE_SHA256", "freeze_sha"),
            ("INVENTORY_SHA256", "inventory_sha"),
            ("ORDER_SHA256", "order_sha"),
            ("LOCAL_MANIFEST_SHA256", "manifest_sha"),
        ):
            patch.setattr(dataset_audit, name, pins[key])
        patch.setattr(TrainerConfig, "num_slices", property(lambda config: jax.process_count()))
        patch.setattr(trainer_module, "create_mesh_from_axis_specs", cpu_mesh)
        factory = sft_launch.sft_config

        def config_factory(*args):
            config = factory(*args)
            data = StartupDataConfig(**{field.name: getattr(config.data, field.name) for field in fields(LmDataConfig)})
            assert config.trainer.distributed.initialize_jax_distributed is True
            trainer = replace(config.trainer, require_accelerator=False)
            return replace(config, data=data, trainer=trainer)

        patch.setattr(sft_launch, "sft_config", config_factory)
        if mode == "early_gather":
            initialize = sft_launch.initialize

            def gather_before_initialize(config):
                gather_host_scalar(np.int32(0))
                initialize(config)

            patch.setattr(sft_launch, "initialize", gather_before_initialize)
        rounding_path = "s3://marin-rno2a/fixture/routing.json"
        storage_path = sft_launch.StoragePath
        patch.setattr(
            sft_launch,
            "StoragePath",
            lambda path: storage_path(str(root / "routing.json") if path == rounding_path else path),
        )
        config_bytes = (root / "source/config.json").read_bytes()
        generation_bytes = (root / "source/generation_config.json").read_bytes()
        data_bytes = (root / "freeze/data-config.json").read_bytes()
        base = factory(
            str(root / "run"),
            "startup",
            config_bytes,
            generation_bytes,
            data_bytes,
            {"runtime_model_config": {"attention_implementation": "torch_order"}},
            sft_launch.SftRun(16, None, None, None),
        )
        _, receipt = runtime_recipe.configure_precision_runtime(
            base,
            config_bytes=config_bytes,
            generation_bytes=generation_bytes,
            rounding_bytes=(root / "routing.json").read_bytes(),
            rounding_path=rounding_path,
            rounding_sha256=pins["rounding_sha"],
            xla_precision="default",
        )
        receipt_bytes = json.dumps(receipt).encode()
        patch.setattr(sft_launch, "RUNTIME_RECEIPT_SHA256S", {digest(receipt_bytes)})
        horizon = 64 if mode == "normal64" else 16
        run = sft_launch.SftRun(16, None, None, None)
        if horizon == 64:
            reference = json.loads((root / "reference.json").read_bytes())
            reference["runtime_receipt_sha256"] = digest(receipt_bytes)
            reference_bytes = json.dumps(reference).encode()
            decision = b"fixture owner decision"
            supplement = b"fixture owner supplement 2"
            patch.setattr(sft_launch, "R3_RESULT_SHA256", digest(reference_bytes))
            patch.setattr(sft_launch, "FRESH64_DECISION_SHA256", digest(decision))
            patch.setattr(sft_launch, "FRESH64_SUPPLEMENT_SHA256", digest(supplement))
            run = sft_launch.SftRun(64, reference_bytes, decision, supplement)

        class BeforeTrainStep(Exception):
            pass

        def before_step(trainer, state, batch):
            assert int(state.step) == 0 and xla_bridge.backends_are_initialized()
            assert jax.distributed.is_initialized() and jax.process_count() == 4
            assert jax.process_index() == rank
            assert trainer.config.num_train_steps == horizon
            assert batch.tokens.array.shape == (8, 8192)
            runtime_recipe.check_xla_precision("default")
            os.environ["XLA_FLAGS"] = "--xla_allow_excess_precision=false"
            with pytest.raises(ValueError, match="XLA precision"):
                runtime_recipe.check_xla_precision("default")
            os.environ.pop("XLA_FLAGS")
            raise BeforeTrainStep

        patch.setattr(Trainer, "train_step", before_step)

        def register(name, address):
            assert name == "jax_coordinator-attempt-0" and address == f"127.0.0.1:{port}"
            (root / "coordinator.json").write_text(json.dumps({"name": name, "address": address}))
            return "fixture-endpoint"

        context = cast(
            IrisContext,
            SimpleNamespace(
                registry=SimpleNamespace(register=register, unregister=lambda identifier: None),
                resolver=SimpleNamespace(
                    resolve=lambda name: SimpleNamespace(
                        is_empty=False, first=lambda: SimpleNamespace(url=f"127.0.0.1:{port}")
                    )
                ),
                client=SimpleNamespace(complete_job=lambda job: None),
            ),
        )
        try:
            with iris_ctx_scope(context):
                sft_launch.bound_sft_worker(
                    str(root / "run"),
                    "startup",
                    receipt_bytes,
                    digest(receipt_bytes),
                    data_bytes,
                    {"path": str(root / "source-receipt.json"), "sha256": pins["source_receipt_sha"]},
                    {"path": str(root / "objects.json"), "sha256": pins["objects_sha"]},
                    run,
                )
        except BeforeTrainStep:
            report = json.loads((root / "run/audits/dataset-step-0.json").read_bytes())
            assert len(report["updates"]) == report["same_iterator_batches_buffered"] == 64
            masters = json.loads((root / "run/audits/master-step-0.json").read_bytes())
            assert masters["tensor_count"] == 502
            print(
                json.dumps(
                    {
                        "process": rank,
                        "before_train_step": True,
                        "buffered_updates": 64,
                        "horizon": horizon,
                    }
                ),
                flush=True,
            )
        else:
            raise AssertionError("Startup did not reach the first train_step boundary")
    jax.distributed.shutdown()


def run_children(root, *, workdir, mode):
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = str(sock.getsockname()[1])
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "8",
        "XLA_FLAGS": "",
        "AWS_ENDPOINT_URL": "http://localhost",
        "FSSPEC_S3": "{}",
        "AWS_ACCESS_KEY_ID": "fixture",
        "AWS_SECRET_ACCESS_KEY": "fixture",
        "CUDA_VISIBLE_DEVICES": "",
        "IRIS_NUM_TASKS": "4",
        "IRIS_ADVERTISE_HOST": "127.0.0.1",
        "IRIS_PORT_JAX": port,
        "PYTHONPATH": os.pathsep.join([str(workdir), *(str(path) for path in workdir.glob("lib/*/src"))]),
    }
    processes = [
        subprocess.Popen(
            [sys.executable, __file__, "--child", str(root), str(rank), port, mode],
            cwd=workdir,
            env={**env, "IRIS_TASK_ID": f"/test/sft-startup/{rank}"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for rank in range(4)
    ]
    try:
        outputs = [process.communicate(timeout=600) for process in processes]
        return [
            (process.returncode, out.decode(), error.decode())
            for process, (out, error) in zip(processes, outputs, strict=True)
        ]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mode", ["normal", "normal64"])
def test_bound_sft_startup_reaches_training_after_the_real_frozen_input_gather(tmp_path, mode):
    write_fixture(tmp_path)
    results = run_children(tmp_path, workdir=Path.cwd(), mode=mode)
    ranks = set()
    for code, output, error in results:
        assert code == 0, error[-6000:]
        report = json.loads(output.strip().splitlines()[-1])
        assert report["before_train_step"] is True and report["buffered_updates"] == 64
        assert report["horizon"] == (64 if mode == "normal64" else 16)
        ranks.add(report["process"])
    assert ranks == set(range(4))
    assert json.loads((tmp_path / "coordinator.json").read_bytes())["name"] == "jax_coordinator-attempt-0"
    assert not (tmp_path / "run/sft-result.json").exists()


@pytest.mark.timeout(900)
def test_sft_startup_rejects_a_collective_before_code_owned_distributed_init(tmp_path):
    write_fixture(tmp_path)
    results = run_children(tmp_path, workdir=Path.cwd(), mode="early_gather")
    for code, _, error in results:
        assert code != 0 and "jax.distributed.initialize() must be called before any JAX calls" in error
    assert not (tmp_path / "run/sft-result.json").exists()


if __name__ == "__main__":
    child(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
