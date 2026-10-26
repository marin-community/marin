# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LR0 startup/order only, through the first train_step boundary on four CPU processes.

Seams: a local synthetic 502-tensor checkpoint and tokenizer; synthetic source,
routing and gate inputs; local aliases for regional storage paths; the CPU slice
metadata property TrainerConfig.num_slices patched to jax.process_count(); a CPU
host-slice mesh adapter and accelerator requirement; external Iris registry/resolver/completion
fakes; synthetic device-memory observations; and a fixture-only evaluate policy that
checks generated positions, zero drops, FP32 router dtypes and gathered memory.
The real gate forward, input readers, audits, HF loader, precision guard, distributed
initializer and collectives run unchanged. This is never a numerical parity verdict.
Trainer.train_step is a stop sentinel. Sensitivity arms deliberately insert a real
gather before initialization or repeat the real entry guard after the input gather.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from iris.client.client import IrisContext, iris_ctx_scope
from jax._src import xla_bridge
from jax.sharding import Mesh
from levanter import trainer as trainer_module
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.trainer import Trainer, TrainerConfig
from rigging.filesystem.storage_path import StoragePath
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from experiments.interp.snowball_identity import precision_gate
from experiments.interp.snowball_identity.multihost import gather_host_scalar
from experiments.post_training.identity_context_distill import (
    lr0_probe,
    precision_audit,
    precision_probe,
    precision_source_audit,
    runtime_recipe,
    sft,
)

REGIONAL_ROOT = "s3://marin-rno2a/fixture/lr0-startup"
MEMORY = (2**33 + 5, 50 * 1024**3, 63_766_003_712)
TOKENS = (128000, 1, 2, 3)


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_fixture(root):
    source = root / "source"
    source.mkdir()
    config = SnowballConfig(
        vocab_size=128032,
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
            lambda: SnowballLMHeadModel.init(
                hax.Axis("vocab", config.vocab_size), config, key=jax.random.key(310)
            ).to_state_dict()
        )
    tensors = {
        name: np.full(shape.shape, 0.125, dtype=np.float32 if name.endswith(".mlp.router.bias") else jnp.bfloat16)
        for name, shape in shapes.items()
    }
    assert len(tensors) == 502
    save_file(tensors, str(source / "weights.safetensors"))
    write_json(
        source / precision_source_audit.INDEX,
        {
            "weight_map": {name: "weights.safetensors" for name in tensors},
            "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        },
    )
    tokenizer_backend = Tokenizer(WordLevel({f"t{index}": index for index in range(config.vocab_size)}, unk_token="t0"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="t0",
        bos_token="t128000",
        eos_token="t128001",
        pad_token="t128001",
        chat_template="{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}t3 t4{% endgeneration %}{% else %}t1 t2 {% endif %}{% endfor %}",
    )
    tokenizer.save_pretrained(source)
    (source / "chat_template.jinja").write_text(tokenizer.chat_template)
    hf = json.loads(config.to_hf_config(config.vocab_size).to_json_string())
    hf.update(bos_token_id=128000, eos_token_id=128001, pad_token_id=128001)
    write_json(source / "config.json", hf)
    write_json(source / "generation_config.json", {"bos_token_id": 128000, "eos_token_id": [128001, 128009]})
    hashes = {name: digest((source / name).read_bytes()) for name in precision_source_audit.SOURCE_FILES}
    rounding = {
        "positions_sha256": runtime_recipe.POSITIONS_SHA256,
        "sequences": 465,
        "top_k": 4,
        "totals": {
            name: {"positions": 805974, "positions_changed": int(name != "bf16_bias")}
            for name in ("bf16_bias", "bf16_logits", "bf16_logits_bias")
        },
    }
    write_json(root / "routing.json", rounding)
    torch.save({"check|startup": {"tokens": list(TOKENS)}}, root / "positions.pt")
    write_json(
        root / "vllm.json",
        {"check|startup": {"tokens": list(TOKENS), "top": [None, {"1": -1.0}, {"1": -1.0}, {"1": -1.0}]}},
    )
    write_json(
        root / "pins.json", {"source_hashes": hashes, "rounding_sha": digest((root / "routing.json").read_bytes())}
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


def startup_policy(inputs, overrides, positions_path, dot_dtypes, moe, memory):
    """Validate fixture outputs without applying the real-model parity thresholds."""
    records = torch.load(positions_path, map_location="cpu")
    assert set(records) == {"check|startup"}
    record = records["check|startup"]
    assert record["tokens"] == list(TOKENS)
    assert record["routes"].shape == (len(TOKENS), 26, 4)
    assert np.isfinite(record["actual"][1:].numpy()).all()
    assert record["reference_ids"][1:, 0].tolist() == [1] * (len(TOKENS) - 1)
    assert np.isfinite(record["reference_lp"][1:, 0].numpy()).all()
    assert moe["dispatch"]["valid_tokens"] == len(TOKENS)
    assert not precision_gate.dispatch_failures(moe["dispatch"])
    assert precision_gate.router_dot_is_fp32(dot_dtypes)
    assert len(memory) == 32 and {row["process"] for row in memory} == set(range(4))
    assert all(tuple(row[field] for field in precision_gate.MEMORY_FIELDS) == MEMORY for row in memory)
    assert not precision_gate.memory_failures(memory)
    return {"startup_only": True, "failures": [], "fixture_tokens": len(TOKENS), "device_memory": memory}


def child(root, rank, port, mode):
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    assert not jax.distributed.is_initialized() and not xla_bridge.backends_are_initialized()
    pins = json.loads((root / "pins.json").read_bytes())
    source = str(root / "source")

    def local_path(path):
        path = str(path)
        if path.startswith(REGIONAL_ROOT + "/"):
            path = str(root / path.removeprefix(REGIONAL_ROOT + "/"))
        return StoragePath(path)

    with pytest.MonkeyPatch.context() as patch:
        for module in (sft, lr0_probe, precision_probe):
            patch.setattr(module, "BASE_MODEL", source)
        patch.setattr(lr0_probe, "SOURCE_FILE_HASHES", pins["source_hashes"])
        for module in (lr0_probe, precision_probe, precision_audit, precision_source_audit):
            patch.setattr(module, "StoragePath", local_path)
        patch.setattr(TrainerConfig, "num_slices", property(lambda config: jax.process_count()))
        patch.setattr(trainer_module, "create_mesh_from_axis_specs", cpu_mesh)
        factory = lr0_probe.base_config

        def config_factory(*args):
            config = factory(*args)
            assert config.trainer.distributed.initialize_jax_distributed is True
            return replace(config, trainer=replace(config.trainer, require_accelerator=False))

        patch.setattr(lr0_probe, "base_config", config_factory)
        gate_inputs = lr0_probe.gate_inputs

        def fixture_gate(*args):
            return replace(
                gate_inputs(*args),
                positions=str(root / "positions.pt"),
                vllm=str(root / "vllm.json"),
                output=str(root / "run/precision-gate"),
                pad_multiple=8,
            )

        patch.setattr(lr0_probe, "gate_inputs", fixture_gate)
        patch.setattr(precision_gate, "evaluate", startup_policy)
        patch.setattr(
            precision_gate,
            "local_device_memory",
            lambda: np.tile(np.array(MEMORY, dtype=np.int64), (jax.local_device_count(), 1)),
        )
        if mode == "early_gather":
            initialize = precision_probe.initialize

            def gather_before_initialize(config):
                gather_host_scalar(np.int32(0))
                initialize(config)

            patch.setattr(precision_probe, "initialize", gather_before_initialize)
        if mode == "late_guard":
            any_rank_failed = precision_gate.any_rank_failed

            def check_after_input_gather(failed):
                result = any_rank_failed(failed)
                runtime_recipe.require_xla_precision("default")
                return result

            patch.setattr(precision_gate, "any_rank_failed", check_after_input_gather)

        config_bytes = (root / "source/config.json").read_bytes()
        generation_bytes = (root / "source/generation_config.json").read_bytes()
        runtime = {
            "schema_version": 6,
            "runtime_model_config": {
                "moe_capacity_factor": 4.0,
                "router_combine_fp32": True,
                "fp32_activations": True,
                "attention_implementation": "torch_order",
            },
        }
        base = factory(f"{REGIONAL_ROOT}/run", "startup", config_bytes, generation_bytes, runtime)
        _, receipt = runtime_recipe.configure_precision_runtime(
            base,
            config_bytes=config_bytes,
            generation_bytes=generation_bytes,
            rounding_bytes=(root / "routing.json").read_bytes(),
            rounding_path=f"{REGIONAL_ROOT}/routing.json",
            rounding_sha256=pins["rounding_sha"],
            xla_precision="default",
        )
        receipt_bytes = json.dumps(receipt).encode()

        class BeforeTrainStep(Exception):
            pass

        def before_step(trainer, state, batch):
            assert jax.distributed.is_initialized() and jax.process_count() == 4 and jax.process_index() == rank
            assert int(state.step) == 0 and batch.tokens.array.shape == (8, 8192)
            assert str(batch.tokens.array.sharding.spec) == "P(('replica_dcn', 'data', 'expert'), 'context')"
            gate = json.loads((root / "run/precision-gate/gate.json").read_bytes())
            assert gate["startup_only"] and gate["failures"] == []
            master = json.loads((root / "run/audits/master-step-0.json").read_bytes())
            source_audit = json.loads((root / "run/audits/source-tensor-hashes.json").read_bytes())
            assert master["tensor_count"] == len(source_audit["tensors"]) == 502
            write_json(root / f"before-train-{rank}.json", {"rank": rank, "step": 0})
            raise BeforeTrainStep

        patch.setattr(Trainer, "train_step", before_step)

        def register(name, address):
            assert name == "jax_coordinator-attempt-0" and address == f"127.0.0.1:{port}"
            write_json(root / "coordinator.json", {"name": name, "address": address})
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
                lr0_probe.lr0_worker(f"{REGIONAL_ROOT}/run", "startup", receipt_bytes, digest(receipt_bytes), "G1")
        except BeforeTrainStep:
            print(json.dumps({"process": rank, "before_train_step": True}), flush=True)
        else:
            raise AssertionError("LR0 startup did not reach the first train_step boundary")
    jax.distributed.shutdown()


def run_children(root, mode):
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = str(sock.getsockname()[1])
    workdir = Path.cwd()
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "8",
        "XLA_FLAGS": "",
        "CUDA_VISIBLE_DEVICES": "",
        "AWS_ENDPOINT_URL": "http://localhost",
        "FSSPEC_S3": "{}",
        "AWS_ACCESS_KEY_ID": "fixture",
        "AWS_SECRET_ACCESS_KEY": "fixture",
        "IRIS_NUM_TASKS": "4",
        "IRIS_ADVERTISE_HOST": "127.0.0.1",
        "IRIS_PORT_JAX": port,
        "PYTHONPATH": os.pathsep.join([str(workdir), *(str(path) for path in workdir.glob("lib/*/src"))]),
    }
    processes = [
        subprocess.Popen(
            [sys.executable, __file__, "--child", str(root), str(rank), port, mode],
            cwd=workdir,
            env={**env, "IRIS_TASK_ID": f"/test/lr0-startup/{rank}"},
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
def test_lr0_startup_order_reaches_training_with_code_owned_distributed_init(tmp_path):
    write_fixture(tmp_path)
    reports = run_children(tmp_path, "normal")
    ranks = set()
    for code, output, error in reports:
        assert code == 0, error[-6000:]
        report = json.loads(output.strip().splitlines()[-1])
        assert report["before_train_step"]
        ranks.add(report["process"])
    assert ranks == set(range(4))
    for rank in ranks:
        assert json.loads((tmp_path / f"before-train-{rank}.json").read_bytes()) == {"rank": rank, "step": 0}
        receipt = json.loads((tmp_path / f"run/precision-gate/ranks/{rank}.json").read_bytes())
        assert receipt["stage"] == "gated" and receipt["failures"] == 0
    assert not (tmp_path / "run/lr0-result.json").exists()


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mode,message",
    [
        ("early_gather", "jax.distributed.initialize() must be called before any JAX calls"),
        ("late_guard", "XLA's backend started before the XLA precision setting was checked"),
    ],
)
def test_lr0_startup_order_rejects_collective_and_guard_reordering(tmp_path, mode, message):
    write_fixture(tmp_path)
    reports = run_children(tmp_path, mode)
    for code, _, error in reports:
        assert code != 0 and message in error
    assert not list(tmp_path.glob("before-train-*.json"))
    assert not (tmp_path / "run/lr0-result.json").exists()


if __name__ == "__main__":
    child(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
