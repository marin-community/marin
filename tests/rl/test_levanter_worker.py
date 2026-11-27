# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess update and checkpoint continuation at the learner boundary."""

import concurrent.futures
import copy
import os
import socket
import sys
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from marin.rl.levanter_worker_protocol import LearnerProcess
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


def _hf_initializer(path):
    vocab = {"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>")),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", 64), config, key=jax.random.key(31))
        config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).save_pretrained(
            model,
            str(path),
            save_tokenizer=False,
            save_reference_code=False,
            dtype=jnp.float32,
        )


def _request_all(children, method, payloads):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(children)) as pool:
        futures = [
            pool.submit(child.request, method, payload) for child, payload in zip(children, payloads, strict=True)
        ]
        done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        error = next((future.exception() for future in done if future.exception() is not None), None)
        if error is not None:
            for child in children:
                child.close()
            raise error
        return [future.result() for future in futures]


def _same_request(children, method, payload):
    return _request_all(children, method, [payload] * len(children))


@contextmanager
def _learner_processes(config, process_count):
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo"}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    children = [
        LearnerProcess(sys.executable, environment=environment, timeout_seconds=60) for _ in range(process_count)
    ]
    try:
        configs = []
        for rank in range(process_count):
            local = copy.deepcopy(config)
            local["distributed"].update(process_id=rank, coordinator_address=address)
            configs.append(local)
        receipts = _request_all(children, "initialize", configs)
        yield children, receipts
    finally:
        for child in children:
            child.close()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("process_count", [1, pytest.param(2, marks=pytest.mark.multidevice)])
def test_native_learner_subprocess_updates_and_resumes_exactly(tmp_path, process_count):
    initializer = tmp_path / "initializer"
    initializer.mkdir()
    _hf_initializer(initializer)
    local_count = jax.local_device_count(backend="cpu")
    count = local_count * process_count
    batch_size = count * 2
    config = {
        "runtime": {
            "mesh_axes": {"replica_dcn": 1, "replica": 1, "data": count, "expert": 1, "model": 1, "context": 1},
            "checkpoint_dir": str(tmp_path),
            "projection_block_size": 16,
            "attention_implementation": "reference",
            "moe_implementation": "ring_gather_combine",
            "router_dot_fp32": True,
            "moe_capacity_factor": 8.0,
            "fp32_activations": False,
        },
        "distributed": {
            "num_processes": process_count,
            "process_id": 0,
            "local_device_count": local_count,
            "coordinator_address": None,
        },
        "initialization_timeout_seconds": 60,
        "run_id": "subprocess-policy-regression",
        "seed": 41,
        "bf16": True,
        "num_training_steps": 3,
        "mini_batch_size": batch_size,
        "micro_batch_size_per_device": 1,
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "require_accelerator": False,
        "model": {"path": str(initializer), "revision": None},
        "optimizer": {
            "max_grad_norm": 0.5,
            "lr": 0.001,
            "adam_betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
    }
    tokens = np.tile(np.array([0, 2, 3, 4, 5, 6], dtype=np.int32), (batch_size, 1))
    arrays = {
        "tokens": tokens,
        "attention_mask": tokens != 0,
        "position_ids": np.tile(np.array([0, 0, 1, 2, 3, 4], dtype=np.int32), (batch_size, 1)),
        "response_mask": np.ones((batch_size, 2), dtype=np.bool_),
    }
    payload = {"arrays": arrays, "response_length": 2, "global_step": 1}
    checkpoint = {"ckpt_dir": str(tmp_path / "checkpoint"), "include_fingerprints": True}
    with _learner_processes(config, process_count) as (children, receipts):
        assert all(row["global_devices"] == count and row["microbatches"] == 2 for row in receipts)
        scores = _same_request(children, "forward", payload)[0]["output"]
        arrays.update(
            {
                "old_logprobs": scores.copy(),
                "advantages": np.ones((batch_size, 2), dtype=np.float32),
                "policy_weights": np.full((batch_size, 2), 1 / (batch_size * 2), dtype=np.float32),
                "kl_weights": np.zeros((batch_size, 2), dtype=np.float32),
            }
        )
        updates = _same_request(children, "train", payload)
        assert all(update["optimizer_step"] == 1 and update["router_bias_unchanged"] == 1 for update in updates)
        after = _same_request(children, "forward", payload)[0]["output"]
        assert np.max(np.abs(after - scores)) > 1e-5
        saved = _same_request(children, "save_checkpoint", checkpoint)
        for row in saved:
            for name in ("model", "opt_state", "training_key", "step"):
                assert row["state_fingerprints"]["components"][name]["addressable_bytes"] > 0
        payload["global_step"] = 2
        expected = _same_request(children, "train", payload)
        expected_scores = _same_request(children, "forward", payload)[0]["output"]
        expected_state = _same_request(children, "state_fingerprints", {})
        _same_request(children, "shutdown", {})
        assert all(child.process.returncode == 0 for child in children)

    with _learner_processes(config, process_count) as (children, _):
        restored = _same_request(children, "load_checkpoint", checkpoint)
        for expected_saved, actual_saved in zip(saved, restored, strict=True):
            assert actual_saved["step"] == actual_saved["version"] == 1
            assert actual_saved["state_fingerprints"] == expected_saved["state_fingerprints"]
        np.testing.assert_array_equal(_same_request(children, "forward", payload)[0]["output"], after)
        assert _same_request(children, "train", payload) == expected
        np.testing.assert_array_equal(_same_request(children, "forward", payload)[0]["output"], expected_scores)
        assert _same_request(children, "state_fingerprints", {}) == expected_state
        np.testing.assert_array_equal(arrays["old_logprobs"], scores)
        _same_request(children, "shutdown", {})
        assert all(child.process.returncode == 0 for child in children)
