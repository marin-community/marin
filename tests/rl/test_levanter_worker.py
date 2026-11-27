# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess update and checkpoint continuation at the learner boundary."""

import concurrent.futures
import copy
import os
import socket
import sys
from contextlib import contextmanager

import grpc
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference import inference_pb2, inference_pb2_grpc
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.snowball_publication import publish_snowball_checkpoint
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.testing.devices import use_cpu_devices
from levanter.tokenizers import load_tokenizer
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
        initializer_std=0.12,
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
    return model


def _rollout_with_routes(tokens, scores, expert_ids, policy_id):
    valid = np.ones(tokens.size, dtype=bool)
    valid[-1] = False
    ids = np.broadcast_to(np.asarray(expert_ids, np.uint8), (2, tokens.size, 2))
    response = inference_pb2.GenerateResponse(
        prompt_token_ids=tokens[: -scores.size].tolist(),
        logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
        choices=[
            inference_pb2.Choice(
                token_ids=tokens[-scores.size :].tolist(),
                policy_logprobs=scores,
                policy_id=policy_id,
                router_replay=inference_pb2.RouterReplay(
                    num_layers=2,
                    num_positions=tokens.size,
                    experts_per_token=2,
                    expert_ids=ids.tobytes(),
                    expert_id_dtype=inference_pb2.EXPERT_ID_DTYPE_UINT8,
                    valid=valid.tolist(),
                ),
            )
        ],
    )
    return response.SerializeToString()


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


def _worker_config(tmp_path, initializer, process_count):
    local_count = jax.local_device_count(backend="cpu")
    count = local_count * process_count
    return {
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
        "mini_batch_size": count * 2,
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


@pytest.mark.timeout(180)
@pytest.mark.parametrize("process_count", [1, pytest.param(2, marks=pytest.mark.multidevice)])
def test_native_learner_subprocess_updates_and_resumes_exactly(tmp_path, process_count):
    initializer = tmp_path / "initializer"
    initializer.mkdir()
    _hf_initializer(initializer)
    config = _worker_config(tmp_path, initializer, process_count)
    count = config["runtime"]["mesh_axes"]["data"]
    batch_size = config["mini_batch_size"]
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
        policy_id = "published-fixture"

        def with_served_routes(expert_ids):
            rollouts = [
                (_rollout_with_routes(row[1:], values, expert_ids, policy_id), 0)
                for row, values in zip(tokens, scores, strict=True)
            ]
            return payload | {"policy_id": policy_id, "grpc_rollouts": rollouts}

        replayed = _same_request(children, "forward", with_served_routes([0, 1]))[0]["output"]
        other = _same_request(children, "forward", with_served_routes([2, 3]))[0]["output"]
        assert np.max(np.abs(replayed - other)) > 1e-5
        payload = with_served_routes([0, 1])
        updates = _same_request(children, "train", payload)
        assert all(update["optimizer_step"] == 1 and update["router_bias_unchanged"] == 1 for update in updates)
        after = _same_request(children, "forward", payload)[0]["output"]
        assert np.max(np.abs(after - replayed)) > 1e-5
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

    resumed_config = config | {"run_id": "resumed-policy-regression"}
    with _learner_processes(resumed_config, process_count) as (children, _):
        restored = _same_request(children, "load_checkpoint", checkpoint)
        assert [row["policy_id"] for row in restored] == [row["policy_id"] for row in saved]
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


@pytest.mark.timeout(180)
@use_cpu_devices(1)
def test_served_snowball_rollout_updates_and_publishes_in_learner_worker(tmp_path):
    initializer = tmp_path / "initializer"
    initializer.mkdir()
    model = _hf_initializer(initializer)
    config = _worker_config(tmp_path, initializer, process_count=1)
    batch_size = config["mini_batch_size"]
    prompt = np.array([2, 5, 7], dtype=np.int32)
    options = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=32,
            window_cache_max_pages=24,
            max_seqs=2,
            max_rounds=2,
            page_size=2,
            max_seqs_in_prefill=2,
            max_prefill_size=8,
            max_queued_tokens=4,
            logprob_mode=LogprobMode.POLICY,
            compute_dtype=jnp.float32,
        ),
        initial_policy_id="initializer",
        host="127.0.0.1",
        grpc_port=0,
    )
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        server = InferenceServer.create(
            options, model.with_packed_experts().with_split_decode_cache(), load_tokenizer(str(initializer)), mesh=mesh
        )
        try:
            with (
                TestClient(server.app),
                grpc.insecure_channel(f"{server.config.host}:{server.app.state.grpc_port}") as channel,
            ):
                stub = inference_pb2_grpc.InferenceStub(channel)

                def generate(policy_id):
                    request = inference_pb2.GenerateRequest(
                        prompt_token_ids=prompt.tolist(),
                        max_new_tokens=2,
                        temperature=0,
                        capture_router_replay=True,
                        expected_policy_id=policy_id,
                        expected_logprob_mode=inference_pb2.LOGPROB_MODE_POLICY,
                    )
                    return stub.Generate(request, timeout=30)

                before = generate("initializer")
                original_scores = np.asarray(before.choices[0].policy_logprobs, dtype=np.float32)
                tokens = np.tile(np.r_[prompt, before.choices[0].token_ids].astype(np.int32), (batch_size, 1))
                response_length = len(original_scores)
                assert response_length == 2
                payload = {
                    "arrays": {
                        "tokens": tokens,
                        "attention_mask": np.ones_like(tokens, dtype=bool),
                        "position_ids": np.tile(np.arange(tokens.shape[1], dtype=np.int32), (batch_size, 1)),
                        "response_mask": np.ones((batch_size, response_length), dtype=bool),
                        "advantages": np.ones((batch_size, response_length), dtype=np.float32),
                        "policy_weights": np.full(
                            (batch_size, response_length), 1 / (batch_size * response_length), dtype=np.float32
                        ),
                        "kl_weights": np.zeros((batch_size, response_length), dtype=np.float32),
                    },
                    "response_length": response_length,
                    "global_step": 1,
                    "policy_id": "initializer",
                    "grpc_rollouts": [(before.SerializeToString(), 0)] * batch_size,
                }
                checkpoint = {"ckpt_dir": str(tmp_path / "updated")}
                with _learner_processes(config, 1) as (children, _):
                    assert _same_request(children, "train", payload)[0]["optimizer_step"] == 1
                    receipt = _same_request(children, "save_checkpoint", checkpoint)[0]
                    _same_request(children, "shutdown", {})
                assert (
                    publish_snowball_checkpoint(
                        server, checkpoint["ckpt_dir"], policy_version=receipt["version"], expected_serving_version=0
                    )
                    == 1
                )
                with pytest.raises(grpc.RpcError) as stale:
                    generate("initializer")
                assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                choice = generate(receipt["policy_id"]).choices[0]
                assert choice.policy_id == receipt["policy_id"]
                assert not np.array_equal(choice.policy_logprobs, original_scores)
        finally:
            server.shutdown()
