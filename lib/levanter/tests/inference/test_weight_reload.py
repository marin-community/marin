# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import multiprocessing
import socket
import subprocess
import sys
from pathlib import Path
from multiprocessing.connection import Connection
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.state_dict import flatten_modules_for_export, to_state_dict

from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.weight_reload import WeightTransferConfig
from levanter.models.llama import LlamaLMHeadModel
from levanter.testing.helpers import skip_if_no_torch
from levanter.testing.model_configs import llama_test_config

try:
    import torch
    import torch.distributed as dist
    from fastapi.testclient import TestClient
    from torch.distributed.distributed_c10d import _new_process_group_helper, _world
except ImportError:
    pytest.skip("Torch and serving dependencies are required", allow_module_level=True)


def _weight_source(port, connection):
    group = None
    try:
        store = dist.TCPStore("127.0.0.1", port, 2, True, timeout=timedelta(seconds=15))
        group, _ = _new_process_group_helper(
            2, 0, [], "gloo", dist.PrefixStore("skyrl", store), group_name="skyrl", timeout=timedelta(seconds=15)
        )
        _world.pg_group_ranks[group] = {0: 0, 1: 1}
        connection.send(None)
        while (command := connection.recv()) is not None:
            dtype, values = command
            tensor = torch.tensor(values, dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype])
            dist.broadcast(tensor, src=0, group=group)
            connection.send(None)
    except Exception:
        connection.send(traceback.format_exc())
        raise
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        connection.close()


@skip_if_no_torch
@pytest.mark.timeout(90)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_http_broadcast_installs_only_complete_weights_and_preserves_failed_version(
    local_gpt2_marin_tokenizer, tmp_path, dtype
):
    model_config = dataclasses.replace(
        llama_test_config(seq_len=8), hidden_dim=8, intermediate_dim=16, num_layers=1, num_heads=2, num_kv_heads=2
    )
    model = LlamaLMHeadModel.init(hax.Axis("vocab", 16), model_config, key=jax.random.PRNGKey(0))
    model = jax.tree.map(lambda value: value.astype(dtype) if eqx.is_array(value) else value, model)
    config = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=1,
            page_size=4,
            max_queued_tokens=4,
            max_seqs_in_prefill=1,
            compute_dtype=jnp.dtype(dtype),
        ),
        weight_transfer=WeightTransferConfig(backend="gloo", max_staging_bytes=100_000, timeout=15),
    )
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, model, local_gpt2_marin_tokenizer)
    weights = {key: np.asarray(value) for key, value in to_state_dict(flatten_modules_for_export(model)).items()}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    connection, child_connection = multiprocessing.Pipe()
    worker_log = tmp_path / "broadcast-source.log"
    with worker_log.open("wb") as output:
        source = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(port), str(child_connection.fileno())],
            pass_fds=(child_connection.fileno(),),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    child_connection.close()
    body = {
        "model": "gpt2",
        "prompt": [1, 2],
        "max_tokens": 2,
        "temperature": 0,
        "return_token_ids": True,
        "logprobs": 0,
    }
    try:
        with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=1) as pool:
            initialized = client.post(
                "/init_weight_update_communicator",
                json={
                    "master_address": "127.0.0.1",
                    "master_port": port,
                    "rank_offset": 1,
                    "world_size": 2,
                    "group_name": "skyrl",
                    "backend": "gloo",
                },
            )
            assert initialized.status_code == 200
            assert connection.poll(15) and connection.recv() is None

            def generate():
                response = client.post("/v1/completions", json=body)
                assert response.status_code == 200
                return response.json()["choices"][0]

            def transfer(publication, name):
                weight = np.zeros_like(weights[name])
                pending = pool.submit(
                    client.post,
                    "/update_weights",
                    json={
                        **publication,
                        "name": name,
                        "shape": list(weight.shape),
                        "dtype": f"torch.{dtype}",
                    },
                )
                connection.send((dtype, weight.tolist()))
                assert connection.poll(15) and connection.recv() is None
                response = pending.result(timeout=15)
                assert response.status_code == 200, response.text

            old = generate()
            cache_arrays = lambda: jax.tree.leaves(
                eqx.filter(server.inference_context.engine.gen_state.cache, eqx.is_array)
            )
            assert any(np.any(np.asarray(value)) for value in cache_arrays())
            assert client.post("/reset_prefix_cache").status_code == 200
            assert all(not np.any(np.asarray(value)) for value in cache_arrays())
            assert generate() == old
            server.pause_generation()
            assert client.post("/reset_prefix_cache").status_code == 200
            assert server.inference_context.pause_event.is_set()
            server.resume_generation()
            incomplete = client.post("/begin_weight_reload").json()
            transfer(incomplete, next(iter(weights)))
            assert generate() == old
            assert client.post("/finish_weight_reload", json=incomplete).status_code == 409
            assert generate() == old
            assert server.model_version == 0

            invalid = client.post("/begin_weight_reload").json()
            rejected = client.post(
                "/update_weights",
                json={**invalid, "name": next(iter(weights)), "shape": [999], "dtype": f"torch.{dtype}"},
            )
            assert rejected.status_code == 409
            assert client.post("/finish_weight_reload", json=invalid).status_code == 409
            assert generate() == old

            staged = client.post("/begin_weight_reload").json()
            for name in weights:
                transfer(staged, name)
            assert generate() == old
            # A concurrent local publication invalidates the staged remote version.
            server.reload(lambda current: current, expected_version=0)
            assert client.post("/finish_weight_reload", json=staged).status_code == 409
            assert generate()["token_ids"] == old["token_ids"]
            assert server.model_version == 1

            ready = client.post("/begin_weight_reload").json()
            # A delayed packet from a discarded publication cannot poison the new one.
            assert (
                client.post(
                    "/update_weights",
                    json={**staged, "name": next(iter(weights)), "shape": [999], "dtype": f"torch.{dtype}"},
                ).status_code
                == 409
            )
            for name in weights:
                transfer(ready, name)
            installed = client.post("/finish_weight_reload", json=ready)
            assert installed.status_code == 200, installed.text
            assert installed.json()["model_version"] == 2
            assert all(not np.any(np.asarray(value)) for value in cache_arrays())
            new = generate()
            assert new["token_ids"] == [0, 0]
            assert new["token_ids"] != old["token_ids"]
            assert new["model_version"] == 2
            assert client.post("/finish_weight_reload", json=ready).status_code == 409
    finally:
        if source.poll() is None:
            connection.send(None)
        try:
            source.wait(timeout=15)
        except subprocess.TimeoutExpired:
            source.kill()
            source.wait()
        connection.close()
        server.inference_context.shutdown()
    assert source.returncode == 0, worker_log.read_text()


if __name__ == "__main__":
    _weight_source(int(sys.argv[1]), Connection(int(sys.argv[2])))
