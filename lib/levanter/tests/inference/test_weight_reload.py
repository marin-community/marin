# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.state_dict import flatten_modules_for_export, to_state_dict

from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.weight_reload import WeightTransferConfig
from levanter.models.llama import LlamaLMHeadModel
from levanter.testing.model_configs import llama_test_config
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.testing.helpers import skip_if_no_torch

try:
    from fastapi.testclient import TestClient
    from levanter.inference.openai import InferenceServer, InferenceServerConfig
    from levanter.testing.weight_broadcast import broadcast_source
except ImportError:
    pytest.skip("Torch and serving dependencies are required", allow_module_level=True)


@skip_if_no_torch
@pytest.mark.timeout(90)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("recipe", ["llama", "snowball", "hero"])
def test_http_broadcast_installs_only_complete_weights_and_preserves_failed_version(
    local_gpt2_marin_tokenizer, tmp_path, dtype, recipe
):
    trainer = TrainerConfig(
        mesh=MeshConfig(
            axes={"data": -1, "model": 1, "expert": 1, "context": 1},
            dcn_axes={"replica_dcn": -1},
            shared_mapping={"batch": ["replica_dcn", "data"], "heads": "model", "kv_head": "model"},
            param_mapping={},
        ),
        use_explicit_mesh_axes=True,
    )
    common = dict(
        vocab_size=16,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=2,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    if recipe == "llama":
        trainer = TrainerConfig()
        model_config = dataclasses.replace(
            llama_test_config(seq_len=8), hidden_dim=8, intermediate_dim=16, num_layers=1, num_heads=2, num_kv_heads=2
        )
        model_type = LlamaLMHeadModel
    elif recipe == "hero":
        model_config = HeroConfig(
            **common, latent_dim=8, local_kv_heads=2, global_kv_heads=1, global_every=2, sconv_kernel=3
        )
        model_type = HeroLMHeadModel
    else:
        model_config = SnowballConfig(**common)
        model_type = SnowballLMHeadModel
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        model = model_type.init(hax.Axis("vocab", 16), model_config, key=jax.random.PRNGKey(0))
        model = jax.tree.map(lambda value: value.astype(dtype) if eqx.is_array(value) else value, model)
    config = InferenceServerConfig(
        trainer=trainer,
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=1,
            page_size=4,
            max_queued_tokens=4,
            max_seqs_in_prefill=1,
            compute_dtype=jnp.dtype(dtype),
        ),
        weight_transfer=WeightTransferConfig(backend="gloo", max_staging_bytes=1_000_000, timeout=15),
    )
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(config, model, local_gpt2_marin_tokenizer)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        weights = {}
        for name, value in to_state_dict(flatten_modules_for_export(model)).items():
            array = np.asarray(value)
            if ".experts." in name:
                prefix, parameter = name.split(".experts.")
                for expert, part in enumerate(array):
                    weights[f"{prefix}.experts.{expert}.{parameter}"] = part
            else:
                weights[name] = array
    body = {
        "model": "gpt2",
        "prompt": [1, 2],
        "max_tokens": 2,
        "temperature": 0,
        "return_token_ids": True,
        "logprobs": 0,
    }
    try:
        with (
            broadcast_source(tmp_path / "broadcast-source.log") as source,
            TestClient(server.app) as client,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            initialized = client.post(
                "/init_weight_update_communicator",
                json={
                    "master_address": "127.0.0.1",
                    "master_port": source.port,
                    "rank_offset": 1,
                    "world_size": 2,
                    "group_name": "skyrl",
                    "backend": "gloo",
                },
            )
            assert initialized.status_code == 200
            assert source.receive()["device"] == "cpu"

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
                source.broadcast(dtype, weight.tolist())
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

            if recipe == "hero":
                caches = server.inference_context.engine.gen_state.cache.caches
                assert any(np.any(np.asarray(layer.k_history.history.array)) for layer in caches)
            if recipe == "hero":
                caches = server.inference_context.engine.gen_state.cache.caches
                assert any(np.any(np.asarray(layer.k_history.history.array)) for layer in caches)
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
            assert all(not np.any(np.asarray(value)) for value in cache_arrays())
            new = generate()
            assert new["token_ids"] == [0, 0]
            assert new["token_ids"] != old["token_ids"]
            assert new["model_version"] == 2
            assert client.post("/finish_weight_reload", json=ready).status_code == 409
    finally:
        server.inference_context.shutdown()
