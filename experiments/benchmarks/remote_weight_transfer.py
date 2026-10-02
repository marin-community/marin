# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the pinned SkyRL client against native serving with real tensor broadcasts.

Add the paired SkyRL checkout's skyrl-train directory to PYTHONPATH. For NCCL,
expose only the receiver GPU through CUDA_VISIBLE_DEVICES and pass distinct physical
sender/receiver device IDs. The sender runs in a separate process. No training runs.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path

import aiohttp
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import torch
import uvicorn
from haliax.state_dict import flatten_modules_for_export, to_state_dict
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.weight_reload import WeightTransferConfig
from levanter.models.llama import LlamaLMHeadModel
from levanter.testing.model_configs import llama_test_config
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.testing.weight_broadcast import broadcast_source
from levanter.tokenizers import load_tokenizer
from skyrl_train.inference_engines import remote_inference_engine

SKYRL_REVISION = "c2ed0d0b795e884ac452841d48455ceba06d5f54"


class ReadyServer(uvicorn.Server):
    def __init__(self, app, ready):
        super().__init__(uvicorn.Config(app, log_level="error"))
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets)
        self.ready.set()


async def run(args):
    skyrl_root = Path(remote_inference_engine.__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "-C", str(skyrl_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != SKYRL_REVISION:
        raise ValueError(f"Expected SkyRL {SKYRL_REVISION}, found {revision}")
    if args.backend == "nccl":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.receiver_device:
            raise ValueError("CUDA_VISIBLE_DEVICES must expose exactly --receiver-device")
        if args.sender_device == args.receiver_device:
            raise ValueError("Sender and receiver must own distinct physical GPUs")
    if jax.device_count() != 1:
        raise ValueError("This gate requires exactly one visible JAX receiver device")
    model_config = dataclasses.replace(
        llama_test_config(seq_len=8), hidden_dim=8, intermediate_dim=16, num_layers=1, num_heads=2, num_kv_heads=2
    )
    model = LlamaLMHeadModel.init(hax.Axis("vocab", 16), model_config, key=jax.random.key(0))
    model = jax.tree.map(lambda value: value.astype(args.dtype) if eqx.is_array(value) else value, model)
    config = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=8,
            max_pages=4,
            max_seqs=1,
            page_size=4,
            max_queued_tokens=4,
            max_seqs_in_prefill=1,
            compute_dtype=jnp.dtype(args.dtype),
        ),
        weight_transfer=WeightTransferConfig(backend=args.backend, max_staging_bytes=1_000_000, timeout=30),
    )
    receiver = {"jax_device": str(jax.devices()[0]), "torch": torch.__version__, "jax": jax.__version__}
    if args.backend == "nccl":
        assert torch.cuda.device_count() == 1 and jax.devices()[0].local_hardware_id == 0
        receiver.update(
            uuid=str(torch.cuda.get_device_properties(0).uuid), cuda=torch.version.cuda, nccl=torch.cuda.nccl.version()
        )
    with tempfile.TemporaryDirectory(prefix="native-weight-gate-") as scratch:
        scratch = Path(scratch)
        repository = Path(__file__).resolve().parents[2]
        (scratch / "tokenizer").mkdir()
        tokenizer = load_tokenizer(str(stage_gpt2_tokenizer(repository / "lib/levanter/tests", scratch / "tokenizer")))
        with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
            server = InferenceServer.create(config, model, tokenizer)
        weights = {
            name: np.zeros_like(np.asarray(value))
            for name, value in to_state_dict(flatten_modules_for_export(model)).items()
        }
        ready = asyncio.Event()
        http_server = ReadyServer(server.app, ready)
        with (
            socket.socket() as listener,
            broadcast_source(
                scratch / "sender.log",
                backend=args.backend,
                timeout=30,
                environment={"CUDA_VISIBLE_DEVICES": args.sender_device if args.backend == "nccl" else ""},
            ) as source,
        ):
            listener.bind(("127.0.0.1", 0))
            serving = asyncio.create_task(http_server.serve(sockets=[listener]))
            try:
                await asyncio.wait_for(ready.wait(), 30)
                client = remote_inference_engine.RemoteInferenceEngine(
                    f"127.0.0.1:{listener.getsockname()[1]}", "gpt2", "vllm", tokenizer
                )
                await client.init_weight_update_communicator("127.0.0.1", source.port, 1, 2, "skyrl", args.backend)
                sender = await asyncio.to_thread(source.receive)
                if args.backend == "nccl":
                    assert sender["uuid"] != receiver["uuid"]
                request = {"prompt_token_ids": [[1, 2]], "sampling_params": {"temperature": 0, "max_tokens": 2}}

                def cache_arrays():
                    return jax.tree.leaves(eqx.filter(server.inference_context.engine.gen_state.cache, eqx.is_array))

                async def transfer(names):
                    async def send():
                        for name in names:
                            await asyncio.to_thread(source.broadcast, args.dtype, weights[name].tolist())

                    await asyncio.gather(
                        client.update_named_weights(
                            {
                                "names": names,
                                "dtypes": [f"torch.{args.dtype}"] * len(names),
                                "shapes": [list(weights[name].shape) for name in names],
                            }
                        ),
                        send(),
                    )

                old = await client.generate(request)
                assert any(np.any(np.asarray(value)) for value in cache_arrays())
                assert await client.reset_prefix_cache() == {"status": "ok"}
                assert all(not np.any(np.asarray(value)) for value in cache_arrays())
                await client.begin_weight_reload()
                await transfer([next(iter(weights))])
                assert await client.generate(request) == old
                try:
                    await client.finish_weight_reload()
                except aiohttp.ClientResponseError as error:
                    assert error.status == 409
                else:
                    raise AssertionError("Incomplete publication was accepted")
                assert server.model_version == 0 and await client.generate(request) == old
                await client.begin_weight_reload()
                await transfer(list(weights))
                assert await client.generate(request) == old
                await asyncio.to_thread(server.reload, lambda current: current, expected_version=0)
                try:
                    await client.finish_weight_reload()
                except aiohttp.ClientResponseError as error:
                    assert error.status == 409
                else:
                    raise AssertionError("Stale publication was accepted")
                assert server.model_version == 1 and await client.generate(request) == old
                await client.begin_weight_reload()
                await transfer(list(weights))
                installed = await client.finish_weight_reload()
                assert installed == {"model_version": 2}
                assert all(not np.any(np.asarray(value)) for value in cache_arrays())
                for leaf in jax.tree.leaves(eqx.filter(server.model, eqx.is_array)):
                    assert leaf.devices() == {jax.devices()[0]}
                new = await client.generate(request)
                assert new["response_ids"] == [[0, 0]] and new["response_ids"] != old["response_ids"]
                # Zero parameters give uniform logits, independent of the prior cache contents.
                np.testing.assert_allclose(new["response_logprobs"], -np.log(16), rtol=0, atol=1e-6)
                await client.teardown()
                report = {
                    "backend": args.backend,
                    "dtype": args.dtype,
                    "skyrl_revision": revision,
                    "marin_revision": (
                        subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
                    ),
                    "receiver": receiver,
                    "sender": sender,
                    "old": old,
                    "new": new,
                    "model_version": server.model_version,
                    "parameter_bytes": sum(v.nbytes for v in weights.values()),
                }
            finally:
                http_server.should_exit = True
                await asyncio.wait_for(serving, 30)
                server.inference_context.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--sender-device", default="0")
    parser.add_argument("--receiver-device", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args), timeout=240))


if __name__ == "__main__":
    main()
