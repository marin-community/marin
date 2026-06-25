# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual benchmarks for Arrow Flight weight transfer.

These are not pytest tests: they require real TPU devices and download a
Llama-1B model from HuggingFace, so they cannot run in CI. Run them directly:

    python -m marin.rl.scripts.benchmark_weight_transfer arrow-flight
    python -m marin.rl.scripts.benchmark_weight_transfer vllm
"""

import argparse
import logging
import sys
import tempfile
import uuid
from functools import partial

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import Mesh
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.llama import LlamaConfig
from marin.rl.environments.inference_ctx import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
from marin.rl.weight_transfer import (
    WeightTransferConfig,
    WeightTransferMode,
    create_weight_transfer_client,
    create_weight_transfer_server,
)
from marin.rl.weight_utils import levanter_to_nnx_state
from transformers import AutoConfig, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def benchmark_arrow_flight_with_llama():
    """Benchmark Arrow Flight weight transfer with a Llama 1B model."""
    # Load model config from HuggingFace
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = LlamaConfig.from_hf_config(hf_config)

    devices = jax.devices("tpu")
    num_devices = len(devices)
    print(f"Found {num_devices} TPU devices")

    # Use tensor parallelism and FSDP like in exp1247_rl_async.py
    mesh = Mesh(np.array(devices).reshape(-1, 4), axis_names=("data", "model"))
    print(f"Mesh created with shape {mesh.shape}: {mesh.devices}")

    # Create axis mapping for sharding - match the pattern from rollout_worker.py
    # FSDP on embed, TP on mlp and heads
    axis_mapping = {
        "embed": "data",  # FSDP
        "mlp": "model",  # TP
        "kv_head": "model",
        "q_heads_per_group": None,
        "head_size": None,
        "layers": None,
    }

    # Create weight transfer config
    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        checkpoint_dir=tempfile.mkdtemp(),
        coordinator_name=f"test_coordinator_{uuid.uuid4().hex[:8]}",
    )

    server = create_weight_transfer_server(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

    client = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

    Vocab = hax.Axis("vocab", hf_config.vocab_size)
    key = jax.random.PRNGKey(0)

    with hax.set_mesh(mesh), hax.axis_mapping(axis_mapping):
        model = model_config.build(Vocab, key=key)
        model = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(model)

    print(f"Model built with vocab size: {hf_config.vocab_size}")

    @partial(jax.jit, donate_argnums=0)
    def _bump_params(params):
        return jax.tree.map(lambda x: x + jnp.ones_like(x) * 0.001, params)

    try:
        # Test weight transfer
        print("Starting weight transfer test...")
        for i in range(10):
            print(f"Transfer iteration {i}")
            model = _bump_params(model)
            server.serve_weights(i, model)
            update = client.receive_weights(model)

            assert update is not None, f"Weight update {i} failed"
            assert update.weight_id == i, f"Expected weight_id {i}, got {update.weight_id}"
            model = update.model

            print(f"Iteration {i} completed successfully")
            print(f"Client metrics: {client.get_metrics()}")

        print("All transfers completed successfully")

    finally:
        server.cleanup()
        client.cleanup()


def benchmark_arrow_flight_transfer_to_vllm():
    """Benchmark Arrow Flight weight transfer to vLLM."""
    from vllm import LLM  # noqa: PLC0415  # optional dep: vllm

    devices = jax.devices("tpu")
    num_devices = len(devices)
    print(f"Found {num_devices} TPU devices")

    # Only use device one for now so we don't have to deal with sharded arrays.
    # This is fine since the model will fit on single TPU.
    mesh = Mesh(
        np.array(devices)[:1].reshape(
            1,
        ),
        axis_names=("data",),
    )
    print(f"Mesh created with shape {mesh.shape}: {mesh.devices}")

    # Create weight transfer config
    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        checkpoint_dir=tempfile.mkdtemp(),
        coordinator_name=f"test_coordinator_{uuid.uuid4().hex[:8]}",
    )

    server = create_weight_transfer_server(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping={},
    )

    client = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping={},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_checkpoint = RepoRef.from_string(MODEL_NAME)
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = LlamaConfig.from_hf_config(hf_config)
    converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)

    # Load pretrained weights from HuggingFace
    with hax.partitioning.set_mesh(mesh):
        model = converter.load_pretrained(
            model_config.model_type,
            ref=hf_checkpoint,
            config=model_config,
            dtype=jmp.get_policy("p=bfloat16,c=bfloat16").compute_dtype,
        )

    print(f"Model built with vocab size: {hf_config.vocab_size}")

    llm = LLM(MODEL_NAME, gpu_memory_utilization=0.50)
    try:
        # Test weight transfer
        print("Starting weight transfer test to vllm...")
        print("Transfer iteration 0")
        server.serve_weights(0, model)
        update = client.receive_weights(model)

        assert update is not None, "Weight update failed"
        assert update.weight_id == 0, f"Expected weight_id 0, got {update.weight_id}"
        model = update.model
        model_nnx_state = levanter_to_nnx_state(model)
        llm.llm_engine.model_executor.driver_worker.sync_weights(
            model_nnx_state,
            mappings=MODEL_MAPPINGS[MODEL_NAME],
            transpose_keys=MODEL_TRANSPOSE_KEYS[MODEL_NAME],
            reshard_fn=None,
        )

        output = llm.generate(["Hello, how are you?"])

        key_phrase = "I'm excited to be here"
        generated_text = output[0].outputs[0].text
        assert key_phrase in generated_text, f"Key phrase: {key_phrase} not found in output: {generated_text}"

        print(f"Client metrics: {client.get_metrics()}")

        print("All transfers completed successfully")

    finally:
        server.cleanup()
        client.cleanup()


BENCHMARKS = {
    "arrow-flight": benchmark_arrow_flight_with_llama,
    "vllm": benchmark_arrow_flight_transfer_to_vllm,
}


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS), help="Which benchmark to run.")
    args = parser.parse_args()
    BENCHMARKS[args.benchmark]()


if __name__ == "__main__":
    main()
