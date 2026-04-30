# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import uuid
from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
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

TRANSFER_TYPES = [
    WeightTransferMode.GCS_CHECKPOINT,
    WeightTransferMode.ARROW_FLIGHT,
]

if os.environ.get("CI"):
    pytest.skip("Skipping slow tests on CI", allow_module_level=True)


class EmbeddingTestModule(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: list[eqx.Module]


def create_sample_pytree(
    seed: int,
    vocab_size: int = 1000,
    hidden_size: int = 32,
    layers: int = 2,
) -> eqx.Module:
    """Create a sample eqx module pytree with random weights for testing."""
    generator = np.random.Generator(np.random.PCG64(seed))

    Vocab = hax.Axis("vocab", vocab_size)
    Hidden = hax.Axis("hidden", hidden_size)
    Layers = hax.Axis("layers", layers)
    return EmbeddingTestModule(
        embedding=hax.named(
            generator.normal(size=(Vocab.size, Hidden.size)).astype(np.float32),
            (Vocab, Hidden),
        ),
        layers=[
            eqx.nn.Linear(
                in_features=Hidden.size,
                out_features=Hidden.size,
                key=jax.random.PRNGKey(seed + i),
                use_bias=True,
            )
            for i in range(Layers.size)
        ],
    )


def create_mesh(devices=None):
    """Create a simple JAX mesh for testing."""
    if devices is None:
        devices = jax.local_devices()[:1]  # Use just one device for tests
    return Mesh(np.array(devices), axis_names=("batch",))


@pytest.fixture(params=TRANSFER_TYPES)
def transfer_mode(request):
    """Parametrized weight transfer mode."""
    return request.param


@pytest.fixture
def sample_params():
    """Generate sample JAX parameters for testing."""
    return create_sample_pytree(seed=42)


def create_test_weight_transfer_pair(weight_transfer_config):
    """Helper function to create server/client pairs for testing with simplified Levanter API."""
    from fray import current_client
    from marin.rl.weight_transfer.arrow_flight import ArrowFlightCoordinator

    # Set unique coordinator name for distributed modes
    coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
    weight_transfer_config.coordinator_name = coordinator_name

    # Create simple mesh and axis mapping for testing
    mesh = create_mesh()
    axis_mapping = {
        "vocab": None,
        "hidden": None,
        "layers": None,
    }

    # Create coordinator handle for Arrow Flight mode
    coordinator_handle = None
    if weight_transfer_config.mode == WeightTransferMode.ARROW_FLIGHT:
        client = current_client()
        coordinator_handle = client.create_actor(
            ArrowFlightCoordinator,
            name=coordinator_name,
        )

    server = create_weight_transfer_server(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping=axis_mapping,
        coordinator_handle=coordinator_handle,
    )

    client = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping=axis_mapping,
        coordinator_handle=coordinator_handle,
    )

    return server, client


@pytest.fixture
def weight_transfer_config(transfer_mode):
    """Create weight transfer config for the specified mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = WeightTransferConfig(
            mode=transfer_mode,
            sync_interval_steps=1,
            checkpoint_dir=temp_dir,
        )
        yield config


@pytest.fixture(autouse=True)
def v2_client():
    """Ensure a v2 LocalClient for weight transfer tests."""
    from fray import LocalClient, set_current_client

    with set_current_client(LocalClient()) as client:
        yield client


def test_multiple_weight_updates(weight_transfer_config, sample_params):
    """Test multiple sequential weight updates."""
    server, client = create_test_weight_transfer_pair(weight_transfer_config)

    # First weight transfer
    server.serve_weights(1, sample_params)
    update_1 = client.receive_weights(sample_params)
    assert update_1 is not None
    assert update_1.weight_id == 1

    # Second weight transfer with new params
    new_params = create_sample_pytree(seed=456)  # Different seed
    server.serve_weights(2, new_params)
    update_2 = client.receive_weights(update_1.model)
    assert update_2 is not None
    assert update_2.weight_id == 2

    # bfloat16 round-trip is lossy (7-bit mantissa → ~0.78% max relative error)
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=4e-3, atol=4e-3),
        update_2.model,
        new_params,
    )

    # Third call should return None (no new weights)
    update_3 = client.receive_weights(update_2.model)
    assert update_3 is None

    server.cleanup()
    client.cleanup()


def test_arrow_flight_server_debug_snapshot_reports_stored_bytes(sample_params):
    config = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    server, client = create_test_weight_transfer_pair(config)

    try:
        server.serve_weights(7, sample_params)

        debug_snapshot = server.get_debug_snapshot()
        latest_store = debug_snapshot["latest_store"]
        assert latest_store["latest_weight_id"] == 7
        assert latest_store["stored_param_count"] > 0
        assert latest_store["stored_record_batch_count"] > 0
        assert latest_store["stored_arrow_bytes"] > 0
        assert latest_store["flight_server_count"] > 0
    finally:
        server.cleanup()
        client.cleanup()


def test_arrow_flight_coordinator_accepts_rollback_weight_ids():
    from fray import current_client
    from marin.rl.weight_transfer.arrow_flight import ArrowFlightCoordinator

    client = current_client()
    coordinator = client.create_actor(
        ArrowFlightCoordinator,
        name=f"test_coordinator_{uuid.uuid4().hex[:8]}",
    )

    param_names = ["param"]
    first_server_locations = [("127.0.0.1", 5001)]
    rollback_server_locations = [("127.0.0.1", 5002)]

    coordinator.update_server.remote(1, param_names, first_server_locations).result()
    coordinator.update_server.remote(-1, param_names, rollback_server_locations).result()
    server_info = coordinator.fetch_server.remote().result()

    assert server_info.weight_id == -1
    assert server_info.server_addresses == ["grpc://127.0.0.1:5002"]


def test_concurrent_clients(weight_transfer_config, sample_params):
    server, client_1 = create_test_weight_transfer_pair(weight_transfer_config)

    client_2 = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=client_1.mesh,
        axis_mapping=client_1.axis_mapping,
        coordinator_handle=client_1._coordinator if hasattr(client_1, "_coordinator") else None,
    )

    try:
        # Serve weights
        server.serve_weights(1, sample_params)

        # Both clients should receive the same weights
        update_1 = client_1.receive_weights(sample_params)
        update_2 = client_2.receive_weights(sample_params)

        assert update_1 is not None
        assert update_2 is not None

        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(x, y),
            update_1.model,
            update_2.model,
        )

    finally:
        server.cleanup()
        client_1.cleanup()
        client_2.cleanup()


def test_arrow_flight_exports_and_tracks_bytes(sample_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            checkpoint_dir=temp_dir,
        )

        server, client = create_test_weight_transfer_pair(config)

        try:
            server.serve_weights(1, sample_params)
            update = client.receive_weights(sample_params)

            assert update is not None
            assert update.weight_id == 1

            server_metrics = server.get_metrics()
            assert server_metrics.transfer_bytes > 0
            assert server_metrics.param_count > 0
            assert server_metrics.materialize_time >= 0

            client_metrics = client.get_metrics()
            assert client_metrics["receive_bytes"] > 0
            assert client_metrics["param_count"] > 0
            assert client_metrics["largest_param_bytes"] > 0

        finally:
            server.cleanup()
            client.cleanup()


@pytest.mark.skip("Manual benchmark test")
def benchmark_arrow_flight_with_llama():
    """Test Arrow Flight weight transfer with a LLama 1B model."""
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

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


@pytest.mark.skip("Manual benchmark test")
@pytest.mark.slow("Uses real Llama model, requires HuggingFace access.")
def test_arrow_flight_transfer_to_vllm():
    """Test Arrow Flight weight transfer to vLLM."""
    from vllm import LLM

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
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


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    benchmark_arrow_flight_with_llama()
