# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import uuid

import equinox as eqx
import haliax as hax
import jax
import numpy as np
import pytest
import ray
from jax.sharding import Mesh

from marin.rl.weight_transfer import (
    WeightTransferConfig,
    WeightTransferMode,
    create_weight_transfer_client,
    create_weight_transfer_server,
)

try:
    import jax.experimental.transfer

    TRANSFER_TYPES = [
        WeightTransferMode.GCS_CHECKPOINT,
        WeightTransferMode.JAX_TRANSFER_SERVER,
        WeightTransferMode.ARROW_FLIGHT,
    ]
except (ImportError, AttributeError):
    TRANSFER_TYPES = [
        WeightTransferMode.GCS_CHECKPOINT,
        WeightTransferMode.ARROW_FLIGHT,
    ]

if os.environ.get("CI"):
    pytest.skip("Skipping slow tests on CI", allow_module_level=True)


class TestModule(eqx.Module):
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
    return TestModule(
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

    return server, client


@pytest.fixture
def weight_transfer_config(transfer_mode):
    """Create weight transfer config for the specified mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = WeightTransferConfig(
            mode=transfer_mode,
            sync_interval_steps=1,
            poll_interval_seconds=0.1,
            checkpoint_dir=temp_dir,
        )
        yield config


def test_multiple_weight_updates(ray_tpu_cluster, weight_transfer_config, sample_params):
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

    jax.tree.map(
        lambda x, y: np.testing.assert_array_equal(x, y),
        update_2.model,
        new_params,
    )

    # Third call should return None (no new weights)
    update_3 = client.receive_weights(update_2.model)
    assert update_3 is None

    server.cleanup()
    client.cleanup()


def test_concurrent_clients(ray_tpu_cluster, weight_transfer_config, sample_params):
    server, client_1 = create_test_weight_transfer_pair(weight_transfer_config)

    client_2 = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=client_1.mesh,
        axis_mapping=client_1.axis_mapping,
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


@pytest.mark.slow("Uses a large buffer, can OOM on CI.")
def test_arrow_flight_with_large_buffer(ray_tpu_cluster):
    """Test Arrow Flight weight transfer with large buffer sizes."""

    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        poll_interval_seconds=0.1,
        checkpoint_dir=tempfile.mkdtemp(),
    )

    server, client = create_test_weight_transfer_pair(weight_transfer_config)
    # 5k * 5k * 4bytes = ~100MB per layer, 40 layers = ~4GB total
    large_params = create_sample_pytree(seed=789, hidden_size=1000, layers=40)

    # put the params on the TPU to test real transfer, shard across the devices
    print("Creating mesh for large params transfer...")
    # mesh = create_mesh(devices=jax.devices("tpu"))
    mesh = create_mesh(devices=jax.local_devices())
    with mesh:
        large_params = jax.device_put(large_params)
    print("Mesh created with devices:", mesh.devices)

    try:
        for i in range(10):
            print(i)
            server.serve_weights(i, large_params)
            update = client.receive_weights(large_params)

            assert update is not None
            assert isinstance(update.model, eqx.Module)

            print(client.get_metrics())

        # walk the pytree and verify all arrays match
        def assert_arrays_equal(x, y):
            np.testing.assert_array_equal(x, y)

        jax.tree.map(assert_arrays_equal, large_params, update.model)
    finally:
        server.cleanup()
        client.cleanup()


if __name__ == "__main__":
    # log to stderr
    import logging
    import sys

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cluster = ray.init("local")
    test_arrow_flight_with_large_buffer(ray_tpu_cluster=cluster)
