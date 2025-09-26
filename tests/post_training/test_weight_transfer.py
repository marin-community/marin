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

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ray
from jax.sharding import Mesh

from marin.post_training.weight_transfer import (
    RayWeightCoordinator,
    WeightTransferConfig,
    WeightTransferMode,
    create_weight_transfer_client,
    create_weight_transfer_server,
)

try:
    import jax.experimental.transfer

    TRANSFER_TYPES = [
        WeightTransferMode.RAY_REMOTING,
        WeightTransferMode.GCS_CHECKPOINT,
        WeightTransferMode.JAX_TRANSFER_SERVER,
        WeightTransferMode.ARROW_FLIGHT,
    ]
except (ImportError, AttributeError):
    TRANSFER_TYPES = [
        WeightTransferMode.RAY_REMOTING,
        WeightTransferMode.GCS_CHECKPOINT,
        WeightTransferMode.ARROW_FLIGHT,
    ]

if os.environ.get("CI"):
    pytest.skip("Skipping slow tests on CI", allow_module_level=True)


def create_sample_pytree(seed: int):
    """Create a sample pytree representing Levanter model parameters."""
    generator = np.random.Generator(np.random.PCG64(seed))

    # Create axes for NamedArrays
    Vocab = hax.Axis("vocab", 1000)
    Hidden = hax.Axis("hidden", 512)
    HiddenOut = hax.Axis("hidden_out", 512)
    FF = hax.Axis("ff", 2048)

    return {
        "embedding": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((1000, 512), dtype=jnp.float32)),
                (Vocab, Hidden),
            ),
        },
        "layers": {
            "0": {
                "attention": {
                    "query": {
                        "weight": hax.named(
                            jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32)),
                            (Hidden, HiddenOut),
                        )
                    },
                    "key": {
                        "weight": hax.named(
                            jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32)),
                            (Hidden, HiddenOut),
                        )
                    },
                    "value": {
                        "weight": hax.named(
                            jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32)),
                            (Hidden, HiddenOut),
                        )
                    },
                },
                "feed_forward": {
                    "linear1": {
                        "weight": hax.named(
                            jnp.array(generator.standard_normal((512, 2048), dtype=jnp.float32)),
                            (Hidden, FF),
                        )
                    },
                    "linear2": {
                        "weight": hax.named(
                            jnp.array(generator.standard_normal((2048, 512), dtype=jnp.float32)),
                            (FF, Hidden),
                        )
                    },
                },
            },
        },
        "output": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((512, 1000), dtype=jnp.float32)),
                (Hidden, Vocab),
            ),
        },
    }


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
    if weight_transfer_config.mode in [
        WeightTransferMode.RAY_REMOTING,
        WeightTransferMode.JAX_TRANSFER_SERVER,
    ]:
        coordinator_name = f"test_coordinator_{uuid.uuid4().hex[:8]}"
        weight_transfer_config.coordinator_name = coordinator_name

    # Create simple mesh and axis mapping for testing
    mesh = create_mesh()
    axis_mapping = None  # Use default Levanter sharding

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


def test_ray_coordinator_no_weights_initially(ray_tpu_cluster):
    """Test coordinator returns None when no weights stored."""
    coordinator = RayWeightCoordinator.remote()

    weight_refs, weight_id = ray.get(coordinator.get_latest_weight_refs.remote())
    assert weight_refs is None
    assert weight_id is None


def test_multiple_weight_updates(ray_tpu_cluster, weight_transfer_config, sample_params):
    """Test multiple sequential weight updates."""
    server, client = create_test_weight_transfer_pair(weight_transfer_config)

    # First weight transfer
    server.serve_weights(1, sample_params)
    received_params_1 = client.receive_weights(sample_params)
    assert received_params_1 is not None

    # Second weight transfer with new params
    new_params = create_sample_pytree(seed=456)  # Different seed
    server.serve_weights(2, new_params)
    received_params_2 = client.receive_weights(received_params_1)
    assert received_params_2 is not None

    # Verify weights are different
    assert not np.array_equal(
        received_params_1["embedding"]["weight"].array,
        received_params_2["embedding"]["weight"].array,
    )

    # Third call should return None (no new weights)
    received_params_3 = client.receive_weights(received_params_2)
    assert received_params_3 is None

    # Cleanup
    server.cleanup()
    client.cleanup()


def test_client_no_new_weights(ray_tpu_cluster, weight_transfer_config, sample_params):
    """Test client behavior when no new weights are available."""
    server, client = create_test_weight_transfer_pair(weight_transfer_config)

    # Serve weights
    server.serve_weights(1, sample_params)

    # First receive should get weights
    received_params_1 = client.receive_weights(sample_params)
    assert received_params_1 is not None

    # Second receive should return None (no new weights)
    received_params_2 = client.receive_weights(received_params_1)
    assert received_params_2 is None

    # Cleanup
    server.cleanup()
    client.cleanup()


def test_concurrent_clients(ray_tpu_cluster, weight_transfer_config, sample_params):
    """Test multiple clients receiving weights concurrently (Ray remoting only)."""

    server, client_1 = create_test_weight_transfer_pair(weight_transfer_config)

    mesh = create_mesh()
    client_2 = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping=None,
    )

    try:
        # Serve weights
        server.serve_weights(1, sample_params)

        # Both clients should receive the same weights
        received_params_1 = client_1.receive_weights(sample_params)
        received_params_2 = client_2.receive_weights(sample_params)

        assert received_params_1 is not None
        assert received_params_2 is not None

        # Verify weights are identical
        np.testing.assert_array_equal(
            received_params_1["embedding"]["weight"].array,
            received_params_2["embedding"]["weight"].array,
        )
    finally:
        server.cleanup()
        client_1.cleanup()
        client_2.cleanup()


def test_with_mesh_sharding(ray_tpu_cluster, weight_transfer_config, sample_params):
    """Test weight transfer with JAX mesh sharding."""
    mesh = create_mesh()

    # Define simple sharding rules
    params_sharding_rules = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), sample_params
    )

    # Create shard functions from sharding rules
    def create_shard_fn(sharding):
        return lambda x: jax.device_put(x, sharding)

    shard_fns = jax.tree.map(create_shard_fn, params_sharding_rules)

    server, client = create_test_weight_transfer_pair(weight_transfer_config)

    # For Ray remoting mode, update the client's shard functions
    if weight_transfer_config.mode == WeightTransferMode.RAY_REMOTING:
        client.shard_fns = shard_fns

    try:
        # Serve and receive weights
        server.serve_weights(1, sample_params)
        received_params = client.receive_weights(sample_params)

        assert received_params is not None

        # For GCS checkpoints, there may be precision loss due to bfloat16 conversion
        if weight_transfer_config.mode == WeightTransferMode.GCS_CHECKPOINT:
            np.testing.assert_allclose(
                received_params["embedding"]["weight"].array,
                sample_params["embedding"]["weight"].array,
                rtol=1e-2,
            )
        else:
            # Verify params are properly sharded (should still have same values)
            np.testing.assert_array_equal(
                received_params["embedding"]["weight"].array,
                sample_params["embedding"]["weight"].array,
            )
    finally:
        server.cleanup()
        client.cleanup()
