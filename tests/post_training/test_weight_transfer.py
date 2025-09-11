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

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ray
from jax.sharding import Mesh

try:
    from marin.post_training.training_config import WeightTransferConfig, WeightTransferMode
    from marin.post_training.weight_transfer_manager import (
        RayRemotingClient,
        RayRemotingServer,
        RayWeightCoordinator,
        create_weight_transfer_client,
        create_weight_transfer_server,
    )
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)


def create_sample_pytree(seed: int):
    """Create a sample JAX pytree representing model parameters."""
    generator = np.random.Generator(np.random.PCG64(seed))
    return {
        "embedding": {
            "weight": jnp.array(generator.standard_normal((1000, 512), dtype=jnp.float32)),
        },
        "layers": {
            "0": {
                "attention": {
                    "query": {"weight": jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32))},
                    "key": {"weight": jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32))},
                    "value": {"weight": jnp.array(generator.standard_normal((512, 512), dtype=jnp.float32))},
                },
                "feed_forward": {
                    "linear1": {"weight": jnp.array(generator.standard_normal((512, 2048), dtype=jnp.float32))},
                    "linear2": {"weight": jnp.array(generator.standard_normal((2048, 512), dtype=jnp.float32))},
                },
            },
        },
        "output": {
            "weight": jnp.array(generator.standard_normal((512, 1000), dtype=jnp.float32)),
        },
    }


def create_mesh(devices=None):
    """Create a simple JAX mesh for testing."""
    if devices is None:
        devices = jax.local_devices()[:1]  # Use just one device for tests
    return Mesh(np.array(devices), axis_names=("batch",))


@pytest.fixture(scope="module")
def ray_cluster():
    """Start a Ray cluster with 2 workers for testing."""
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(
    params=[
        WeightTransferMode.RAY_REMOTING,
        WeightTransferMode.GCS_CHECKPOINT,
    ]
)
def transfer_mode(request):
    """Parametrized weight transfer mode."""
    return request.param


@pytest.fixture
def sample_params():
    """Generate sample JAX parameters for testing."""
    return create_sample_pytree(seed=42)


@pytest.fixture
def weight_transfer_config(transfer_mode):
    """Create weight transfer config for the specified mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield WeightTransferConfig(
            mode=transfer_mode,
            coordinator_name="test_weight_coordinator",
            sync_interval_steps=1,
            poll_interval_seconds=0.1,
            checkpoint_dir=temp_dir,
        )


@pytest.fixture
def weight_transfer_server_client(weight_transfer_config, sample_params):
    """Create weight transfer server and client pair."""
    # For GCS checkpoints, we need some additional setup
    if weight_transfer_config.mode == WeightTransferMode.GCS_CHECKPOINT:
        # we just need something with a to_dict() method
        class DummyConfig:
            def to_dict(self):
                return {}

        # Create mock gather_fns with params attribute
        class MockGatherFns:
            def __init__(self):
                self.params = None

        server = create_weight_transfer_server(
            config=weight_transfer_config, gather_fns=MockGatherFns(), model_config=DummyConfig()
        )

        # For GCS checkpoints, we need to provide a load function
        def mock_load_checkpoint_fn(path):
            # Mock load function that returns the sample params
            return sample_params

        client = create_weight_transfer_client(config=weight_transfer_config, load_checkpoint_fn=mock_load_checkpoint_fn)
    else:
        # Ray remoting and JAX transfer modes
        server = create_weight_transfer_server(config=weight_transfer_config)

        # Create identity shard functions for testing
        def identity_shard(x):
            return x

        shard_fns = jax.tree.map(lambda x: identity_shard, sample_params)
        client = create_weight_transfer_client(
            config=weight_transfer_config, shard_fns=shard_fns, load_checkpoint_fn=None
        )

    yield server, client

    # Cleanup
    server.cleanup()
    client.cleanup()


# Ray Weight Coordinator Tests (Ray remoting specific)


def test_ray_coordinator_basic_storage_retrieval(ray_cluster, sample_params):
    """Test basic weight storage and retrieval."""
    coordinator = RayWeightCoordinator.remote()

    # Convert to numpy for storage
    numpy_params = jax.tree.map(lambda x: np.array(x), sample_params)

    # Store weights
    ray.get(coordinator.put_weights.remote(1, numpy_params))

    # Retrieve weights
    weight_ref, weight_id = ray.get(coordinator.get_latest_weights.remote())
    assert weight_id == 1
    assert weight_ref is not None

    # Get actual weights
    retrieved_params = ray.get(weight_ref)

    # Verify structure matches
    assert retrieved_params.keys() == sample_params.keys()
    assert retrieved_params["embedding"]["weight"].shape == sample_params["embedding"]["weight"].shape


def test_ray_coordinator_version_ordering(ray_cluster, sample_params):
    """Test that newer weights replace older ones."""
    coordinator = RayWeightCoordinator.remote()

    # Store weight version 1
    numpy_params_1 = jax.tree.map(lambda x: np.array(x), sample_params)
    ray.get(coordinator.put_weights.remote(1, numpy_params_1))

    # Store weight version 3 (should replace version 1)
    new_params = create_sample_pytree(seed=123)  # Different seed
    numpy_params_3 = jax.tree.map(lambda x: np.array(x), new_params)
    ray.get(coordinator.put_weights.remote(3, numpy_params_3))

    # Store weight version 2 (should be ignored since 3 > 2)
    ray.get(coordinator.put_weights.remote(2, numpy_params_1))

    # Should get version 3
    weight_ref, weight_id = ray.get(coordinator.get_latest_weights.remote())
    assert weight_id == 3

    retrieved_params = ray.get(weight_ref)
    # Verify it's version 3 by checking it's different from original
    assert not np.array_equal(retrieved_params["embedding"]["weight"], numpy_params_1["embedding"]["weight"])


def test_ray_coordinator_no_weights_initially(ray_cluster):
    """Test coordinator returns None when no weights stored."""
    coordinator = RayWeightCoordinator.remote()

    weight_ref, weight_id = ray.get(coordinator.get_latest_weights.remote())
    assert weight_ref is None
    assert weight_id is None


def test_basic_weight_transfer(ray_cluster, weight_transfer_server_client, sample_params, weight_transfer_config):
    """Test basic weight transfer from server to client."""
    server, client = weight_transfer_server_client

    # Serve weights
    server.serve_weights(1, sample_params)

    # Receive weights
    received_params, metadata = client.receive_weights()

    assert received_params is not None
    assert metadata["weight_id"] == 1

    # Source varies by transfer mode
    expected_sources = {
        WeightTransferMode.RAY_REMOTING: "ray_remoting",
        WeightTransferMode.GCS_CHECKPOINT: "gcs_checkpoint",
    }
    assert metadata["source"] == expected_sources[weight_transfer_config.mode]

    # Verify structure and values match
    assert received_params.keys() == sample_params.keys()

    # For GCS checkpoints, there may be precision loss due to bfloat16 conversion
    if weight_transfer_config.mode == WeightTransferMode.GCS_CHECKPOINT:
        np.testing.assert_allclose(
            received_params["embedding"]["weight"], sample_params["embedding"]["weight"], rtol=1e-2
        )
    else:
        np.testing.assert_array_equal(received_params["embedding"]["weight"], sample_params["embedding"]["weight"])


def test_multiple_weight_updates(ray_cluster, weight_transfer_server_client, sample_params):
    """Test multiple sequential weight updates."""
    server, client = weight_transfer_server_client

    # First weight transfer
    server.serve_weights(1, sample_params)
    received_params_1, metadata_1 = client.receive_weights()
    assert metadata_1["weight_id"] == 1

    # Second weight transfer with new params
    new_params = create_sample_pytree(seed=456)  # Different seed
    server.serve_weights(2, new_params)
    received_params_2, metadata_2 = client.receive_weights()
    assert metadata_2["weight_id"] == 2

    # Verify weights are different
    assert not np.array_equal(received_params_1["embedding"]["weight"], received_params_2["embedding"]["weight"])

    # Third call should return None (no new weights)
    received_params_3, metadata_3 = client.receive_weights()
    assert received_params_3 is None
    assert metadata_3 == {}


def test_client_no_new_weights(ray_cluster, weight_transfer_server_client, sample_params):
    """Test client behavior when no new weights are available."""
    server, client = weight_transfer_server_client

    # Serve weights
    server.serve_weights(1, sample_params)

    # First receive should get weights
    received_params_1, metadata_1 = client.receive_weights()
    assert received_params_1 is not None
    assert metadata_1["weight_id"] == 1

    # Second receive should return None (no new weights)
    received_params_2, metadata_2 = client.receive_weights()
    assert received_params_2 is None
    assert metadata_2 == {}


def test_concurrent_clients(ray_cluster, weight_transfer_config, sample_params):
    """Test multiple clients receiving weights concurrently (Ray remoting only)."""
    # Create mock gather_fns for GCS checkpoint mode
    if weight_transfer_config.mode == WeightTransferMode.GCS_CHECKPOINT:

        class MockGatherFns:
            def __init__(self):
                self.params = None

        class DummyConfig:
            def to_dict(self):
                return {}

        server = create_weight_transfer_server(
            config=weight_transfer_config, gather_fns=MockGatherFns(), model_config=DummyConfig()
        )
    else:
        server = create_weight_transfer_server(config=weight_transfer_config)

    # Create multiple clients with same coordinator
    def mock_load_checkpoint_fn(path):
        return sample_params

    load_fn = mock_load_checkpoint_fn if weight_transfer_config.mode == WeightTransferMode.GCS_CHECKPOINT else None

    # Create identity shard functions for testing Ray remoting
    def identity_shard(x):
        return x

    shard_fns = (
        jax.tree.map(lambda x: identity_shard, sample_params)
        if weight_transfer_config.mode == WeightTransferMode.RAY_REMOTING
        else None
    )

    client_1 = create_weight_transfer_client(
        config=weight_transfer_config, shard_fns=shard_fns, load_checkpoint_fn=load_fn
    )
    client_2 = create_weight_transfer_client(
        config=weight_transfer_config, shard_fns=shard_fns, load_checkpoint_fn=load_fn
    )

    try:
        # Serve weights
        server.serve_weights(1, sample_params)

        # Both clients should receive the same weights
        received_params_1, metadata_1 = client_1.receive_weights()
        received_params_2, metadata_2 = client_2.receive_weights()

        assert received_params_1 is not None
        assert received_params_2 is not None
        assert metadata_1["weight_id"] == 1
        assert metadata_2["weight_id"] == 1

        # Verify weights are identical
        np.testing.assert_array_equal(received_params_1["embedding"]["weight"], received_params_2["embedding"]["weight"])
    finally:
        server.cleanup()
        client_1.cleanup()
        client_2.cleanup()


def test_with_mesh_sharding(ray_cluster, weight_transfer_config, sample_params):
    """Test weight transfer with JAX mesh sharding."""
    mesh = create_mesh()

    # Define simple sharding rules
    params_sharding_rules = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), sample_params
    )

    # Create server first to set up coordinator
    server = RayRemotingServer(weight_transfer_config)

    # Create client with mesh and sharding
    client = RayRemotingClient(
        weight_transfer_config,
        mesh=mesh,
        params_sharding_rules=params_sharding_rules,
    )

    # Serve and receive weights
    server.serve_weights(1, sample_params)
    received_params, metadata = client.receive_weights()

    assert received_params is not None
    assert metadata["weight_id"] == 1

    # Verify params are properly sharded (should still have same values)
    np.testing.assert_array_equal(received_params["embedding"]["weight"], sample_params["embedding"]["weight"])

    server.cleanup()
    client.cleanup()


def test_jax_numpy_conversion(ray_cluster, weight_transfer_server_client):
    """Test JAX array to numpy conversion and back."""
    server, client = weight_transfer_server_client

    # Create JAX arrays with specific values for testing
    jax_params = {
        "weight": jnp.array([1.0, 2.0, 3.0, 4.0]),
        "bias": jnp.array([0.1, 0.2]),
    }

    # Transfer weights
    server.serve_weights(1, jax_params)
    received_params, metadata = client.receive_weights()

    # Verify they are JAX arrays (all clients should return JAX arrays)
    assert isinstance(received_params["weight"], jax.Array)
    assert isinstance(received_params["bias"], jax.Array)

    # For GCS checkpoints, account for precision loss due to bfloat16 conversion
    if metadata["source"] == "gcs_checkpoint":
        np.testing.assert_allclose(received_params["weight"], jax_params["weight"], rtol=1e-2)
        np.testing.assert_allclose(received_params["bias"], jax_params["bias"], rtol=1e-2)
    else:
        # Verify conversion preserved values and dtypes
        np.testing.assert_array_equal(received_params["weight"], jax_params["weight"])
        np.testing.assert_array_equal(received_params["bias"], jax_params["bias"])


def test_cleanup(ray_cluster, weight_transfer_server_client, sample_params):
    """Test proper cleanup of server and client resources."""
    server, client = weight_transfer_server_client

    # Do a basic transfer
    server.serve_weights(1, sample_params)
    received_params, metadata = client.receive_weights()
    assert received_params is not None

    # Note: cleanup is handled by the fixture
    # Verify cleanup was successful (no specific assertions needed,
    # just that no exceptions were raised)
    assert True


def test_empty_params(ray_cluster, weight_transfer_server_client):
    """Test behavior with empty parameter trees."""
    server, client = weight_transfer_server_client
    empty_params = {}

    server.serve_weights(1, empty_params)
    received_params, metadata = client.receive_weights()

    assert received_params == {}
    assert metadata["weight_id"] == 1
