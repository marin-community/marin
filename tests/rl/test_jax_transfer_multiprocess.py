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

"""
Multi-process tests for JAX transfer server weight transfer mode.
"""

import logging
import multiprocessing.pool
import os
import time
import uuid

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

logger = logging.getLogger(__name__)

if os.environ.get("CI"):
    pytest.skip("Skipping slow multiprocess tests on CI", allow_module_level=True)

try:
    from jax.experimental import transfer as jax_transfer

    from marin.rl.weight_transfer import (
        WeightTransferConfig,
        WeightTransferMode,
        create_weight_transfer_client,
        create_weight_transfer_server,
    )

    _ = jax_transfer  # Ensure we can access this module
except (ImportError, AttributeError):
    pytest.skip("Post training imports unavailable", allow_module_level=True)


def create_sample_pytree(seed: int):
    """Create a sample pytree for testing."""
    generator = np.random.Generator(np.random.PCG64(seed))

    # Create axes for NamedArrays
    Vocab = hax.Axis("vocab", 5)
    Hidden = hax.Axis("hidden", 5)

    return {
        "embedding": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((5, 5), dtype=jnp.float32)),
                (Vocab, Hidden),
            ),
        },
        "output": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((5, 5), dtype=jnp.float32)),
                (Hidden, Vocab),
            ),
        },
    }


def create_mesh():
    """Create a simple JAX mesh for testing."""
    devices = jax.local_devices()[:1]
    return jax.sharding.Mesh(np.array(devices), axis_names=("batch",))


def run_server(coordinator_name: str, num_processes: int, coordinator_address: str):
    assert os.environ["RAY_ADDRESS"]
    logger.info("Connecting to Ray at %s", os.environ.get("RAY_ADDRESS"))
    # Set up JAX environment before any JAX imports
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_DISABLE_JIT"] = "true"
    os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator_address
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_processes}"

    config = WeightTransferConfig(
        mode=WeightTransferMode.JAX_TRANSFER_SERVER,
        sync_interval_steps=1,
        poll_interval_seconds=0.1,
        coordinator_name=coordinator_name,
    )
    logger.info("Creating server with config: %s", config)

    mesh = create_mesh()
    server = create_weight_transfer_server(config, mesh=mesh)

    # ray takes a while to warm up...
    time.sleep(5)

    for i in range(10):
        params = create_sample_pytree(seed=i)
        server.serve_weights(i, params)
        logger.info("Switching to new weights.")
        time.sleep(1)

    # wait for clients to catch up
    time.sleep(5)
    server.cleanup()

    # jax.distributed.shutdown()
    return True


def run_client(coordinator_name: str, process_id: int, num_processes: int, coordinator_address: str):
    assert os.environ["RAY_ADDRESS"]
    logger.info("Connecting to Ray at %s", os.environ.get("RAY_ADDRESS"))
    # Set up JAX environment
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_DISABLE_JIT"] = "true"
    os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator_address
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_processes}"

    config = WeightTransferConfig(
        mode=WeightTransferMode.JAX_TRANSFER_SERVER,
        sync_interval_steps=1,
        poll_interval_seconds=0.1,
        transfer_timeout=10,
        coordinator_name=coordinator_name,
    )
    logger.info("Created client with config: %s", config)

    mesh = create_mesh()
    client = create_weight_transfer_client(config, mesh=mesh)

    # Receive weights
    logger.info("Receiving weights.")
    params_received = []
    placeholder = create_sample_pytree(seed=0)
    for _ in range(10):
        logger.info("Attempting to receive weights...")
        update = client.receive_weights(placeholder)
        logger.info("Received weights: %s", update)
        if update is not None:
            params_received.append(update.model)
        time.sleep(1)

    # count unique weights received
    unique_weights = set()
    for params in params_received:
        weight_hash = hash(str(jax.tree_util.tree_leaves(params)))
        unique_weights.add(weight_hash)

    logger.info("Received %d unique weight sets.", len(unique_weights))
    assert len(unique_weights) > 1, "Did not receive multiple unique weight sets."
    client.cleanup()
    return True


@pytest.mark.parametrize("num_clients", [1, 2])
def test_jax_transfer_multiprocess(ray_tpu_cluster, num_clients):
    num_processes = num_clients + 1  # 1 server + N clients
    coordinator_name = f"jax_coordinator_{uuid.uuid4().hex[:8]}"
    coordinator_address = f"localhost:{12321 + (uuid.uuid4().int % 1000)}"

    # ray multiprocessing doesn't work very well....
    with multiprocessing.pool.ThreadPool(processes=num_processes) as pool:
        server = pool.apply_async(run_server, args=(coordinator_name, num_processes, coordinator_address))
        clients = []
        for i in range(num_clients):
            client = pool.apply_async(run_client, args=(coordinator_name, i + 1, num_processes, coordinator_address))
            clients.append(client)

        server_result = server.get(timeout=60)
        assert server_result, "Server process failed"

        clients[0].get(timeout=60)

        for client in clients:
            client_result = client.get()
            assert client_result is True, f"Client process {clients.index(client)} failed"
    logger.info("All server and client processes completed successfully.")
