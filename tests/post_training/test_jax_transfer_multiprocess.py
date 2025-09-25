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

import multiprocessing
import os
import time
import uuid

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ray

try:
    from jax.experimental import transfer as jax_transfer

    from marin.post_training.weight_transfer import (
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
    Vocab = hax.Axis("vocab", 100)
    Hidden = hax.Axis("hidden", 64)

    return {
        "embedding": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((100, 64), dtype=jnp.float32)),
                (Vocab, Hidden),
            ),
        },
        "output": {
            "weight": hax.named(
                jnp.array(generator.standard_normal((64, 100), dtype=jnp.float32)),
                (Hidden, Vocab),
            ),
        },
    }


def create_mesh():
    """Create a simple JAX mesh for testing."""
    devices = jax.local_devices()[:1]
    return jax.sharding.Mesh(np.array(devices), axis_names=("batch",))


def run_server(coordinator_name: str, process_id: int, num_processes: int, coordinator_address: str):
    """Run server process."""
    try:
        # Set up JAX environment before any JAX imports
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_DISABLE_JIT"] = "true"
        os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator_address
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_processes}"

        # Skip JAX distributed for this test - focus on weight transfer coordination

        # Connect to Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)

        config = WeightTransferConfig(
            mode=WeightTransferMode.JAX_TRANSFER_SERVER,
            sync_interval_steps=1,
            poll_interval_seconds=0.1,
            coordinator_name=coordinator_name,
        )

        mesh = create_mesh()
        server = create_weight_transfer_server(config, mesh=mesh)

        # Create and serve weights
        params = create_sample_pytree(seed=42)
        server.serve_weights(1, params)

        # Wait for clients
        time.sleep(3)

        # Serve updated weights
        new_params = create_sample_pytree(seed=123)
        server.serve_weights(2, new_params)

        time.sleep(2)
        server.cleanup()

        # jax.distributed.shutdown()
        return True

    except Exception as e:
        print(f"Server process error: {e}")
        return False


def run_client(coordinator_name: str, process_id: int, num_processes: int, coordinator_address: str):
    """Run client process."""
    try:
        # Set up JAX environment
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_DISABLE_JIT"] = "true"
        os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator_address
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_processes}"

        ray.init(address="auto", ignore_reinit_error=True)

        # Wait for server to initialize
        time.sleep(1)

        config = WeightTransferConfig(
            mode=WeightTransferMode.JAX_TRANSFER_SERVER,
            sync_interval_steps=1,
            poll_interval_seconds=0.1,
            coordinator_name=coordinator_name,
        )

        mesh = create_mesh()
        client = create_weight_transfer_client(config, mesh=mesh)

        # Receive weights
        placeholder = create_sample_pytree(seed=0)
        received_params = client.receive_weights(placeholder)

        if received_params is None:
            print(f"Client {process_id} failed to receive weights")
            return False

        # Try to receive updated weights
        time.sleep(2)
        received_params_2 = client.receive_weights(received_params)

        client.cleanup()
        # jax.distributed.shutdown()

        return received_params_2 is not None

    except Exception as e:
        print(f"Client process {process_id} error: {e}")
        return False


@pytest.mark.parametrize("num_clients", [1, 2, 3])
def test_jax_transfer_multiprocess(ray_tpu_cluster, num_clients):
    num_processes = num_clients + 1  # 1 server + N clients
    coordinator_name = f"jax_coordinator_{uuid.uuid4().hex[:8]}"
    coordinator_address = f"localhost:{12321 + (uuid.uuid4().int % 1000)}"

    # Create server process
    server_process = multiprocessing.Process(
        target=run_server, args=(coordinator_name, 0, num_processes, coordinator_address)
    )

    # Create client processes
    client_processes = []
    for i in range(num_clients):
        client_process = multiprocessing.Process(
            target=run_client, args=(coordinator_name, i + 1, num_processes, coordinator_address)
        )
        client_processes.append(client_process)

    all_processes = [server_process, *client_processes]

    try:
        # Start server first
        server_process.start()
        time.sleep(1)  # Give server a head start

        # Start clients with slight delays
        for i, client_process in enumerate(client_processes):
            client_process.start()
            if i < len(client_processes) - 1:  # Don't sleep after the last client
                time.sleep(0.5)

        # Wait for completion
        for process in all_processes:
            process.join(timeout=30)

        # Check results
        for i, process in enumerate(all_processes):
            if process.exitcode != 0:
                process_name = "server" if i == 0 else f"client{i}"
                pytest.fail(f"{process_name} process failed with exit code {process.exitcode}")

    finally:
        # Clean up processes
        for process in all_processes:
            process.kill()
