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

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import pytest
import ray
from fray.cluster.ray.tpu import run_on_pod
from ray.exceptions import RayTaskError

# Mesh axis names
DATA_AXIS = "data"
MODEL_AXIS = "model"


@pytest.fixture(scope="module", autouse=True)
def setup_ray_tpu_tests():
    """Start a dedicated Ray TPU cluster for testing."""

    # Add the tests directory to sys.path so Ray workers can import test modules
    # This is needed because Ray pickles objects that reference the test module,
    # and workers need to be able to import it for deserialization
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{tests_dir}:{current_pythonpath}"

    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_SCHEDULER_EVENTS"] = "0"

    tpu_resource_name = "TPU-v5litepod-4-head"
    chip_count = 4

    print(f"Starting TPU Ray cluster with resources TPU:{chip_count}, {tpu_resource_name}:1")
    ctx = ray.init(
        address="local",
        namespace="fray",
        resources={"TPU": chip_count, tpu_resource_name: 1, "head_node": 1},
        num_cpus=120,
        ignore_reinit_error=True,
    )

    # update environment variable to pass Ray address to subprocesses
    os.environ["RAY_ADDRESS"] = ctx.address_info["address"]
    if ctx.address_info["webui_url"] is not None:
        os.environ["RAY_DASHBOARD_URL"] = ctx.address_info["webui_url"]
    os.environ["RAY_API_SERVER_ADDRESS"] = ctx.address_info["gcs_address"]

    print(
        f"Initialized ray with address={ctx.address_info['address']}",
        f"webui_url={ctx.address_info['webui_url']}, gcs_address={ctx.address_info['gcs_address']}",
    )

    yield

    ray.shutdown(_exiting_interpreter=True)


def simple_jax_fn():
    # Import JAX only inside the remote function to avoid claiming TPU in main process
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from jax.lax import with_sharding_constraint
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    jax.devices()

    try:
        if jax.default_backend() != "tpu":
            print(f"Warning: JAX default backend is {jax.default_backend()}, not TPU.")

        devices = jax.devices("tpu")
        if not devices:
            raise RuntimeError("No JAX TPU devices found on the worker.")

        # Create mesh directly without context manager
        if len(devices) % 1 != 0:
            raise ValueError(f"Cannot create mesh with tensor_parallelism=1 using {len(devices)} devices")

        mesh_devices = np.array(devices).reshape(-1, 1)
        mesh = Mesh(mesh_devices, (DATA_AXIS, MODEL_AXIS))

    except Exception as e:
        print(f"Error during JAX device/mesh setup on worker: {e}")
        raise RuntimeError(f"JAX TPU initialization failed on worker: {e}") from e

    key_x, key_weights, key_bias = jrandom.split(jrandom.PRNGKey(0), 3)

    # Define array dimensions
    dim_in = 8  # factor of num_tpus_per_host usually
    dim_out = 4

    with mesh:
        x = jrandom.normal(key_x, (dim_in,))
        weights = jrandom.normal(key_weights, (dim_in, dim_out))
        bias = jrandom.normal(key_bias, (dim_out,))

        # Shard inputs
        x_sharded = with_sharding_constraint(x, P(DATA_AXIS))
        weights_sharded = with_sharding_constraint(weights, P(DATA_AXIS))
        bias_sharded = with_sharding_constraint(bias, P())

        @jax.jit
        def layer(x_arg, weights_arg, bias_arg):
            return with_sharding_constraint(jnp.dot(x_arg, weights_arg) + bias_arg, P())

        output = layer(x_sharded, weights_sharded, bias_sharded)

    return np.array(output)


def simple_jax_workload():
    return simple_jax_fn()


@ray.remote
class CounterActor:
    def __init__(self):
        self._count = 0

    def increment(self) -> None:
        self._count += 1

    def count(self) -> int:
        return self._count


@pytest.mark.tpu_ci
def test_single_slice_simple_run():
    """1. Run a simple function on a single slice and verify it runs correctly."""
    num_slices = 1
    results = run_on_pod(simple_jax_workload, "v5litepod-4", num_slices=num_slices)

    assert results is not None
    assert len(results) == num_slices

    # For `num_slices=1` with "v5litepod-4" (1 host per slice):
    assert len(results) == 1  # One result because one host in total for one v5litepod-4 slice.
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (4,)  # Based on simple_jax_fn's output dim_out

    # Verify a second run works
    results_2 = run_on_pod(simple_jax_workload, "v5litepod-4", num_slices=num_slices)
    assert len(results_2) == 1
    assert isinstance(results_2[0], np.ndarray)
    assert np.array_equal(results[0], results_2[0])  # Deterministic function


@pytest.mark.tpu_ci
def test_single_slice_run_twice():
    """2. Run a second function after the first one and verify it runs correctly."""
    num_slices = 1
    # First run
    results1 = run_on_pod(simple_jax_workload, "v5litepod-4", num_slices=num_slices)
    assert len(results1) == 1
    assert isinstance(results1[0], np.ndarray)
    assert results1[0].shape == (4,)

    # Second run
    results2 = run_on_pod(simple_jax_workload, "v5litepod-4", num_slices=num_slices)
    assert len(results2) == 1
    assert isinstance(results2[0], np.ndarray)
    assert results2[0].shape == (4,)

    # Check if results are the same (since PRNGKey is fixed)
    assert np.array_equal(results1[0], results2[0])


@pytest.mark.tpu_ci
@pytest.mark.skip("Seems to claim TPU on first failure attempt")
def test_single_slice_fail_once():
    """1. Run a simple function on a single slice and verify it runs correctly."""
    counter_actor = CounterActor.remote()

    def fail_once_jax_fn() -> None:
        # Check if we should fail BEFORE initializing JAX/TPU
        # This avoids claiming the TPU on the first attempt
        count = ray.get(counter_actor.count.remote())
        ray.get(counter_actor.increment.remote())
        if count == 0:
            raise DeliberatelyRaisedException(f"Failing deliberately because count is {count}")

        # Only do JAX work after we know we won't fail
        result = simple_jax_fn()
        return result

    num_slices = 1
    results = run_on_pod(fail_once_jax_fn, "v5litepod-4", num_slices=num_slices, max_retries_failure=1)

    assert results is not None
    assert len(results) == num_slices

    # For `num_slices=1` with "v5litepod-4" (1 host per slice):
    assert len(results) == 1  # One result because one host in total for one v5litepod-4 slice.
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (4,)  # Based on simple_jax_fn's output dim_out

    # Verify a second run works
    results_2 = run_on_pod(simple_jax_workload, "v5litepod-4", num_slices=num_slices)
    assert len(results_2) == 1
    assert isinstance(results_2[0], np.ndarray)
    assert np.array_equal(results[0], results_2[0])  # Deterministic function


# --- Multislice Tests ---


@pytest.mark.tpu_ci
@pytest.mark.skip("Multislice not available in CI")
def test_multislice_simple_run():
    """1. Run a simple function on a multislice and verify it runs correctly."""
    num_slices = 2
    tpu_type = "v5litepod-4"  # Each slice is a v5litepod-4

    results = run_on_pod(simple_jax_workload, tpu_type, num_slices=num_slices)

    # run_on_pod_new returns a flat list of results from all hosts across all slices.
    # If each v5litepod-4 slice has 1 host (as per TPU-v5litepod-4-head resource meaning),
    # then for num_slices=2, we expect 2 results in the list.
    assert results is not None
    assert len(results) == num_slices  # num_slices * hosts_per_slice (assuming 1 host per v5litepod-4 slice)

    for i in range(num_slices):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            # Due to MEGASCALE_SLICE_ID, the PRNG key might differ effectively if the code used it.
            # simple_jax_fn uses a fixed PRNGKey(0) so all slices should produce identical results.
            assert np.array_equal(results[i], results[0])


@pytest.mark.tpu_ci
@pytest.mark.skip("Multislice not available in CI")
def test_variable_multislice_run():
    """1. Run a simple function on a multislice and verify it runs correctly."""
    num_slices = [1, 2]
    tpu_type = "v5litepod-4"  # Each slice is a v5litepod-4

    results = run_on_pod(simple_jax_fn, tpu_type, num_slices=num_slices)

    assert results is not None
    assert len(results) in num_slices  # num_slices * hosts_per_slice (assuming 1 host per v5litepod-4 slice)

    for i in range(len(results)):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            assert np.array_equal(results[i], results[0])


@pytest.mark.tpu_ci
@pytest.mark.skip("Multislice not available in CI")
def test_multislice_run_twice():
    """2. Run a second function after the first one and verify it runs correctly."""
    num_slices = 2
    tpu_type = "v5litepod-4"

    # First run
    results1 = run_on_pod(simple_jax_workload, tpu_type, num_slices=num_slices)
    assert len(results1) == num_slices
    for i in range(num_slices):
        assert isinstance(results1[i], np.ndarray)
        assert np.array_equal(results1[i], results1[0])  # All slices should be same

    # Second run
    results2 = run_on_pod(simple_jax_workload, tpu_type, num_slices=num_slices)
    assert len(results2) == num_slices
    for i in range(num_slices):
        assert isinstance(results2[i], np.ndarray)
        assert np.array_equal(results2[i], results2[0])

    # Compare first and second run (should be identical)
    for i in range(num_slices):
        assert np.array_equal(results1[i], results2[i])


@pytest.mark.tpu_ci
@pytest.mark.skip("Multislice not available in CI")
def test_multislice_fail_once():
    """Run a simple function on two slices and verify it runs correctly
    when the first slice will fail on the first run."""
    # NOTE: This is currently causing a TPU initialization failure:
    # https://gist.github.com/yifanmai/88c7d56f31c2558ee79cd45b97ad5de0
    num_slices = 2
    counter_actor = CounterActor.remote()

    def fail_once_on_first_slice_jax_fn() -> None:
        import time

        # do JAX work first
        result = simple_jax_fn()
        # fail on the first run one the first slice
        slice_id_str = os.getenv("MEGASCALE_SLICE_ID")
        if slice_id_str == "0":
            count = ray.get(counter_actor.count.remote())
            ray.get(counter_actor.increment.remote())
            if count == 0:
                raise DeliberatelyRaisedException(f"Failing deliberately because count is {count}")
        # sleeping for a while makes the TPU initialization error repro more consistent
        time.sleep(5)
        return result

    results = run_on_pod(fail_once_on_first_slice_jax_fn, "v5litepod-4", num_slices=num_slices, max_retries_failure=1)

    # run_on_pod_new returns a flat list of results from all hosts across all slices.
    # If each v5litepod-4 slice has 1 host (as per TPU-v5litepod-4-head resource meaning),
    # then for num_slices=2, we expect 2 results in the list.
    assert results is not None
    assert len(results) == num_slices  # num_slices * hosts_per_slice (assuming 1 host per v5litepod-4 slice)

    for i in range(num_slices):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            # Due to MEGASCALE_SLICE_ID, the PRNG key might differ effectively if the code used it.
            # simple_jax_fn uses a fixed PRNGKey(0) so all slices should produce identical results.
            assert np.array_equal(results[i], results[0])


def failing_fn():
    print("Executing failing_fn. This should fail.")
    raise deliberately_raised_exception  # Use a unique exception name


class DeliberatelyRaisedException(Exception):
    pass


deliberately_raised_exception = DeliberatelyRaisedException("This function is designed to fail.")


@pytest.mark.tpu_ci
def test_single_slice_catches_failure():
    """Test that run_on_pod_new correctly handles a failing function after retries."""
    with pytest.raises(RayTaskError) as excinfo:
        run_on_pod(failing_fn, "v5litepod-4", num_slices=1, max_retries_failure=0, max_retries_preemption=0)

    assert "DeliberatelyRaisedException" in str(
        excinfo.value
    ), f"Expected 'Failed too many times' but got: {excinfo.value}"


# Simulating preemption is tricky.
# We can define a function that, after a few calls, starts raising an error that `_handle_ray_error`
# would interpret as a preemption (e.g. a TimeoutError, or by mocking `get_current_tpu_is_preempted`).
@ray.remote
class PreemptionCountingActor:
    """Actor to count calls to a preemptible function."""

    def __init__(self, fn_id: str, preempt_until_n_calls: int):
        self.fn_id = fn_id
        self.preempt_until_n_calls = preempt_until_n_calls
        self.call_count = 0

    def run(self):
        self.call_count += 1
        print(f"Running {self.fn_id}, call count: {self.call_count}")
        if self.call_count < self.preempt_until_n_calls:
            raise TimeoutError("Simulated preemption via TimeoutError")

        return np.zeros(1)


@pytest.mark.tpu_ci
def test_single_slice_handles_preemption():
    """4. Run a function that preempts and verify it retries and eventually fails due to preemption retries."""
    actor = PreemptionCountingActor.remote("preemptible_fn", preempt_until_n_calls=4)

    def preempted_until_n():
        return ray.get(actor.run.remote())

    with pytest.raises(RuntimeError) as excinfo:
        run_on_pod(
            # We need to curry arguments into preemptible_fn or wrap it
            preempted_until_n,
            "v5litepod-4",
            num_slices=1,
            max_retries_failure=1,
            max_retries_preemption=2,  # Should retry preemption twice
        )

    assert "preempted too many times" in str(
        excinfo.value
    ), f"Expected 'Preempted too many times' but got: {excinfo.value}"

    # now let's call with a lower preempt_until_n_calls to ensure it succeeds

    actor = PreemptionCountingActor.remote("preemptible_fn_always", preempt_until_n_calls=2)

    def preempted_always():
        return ray.get(actor.run.remote())

    # This should succeed after 2 retries
    results = run_on_pod(
        preempted_always,
        "v5litepod-4",
        num_slices=1,
        max_retries_failure=0,  # No failure retries
        max_retries_preemption=2,  # Should retry preemption twice
    )

    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)


def fail_on_slice_0_fn():
    # Import JAX only inside the remote function
    import jax.distributed
    import jax.random as jrandom

    # need to ensure JAX is initialized or else we get weird crashes
    jax.distributed.initialize()
    slice_id_str = os.getenv("MEGASCALE_SLICE_ID")
    if slice_id_str == "0":
        print("Slice 0 is failing deliberately.")
        raise DeliberatelyRaisedException("Slice 0 is failing.")

    # Do simple JAX work for other slices
    key = jrandom.PRNGKey(int(slice_id_str) if slice_id_str else 42)
    data = jrandom.normal(key, (4,))
    return np.array(data)


# Multislice failure: one slice fails, the whole thing should retry and eventually fail.
@pytest.mark.tpu_ci
@pytest.mark.skip("Multislice not available in CI")
def test_multislice_one_slice_fails():
    """3. Run a function where one slice fails, verify retries and eventual failure."""
    num_slices = 2
    tpu_type = "v5litepod-4"

    with pytest.raises(RayTaskError) as excinfo:
        run_on_pod(
            fail_on_slice_0_fn,
            tpu_type,
            num_slices=num_slices,
            max_retries_failure=2,  # low retry
            max_retries_preemption=1,
        )

    assert "DeliberatelyRaisedException" in str(excinfo.value)
