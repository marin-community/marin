# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os

import jax.distributed
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
import ray
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec as P
from ray.exceptions import RayTaskError

from levanter.infra.ray_tpu import run_on_pod
from haliax.partitioning import ResourceAxis
from tests.test_utils import create_test_mesh, skip_in_ci, use_test_mesh

# Store whether TPUs are available and if multislice is possible
_TPU_AVAILABLE = False
_MULTISLICE_POSSIBLE = False


@pytest.fixture(scope="module", autouse=True)
def setup_ray_tpu_tests():
    global _TPU_AVAILABLE, _MULTISLICE_POSSIBLE

    # TODO: I don't understand why this is needed - without it, the tests hang indefinitely in CI
    skip_in_ci("Skipping TPU tests in CI environment")

    try:
        ray.init(ignore_reinit_error=True)
    except Exception as e:
        pytest.skip(f"Ray initialization failed: {e}", allow_module_level=True)

    available_resources = ray.cluster_resources()
    tpu_v4_8_head_count = available_resources.get("TPU-v4-8-head", 0)

    if tpu_v4_8_head_count < 1:
        pytest.skip("No TPU-v4-8-head resources available", allow_module_level=True)

    _TPU_AVAILABLE = True
    _MULTISLICE_POSSIBLE = tpu_v4_8_head_count >= 2

    yield

    ray.shutdown()


# Helper to skip multislice tests if not enough TPUs
skip_if_no_multislice = pytest.mark.skipif(
    not _MULTISLICE_POSSIBLE, reason="Less than 2 TPU-v4-8-head resources available for multislice tests"
)


# Base function for tests, similar to the one in ray_tpu.py
def simple_jax_fn():
    import jax

    jax.devices()

    # Check if we are on a TPU
    try:
        if jax.default_backend() != "tpu":
            # This can happen if the test is run on a CPU node by mistake or if JAX doesn't see the TPU
            # We won't raise an error here, but the test will likely fail later if it expects TPU behavior.
            # Or, more likely, jax.devices() will be empty or only show CPUs.
            print(f"Warning: JAX default backend is {jax.default_backend()}, not TPU.")

        devices = jax.devices("tpu")
        if not devices:
            raise RuntimeError("No JAX TPU devices found on the worker.")

        mesh = create_test_mesh()
        # print(f"JAX devices found on worker: {devices}")
        # print(f"Mesh created: {mesh}")

    except Exception as e:
        print(f"Error during JAX device/mesh setup on worker: {e}")
        # Raise an error that can be caught by the test runner if JAX setup fails.
        raise RuntimeError(f"JAX TPU initialization failed on worker: {e}")

    key_x, key_weights, key_bias = jrandom.split(jrandom.PRNGKey(0), 3)

    # Define array dimensions
    dim_in = 8  # factor of num_tpus_per_host usually
    dim_out = 4

    with use_test_mesh(mesh=mesh):
        x = jrandom.normal(key_x, (dim_in,))
        weights = jrandom.normal(key_weights, (dim_in, dim_out))
        bias = jrandom.normal(key_bias, (dim_out,))

        # Shard inputs - simple 1D sharding for x and weights
        # Adjust PartitionSpec based on your actual sharding strategy.
        # For a single host, this might just be P(None) or P(ResourceAxis.DATA) if you intend to shard across cores.
        x_sharded = with_sharding_constraint(x, P(ResourceAxis.DATA))
        weights_sharded = with_sharding_constraint(weights, P(ResourceAxis.DATA))
        # Bias is usually replicated or not sharded
        bias_sharded = with_sharding_constraint(bias, P())

        @jax.jit
        def layer(x_arg, weights_arg, bias_arg):
            return with_sharding_constraint(jnp.dot(x_arg, weights_arg) + bias_arg, P())

        output = layer(x_sharded, weights_sharded, bias_sharded)
    return np.array(output)


@ray.remote(max_calls=1)
def remote_simple_jax_fn():
    return simple_jax_fn()


@ray.remote
class CounterActor:
    def __init__(self):
        self._count = 0

    def increment(self) -> None:
        self._count += 1

    def count(self) -> int:
        return self._count


# Want to try:
# Task fails on first slice but not second
# Some amount of sleeping


# --- Single Slice Tests ---


@pytest.mark.ray
def test_single_slice_simple_run():
    """1. Run a simple function on a single slice and verify it runs correctly."""
    if not _TPU_AVAILABLE:
        pytest.skip("TPU not available for single slice test")

    num_slices = 1
    results = run_on_pod(remote_simple_jax_fn, "v4-8", num_slices=num_slices)

    assert results is not None
    assert len(results) == num_slices

    # For `num_slices=1` with "v4-8" (1 host per slice):
    assert len(results) == 1  # One result because one host in total for one v4-8 slice.
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (4,)  # Based on simple_jax_fn's output dim_out

    # Verify a second run works
    results_2 = run_on_pod(remote_simple_jax_fn, "v4-8", num_slices=num_slices)
    assert len(results_2) == 1
    assert isinstance(results_2[0], np.ndarray)
    assert np.array_equal(results[0], results_2[0])  # Deterministic function


@pytest.mark.ray
def test_single_slice_run_twice():
    """2. Run a second function after the first one and verify it runs correctly."""
    if not _TPU_AVAILABLE:
        pytest.skip("TPU not available for single slice test")

    num_slices = 1
    # First run
    results1 = run_on_pod(remote_simple_jax_fn, "v4-8", num_slices=num_slices)
    assert len(results1) == 1
    assert isinstance(results1[0], np.ndarray)
    assert results1[0].shape == (4,)

    # Second run
    results2 = run_on_pod(remote_simple_jax_fn, "v4-8", num_slices=num_slices)
    assert len(results2) == 1
    assert isinstance(results2[0], np.ndarray)
    assert results2[0].shape == (4,)

    # Check if results are the same (since PRNGKey is fixed)
    assert np.array_equal(results1[0], results2[0])


@pytest.mark.ray
def test_single_slice_fail_once():
    """1. Run a simple function on a single slice and verify it runs correctly."""
    if not _TPU_AVAILABLE:
        pytest.skip("TPU not available for single slice test")

    num_slices = 1
    results = run_on_pod(fail_once_jax_fn, "v4-8", num_slices=num_slices, max_retries_failure=1)

    counter_actor = CounterActor.remote()

    @ray.remote(max_calls=1)
    def fail_once_jax_fn() -> None:
        # do JAX work first
        result = simple_jax_fn()
        # fail on the first run
        count = ray.get(counter_actor.count.remote())
        ray.get(counter_actor.increment.remote())
        if count == 0:
            raise DeliberatelyRaisedException(f"Failing deliberately because count is {count}")
        return result

    assert results is not None
    assert len(results) == num_slices

    # For `num_slices=1` with "v4-8" (1 host per slice):
    assert len(results) == 1  # One result because one host in total for one v4-8 slice.
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (4,)  # Based on simple_jax_fn's output dim_out

    # Verify a second run works
    results_2 = run_on_pod(remote_simple_jax_fn, "v4-8", num_slices=num_slices)
    assert len(results_2) == 1
    assert isinstance(results_2[0], np.ndarray)
    assert np.array_equal(results[0], results_2[0])  # Deterministic function


# --- Multislice Tests ---


@pytest.mark.ray
def test_multislice_simple_run():
    """1. Run a simple function on a multislice and verify it runs correctly."""
    if not _MULTISLICE_POSSIBLE:  # Redundant due to marker, but good for clarity
        pytest.skip("Not enough TPUs for multislice test")

    num_slices = 2
    tpu_type = "v4-8"  # Each slice is a v4-8

    results = run_on_pod(remote_simple_jax_fn, tpu_type, num_slices=num_slices)

    # run_on_pod_new returns a flat list of results from all hosts across all slices.
    # If each v4-8 slice has 1 host (as per TPU-v4-8-head resource meaning),
    # then for num_slices=2, we expect 2 results in the list.
    assert results is not None
    assert len(results) == num_slices  # num_slices * hosts_per_slice (assuming 1 host per v4-8 slice)

    for i in range(num_slices):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            # Due to MEGASCALE_SLICE_ID, the PRNG key might differ effectively if the code used it.
            # simple_jax_fn uses a fixed PRNGKey(0) so all slices should produce identical results.
            assert np.array_equal(results[i], results[0])


@pytest.mark.ray
def test_variable_multislice_run():
    """1. Run a simple function on a multislice and verify it runs correctly."""
    if not _MULTISLICE_POSSIBLE:  # Redundant due to marker, but good for clarity
        pytest.skip("Not enough TPUs for multislice test")

    num_slices = [1, 2]
    tpu_type = "v4-8"  # Each slice is a v4-8

    results = run_on_pod(simple_jax_fn, tpu_type, num_slices=num_slices)

    assert results is not None
    assert len(results) in num_slices  # num_slices * hosts_per_slice (assuming 1 host per v4-8 slice)

    for i in range(len(results)):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            assert np.array_equal(results[i], results[0])


@pytest.mark.ray
def test_multislice_run_twice():
    """2. Run a second function after the first one and verify it runs correctly."""
    if not _MULTISLICE_POSSIBLE:
        pytest.skip("Not enough TPUs for multislice test")

    num_slices = 2
    tpu_type = "v4-8"

    # First run
    results1 = run_on_pod(remote_simple_jax_fn, tpu_type, num_slices=num_slices)
    assert len(results1) == num_slices
    for i in range(num_slices):
        assert isinstance(results1[i], np.ndarray)
        assert np.array_equal(results1[i], results1[0])  # All slices should be same

    # Second run
    results2 = run_on_pod(remote_simple_jax_fn, tpu_type, num_slices=num_slices)
    assert len(results2) == num_slices
    for i in range(num_slices):
        assert isinstance(results2[i], np.ndarray)
        assert np.array_equal(results2[i], results2[0])

    # Compare first and second run (should be identical)
    for i in range(num_slices):
        assert np.array_equal(results1[i], results2[i])


@pytest.mark.ray
def test_multislice_fail_once():
    """Run a simple function on two slices and verify it runs correctly
    when the first slice will fail on the first run."""
    # NOTE: This is currently causing a TPU initialization failure:
    # https://gist.github.com/yifanmai/88c7d56f31c2558ee79cd45b97ad5de0

    if not _MULTISLICE_POSSIBLE:
        pytest.skip("Not enough TPUs for multislice test")

    num_slices = 2
    counter_actor = CounterActor.remote()

    @ray.remote(max_calls=1)
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

    results = run_on_pod(fail_once_on_first_slice_jax_fn, "v4-8", num_slices=num_slices, max_retries_failure=1)

    # run_on_pod_new returns a flat list of results from all hosts across all slices.
    # If each v4-8 slice has 1 host (as per TPU-v4-8-head resource meaning),
    # then for num_slices=2, we expect 2 results in the list.
    assert results is not None
    assert len(results) == num_slices  # num_slices * hosts_per_slice (assuming 1 host per v4-8 slice)

    for i in range(num_slices):
        assert isinstance(results[i], np.ndarray)
        assert results[i].shape == (4,)
        if i > 0:
            # Due to MEGASCALE_SLICE_ID, the PRNG key might differ effectively if the code used it.
            # simple_jax_fn uses a fixed PRNGKey(0) so all slices should produce identical results.
            assert np.array_equal(results[i], results[0])


@ray.remote(max_calls=1)
def failing_fn():
    print("Executing failing_fn. This should fail.")
    raise deliberately_raised_exception  # Use a unique exception name


class DeliberatelyRaisedException(Exception):
    pass


deliberately_raised_exception = DeliberatelyRaisedException("This function is designed to fail.")


def test_single_slice_catches_failure():
    """Test that run_on_pod_new correctly handles a failing function after retries."""
    if not _TPU_AVAILABLE:
        pytest.skip("TPU not available for failure test")

    with pytest.raises(RayTaskError) as excinfo:
        run_on_pod(failing_fn, "v4-8", num_slices=1, max_retries_failure=0, max_retries_preemption=0)

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


@pytest.mark.ray
def test_single_slice_handles_preemption():
    """4. Run a function that preempts and verify it retries and eventually fails due to preemption retries."""
    if not _TPU_AVAILABLE:
        pytest.skip("TPU not available for preemption test")

    actor = PreemptionCountingActor.remote("preemptible_fn", preempt_until_n_calls=4)

    @ray.remote(max_calls=1)
    def preempted_until_n():
        return ray.get(actor.run.remote())

    with pytest.raises(RuntimeError) as excinfo:
        run_on_pod(
            # We need to curry arguments into preemptible_fn or wrap it
            preempted_until_n,
            "v4-8",
            num_slices=1,
            max_retries_failure=1,
            max_retries_preemption=2,  # Should retry preemption twice
        )

    assert "preempted too many times" in str(
        excinfo.value
    ), f"Expected 'Preempted too many times' but got: {excinfo.value}"

    # now let's call with a lower preempt_until_n_calls to ensure it succeeds

    actor = PreemptionCountingActor.remote("preemptible_fn_always", preempt_until_n_calls=2)

    @ray.remote(max_calls=1)
    def preempted_always():
        return ray.get(actor.run.remote())

    # This should succeed after 2 retries
    results = run_on_pod(
        preempted_always,
        "v4-8",
        num_slices=1,
        max_retries_failure=0,  # No failure retries
        max_retries_preemption=2,  # Should retry preemption twice
    )

    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)


@ray.remote(max_calls=1)
def fail_on_slice_0_fn():
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
@pytest.mark.ray
def test_multislice_one_slice_fails():
    """3. Run a function where one slice fails, verify retries and eventual failure."""
    if not _MULTISLICE_POSSIBLE:
        pytest.skip("Not enough TPUs for multislice failure test")

    num_slices = 2
    tpu_type = "v4-8"

    with pytest.raises(RayTaskError) as excinfo:
        run_on_pod(
            fail_on_slice_0_fn,
            tpu_type,
            num_slices=num_slices,
            max_retries_failure=2,  # low retry
            max_retries_preemption=1,
        )

    assert "DeliberatelyRaisedException" in str(excinfo.value)


if __name__ == "__main__":
    # test_single_slice_catches_failure()
    test_multislice_one_slice_fails()
