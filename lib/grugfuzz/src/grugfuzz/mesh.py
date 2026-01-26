"""Mesh utilities for multi-device testing."""

import os
from contextlib import contextmanager

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src import mesh as mesh_lib


def ensure_devices(n: int) -> None:
    """Ensure JAX has at least n devices available.

    On CPU, this sets XLA_FLAGS to create fake devices if needed.
    Must be called before JAX is imported elsewhere.
    """
    current_flags = os.environ.get("XLA_FLAGS", "")
    device_flag = f"--xla_force_host_platform_device_count={n}"
    if device_flag not in current_flags:
        os.environ["XLA_FLAGS"] = f"{current_flags} {device_flag}".strip()


def create_mesh(data_parallel: int = 1, model_parallel: int = 1) -> Mesh:
    """Create a mesh with the specified parallelism.

    Args:
        data_parallel: Number of devices for data parallelism
        model_parallel: Number of devices for model/tensor parallelism

    Returns:
        A JAX Mesh with axes ("data", "model")
    """
    devices = jax.devices()
    n_devices = len(devices)
    expected = data_parallel * model_parallel

    if n_devices < expected:
        raise RuntimeError(
            f"Need {expected} devices ({data_parallel}x{model_parallel}), "
            f"only have {n_devices}. Call ensure_devices({expected}) before importing JAX."
        )

    device_array = np.array(devices[:expected]).reshape(data_parallel, model_parallel)
    return Mesh(
        device_array,
        axis_names=("data", "model"),
        axis_types=(mesh_lib.AxisType.Explicit, mesh_lib.AxisType.Explicit),
    )


@contextmanager
def with_mesh(data_parallel: int = 1, model_parallel: int = 1):
    """Context manager that creates and sets a mesh.

    Args:
        data_parallel: Number of devices for data parallelism
        model_parallel: Number of devices for model/tensor parallelism

    Yields:
        The created Mesh

    Example:
        with with_mesh(2, 2) as mesh:
            # mesh is set globally via jax.set_mesh
            params = init_parameters(cfg, key=key)
    """
    mesh = create_mesh(data_parallel, model_parallel)
    old_mesh = mesh_lib.get_abstract_mesh()

    jax.set_mesh(mesh)
    try:
        with mesh:
            yield mesh
    finally:
        if old_mesh is not None and not old_mesh.empty:
            jax.set_mesh(old_mesh)


def sharded_zeros(shape: tuple[int, ...], mesh: Mesh, spec: P) -> jax.Array:
    """Create a sharded array of zeros."""
    return jax.device_put(
        jax.numpy.zeros(shape),
        NamedSharding(mesh, spec)
    )


def sharded_ones(shape: tuple[int, ...], mesh: Mesh, spec: P) -> jax.Array:
    """Create a sharded array of ones."""
    return jax.device_put(
        jax.numpy.ones(shape),
        NamedSharding(mesh, spec)
    )


def sharded_randn(
    key: jax.Array,
    shape: tuple[int, ...],
    mesh: Mesh,
    spec: P,
    scale: float = 1.0,
) -> jax.Array:
    """Create a sharded array of random normal values."""
    return jax.device_put(
        jax.random.normal(key, shape) * scale,
        NamedSharding(mesh, spec)
    )
