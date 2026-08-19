# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the fused MoK call against the pure-JAX reference on a real expert mesh.

Runs under the Iris four-process-per-node GPU supervisor, exactly like the MoK training arm:
one JAX process per GB200, a Torch/NCCL group for the symmetric workspace, and a JAX mesh whose
``expert`` axis covers the world. Everything below is what no local test could reach -- the native
handlers, the ``_v2`` scratch queries, the workspace registration, the ``shard_map`` boundary with
a real expert axis, and the numerics of the fused forward and backward.

The reference (``levanter.kernels.mok.mok_bf16_reference``) is dense, single-device and float32,
so the errors printed here are BF16 kernel versus FP32 math; the per-tensor "ref |max|" and
"bf16 floor" columns are the scale to read them against.
"""

from __future__ import annotations

import os
import sys


def _apply_runtime_env() -> None:
    os.environ.setdefault("JAX_ENABLE_PGLE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    os.environ.setdefault("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", "64")


_apply_runtime_env()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# JAX 0.11 renamed the mesh context manager; haliax probes for the same pair.
_set_mesh = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh


# One case per ABI branch. The hero token widths, a small batch, and independent
# shared/routed intermediate widths so `is` and `ir` cannot stand in for each other.
HIDDEN = 6144
LATENT = 3072
SHARED_INTERMEDIATE = 512
ROUTED_INTERMEDIATE = 1024
LOCAL_TOKENS = 512
TOPK = 4
EXPERTS_PER_RANK = 2
NORM_EPS = 1e-5
INIT_STD = 0.02


def _log(message: str) -> None:
    if jax.process_index() == 0:
        print(message, flush=True)


def _init_torch_distributed() -> tuple[int, int, int]:
    """Join the same NCCL group the MoK training arm builds, off the Iris JAX coordinator."""
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from iris.client.client import iris_ctx  # noqa: PLC0415
    from iris.cluster.client.job_info import get_job_info  # noqa: PLC0415
    from iris.hooks.multigpu import (  # noqa: PLC0415
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
        IRIS_MULTIGPU_PROCESS_COUNT_ENV,
        IRIS_MULTIGPU_PROCESS_INDEX_ENV,
    )
    from iris.runtime.jax_init import attempt_scoped_endpoint_name  # noqa: PLC0415

    world_size = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV])
    rank = int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV])
    device_id = int(os.environ[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV].split(",", 1)[0])

    torch.cuda.set_device(device_id)
    job_info = get_job_info()
    if job_info is None or job_info.task_index == 0:
        coordinator_host = job_info.advertise_host if job_info is not None else "127.0.0.1"
    else:
        endpoint_name = attempt_scoped_endpoint_name("jax_coordinator", job_info)
        coordinator_host = iris_ctx().resolver.resolve(endpoint_name).first().url.rsplit(":", 1)[0]
    coordinator_host = os.environ.get("MOK_MASTER_ADDR", coordinator_host)
    coordinator_port = int(os.environ.get("MOK_MASTER_PORT", "29500"))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{coordinator_host}:{coordinator_port}",
            rank=rank,
            world_size=world_size,
        )
    return rank, world_size, device_id


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.array(jax.devices()).reshape(1, ep_size)
    return Mesh(devices, ("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def _normal(key, shape, dtype, spec, mesh, scale=INIT_STD):
    def build(k):
        return (jax.random.normal(k, shape, jnp.float32) * scale).astype(dtype)

    return jax.jit(build, out_shardings=NamedSharding(mesh, spec))(key)


def _replicate(mesh, value):
    """Fully replicate an operand for the dense reference.

    The reference is deliberately single-device: it indexes the expert stack with a traced index,
    which JAX cannot resolve against an ``expert``-sharded operand under an explicit mesh. Giving
    it replicated copies keeps it exactly the arithmetic ``mok_bf16`` is supposed to preserve,
    with none of the sharding.
    """
    if value is None:
        return None
    return jax.device_put(value, NamedSharding(mesh, P(*([None] * value.ndim))))


def _stats(name: str, expected: jax.Array, actual: jax.Array) -> dict[str, float]:
    expected32 = expected.astype(jnp.float32)
    actual32 = actual.astype(jnp.float32)
    diff = jnp.abs(expected32 - actual32)
    # What BF16 rounding alone costs on this tensor, as the floor to read the error against.
    floor = jnp.max(jnp.abs(expected32 - expected32.astype(jnp.bfloat16).astype(jnp.float32)))
    values = jax.device_get(
        (
            jnp.max(diff),
            jnp.mean(diff),
            jnp.max(jnp.abs(expected32)),
            jnp.sqrt(jnp.mean(jnp.square(expected32))),
            floor,
            jnp.max(diff) / jnp.maximum(jnp.max(jnp.abs(expected32)), 1e-30),
        )
    )
    max_abs, mean_abs, ref_max, ref_rms, bf16_floor, rel = (float(v) for v in values)
    _log(
        f"    {name:<22} max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
        f"ref|max|={ref_max:.3e}  ref_rms={ref_rms:.3e}  bf16_floor={bf16_floor:.3e}  "
        f"max_abs/ref|max|={rel:.3e}"
    )
    if not np.isfinite(max_abs):
        raise RuntimeError(f"{name}: non-finite error")
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_max": ref_max,
        "ref_rms": ref_rms,
        "bf16_floor": bf16_floor,
        "max_abs_over_ref_max": rel,
    }


def _build_inputs(mesh, ep_size, latent_size, seed):
    from levanter.kernels.mok import mok_bf16  # noqa: F401, PLC0415  (import check only)

    hidden = HIDDEN
    routed = latent_size or hidden
    experts = EXPERTS_PER_RANK * ep_size
    tokens = LOCAL_TOKENS * ep_size

    batch_spec = P(("data", "expert"), None)
    shared_spec = P(None, None)
    routed_spec = P("expert", None, None)

    keys = list(jax.random.split(jax.random.key(seed), 20))
    x = _normal(keys[0], (tokens, hidden), jnp.bfloat16, batch_spec, mesh, scale=1.0)

    def _routing(key):
        logits = jax.random.uniform(key, (tokens, experts), jnp.float32)
        order = jnp.argsort(logits, axis=-1)[:, :TOPK].astype(jnp.int32)
        chosen = jnp.take_along_axis(logits, order, axis=-1)
        return order, jax.nn.softmax(chosen * 4.0, axis=-1)

    selected, router = jax.jit(_routing, out_shardings=(NamedSharding(mesh, batch_spec),) * 2)(keys[1])

    shared = []
    for index in range(2):
        shared.append(_normal(keys[2 + index * 3], (hidden, SHARED_INTERMEDIATE), jnp.bfloat16, shared_spec, mesh))
        shared.append(_normal(keys[3 + index * 3], (hidden, SHARED_INTERMEDIATE), jnp.bfloat16, shared_spec, mesh))
        shared.append(_normal(keys[4 + index * 3], (SHARED_INTERMEDIATE, hidden), jnp.bfloat16, shared_spec, mesh))
    shared0 = tuple(shared[:3])
    shared1 = tuple(shared[3:6])

    routed_weights = (
        _normal(keys[8], (experts, routed, ROUTED_INTERMEDIATE), jnp.bfloat16, routed_spec, mesh),
        _normal(keys[9], (experts, routed, ROUTED_INTERMEDIATE), jnp.bfloat16, routed_spec, mesh),
        _normal(keys[10], (experts, ROUTED_INTERMEDIATE, routed), jnp.bfloat16, routed_spec, mesh),
    )

    if latent_size:
        # A non-trivial gain, still positive and O(1), so the norm backward is exercised.
        gain = 1.0 + _normal(keys[12], (latent_size,), jnp.float32, P(None), mesh, scale=0.1)
        latent_weights = (
            _normal(keys[11], (hidden, latent_size), jnp.bfloat16, shared_spec, mesh),
            gain,
            _normal(keys[13], (latent_size, hidden), jnp.bfloat16, shared_spec, mesh),
        )
    else:
        latent_weights = (None, None, None)

    cotangent = _normal(keys[15], (tokens, hidden), jnp.float32, batch_spec, mesh, scale=1.0)
    return x, selected, router, shared0, shared1, routed_weights, latent_weights, cotangent


def _run_case(mesh, ep_size, latent_size, seed):
    from levanter.kernels.mok import MokBf16Config, initialize_mok_runtime, mok_bf16, mok_bf16_reference  # noqa: PLC0415

    label = f"latent_size={latent_size}"
    _log(f"\n=== case {label} (hidden={HIDDEN}, routed={latent_size or HIDDEN}) ===")

    with _set_mesh(mesh):
        x, selected, router, shared0, shared1, routed_weights, latent_weights, cotangent = _build_inputs(
            mesh, ep_size, latent_size, seed
        )

    config = MokBf16Config(
        workspace_id=0 if latent_size else 1,
        minibatch_size=4096,
        macrobatch_size=12288,
        schedule_capacity_multiplier=1.5,
        latent_size=latent_size,
        latent_norm_eps=NORM_EPS,
    )
    runtime = initialize_mok_runtime(
        config=config,
        num_local_tokens=LOCAL_TOKENS,
        hidden_size=HIDDEN,
        latent_size=latent_size or None,
        topk=TOPK,
    )
    _log(f"  workspace registered (workspace_id={config.workspace_id})")

    args = (x, selected, router, *shared0, *shared1, *routed_weights, *latent_weights)
    diff_argnums = tuple(index for index in range(len(args)) if index != 1 and args[index] is not None)

    def fused(*call_args):
        return mok_bf16(*call_args, config=config)

    def reference(*call_args):
        return mok_bf16_reference(*call_args, latent_norm_eps=NORM_EPS)

    # The cotangent is sharded across processes, so it has to be an argument rather than a
    # closure: JAX refuses to constant-fold an array whose shards this process cannot address.
    def _loss(fn):
        def loss(cot, *call_args):
            return jnp.sum(fn(*call_args).astype(jnp.float32) * cot)

        return loss

    try:
        with _set_mesh(mesh):
            ref_args = tuple(_replicate(mesh, arg) for arg in args)
            ref_cotangent = _replicate(mesh, cotangent)
            fused_y = jax.jit(fused)(*args)
            reference_y = jax.jit(reference)(*ref_args)
            results = {"forward": _stats("y", reference_y, _replicate(mesh, fused_y))}

            grad_argnums = tuple(index + 1 for index in diff_argnums)
            fused_grads = jax.jit(jax.grad(_loss(fused), argnums=grad_argnums))(cotangent, *args)
            reference_grads = jax.jit(jax.grad(_loss(reference), argnums=grad_argnums))(ref_cotangent, *ref_args)
            fused_grads = tuple(_replicate(mesh, grad) for grad in fused_grads)
            names = (
                "d_x",
                "d_router_weights",
                "d_shared0_gate",
                "d_shared0_up",
                "d_shared0_down",
                "d_shared1_gate",
                "d_shared1_up",
                "d_shared1_down",
                "d_routed_gate",
                "d_routed_up",
                "d_routed_down",
                "d_latent_down",
                "d_latent_norm_gain",
                "d_latent_up",
            )
            for name, expected, actual in zip(names[: len(fused_grads)], reference_grads, fused_grads, strict=True):
                results[name] = _stats(name, expected, actual)
    except Exception as exc:
        _log(f"  CASE FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        runtime.close()
    return results


def main() -> int:
    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    initialize_jax()
    _rank, world_size, _device_id = _init_torch_distributed()
    _log(f"jax processes={jax.process_count()} devices={jax.device_count()} torch world={world_size}")

    from levanter.kernels.mok import mok_preflight_status  # noqa: PLC0415

    status = mok_preflight_status()
    _log(
        f"preflight: torch={status.torch_version} cuda={status.torch_cuda_version} "
        f"native={status.native_extension_loaded}"
    )
    if status.errors:
        _log("preflight errors: " + "; ".join(status.errors))
        return 1
    import mok._C as native  # noqa: PLC0415

    _log(f"native abi version = {int(native.levanter_mok_ffi_abi_version())}")

    ep_size = jax.device_count()
    mesh = _make_mesh(ep_size)
    all_results = {}
    for latent_size in (LATENT, 0):
        all_results[latent_size] = _run_case(mesh, ep_size, latent_size, seed=1234 + latent_size)

    _log("\nRESULT_JSON " + repr(all_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
