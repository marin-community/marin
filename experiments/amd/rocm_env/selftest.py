# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-test of a freshly built AMD ROCm training environment on one GPU node.

Checks, in order: the RCCL that JAX loads is the venv's 2.30.7 and not a system ROCm copy; the local device count;
an exact all-reduce over every local GPU; a jax_triton kernel on every GPU (the trainer's ragged_dot is Triton);
and a small jitted training step whose loss falls. With --anp-plugin, RCCL is pointed at the Pensando ANP net
plugin and the test confirms RCCL loaded it.

Run it through selftest.sh, which sets the library path and RCCL logging before Python starts:
    experiments/amd/rocm_env/selftest.sh $PREFIX/venv/bin/python --expect-devices 8
"""

import argparse
import ctypes
import os
import sys

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl

RCCL_VERSION_CODE = 23007


def loaded_rccl_path() -> str:
    with open("/proc/self/maps") as maps:
        paths = {line.split()[-1] for line in maps if "librccl.so" in line}
    assert len(paths) == 1, f"expected one librccl loaded, found {sorted(paths)}"
    return paths.pop()


def rccl_log_text(log_dir: str) -> str:
    text = []
    for name in os.listdir(log_dir):
        with open(os.path.join(log_dir, name)) as log:
            text.append(log.read())
    return "".join(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-devices", type=int, required=True)
    parser.add_argument("--anp-plugin", help="path to librccl-anp.so; checks that RCCL loads it")
    parser.add_argument("--rccl-log-dir", required=True, help="directory of NCCL_DEBUG_FILE (set by selftest.sh)")
    args = parser.parse_args()
    system_rocm = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p.startswith("/opt/rocm")]
    assert not system_rocm, f"system ROCm on LD_LIBRARY_PATH would shadow the venv's libraries: {system_rocm}"

    print(f"python {sys.version.split()[0]} jax {jax.__version__} triton {triton.__version__}", flush=True)
    devices = jax.local_devices()
    kinds = {d.device_kind for d in devices}
    print(f"{len(devices)} devices {kinds} platform {devices[0].platform}", flush=True)
    assert devices[0].platform == "gpu", devices[0].platform
    assert len(devices) == args.expect_devices, f"expected {args.expect_devices} devices, found {len(devices)}"

    # All-reduce: device i contributes i + 1 everywhere, so every element of the sum is n(n+1)/2 exactly.
    n = len(devices)
    mesh = jax.make_mesh((n,), ("d",), axis_types=(jax.sharding.AxisType.Auto,), devices=devices)
    sharding = jax.NamedSharding(mesh, jax.P("d"))
    x = jax.device_put(np.repeat(np.arange(1, n + 1, dtype=np.float32), 1 << 20).reshape(n, 1 << 20), sharding)
    total = jax.jit(jax.shard_map(lambda v: jax.lax.psum(v, "d"), mesh=mesh, in_specs=jax.P("d"), out_specs=jax.P("d")))(
        x
    )
    total = np.asarray(total)
    assert (total == n * (n + 1) / 2).all(), f"all-reduce wrong: {np.unique(total)}"
    print(f"all-reduce over {n} devices ok ({total.nbytes // n >> 20} MiB per device)", flush=True)

    rccl = loaded_rccl_path()
    assert ".venv" in rccl or "/venv/" in rccl, f"RCCL loaded from {rccl}, not the venv"
    version = ctypes.c_int()
    ctypes.CDLL(rccl).ncclGetVersion(ctypes.byref(version))
    assert version.value == RCCL_VERSION_CODE, f"RCCL {version.value}, expected {RCCL_VERSION_CODE}"
    print(f"RCCL {version.value} from {rccl}", flush=True)
    if args.anp_plugin:
        text = rccl_log_text(args.rccl_log_dir)
        # RCCL also probes the plugin for optional RMA and GIN entry points and logs their absence; only the
        # network plugin matters here.
        loaded = [line for line in text.splitlines() if "ANP plugin loaded successfully" in line]
        assert loaded, "RCCL did not report loading the ANP net plugin"
        print(loaded[0], flush=True)
        print("ANP net plugin loaded", flush=True)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, size, out_ptr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)

    def triton_add(a, b):
        return jt.triton_call(
            a,
            b,
            a.size,
            kernel=add_kernel,
            out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
            grid=(triton.cdiv(a.size, 1024),),
            BLOCK=1024,
        )

    values = np.arange(n * 4096, dtype=np.float32).reshape(n, 4096)
    doubled = jax.jit(jax.shard_map(lambda v: triton_add(v, v), mesh=mesh, in_specs=jax.P("d"), out_specs=jax.P("d")))(
        jax.device_put(values, sharding)
    )
    assert np.array_equal(np.asarray(doubled), 2 * values), "jax_triton kernel wrong"
    print(f"jax_triton kernel on {n} devices ok", flush=True)

    # A two-layer bf16 MLP regression, data-parallel over every device, 20 SGD steps.
    key_w1, key_w2, key_x = jax.random.split(jax.random.key(0), 3)
    params = {
        "w1": jax.random.normal(key_w1, (512, 1024), jnp.float32) / 512**0.5,
        "w2": jax.random.normal(key_w2, (1024, 512), jnp.float32) / 1024**0.5,
    }
    inputs = jax.device_put(jax.random.normal(key_x, (n * 64, 512), jnp.float32), sharding)
    targets = jnp.sin(inputs)

    def loss_fn(p, xb, yb):
        hidden = jax.nn.gelu(xb.astype(jnp.bfloat16) @ p["w1"].astype(jnp.bfloat16))
        out = (hidden @ p["w2"].astype(jnp.bfloat16)).astype(jnp.float32)
        return jnp.mean((out - yb) ** 2)

    @jax.jit
    def train_step(p, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        return jax.tree.map(lambda w, g: w - 0.5 * g, p, grads), loss

    losses = []
    for _ in range(20):
        params, loss = train_step(params, inputs, targets)
        losses.append(float(loss))
    assert all(np.isfinite(losses)), losses
    assert losses[-1] < 0.9 * losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"jitted train step ok: loss {losses[0]:.4f} -> {losses[-1]:.4f} over 20 steps", flush=True)
    print("SELFTEST PASSED", flush=True)


if __name__ == "__main__":
    main()
