# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate TPU lowering failures by compiling GDN-2 stages independently.

This diagnostic uses synthetic inputs and reports compile/run success, not
kernel correctness or accepted performance. All arrays reside on one TPU.
"""

import argparse
from functools import partial
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from levanter.kernels.pallas import gdn2
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


def primitive_operation(value, operation):
    if operation == "exp":
        return jnp.exp(value)
    if operation == "exp2":
        return jnp.exp2(value)
    if operation == "where":
        return jnp.where(value < 0, value, -jnp.inf)
    if operation == "masked_exp":
        return jnp.exp(jnp.where(value < 0, value, -jnp.inf))
    if operation == "last_row":
        return jnp.broadcast_to(value[-1:], value.shape)
    if operation == "sum_last":
        return jnp.sum(value, axis=-1)
    if operation == "sum_middle":
        return jnp.sum(value, axis=1)
    if operation == "sum_first":
        return jnp.sum(value, axis=0)
    if operation == "sum_last_causal":
        reduced = jnp.sum(value, axis=-1)
        row = jnp.arange(reduced.shape[0])[:, None]
        col = jnp.arange(reduced.shape[1])[None, :]
        return reduced * (row >= col).astype(value.dtype)
    if operation == "masked_sum_last":
        row = jax.lax.broadcasted_iota(jnp.int32, value.shape, 0)
        col = jax.lax.broadcasted_iota(jnp.int32, value.shape, 1)
        return jnp.sum(jnp.where(row >= col, value, 0.0), axis=-1)
    if operation == "feature_major_sum":
        return jnp.sum(jnp.moveaxis(value, -1, 0), axis=0)
    if operation == "weighted_masked_sum":
        left = value[:, 0, :]
        right = value[0, :, :]
        row = jax.lax.broadcasted_iota(jnp.int32, value.shape, 0)
        col = jax.lax.broadcasted_iota(jnp.int32, value.shape, 1)
        decay = jnp.exp(jnp.where(row >= col, value, -jnp.inf))
        return jnp.sum(left[:, None, :] * decay * right[None, :, :], axis=-1)
    raise ValueError(operation)


def primitive_call(x, operation):
    def body(x_ref, out_ref):
        out_ref[...] = primitive_operation(x_ref[...], operation)

    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    output_spec = jax.eval_shape(partial(primitive_operation, operation=operation), spec)
    return pl.pallas_call(
        body,
        out_shape=output_spec,
        compiler_params=pltpu.CompilerParams(),
        cost_estimate=with_io_bytes_accessed(
            pl.estimate_cost(partial(primitive_operation, operation=operation), spec),
            kernel_inputs_specs=(spec,),
            kernel_outputs_specs=(output_spec,),
        ),
    )(x)


def stage_a(module, config, q, k, b, g):
    return module.build_chunk_scores_pallas(q, k, b, g, 128**-0.5, config=config)


def stage_b(module, config, akk):
    return module.wy_solve_pallas(akk, config=config)


def stage_c(module, config, q, k, v, w, b, g, a):
    return module.recompute_wy_pallas(q, k, v, w, b, g, a, config=config)


def stage_b2(module, config, aqk, v_new, do):
    return module.dav_backward_pallas(aqk, v_new, do, config=config)


def stage_b3(module, config, q, k, b, w, v, gc, a, akk, h_pre, v_new, do, dv, dh_next):
    return module.wy_dqkg_backward_pallas(
        q, k, b, w, v, gc, a, akk, h_pre, v_new, do, dv, dh_next, 128**-0.5, config=config
    )


def stage_b4(module, config, daqk, dakk, q, k, b, g):
    return module.intra_backward_pallas(daqk, dakk, q, k, b, g, 128**-0.5, config=config)


def run_probe(function, inputs, metadata, stream):
    row = dict(metadata, error=None, success=False, compile_time=None, execution_time=None)
    phase = "compile"
    try:
        start = time.perf_counter()
        executable = jax.jit(function).lower(*inputs).compile()
        row["compile_time"] = time.perf_counter() - start
        phase = "execute"
        start = time.perf_counter()
        jax.block_until_ready(executable(*inputs))
        row.update(success=True, execution_time=time.perf_counter() - start)
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", failed_phase=phase, traceback=traceback.format_exc())
    stream.write(json.dumps(row) + "\n")
    stream.flush()
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bt", type=int, choices=(128, 256), default=128)
    parser.add_argument("--mb", type=int, choices=(16, 32), default=16)
    parser.add_argument("--implementation", choices=("candidate", "upstream"), action="append")
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.platform != "tpu":
        parser.error("This compiler diagnostic requires TPU")
    path = args.output or Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / f"gdn2-probe-{time.time_ns()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    source_root = Path(gdn2.__file__).parent
    for source in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source.relative_to(source_root)).encode())
        source_digest.update(source.read_bytes())
    source_digest.update(Path(__file__).read_bytes())
    common = {
        "diagnostic": "gdn2_stage_compile",
        "device_type": device.device_kind,
        "jax_version": jax.__version__,
        "git_sha": os.environ.get("GDN2_GIT_SHA"),
        "source_sha256": source_digest.hexdigest(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    with path.open("x") as stream, jax.default_device(device):
        shape = (1, 256, 1, 128)
        q = jax.random.normal(jax.random.key(0), shape) * 0.05
        k = jax.random.normal(jax.random.key(1), shape) * 0.05
        v = jax.random.normal(jax.random.key(2), shape)
        w = jnp.full(shape, 0.5, jnp.float32)
        b = jnp.full(shape, 0.4, jnp.float32)
        g = jnp.full(shape, -0.005, jnp.float32)
        matrix_shape = (1, 1, 256 // args.bt, args.bt, args.bt)
        akk = jnp.broadcast_to(jnp.tril(jnp.full((args.bt, args.bt), 0.001), -1), matrix_shape)
        a = jnp.broadcast_to(jnp.eye(args.bt), matrix_shape)
        q_chunk, k_chunk, v_chunk, w_chunk, b_chunk, g_chunk = (
            jnp.moveaxis(x.reshape(1, 256 // args.bt, args.bt, 1, 128), 3, 1) for x in (q, k, v, w, b, g)
        )
        gc = jnp.cumsum(g_chunk, axis=3)
        state_shape = (1, 1, 256 // args.bt, 128, 128)
        h_pre = jax.random.normal(jax.random.key(3), state_shape) * 0.05
        dh_next = jax.random.normal(jax.random.key(4), state_shape)
        do = jax.random.normal(jax.random.key(5), v_chunk.shape)
        dv = jax.random.normal(jax.random.key(6), v_chunk.shape)
        for implementation in args.implementation or ("upstream", "candidate"):
            prefix = f"levanter.kernels.pallas.gdn2.{implementation}"
            module = importlib.import_module(prefix + ".gdn2_fwd")
            backward_module = importlib.import_module(prefix + ".gdn2_bwd")
            config_module = importlib.import_module(prefix + ".configs")
            if implementation == "candidate":
                layout = (
                    config_module.ScoreLayout.FEATURE_FIRST
                    if "tpu v4" in device.device_kind.lower()
                    else config_module.ScoreLayout.FEATURE_LAST
                )
                config = config_module.KernelConfig(
                    bt=args.bt, bc=args.bt // 2, mb=args.mb, wy_eps=0.0, score_layout=layout
                )
            else:
                config = config_module.KernelConfig(bt=args.bt, bc=args.bt // 2, mb=args.mb, wy_eps=0.0)
            for stage, function, stage_module, inputs in (
                ("A", stage_a, module, (q, k, b, g)),
                ("B", stage_b, module, (akk,)),
                ("C", stage_c, module, (q, k, v, w, b, g, a)),
                ("B2", stage_b2, backward_module, (a, v_chunk, do)),
                (
                    "B3",
                    stage_b3,
                    backward_module,
                    (q_chunk, k_chunk, b_chunk, w_chunk, v_chunk, gc, a, akk, h_pre, v_chunk, do, dv, dh_next),
                ),
                ("B4", stage_b4, backward_module, (a, akk, q, k, b, g)),
            ):
                metadata = dict(
                    common,
                    implementation=implementation,
                    stage=stage,
                    shape=shape,
                    block_sizes={"bt": args.bt, "bc": args.bt // 2, "mb": args.mb},
                )
                run_probe(partial(function, stage_module, config), inputs, metadata, stream)
        for primitive_shape in ((64, 128), (64, 64, 128)):
            value = jnp.linspace(-1.0, 1.0, math.prod(primitive_shape)).reshape(primitive_shape)
            operations = ("exp", "exp2", "where", "masked_exp", "last_row")
            if len(primitive_shape) == 3:
                operations += (
                    "sum_last",
                    "sum_middle",
                    "sum_first",
                    "sum_last_causal",
                    "masked_sum_last",
                    "feature_major_sum",
                    "weighted_masked_sum",
                )
            for operation in operations:
                metadata = dict(common, implementation="primitive", stage=operation, shape=primitive_shape)
                run_probe(partial(primitive_call, operation=operation), (value,), metadata, stream)


if __name__ == "__main__":
    main()
