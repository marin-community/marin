# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe which tensor layouts cuDNN's MoE grouped matmul backward accepts, straight through
cudnn-frontend's Python bindings and without XLA in the way.

XLA's ragged-dot fusion hands the node ``dweight[G, K, N] = token[M, K]^T @ doutput[M, N]`` and
cuDNN answers ``cublasLt grouped GEMM heuristic failed -- check that the datatype/layout combination
is supported``. This builds the same graph at the hero chunk shape under several layouts and
reports which ones cuDNN will plan, and times the ones that run.

Run on one GPU with ``nvidia-cudnn-frontend`` installed (``run_cudnn_probe.sh``)::

    python lib/levanter/scripts/bench/probe_cudnn_moe_bwd.py

Device buffers are JAX arrays: cudnn-frontend reads them through ``__cuda_array_interface__``.
"""

import itertools
import time

import cudnn
import jax
import jax.numpy as jnp
import numpy as np

M, K, N = 301_466, 3072, 6144
SIZES = [86153, 108681, 67209]
G = len(SIZES)
OFFSETS = np.concatenate([[0], np.cumsum(SIZES)[:-1]]).astype(np.int32)


def _row_major(*dims):
    strides = [1]
    for d in reversed(dims[1:]):
        strides.insert(0, strides[0] * d)
    return list(dims), strides


def _col_major(g, a, b):
    """dims (g, a, b) with a fastest: strides (a*b, 1, a)."""
    return [g, a, b], [a * b, 1, a]


def probe(label, token_layout, doutput_layout, dweight_layout, offset_dtype, run):
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    token = graph.tensor(name="token", dim=token_layout[0], stride=token_layout[1], data_type=cudnn.data_type.BFLOAT16)
    doutput = graph.tensor(
        name="doutput", dim=doutput_layout[0], stride=doutput_layout[1], data_type=cudnn.data_type.BFLOAT16
    )
    offsets = graph.tensor(name="offsets", dim=[G, 1, 1], stride=[1, 1, 1], data_type=offset_dtype)
    dweight = graph.moe_grouped_matmul_bwd(doutput=doutput, token=token, first_token_offset=offsets, name="bwd")
    dweight.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    if dweight_layout is not None:
        dweight.set_dim(dweight_layout[0]).set_stride(dweight_layout[1])
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
        graph.check_support()
        graph.build_plans()
    except Exception as exc:  # noqa: BLE001 -- the failure text is the datum
        print(f"{label:<60} UNSUPPORTED: {str(exc).splitlines()[0][:160]}")
        return
    print(f"{label:<60} SUPPORTED dweight dim={dweight.get_dim()} stride={dweight.get_stride()}")
    if not run:
        return
    key = jax.random.key(0)
    tok = jax.random.normal(key, token_layout[0], jnp.bfloat16)
    dout = jax.random.normal(key, doutput_layout[0], jnp.bfloat16)
    off_np = OFFSETS.astype(np.int32 if offset_dtype == cudnn.data_type.INT32 else np.int64)
    off = jnp.asarray(off_np).reshape(G, 1, 1)
    dw = jnp.zeros(dweight.get_dim(), jnp.bfloat16)
    ws = jnp.zeros((max(graph.get_workspace_size(), 1),), jnp.uint8)
    for a in (tok, dout, off, dw, ws):
        a.block_until_ready()
    pack = {token: tok, doutput: dout, offsets: off, dweight: dw}
    graph.execute(pack, ws, handle=handle)
    jax.effects_barrier()
    t = time.perf_counter()
    for _ in range(10):
        graph.execute(pack, ws, handle=handle)
    jax.effects_barrier()
    ms = (time.perf_counter() - t) / 10 * 1e3
    flops = 2.0 * sum(SIZES) * K * N
    print(f"{'':<60}   {ms:8.3f} ms  {flops / ms / 1e9:8.1f} TFLOP/s")


def main():
    print("cudnn frontend", cudnn.__version__, "backend", cudnn.backend_version())
    tok_rm = _row_major(1, M, K)
    dout_rm = _row_major(1, M, N)
    cases = [
        ("sample: rm token/doutput, inferred dweight", tok_rm, dout_rm, None),
        ("rm token/doutput, dweight col-major [G,K,N]", tok_rm, dout_rm, _col_major(G, K, N)),
        ("rm token/doutput, dweight row-major [G,K,N]", tok_rm, dout_rm, _row_major(G, K, N)),
        ("swapped roles (doutput:=token), inferred", dout_rm, tok_rm, None),
        ("swapped roles, dweight col-major [G,N,K]", dout_rm, tok_rm, _col_major(G, N, K)),
        ("swapped roles, dweight row-major [G,N,K]", dout_rm, tok_rm, _row_major(G, N, K)),
    ]
    for (label, tok, dout, dw), dtype in itertools.product(cases, (cudnn.data_type.INT32, cudnn.data_type.INT64)):
        probe(f"{label} / offsets {dtype.name.lower()}", tok, dout, dw, dtype, run=dtype == cudnn.data_type.INT32)


if __name__ == "__main__":
    main()
