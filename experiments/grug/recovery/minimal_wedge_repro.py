# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Minimal single-file JAX reproducer for the NCCL 2.28.9 proxy-slot wedge on GB200 NVL72.

Symptom: a multi-node JAX/XLA job stops making progress with no error. Every rank
stays alive, no timeout fires, GPUs sit at 100% SM utilisation with 0% memory
traffic, and NCCL RAS reports one rank a few collective operations behind the rest,
static forever. NCCL 2.29.3-1 and later contain the upstream aarch64 proxy-slot
fix and should complete the requested step count.

That applies to this synthetic collective only. The 300B hero wedge tracked in
#7344 is not the same fault: #8029 records it wedging on 2.30.7 behind a verified
provenance gate, so a clean run here does not clear a build for the hero shape.

This file contains no model framework and no custom CUDA kernels -- no attention,
no MoE, no flash-attn, no CuTe/CUTLASS. It is a dense MLP stack that keeps only the
things that plausibly matter:

  * an explicit ``(replica, data, expert, model)`` mesh sized from ``--dp-racks``
    (default 16 nodes/rack x 4 GPUs/node, replica axis = dp_racks, expert = 1),
  * FSDP-style parameter sharding over ``data``: one all-gather per layer over the
    64-device (one rack) ``data`` group in the forward pass, and a gradient
    all-reduce over all ``replica x data x expert`` devices in the backward
    (XLA:GPU is expected to rewrite the data-axis part into a reduce-scatter;
    verified as all-gather + all-reduce on the CPU backend),
  * the batch sharded over ``(replica, data, expert)``,
  * ``XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async``, ``JAX_ENABLE_PGLE=1`` and
    ``--xla_gpu_enable_command_buffer=`` (command buffers disabled),
  * mixed precision params=float32 / compute=bfloat16,
  * a device->host scalar sync every step, and a flushed per-step progress line
    (the line a wedge detector watches for silence).

Every one of the above is overridable from the command line or the environment so
each can be ablated independently.

Dependencies: jax, and numpy (used only to reshape the device list).

Usage (see --help). Nothing is loaded from disk or the network; all data is
synthetic and generated on device.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Runtime environment. MUST happen before `import jax`.
#
#   RUNTIME_ENV = {"JAX_ENABLE_PGLE": "1",
#                            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async"}
#   XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
# every one of these can be ablated from the launch environment.
# --------------------------------------------------------------------------
RUNTIME_ENV = {
    "JAX_ENABLE_PGLE": "1",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="

PROVENANCE_ENV_KEYS = (
    "CUDA_CACHE_DISABLE",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "JAX_ENABLE_PGLE",
    "NCCL_BUFFSIZE",
    "NCCL_CTA_POLICY",
    "NCCL_CUMEM_ENABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_LAUNCH_ORDER_IMPLICIT",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NCHANNELS_PER_PEER",
    "NCCL_NVLS_ENABLE",
    "NCCL_PROTO",
    "NCCL_RUNTIME_CONNECT",
    "NCCL_WORK_FIFO_BYTES",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
)
PROVENANCE_LIBRARY_NAMES = ("libcuda.so", "libcudart.so", "libnccl.so")


def _apply_runtime_defaults() -> None:
    for key, value in RUNTIME_ENV.items():
        os.environ.setdefault(key, value)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    flag_name = XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG.partition("=")[0]
    if any(flag.partition("=")[0] == flag_name for flag in xla_flags.split()):
        return  # already set (or explicitly overridden) by the launch environment
    os.environ["XLA_FLAGS"] = f"{xla_flags} {XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG}".strip()


_apply_runtime_defaults()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, Mesh, reshard  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# --------------------------------------------------------------------------
# Defaults matching the production job this reproduces.
# --------------------------------------------------------------------------
BATCH_PER_RACK = 1024
NODES_PER_RACK = 16
GPUS_PER_NODE = 4
DEFAULT_STEPS = 25
EXPERT_AXIS_SIZE = 1
MODEL_AXIS_SIZE = 1

HIDDEN_DIM = 6144
NUM_LAYERS = 48
VOCAB_SIZE = 128_256
MAX_SEQ_LEN = 4096

# Mesh axis names, outermost first.
GRUG_MESH_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
BATCH_AXES = ("replica_dcn", "data", "expert")
#   dense in->out weights   P("data", "model")     <- FSDP shard over `data`
#   dense out->in weights   P("model", "data")
#   lm head                 P("data", "model")     (_LM_HEAD_PARTITION_SPEC)
#   activations / batch     P(_BATCH_AXES, None)
FSDP_IN_OUT_SPEC = P("data", "model")
FSDP_OUT_IN_SPEC = P("model", "data")
BATCH_SPEC = P(BATCH_AXES, None)
REPLICATED = P(None, None)


def print_runtime_provenance() -> None:
    """Print filtered child-process environment and loaded GPU library paths."""
    environ_entries = Path("/proc/self/environ").read_bytes().split(b"\0")
    environ = dict(entry.decode(errors="replace").split("=", 1) for entry in environ_entries if b"=" in entry)
    selected_environ = " ".join(f"{key}={environ[key]!r}" for key in PROVENANCE_ENV_KEYS if key in environ)
    print(f"[repro] /proc/self/environ {selected_environ}", flush=True)

    loaded_paths = {
        line.rsplit(maxsplit=1)[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if any(library in line for library in PROVENANCE_LIBRARY_NAMES)
    }
    print(f"[repro] /proc/self/maps GPU libraries={sorted(loaded_paths)}", flush=True)


def build_mesh(
    *,
    dp_racks: int,
    nodes_per_rack: int,
    gpus_per_node: int,
    expert_axis_size: int = EXPERT_AXIS_SIZE,
    model_axis_size: int = MODEL_AXIS_SIZE,
) -> Mesh:
    """The compact ``(replica_dcn, data, expert, model)`` mesh.

    Builds the explicit mesh with the production
    launcher's arguments: ``replica_axis_size=dp_racks``, ``expert_axis_size=1``,
    ``model_axis_size=1``. ``data`` absorbs the remainder, so with the real
    16 nodes/rack x 4 GPUs/node the mesh is ``(dp_racks, 64, 1, 1)`` -- i.e.
    FSDP within a rack, pure data-parallel replication across racks.
    """
    replica_axis_size = dp_racks
    expected_devices = dp_racks * nodes_per_rack * gpus_per_node
    global_device_count = jax.device_count()
    if global_device_count != expected_devices:
        raise SystemExit(
            f"device count mismatch: jax.device_count()={global_device_count} but "
            f"dp_racks={dp_racks} x nodes_per_rack={nodes_per_rack} x "
            f"gpus_per_node={gpus_per_node} = {expected_devices}. "
            f"(On CPU, set XLA_FLAGS=--xla_force_host_platform_device_count=N and pass "
            f"--nodes-per-rack/--gpus-per-node to match.)"
        )

    fixed = replica_axis_size * expert_axis_size * model_axis_size
    if global_device_count % fixed != 0:
        raise SystemExit(
            f"global_device_count ({global_device_count}) must be divisible by "
            f"replica({replica_axis_size}) * expert({expert_axis_size}) * model({model_axis_size})"
        )
    data_axis_size = global_device_count // fixed
    shape = (replica_axis_size, data_axis_size, expert_axis_size, model_axis_size)

    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    axis_types = tuple(AxisType.Explicit for _ in GRUG_MESH_AXIS_NAMES)
    return Mesh(devices, GRUG_MESH_AXIS_NAMES, axis_types=axis_types)


# --------------------------------------------------------------------------
# Model: a stack of dense hidden_dim x hidden_dim layers plus an "lm head".
# No attention, no MoE, no custom kernels -- but the FSDP collective structure
# (per-layer all-gather forward, reduce-scatter + all-reduce backward) and the
# batch-axis all-reduce in the loss are the same as the real program's.
# --------------------------------------------------------------------------
def init_params(key, *, num_layers: int, hidden_dim: int, vocab_size: int):
    keys = jax.random.split(key, num_layers + 1)
    std = 0.5 / np.sqrt(hidden_dim)

    layers = []
    for i in range(num_layers):
        w = std * jax.random.truncated_normal(keys[i], -3.0, 3.0, (hidden_dim, hidden_dim), dtype=jnp.float32)
        # alternate the two dense specs the real model uses (in->out / out->in)
        spec = FSDP_IN_OUT_SPEC if i % 2 == 0 else FSDP_OUT_IN_SPEC
        layers.append(reshard(w, spec))

    head = std * jax.random.truncated_normal(keys[-1], -3.0, 3.0, (hidden_dim, vocab_size), dtype=jnp.float32)
    head = reshard(head, FSDP_IN_OUT_SPEC)
    return {"layers": layers, "head": head}


def rms_norm(h, eps=1e-6):
    """Paramless RMSNorm. The real model has RMSNorm + residual around every block;
    without them a 48-layer stack at initializer_std=0.5/sqrt(d) decays to zero and
    trains nothing (verified: loss pinned at exactly ln(vocab_size))."""
    v = jnp.mean(jnp.square(h.astype(jnp.float32)), axis=-1, keepdims=True)
    return (h.astype(jnp.float32) * jax.lax.rsqrt(v + eps)).astype(h.dtype)


def forward(params, x, *, compute_dtype):
    """Forward pass. Each `reshard(w, REPLICATED)` is one FSDP all-gather over `data`."""
    h = x.astype(compute_dtype)
    for w in params["layers"]:
        # params=float32 -> compute=bfloat16 (launch.py MIXED_PRECISION),
        # cast BEFORE the gather so the collective moves bf16, as jmp does.
        w_full = reshard(w.astype(compute_dtype), REPLICATED)
        h = h + jax.nn.gelu(rms_norm(h) @ w_full)  # pre-norm residual block
        h = reshard(h, BATCH_SPEC)  # pin activations to the batch axes
    head_full = reshard(params["head"].astype(compute_dtype), REPLICATED)
    return rms_norm(h) @ head_full


def loss_fn(params, x, labels, *, compute_dtype):
    logits = forward(params, x, compute_dtype=compute_dtype).astype(jnp.float32)
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    nll = -jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    # mean over the token axis == all-reduce over (replica_dcn, data, expert)
    return jnp.mean(nll)


# --------------------------------------------------------------------------
# Adam. State is sharded exactly like the params, so the update is local.
# --------------------------------------------------------------------------
def init_opt_state(params):
    zeros = lambda p: jnp.zeros_like(p)  # noqa: E731
    return {"m": jax.tree.map(zeros, params), "v": jax.tree.map(zeros, params), "count": jnp.asarray(0, jnp.int32)}


def adam_update(params, opt_state, grads, *, lr, b1=0.9, b2=0.95, eps=1e-8):
    count = opt_state["count"] + 1
    bc1 = 1.0 - b1 ** count.astype(jnp.float32)
    bc2 = 1.0 - b2 ** count.astype(jnp.float32)

    def upd(p, m, v, g):
        g = g.astype(jnp.float32)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * jnp.square(g)
        p = p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps)
        return p, m, v

    out = jax.tree.map(upd, params, opt_state["m"], opt_state["v"], grads, is_leaf=lambda t: isinstance(t, jax.Array))
    new_params = jax.tree.map(lambda t: t[0], out, is_leaf=lambda t: isinstance(t, tuple))
    new_m = jax.tree.map(lambda t: t[1], out, is_leaf=lambda t: isinstance(t, tuple))
    new_v = jax.tree.map(lambda t: t[2], out, is_leaf=lambda t: isinstance(t, tuple))
    return new_params, {"m": new_m, "v": new_v, "count": count}


# --------------------------------------------------------------------------
# Synthetic data, generated on device. Labels come from a fixed frozen
# "teacher" projection so the loss actually decreases (a repro that silently
# computes garbage is useless).
# --------------------------------------------------------------------------
def make_batch(key, teacher, *, global_batch: int, hidden_dim: int):
    """`teacher` is (hidden_dim, num_classes) with num_classes << vocab_size, so the
    task is learnable within a few dozen steps while the lm head keeps its real
    vocab_size (and therefore its real collective size)."""
    x = jax.random.normal(key, (global_batch, hidden_dim), dtype=jnp.float32)
    x = reshard(x, BATCH_SPEC)
    labels = jnp.argmax(x @ reshard(teacher, REPLICATED), axis=-1)
    return x, reshard(labels, P(BATCH_AXES))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp-racks", type=int, required=True, help="Data-parallel NVL72 rack count.")
    ap.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nodes-per-rack", type=int, default=NODES_PER_RACK)
    ap.add_argument("--gpus-per-node", type=int, default=GPUS_PER_NODE)
    ap.add_argument("--batch-per-rack", type=int, default=BATCH_PER_RACK)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help=(
            f"Tokens per sequence. The production job uses {MAX_SEQ_LEN}; the default here is 1 because "
            "this repro has no attention, so seq_len only scales activation size. Global token count is "
            "dp_racks * batch_per_rack * seq_len."
        ),
    )
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument(
        "--num-classes",
        type=int,
        default=64,
        help="Distinct synthetic teacher classes (<= vocab_size). Keeps the task learnable "
        "so the loss visibly decreases; the lm head still emits vocab_size logits.",
    )
    ap.add_argument("--expert-axis-size", type=int, default=EXPERT_AXIS_SIZE)
    ap.add_argument("--model-axis-size", type=int, default=MODEL_AXIS_SIZE)
    ap.add_argument("--compute-dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--print-sharding", action="store_true", help="Dump param/batch shardings after init.")
    ap.add_argument(
        "--distributed",
        choices=("auto", "on", "off"),
        default="auto",
        help="Call jax.distributed.initialize(). 'auto' does so when Slurm/OMPI/JAX coordinator "
        "env vars are present. Required for any multi-process (>1 node) run.",
    )
    args = ap.parse_args(argv)

    compute_dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32

    # Multi-process bring-up. The production job runs one process per node with 4 local
    # GPUs, i.e. 16 * dp_racks processes.
    #
    # Two supported paths, both dependency-free:
    #   1. Explicit: export JAX_COORDINATOR_ADDRESS=<host:port>, JAX_NUM_PROCESSES=N,
    #      JAX_PROCESS_ID=<0..N-1>. Use this under any launcher that exposes them.
    #   2. Auto: bare jax.distributed.initialize(), which auto-detects Slurm / OMPI.
    #      This is the zero-configuration path on a Slurm cluster.
    # Must happen before anything touches devices.
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    _AUTODETECT_ENV = ("SLURM_JOB_ID", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE")
    want_distributed = args.distributed == "on" or (
        args.distributed == "auto" and (coord is not None or any(v in os.environ for v in _AUTODETECT_ENV))
    )
    if want_distributed:
        if coord is not None:
            jax.distributed.initialize(
                coordinator_address=coord,
                num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
                process_id=int(os.environ["JAX_PROCESS_ID"]),
            )
        else:
            jax.distributed.initialize()
        print("[repro] jax.distributed.initialize() done", flush=True)

    print(f"[repro] jax {jax.__version__}  backend={jax.default_backend()}", flush=True)
    print(f"[repro] process {jax.process_index()}/{jax.process_count()}  devices={jax.device_count()}", flush=True)
    print(f"[repro] XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}", flush=True)
    print(
        f"[repro] JAX_ENABLE_PGLE={os.environ.get('JAX_ENABLE_PGLE')!r} "
        f"XLA_PYTHON_CLIENT_ALLOCATOR={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')!r}",
        flush=True,
    )
    print_runtime_provenance()

    mesh = build_mesh(
        dp_racks=args.dp_racks,
        nodes_per_rack=args.nodes_per_rack,
        gpus_per_node=args.gpus_per_node,
        expert_axis_size=args.expert_axis_size,
        model_axis_size=args.model_axis_size,
    )
    global_batch = args.dp_racks * args.batch_per_rack * args.seq_len
    batch_axis_devices = int(np.prod([mesh.shape[a] for a in BATCH_AXES]))
    if global_batch % batch_axis_devices != 0:
        raise SystemExit(f"global batch {global_batch} not divisible by batch-axis device count {batch_axis_devices}")

    print(f"[repro] mesh axes={mesh.axis_names} shape={tuple(mesh.shape.values())}", flush=True)
    print(
        f"[repro] global_batch={global_batch} rows  per-device={global_batch // batch_axis_devices} rows  "
        f"layers={args.num_layers} hidden={args.hidden_dim} vocab={args.vocab_size}",
        flush=True,
    )
    n_params = args.num_layers * args.hidden_dim**2 + args.hidden_dim * args.vocab_size
    print(f"[repro] params={n_params / 1e9:.3f}B (float32 = {n_params * 4 / 2**30:.1f} GiB replicated)", flush=True)

    with jax.sharding.set_mesh(mesh):
        key = jax.random.key(args.seed)
        pkey, tkey, dkey = jax.random.split(key, 3)

        params = init_params(pkey, num_layers=args.num_layers, hidden_dim=args.hidden_dim, vocab_size=args.vocab_size)
        opt_state = init_opt_state(params)
        num_classes = min(args.num_classes, args.vocab_size)
        teacher = jax.random.normal(tkey, (args.hidden_dim, num_classes), dtype=jnp.float32)
        teacher = reshard(teacher, REPLICATED)

        if args.print_sharding:
            print(f"[shard] layer0     {params['layers'][0].shape} {params['layers'][0].sharding.spec}", flush=True)
            print(f"[shard] layer1     {params['layers'][1].shape} {params['layers'][1].sharding.spec}", flush=True)
            print(f"[shard] head       {params['head'].shape} {params['head'].sharding.spec}", flush=True)
            print(f"[shard] adam m[0]  {opt_state['m']['layers'][0].sharding.spec}", flush=True)
            x0, y0 = make_batch(dkey, teacher, global_batch=global_batch, hidden_dim=args.hidden_dim)
            print(f"[shard] batch x    {x0.shape} {x0.sharding.spec}", flush=True)
            print(f"[shard] batch y    {y0.shape} {y0.sharding.spec}", flush=True)
            # Optional: needs `rich`, which we do not depend on. Best effort only.
            try:
                jax.debug.visualize_array_sharding(params["layers"][0])
                jax.debug.visualize_array_sharding(x0)
            except Exception as exc:  # pragma: no cover
                print(f"[shard] visualize_array_sharding unavailable ({exc})", flush=True)
            for name, arr in (("layer0", params["layers"][0]), ("batch x", x0)):
                shards = sorted((s.device.id, s.index) for s in arr.addressable_shards)
                print(f"[shard] {name} device->slice: {shards}", flush=True)

        @jax.jit(donate_argnums=(0, 1))
        def train_step(params, opt_state, x, labels):
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, labels, compute_dtype=compute_dtype))(params)
            params, opt_state = adam_update(params, opt_state, grads, lr=args.lr)
            return params, opt_state, loss

        print("[repro] starting train loop", flush=True)
        wall0 = time.perf_counter()
        for step in range(args.num_steps):
            dkey, bkey = jax.random.split(dkey)
            x, labels = make_batch(bkey, teacher, global_batch=global_batch, hidden_dim=args.hidden_dim)
            t0 = time.perf_counter()
            params, opt_state, loss = train_step(params, opt_state, x, labels)
            # Device -> host scalar sync every step, exactly like the real train
            # loop (train.py: block_until_ready + `if not jnp.isfinite(...)`).
            loss_value = float(loss)
            dt = time.perf_counter() - t0
            # The per-step progress line the wedge detector watches for silence.
            print(
                f"[step {step:5d}/{args.num_steps}] loss={loss_value:.6f} "
                f"step_time={dt:8.3f}s elapsed={time.perf_counter() - wall0:9.1f}s",
                flush=True,
            )
            if not np.isfinite(loss_value):
                print(f"[repro] non-finite loss at step {step}", flush=True)
                return 1

        print(f"[repro] done, {args.num_steps} steps in {time.perf_counter() - wall0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
