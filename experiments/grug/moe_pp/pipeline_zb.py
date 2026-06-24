# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device-group microbatched pipeline for the grug-MoE (toward zero-bubble).

Unlike :mod:`experiments.grug.moe_pp.pipeline_manual` (logical stages on one
shared mesh, no overlap), here each stage owns a DISJOINT group of devices, so
different stages run concurrently on different hardware -- the prerequisite for
any pipeline speedup. There is no ``stage`` mesh axis, so the stage-stacked
weight-grad that OOMs the GPU partitioner never forms.

Each stage's forward and backward are compiled ONCE (``jax.jit``) and the Python
scheduler calls those compiled functions per microbatch under the stage's
sub-mesh. Compiled stage calls on disjoint device slices dispatch asynchronously,
so the runtime overlaps them. Activations cross stage boundaries by an explicit
``jax.device_put`` to the next stage's sub-mesh; cotangents flow back the same
way. Each stage's blocks are rematerialized (via :func:`_stage_forward`), so only
stage-boundary activations are held and the backward recomputes block internals.

The op order is a :class:`Schedule`. The ``ZERO_BUBBLE`` (ZB-H1) path runs the whole
step from one interleaved :func:`pipeline_schedule`: F, B (input-grad) and W
(weight-grad) ops wavefront across the stages, the backward split so deferred W work
fills the bubble the last stages leave while the backward drains to stage 0.
``ONE_F_ONE_B`` interleaves F with the combined backward in the same wavefront (no W to
defer); ``GPIPE`` is the baseline -- a full forward sweep then a combined-``vjp``
backward sweep, two separate fills with nothing covering the bubble.
"""

from __future__ import annotations

import functools
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import _GRUG_MESH_AXIS_NAMES
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.pipeline_manual import (
    _embed_forward,
    _embed_head_tuple,
    _head_forward,
    _stage_forward,
)

_REPLICATED = P()
# Muon's production quintic Newton-Schulz coefficients (levanter.optim.util). The
# levanter helper pins a ``with_sharding_constraint`` that asserts (not reshards) under
# Explicit-axis meshes, so the iteration is inlined here; the math is identical.
_MUON_QUINTIC = NEWTON_SCHULZ_COEFFICIENTS["quintic"]


def _newton_schulz(x: jax.Array, eps: float = 1e-7) -> jax.Array:
    """Orthogonalize a 2D matrix via Muon's 5-step quintic Newton-Schulz iteration.

    The ``x @ x.T`` Gram matmul contracts ``x``'s wide dimension; ``out_sharding=P()``
    makes it replicated, so when that dimension is sharded (an FSDP grad) XLA emits the
    all-reduce -- the optimizer's grad exchange -- and when it is not (a single-device
    stage grad) the matmul is local. The rest of the iteration follows from the Gram.
    """
    x = x / (jnp.linalg.norm(x) + eps)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    for a, b, c in _MUON_QUINTIC:
        gram = jnp.matmul(x, x.T, out_sharding=_REPLICATED)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transpose else x


def _orthogonalize(g: jax.Array) -> jax.Array:
    """Muon-orthogonalize a weight-grad over its last two dims (the optimizer's heavy op).

    Vmaps any leading stack/expert dims; leaves of rank < 2 (norm scales, biases -- the
    adamw side of Muon) pass through. The iteration's ``x @ x.T`` matmuls contract a
    dimension that is sharded under a data-sharded (FSDP) mesh, so XLA inserts the
    cross-device reduction every step; under a single-device stage sub-mesh the same
    matmuls are local. Orthogonalizing the per-stage grads on each stage's own devices --
    instead of all-gathering the full model's grads -- is the cost the pipeline amortizes.
    """
    if g.ndim < 2:
        return g
    if g.ndim == 2:
        return _newton_schulz(g)
    flat = g.reshape((-1, *g.shape[-2:]))
    return jax.vmap(_newton_schulz)(flat).reshape(g.shape)


def orthogonalize_tree(tree):
    """Muon-orthogonalize every rank>=2 array leaf of a grad pytree (rank<2 passes through)."""
    return jax.tree_util.tree_map(_orthogonalize, tree)


# Diagnostic: when MOE_PP_TRACE=1, step() blocks after the forward sweep and again after
# the backward sweep and logs each wall time, to see whether a sweep overlaps stages or runs
# serially. Off by default (the block points would otherwise kill cross-step pipelining).
_TRACE = os.environ.get("MOE_PP_TRACE") == "1"
_t: dict = {}

# Diagnostic: when MOE_PP_NULL_XPORT=1, each cross-host boundary hop returns a local zeros
# array (no collective) instead of moving the activation. Gradients are wrong, but the step's
# op graph keeps its shape, so the step time isolates everything-but-the-cross-host-transport.
_NULL_XPORT = os.environ.get("MOE_PP_NULL_XPORT") == "1"


def _stage_submesh(group_devices, *, expert: int, data: int) -> Mesh:
    """A grug sub-mesh ``(stage, replica_dcn, data, expert, model)`` over one device slice."""
    arr = np.array(group_devices, dtype=object).reshape(1, 1, data, expert, 1)
    return Mesh(arr, _GRUG_MESH_AXIS_NAMES, axis_types=tuple(AxisType.Explicit for _ in _GRUG_MESH_AXIS_NAMES))


def _multihost() -> bool:
    return jax.process_count() > 1


def _make_global(full, mesh: Mesh, spec: P, shape, dtype) -> jax.Array:
    """Assemble a global ``jax.Array`` on ``(mesh, spec)`` from a host-local source.

    ``full`` is the entire array as a host-local value (numpy/single-host) -- identical
    on every process for replicated inputs (params, inputs), or the broadcast result of
    a cross-host transfer for activations -- or ``None`` when this process owns none of
    the target devices. Each process contributes only its local devices' shards, the
    multi-controller way to place data on a device set this process may not fully own.
    """
    sharding = NamedSharding(mesh, spec)
    shards = []
    if full is not None:
        host = np.asarray(full)
        idx_map = sharding.devices_indices_map(tuple(shape))
        for d in mesh.devices.flat:
            if d.process_index == jax.process_index():
                shards.append(jax.device_put(host[idx_map[d]], d))
    # ``dtype`` is required when this process contributes no shards (the target sub-mesh is
    # entirely on another host), since it cannot be inferred from an empty shard list.
    return jax.make_array_from_single_device_arrays(tuple(shape), sharding, shards, dtype=jnp.dtype(dtype))


def _put_params(tree, mesh: Mesh):
    """Replicate a param pytree onto ``mesh`` (fully replicated within the stage slice).

    Multi-host: params are identically initialized on every process, so each builds its
    local shards from its own host-local copy -- no cross-host transfer.
    """
    if not _multihost():
        return jax.device_put(tree, NamedSharding(mesh, _REPLICATED))
    return jax.tree_util.tree_map(lambda x: _make_global(x, mesh, _REPLICATED, x.shape, x.dtype), tree)


def _put_act(x, mesh: Mesh) -> jax.Array:
    """Place a HOST-LOCAL input (tokens/labels/weight) onto ``mesh``, batch-sharded.

    The input is identical on every process (same PRNG), so this needs no transfer; for
    runtime activations crossing stages use :func:`_transport` instead.
    """
    spec = grug_model._batch_spec()
    if not _multihost():
        return jax.device_put(x, NamedSharding(mesh, spec))
    return _make_global(x, mesh, spec, x.shape, x.dtype)


def _transport(x: jax.Array, mesh: Mesh, spec: P = grug_model._batch_spec()) -> jax.Array:
    """Move a runtime global array ``x`` onto ``mesh`` (batch sharding by default).

    Single-host: a plain ``device_put``. Multi-host: each stage sub-mesh lives entirely
    on one process, so ``x`` is fully addressable on the process(es) owning its devices;
    that process pulls it to the host, and when the target is on a different process the
    data crosses the wire via ``broadcast_one_to_all``. Every process then builds its local
    shards. Only the one host-boundary hop actually crosses; intra-host hops stay local. The
    on-device :func:`_build_ppermute_hop` carries the boundary activation without this host
    round-trip; ``_transport`` remains for intra-host moves and the embed/head/loss reductions.
    """
    if not _multihost():
        return jax.device_put(x, NamedSharding(mesh, spec))
    pid = jax.process_index()
    src_procs = {d.process_index for d in x.sharding.device_set}
    dst_procs = {d.process_index for d in mesh.devices.flat}
    full = np.asarray(jax.device_get(x)) if pid in src_procs else None
    if src_procs != dst_procs:
        src = min(src_procs)
        send = full if pid == src else np.zeros(x.shape, x.dtype)
        full = multihost_utils.broadcast_one_to_all(send, is_source=(pid == src))
    return _make_global(full if pid in dst_procs else None, mesh, spec, x.shape, x.dtype)


def _tree_sum(trees: list):
    """Sum a non-empty list of like-structured pytrees leafwise."""
    return functools.reduce(lambda a, b: jax.tree_util.tree_map(jnp.add, a, b), trees)


def _mesh_procs(mesh: Mesh) -> frozenset[int]:
    return frozenset(int(d.process_index) for d in mesh.devices.flat)


def _crosses_host(src: Mesh, dst: Mesh) -> bool:
    """Whether moving an activation from ``src`` to ``dst`` traverses the host boundary."""
    return _mesh_procs(src) != _mesh_procs(dst)


def _build_ppermute_hop(low_mesh: Mesh, high_mesh: Mesh):
    """Compile the on-device activation hop across one cross-host stage boundary.

    The two stage slices are the ``pp=0`` (``low_mesh``) and ``pp=1`` (``high_mesh``) ranks of
    a single boundary mesh, so ONE NCCL clique and one compiled ``ppermute`` carry both
    directions: the forward activation (low->high) and the backward cotangent (high->low). A
    single clique is deliberate -- two separate communicators over the same GPUs can deadlock
    on init-order across hosts.

    Returns ``transport(x, from_pp) -> y``. ``from_pp`` is the ``pp`` rank holding the live
    data (0 forward, 1 backward); that rank fills its slice from ``x``'s on-device shards (a
    reshape, never ``device_get``), the other supplies zeros, the ``ppermute`` swaps the two,
    and the arriving rank rebuilds the result onto its stage submesh from its local shards.
    Reassembling from local shards (not indexing ``by[to_pp]``) keeps the relabel host-local;
    a slice index would let XLA insert a cross-host gather the other process never joins.
    Every process calls it (it is a collective), mirroring :func:`_transport`.
    """
    low_devs = list(low_mesh.devices.flat)
    high_devs = list(high_mesh.devices.flat)
    dps = len(low_devs)
    boundary = Mesh(np.array(low_devs + high_devs, dtype=object).reshape(2, dps), ("pp", "data"))
    bspec = P("pp", "data")
    meshes = (low_mesh, high_mesh)
    procs = ({d.process_index for d in low_devs}, {d.process_index for d in high_devs})

    @jax.jit
    @functools.partial(shard_map, mesh=boundary, in_specs=bspec, out_specs=bspec)
    def _hop(a: jax.Array) -> jax.Array:
        return jax.lax.ppermute(a, "pp", [(0, 1), (1, 0)])

    # The off-source slice is filled with zeros the ppermute immediately overwrites, so their
    # value is irrelevant -- allocate once per (shape, device) and reuse across all crossings.
    zero_cache: dict = {}

    def _zeros_on(d, bshard, dtype):
        key = (d.id, bshard, dtype)
        if key not in zero_cache:
            zero_cache[key] = jax.device_put(jnp.zeros(bshard, dtype), d)
        return zero_cache[key]

    def transport(x: jax.Array, from_pp: int) -> jax.Array:
        pid = jax.process_index()
        to_pp = 1 - from_pp
        bshape = (2, *x.shape)
        bsharding = NamedSharding(boundary, bspec)
        bshard = bsharding.shard_shape(bshape)
        # Source rank: each local shard of x, reshaped (+leading pp dim) in place. Other rank
        # (and any local device off the source slice): zeros. Each process supplies only its own.
        is_source = pid in procs[from_pp]
        on_dev = {s.device: s.data.reshape((1, *s.data.shape)) for s in x.addressable_shards} if is_source else {}
        shards = [
            on_dev[d] if d in on_dev else _zeros_on(d, bshard, x.dtype)
            for d in boundary.devices.flat
            if d.process_index == pid
        ]
        bx = jax.make_array_from_single_device_arrays(bshape, bsharding, shards)
        by = _hop(bx)
        dst_mesh = meshes[to_pp]
        dst_sharding = NamedSharding(dst_mesh, grug_model._batch_spec())
        if pid in procs[to_pp]:
            arrived = {s.device: s.data.reshape(s.data.shape[1:]) for s in by.addressable_shards}
            out_shards = [arrived[d] for d in dst_mesh.devices.flat if d.process_index == pid]
            return jax.make_array_from_single_device_arrays(x.shape, dst_sharding, out_shards)
        return _make_global(None, dst_mesh, dst_sharding.spec, x.shape, x.dtype)

    return transport


class _TransportWorker:
    """Run cross-host boundary hops on a side thread so the main thread keeps dispatching.

    The synchronous boundary hop (``device_get`` then a cross-host ``broadcast``) blocks
    the single dispatch thread, serializing the pipeline at the host boundary. This worker
    moves that hop off-thread: the main thread submits the hop and gets a :class:`Future`,
    continues issuing other microbatches' stage compute, and blocks on the future only when
    it reaches the op that consumes the transported activation.

    Both hosts own one worker and submit hops in identical 1f1b ``op_order``; the worker
    processes its queue FIFO, so the two hosts' ``broadcast`` collectives stay paired and
    ordered -- the invariant that makes interleaved dispatch safe across hosts.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="moe-pp-transport", daemon=True)
        self._thread.start()

    def submit(self, x: jax.Array, mesh: Mesh, spec: P) -> Future:
        fut: Future = Future()
        self._q.put((fut, x, mesh, spec))
        return fut

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            fut, x, mesh, spec = job
            try:
                fut.set_result(_transport(x, mesh, spec))
            except BaseException as exc:  # surface to the consumer's .result()
                fut.set_exception(exc)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()


def _resolve(v):
    """Block for a transported value if it is a pending :class:`Future`, else pass through."""
    return v.result() if isinstance(v, Future) else v


# Op kinds in the wavefront schedule. F = stage forward, B = input-grad (the critical
# path, threaded upstream), W = weight-grad (deferrable, fills bubbles). When the
# backward is not split, B is the combined vjp and carries the weight-grad itself.
_F, _B, _W = "F", "B", "W"
_PRIORITY = {_B: 0, _F: 1, _W: 2}


class Schedule(StrEnum):
    """How the microbatched forward/backward ops are ordered across the stages.

    * ``GPIPE`` -- a full forward sweep, then a combined-``vjp`` backward sweep. Two
      separate pipeline fills; nothing covers the backward-warmup bubble. 3F FLOPs.
    * ``ONE_F_ONE_B`` -- F and the combined (input+weight) backward interleaved in one
      wavefront, so later microbatches' forwards overlap earlier ones' backwards. One
      warmup/drain bubble remains (no deferred W to fill it). 3F FLOPs.
    * ``ZERO_BUBBLE`` -- ZB-H1: the backward split into B (input-grad, critical path)
      and W (weight-grad, deferred), wavefronted so W fills the tail bubble. ~0 bubble,
      but the split uses two ``jax.vjp`` passes per stage, so 5F FLOPs (double forward
      recompute) -- a win only while the bubble it removes exceeds that extra compute.
    """

    GPIPE = "gpipe"
    ONE_F_ONE_B = "1f1b"
    ZERO_BUBBLE = "zb"


class TransportMode(StrEnum):
    """How a runtime activation crosses the one stage boundary that splits the two hosts.

    Single-host pipelines never cross, so the mode is moot. Multi-host:

    * ``INLINE`` -- a host-staged ``device_get`` -> ``broadcast_one_to_all`` -> rebuild, run
      synchronously on the dispatch thread (it blocks there until the hop completes).
    * ``ASYNC`` -- the same host-staged broadcast, run on a :class:`_TransportWorker` side
      thread so the main thread keeps dispatching; both hosts submit in identical ``op_order``
      so the broadcasts stay paired.
    * ``PPERMUTE`` -- on-device ``ppermute`` (NCCL send/recv -> GPUDirect RDMA over the
      fabric), no host round-trip. The fast path (see :func:`_build_ppermute_hop`).
    """

    INLINE = "inline"
    ASYNC = "async"
    PPERMUTE = "ppermute"


def pipeline_schedule(num_stages: int, num_microbatches: int, *, split_w: bool) -> list[tuple[str, int, int]]:
    """Wavefront op order: a list of ``(kind, microbatch, stage)`` to dispatch.

    The schedule is the heuristic. Each stage is a resource that runs one op per time
    slot; an op is eligible once its data deps are met:

    * ``F(m,s)`` needs ``F(m,s-1)`` (the upstream activation),
    * ``B(m,s)`` needs ``F(m,s)`` and ``B(m,s+1)`` (the downstream input-grad),
    * ``W(m,s)`` needs ``B(m,s)`` (it reuses the same cotangent ``B`` consumed).

    Per slot every free stage greedily takes its highest-priority eligible op with
    ``B > F > W``: advance the backward critical path first, otherwise push a forward,
    and only when neither is ready spend the slot on a deferred weight-grad. With
    ``split_w`` the backward emits separate B and W ops (ZB-H1): the last stage drains
    its B early and idles while the backward marches to stage 0, so its W work slides
    into that tail -- the bubble the plain forward-then-backward order leaves empty.
    Without ``split_w`` the backward is a single combined op (1F1B); there is no W to
    defer, so a warmup/drain bubble remains, but the forward/backward sweeps still merge
    into one wavefront.

    Forwards are issued eagerly (no run-ahead cap), so all ``M`` activations stay live --
    the memory cost the device-group pipeline already pays; this targets the bubble, not
    activation memory (a bounded ZB-2p variant would cap concurrent forwards).
    """
    kinds = (_F, _B, _W) if split_w else (_F, _B)

    def deps(op: tuple[str, int, int]) -> list[tuple[str, int, int]]:
        kind, m, s = op
        if kind == _F:
            return [(_F, m, s - 1)] if s > 0 else []
        if kind == _B:
            return [(_F, m, s)] + ([(_B, m, s + 1)] if s < num_stages - 1 else [])
        return [(_B, m, s)]

    remaining = {(k, m, s) for k in kinds for m in range(num_microbatches) for s in range(num_stages)}
    done: set[tuple[str, int, int]] = set()
    schedule: list[tuple[str, int, int]] = []
    while remaining:
        eligible = [op for op in remaining if all(d in done for d in deps(op))]
        picked = []
        for s in range(num_stages):
            cand = [op for op in eligible if op[2] == s]
            if cand:
                picked.append(min(cand, key=lambda op: (_PRIORITY[op[0]], op[1])))
        if not picked:
            raise RuntimeError("pipeline schedule deadlocked (dependency cycle)")
        schedule.extend(picked)
        remaining.difference_update(picked)
        done.update(picked)
    return schedule


class _StageFns:
    """Jitted forward + the three backward flavours for one stage, compiled once.

    All are called by the scheduler under the stage's sub-mesh (``set_mesh`` cannot
    live inside ``jax.jit``). Each backward recomputes the stage forward (block
    remat) from ``x`` before differentiating.

    * ``forward(params, x) -> (y, z)``
    * ``backward(params, x, dy, dz) -> (dparams, dx)`` -- combined (GPipe baseline)
    * ``b(params, x, dy, dz) -> dx`` -- INPUT-gradient only (the pipeline critical
      path; ``params`` held constant so no weight-grad matmul is computed)
    * ``w(params, x, dy, dz) -> dparams`` -- WEIGHT-gradient only (deferrable off the
      critical path; ``x`` held constant)

    Splitting B from W is the zero-bubble move: the combined backward gates each
    stage's ``dx`` on the expensive weight-grad, serializing the backward chain;
    computing ``dx`` alone keeps the chain cheap and the ``w`` work fills bubbles.
    """

    def __init__(self, block_static, masks, remat: bool = True):
        fwd = lambda params, x: _stage_forward(params, block_static, x, masks, remat)  # noqa: E731

        @jax.jit
        def forward(params, x):
            return fwd(params, x)

        @jax.jit
        def backward(params, x, dy, dz):
            _, vjp = jax.vjp(lambda p, h: fwd(p, h), params, x)
            return vjp((dy, dz))

        @jax.jit
        def b(params, x, dy, dz):
            _, vjp = jax.vjp(lambda h: fwd(params, h), x)
            (dx,) = vjp((dy, dz))
            return dx

        @jax.jit
        def w(params, x, dy, dz):
            _, vjp = jax.vjp(lambda p: fwd(p, x), params)
            (dparams,) = vjp((dy, dz))
            return dparams

        self.forward, self.backward, self.b, self.w = forward, backward, b, w


def zb_build(
    transformer: Transformer,
    *,
    num_stages: int,
    num_microbatches: int,
    expert_per_stage: int = 1,
    data_per_stage: int = 1,
    schedule: Schedule = Schedule.ZERO_BUBBLE,
    remat: bool = True,
    muon: bool = False,
    transport: TransportMode = TransportMode.INLINE,
):
    """Place params on per-stage sub-meshes and compile the stage fns ONCE.

    Returns a ``step(token_ids, loss_weight) -> (loss, embed_head_grads, block_grads)``
    closure that runs the device-group ``schedule``, reusing the placed params and
    compiled stage fns across calls. Hoisting this setup out of the step is what makes
    the pipeline timeable (and usable in a real training loop) instead of recompiling
    every iteration. See :class:`Schedule` for the op-order variants.

    With ``muon`` each block's weight-grad is Muon-orthogonalized (Newton-Schulz) on its
    own stage's devices as part of the step, so the optimizer's heavy matmuls pipeline
    across stages with no cross-stage all-gather -- the FSDP grad-exchange the pipeline
    is meant to amortize.

    Stages own disjoint device slices of ``expert_per_stage * data_per_stage`` devices
    each (so ``num_stages * expert_per_stage * data_per_stage == device_count``).

    ``transport`` selects how the one cross-host boundary hop moves (see
    :class:`TransportMode`): ``PPERMUTE`` (on-device, the fast path), ``ASYNC`` (host-staged
    broadcast off the dispatch thread), or ``INLINE`` (host-staged broadcast inline). No
    effect single-host -- intra-host hops are always a local ``device_put``.
    """
    devices = jax.devices()
    dps = expert_per_stage * data_per_stage
    if num_stages * dps != len(devices):
        raise ValueError(f"num_stages*{dps} ({num_stages * dps}) must equal device_count ({len(devices)})")
    submeshes = [
        _stage_submesh(devices[s * dps : (s + 1) * dps], expert=expert_per_stage, data=data_per_stage)
        for s in range(num_stages)
    ]

    cfg = transformer.config
    num_layers = cfg.num_layers
    if num_layers % num_stages != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by num_stages={num_stages}")
    lps = num_layers // num_stages
    coef = cfg.router_z_loss_coef

    base_mask = grug_model.AttentionMask.causal()
    short_mask, long_mask = grug_model._layer_attention_masks(base_mask, sliding_window=cfg.sliding_window)
    per_layer_masks = [long_mask if (i % 4 == 3) else short_mask for i in range(num_layers)]
    stage_masks = [tuple(per_layer_masks[s * lps : (s + 1) * lps]) for s in range(num_stages)]

    # --- params: embed/head tuple on stage 0 (embed) + last stage (head); blocks per stage ---
    embed_head = _embed_head_tuple(transformer)
    eh_arrays, eh_static = eqx.partition(embed_head, eqx.is_array)
    eh0 = _put_params(eh_arrays, submeshes[0])
    ehL = _put_params(eh_arrays, submeshes[-1])

    block_static = eqx.partition(transformer.blocks[0], eqx.is_array)[1]
    block_arrays = [eqx.partition(b, eqx.is_array)[0] for b in transformer.blocks]
    stage_params = [_put_params(block_arrays[s * lps : (s + 1) * lps], submeshes[s]) for s in range(num_stages)]

    # --- jitted stage / embed / head fns (compiled once, called per microbatch) ---
    stage_fns = [_StageFns(block_static, stage_masks[s], remat=remat) for s in range(num_stages)]
    mesh0, meshL = submeshes[0], submeshes[-1]

    @jax.jit
    def embed_fwd(params, tok):
        return _embed_forward(params, eh_static, tok)

    @jax.jit
    def embed_bwd(params, tok, dh):
        _, vjp = jax.vjp(lambda p: _embed_forward(p, eh_static, tok), params)
        return vjp(dh)

    @jax.jit
    def head_fwd(params, h, labels, w):
        return _head_forward(params, eh_static, h, labels, w)

    @jax.jit
    def head_bwd(params, h, labels, w, scale):
        _, vjp = jax.vjp(lambda p, hh: _head_forward(p, eh_static, hh, labels, w), params, h)
        return vjp(scale)

    inv_m = 1.0 / num_microbatches
    dz = jnp.asarray(inv_m * coef / num_layers, jnp.float32)
    split_w = schedule is Schedule.ZERO_BUBBLE
    use_schedule = schedule is not Schedule.GPIPE
    op_order = pipeline_schedule(num_stages, num_microbatches, split_w=split_w) if use_schedule else []
    muon_fn = jax.jit(orthogonalize_tree) if muon else None

    # On-device ppermute hops: one per cross-host boundary (keyed by its low stage), carrying
    # both the forward and backward crossing on a single clique. Compiled once, reused.
    ppermute_hops: dict[int, Callable[[jax.Array, int], jax.Array]] = {}
    if transport == TransportMode.PPERMUTE and _multihost():
        for s in range(num_stages - 1):
            if _crosses_host(submeshes[s], submeshes[s + 1]):
                ppermute_hops[s] = _build_ppermute_hop(submeshes[s], submeshes[s + 1])

    def step(token_ids: jax.Array, loss_weight: jax.Array) -> tuple[jax.Array | float, tuple | None, list]:
        """Run one pipelined forward+backward over the global batch; returns ``(loss, g_eh, g_blocks)``.

        For a non-GPipe ``schedule`` the F/B/W ops follow the wavefront; otherwise a
        GPipe forward sweep then combined backward. Either way the loss/grads match the
        non-pipelined oracle over the same global batch by construction (same embed /
        masks / blocks / head / router z-loss, averaged over microbatches), to float
        reassociation tolerance.
        """
        global_batch = token_ids.shape[0]
        if global_batch % num_microbatches != 0:
            raise ValueError(f"global_batch={global_batch} must divide by num_microbatches={num_microbatches}")
        mb = global_batch // num_microbatches
        if _TRACE:
            _t["start"] = time.perf_counter()

        # Per-microbatch inputs on the terminal stages' meshes (tokens feed stage 0;
        # labels/loss-weight feed the head on the last stage). They are computed as
        # HOST-LOCAL numpy (identical on every process), then placed onto the terminal
        # meshes. Keeping them numpy -- never a device array -- avoids a reshard when
        # slicing under multi-host (a global array is not host-addressable).
        tok_mb: list = [None] * num_microbatches
        labels_mb: list = [None] * num_microbatches
        weight_mb: list = [None] * num_microbatches
        tok_host = np.asarray(token_ids)
        weight_host = np.asarray(loss_weight, np.float32)
        for m in range(num_microbatches):
            tok_slice = tok_host[m * mb : (m + 1) * mb]
            labels_slice = np.concatenate([tok_slice[:, 1:], tok_slice[:, :1] * 0], axis=1).astype(np.int32)
            tok_mb[m] = _put_act(tok_slice, mesh0)
            labels_mb[m] = _put_act(labels_slice, meshL)
            weight_mb[m] = _put_act(weight_host[m * mb : (m + 1) * mb], meshL)

        # Activation/cotangent buffers, indexed [microbatch][stage]. ``saved_x`` is each
        # stage's block INPUT (embed output for stage 0); ``saved_dy`` the cotangent fed
        # into its backward (head seed for the last stage). Both are kept for every
        # microbatch so the W ops can reuse them whenever the schedule defers them.
        saved_x: list = [[None] * num_stages for _ in range(num_microbatches)]
        saved_dy: list = [[None] * num_stages for _ in range(num_microbatches)]
        head_h: list = [None] * num_microbatches
        ce_mb: list = [None] * num_microbatches
        z_mb: list = [[None] * num_stages for _ in range(num_microbatches)]
        w_grads: list = [[None] * num_stages for _ in range(num_microbatches)]
        g_embed_mb: list = [None] * num_microbatches
        g_head_mb: list = [None] * num_microbatches
        g_eh = None
        g_blocks: list = [None] * num_layers

        # Multi-host: every process walks the full op_order in lockstep, but COMPUTE for a
        # stage runs only on the process owning its sub-mesh; other processes supply a
        # placeholder so the shared transports (a cross-host broadcast inside ``_transport``)
        # still pair up. Single-host: ``_local`` is always true, so every guard passes and
        # this reduces to the original straight-line dispatch.
        seq = token_ids.shape[1]
        act_shape = (mb, seq, cfg.hidden_dim)

        def _local(mesh: Mesh) -> bool:
            return any(d.process_index == jax.process_index() for d in mesh.devices.flat)

        def _act_ph(mesh: Mesh) -> jax.Array:
            return _make_global(None, mesh, grug_model._batch_spec(), act_shape, jnp.float32)

        def _scalar_ph(mesh: Mesh) -> jax.Array:
            return _make_global(None, mesh, _REPLICATED, (), jnp.float32)

        if _multihost() and not use_schedule:
            raise NotImplementedError("multi-host pipeline requires a 1f1b/zb schedule (GPipe path is single-host)")

        def _accum_embed_head(g_embed_m, g_head_m, prev):
            # The head grad lives on the last stage's host; transport it to stage 0's host
            # (a broadcast on both processes) and accumulate there.
            g_head_on0 = jax.tree_util.tree_map(lambda g: _transport(g, mesh0, _REPLICATED), g_head_m)
            if not _local(mesh0):
                return prev
            g_eh_m = jax.tree_util.tree_map(jnp.add, g_embed_m, g_head_on0)
            return g_eh_m if prev is None else jax.tree_util.tree_map(jnp.add, prev, g_eh_m)

        # Host-staged fallback (``ppermute`` off): run each cross-host broadcast on a side
        # thread so the main thread keeps dispatching other microbatches while the hop is in
        # flight; the consuming op resolves the future just-in-time. Both hosts submit in
        # identical ``op_order`` so the broadcasts stay paired. ``worker`` is None when the flag
        # is off, single-host, or ppermute is on, so ``_send`` collapses to the inline path.
        worker = _TransportWorker() if (transport == TransportMode.ASYNC and _multihost()) else None

        def _send(x: jax.Array, dst: Mesh, src: Mesh, src_stage: int, dst_stage: int) -> jax.Array | Future:
            if not _crosses_host(src, dst):
                return _transport(x, dst)
            if _NULL_XPORT:
                spec = grug_model._batch_spec()
                if _local(dst):
                    return jax.device_put(jnp.zeros(x.shape, x.dtype), NamedSharding(dst, spec))
                return _make_global(None, dst, spec, x.shape, x.dtype)
            hop = ppermute_hops.get(min(src_stage, dst_stage))
            if hop is not None:
                # On-device ppermute: a jax.Array that XLA overlaps with compute -- no thread.
                # from_pp=0 forward (source is the low stage), 1 backward (source is the high).
                return hop(x, 0 if src_stage < dst_stage else 1)
            if worker is not None:
                return worker.submit(x, dst, grug_model._batch_spec())
            return _transport(x, dst)

        if use_schedule:
            # Wavefront: one interleaved dispatch following ``op_order``. F/B/W ops stream
            # across the stages; reductions are deferred so the single Python dispatch
            # thread never stalls (an in-loop tree_map serializes the GPUs). With
            # ``split_w`` the B op is input-grad only and W ops carry the weight-grad;
            # otherwise B is the combined vjp and writes the weight-grad itself.
            try:
                for kind, m, s in op_order:
                    if kind == _F:
                        if s == 0 and _local(mesh0):
                            with set_mesh(mesh0):
                                saved_x[m][0] = embed_fwd(eh0, tok_mb[m])
                        if _local(submeshes[s]):
                            with set_mesh(submeshes[s]):
                                h_out, z_mb[m][s] = stage_fns[s].forward(stage_params[s], _resolve(saved_x[m][s]))
                        else:
                            h_out, z_mb[m][s] = _act_ph(submeshes[s]), _scalar_ph(submeshes[s])
                        if s < num_stages - 1:
                            saved_x[m][s + 1] = _send(h_out, submeshes[s + 1], submeshes[s], s, s + 1)
                        else:
                            head_h[m] = h_out
                            if _local(meshL):
                                with set_mesh(meshL):
                                    ce_mb[m] = head_fwd(ehL, h_out, labels_mb[m], weight_mb[m])
                            else:
                                ce_mb[m] = _scalar_ph(meshL)
                    elif kind == _B:
                        if s == num_stages - 1:
                            if _local(meshL):
                                with set_mesh(meshL):
                                    g_head_mb[m], dy = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
                                saved_dy[m][s] = dy
                            else:
                                g_head_mb[m], saved_dy[m][s] = ehL, _act_ph(meshL)
                        if _local(submeshes[s]):
                            x_s, dy_s = _resolve(saved_x[m][s]), _resolve(saved_dy[m][s])
                            with set_mesh(submeshes[s]):
                                if split_w:
                                    dx = stage_fns[s].b(stage_params[s], x_s, dy_s, dz)
                                else:
                                    w_grads[m][s], dx = stage_fns[s].backward(stage_params[s], x_s, dy_s, dz)
                        else:
                            dx = _act_ph(submeshes[s])
                        if s > 0:
                            saved_dy[m][s - 1] = _send(dx, submeshes[s - 1], submeshes[s], s, s - 1)
                        elif _local(mesh0):
                            with set_mesh(mesh0):
                                (g_embed_mb[m],) = embed_bwd(eh0, tok_mb[m], dx)
                    else:  # _W (emitted only when split_w)
                        if _local(submeshes[s]):
                            x_s, dy_s = _resolve(saved_x[m][s]), _resolve(saved_dy[m][s])
                            with set_mesh(submeshes[s]):
                                w_grads[m][s] = stage_fns[s].w(stage_params[s], x_s, dy_s, dz)
            finally:
                if worker is not None:
                    worker.close()

            if _TRACE:
                # Enqueue time = the Python dispatch loop alone (no wait); the gap to ``sched``
                # is the GPU-completion tail. Splits dispatch-bound from compute-bound.
                _t["enqueue"] = time.perf_counter()
                jax.block_until_ready((ce_mb, z_mb, g_head_mb, g_embed_mb, w_grads))
                _t["sched"] = time.perf_counter()

            # Sum the embed/head weight-grads across microbatches LOCALLY, then cross the host
            # boundary once. The head grad is the [hidden, vocab] unembedding (~hundreds of MB);
            # accumulating per microbatch would broadcast it num_microbatches times.
            g_eh = _accum_embed_head(_tree_sum(g_embed_mb), _tree_sum(g_head_mb), None)
            for s in range(num_stages):
                if not _local(submeshes[s]):
                    continue
                base = s * lps
                acc = list(w_grads[0][s])
                for m in range(1, num_microbatches):
                    acc = [jax.tree_util.tree_map(jnp.add, a, b) for a, b in zip(acc, w_grads[m][s], strict=True)]
                for j, g in enumerate(acc):
                    g_blocks[base + j] = g
        else:
            # GPipe baseline: full forward sweep, then a combined-vjp backward (B and W
            # fused, so dx waits on the weight-grad and nothing fills the bubble).
            for m in range(num_microbatches):
                with set_mesh(mesh0):
                    h = embed_fwd(eh0, tok_mb[m])
                for s in range(num_stages):
                    if s > 0:
                        h = _transport(h, submeshes[s])
                    saved_x[m][s] = h
                    with set_mesh(submeshes[s]):
                        h, z_mb[m][s] = stage_fns[s].forward(stage_params[s], h)
                head_h[m] = h
                with set_mesh(meshL):
                    ce_mb[m] = head_fwd(ehL, h, labels_mb[m], weight_mb[m])

            if _TRACE:
                jax.block_until_ready((head_h, saved_x))
                _t["fwd"] = time.perf_counter()

            def _accum_blocks(base, g_slice):
                for j, g in enumerate(g_slice):
                    cur = g_blocks[base + j]
                    g_blocks[base + j] = g if cur is None else jax.tree_util.tree_map(jnp.add, cur, g)

            for m in range(num_microbatches):
                with set_mesh(meshL):
                    g_head_m, d_hidden = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
                for s in reversed(range(num_stages)):
                    if s < num_stages - 1:
                        d_hidden = _transport(d_hidden, submeshes[s])
                    with set_mesh(submeshes[s]):
                        g_slice, d_hidden = stage_fns[s].backward(stage_params[s], saved_x[m][s], d_hidden, dz)
                    _accum_blocks(s * lps, g_slice)
                d_hidden = _transport(d_hidden, mesh0)
                with set_mesh(mesh0):
                    (g_embed_m,) = embed_bwd(eh0, tok_mb[m], d_hidden)
                g_eh = _accum_embed_head(g_embed_m, g_head_m, g_eh)

        # Muon optimizer: orthogonalize each block's weight-grad on its own stage's
        # devices. Dispatched block-major (round-robin across stages) so all P stages
        # run Newton-Schulz concurrently -- no all-gather, unlike FSDP where each weight's
        # grad must be gathered across the data axis first.
        if muon_fn is not None:
            for j in range(lps):
                for s in range(num_stages):
                    if not _local(submeshes[s]):
                        continue
                    layer = s * lps + j
                    with set_mesh(submeshes[s]):
                        g_blocks[layer] = muon_fn(g_blocks[layer])

        # Loss (deferred reduction; the backward seeds are constants, so this never gates it).
        # Sum each stage's router z-loss across microbatches LOCALLY, then cross to the head mesh
        # once per stage (was a broadcast per microbatch x stage -- M*P crossings for scalars).
        # multi-host then broadcasts the final scalar so every process can read it.
        z_stage_on_head = [
            _transport(functools.reduce(jnp.add, (z_mb[m][s] for m in range(num_microbatches))), meshL, _REPLICATED)
            for s in range(num_stages)
        ]
        if _local(meshL):
            ce_total = functools.reduce(jnp.add, ce_mb)
            z_total = jnp.sum(jnp.stack(z_stage_on_head))
            loss = (ce_total + coef * z_total / num_layers) / num_microbatches
        else:
            loss = None
        if _multihost():
            src = np.asarray(jax.device_get(loss) if _local(meshL) else 0.0, np.float32)
            loss = float(multihost_utils.broadcast_one_to_all(src, is_source=_local(meshL)))

        if _TRACE:
            jax.block_until_ready((loss, g_eh, g_blocks))
            now = time.perf_counter()
            total = (now - _t["start"]) * 1e3
            if use_schedule:
                sched = (_t["sched"] - _t["start"]) * 1e3
                enqueue = (_t["enqueue"] - _t["start"]) * 1e3
                tail = sched - enqueue
                shape = f"M={num_microbatches} P={num_stages} ops={len(op_order)}"
                print(
                    f"TRACE {schedule.value} wavefront={sched:.0f}ms (enqueue={enqueue:.0f}ms gpu_tail={tail:.0f}ms) "
                    f"reduce={total - sched:.0f}ms total={total:.0f}ms ({shape})"
                )
            else:
                fwd, bwd = (_t["fwd"] - _t["start"]) * 1e3, (now - _t["fwd"]) * 1e3
                print(
                    f"TRACE gpipe fwd_sweep={fwd:.0f}ms bwd_sweep={bwd:.0f}ms total={total:.0f}ms "
                    f"(M={num_microbatches} P={num_stages})"
                )

        return loss, g_eh, g_blocks

    return step


def zb_value_and_grad(
    transformer: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    num_stages: int,
    num_microbatches: int,
    expert_per_stage: int = 1,
    data_per_stage: int = 1,
    schedule: Schedule = Schedule.ZERO_BUBBLE,
    remat: bool = True,
    muon: bool = False,
) -> tuple[jax.Array | float, tuple | None, list]:
    """One-shot ``(loss, embed_head_grads, block_grads)`` (builds + steps once).

    Convenience for tests; a training/perf loop should call :func:`zb_build` once and
    reuse the returned ``step`` across iterations.
    """
    step = zb_build(
        transformer,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        expert_per_stage=expert_per_stage,
        data_per_stage=data_per_stage,
        schedule=schedule,
        remat=remat,
        muon=muon,
    )
    return step(token_ids, loss_weight)
