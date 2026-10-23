"""Communication utilities.

This file currently provides the following functionalities, with code mainly
sourced from PyTorch:

1. Provides init process group capability without being restricted by default
   process group. PyTorch assumes users always use its default process group.
2. Provides CUDAIPC capability, which allows bypassing torch's multiprocessing
   to use GPU shared memory, for example to communicate with vllm workers using
   shared memory.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from typing import Any, Optional, Union, Tuple

import torch
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _get_default_group,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import torch.nn as nn
from torch.optim import Optimizer
from loguru import logger

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


def init_worker_process_group_with_device(timeout_seconds: int, backend: str = "nccl") -> None:
    """Pin this actor's CUDA device to its resolved ``LOCAL_RANK`` and create the default
    torch.distributed process group with an explicit ``device_id`` — so ProcessGroupNCCL never
    falls back to *"Guessing device ID based on global rank"*.

    Single source of truth shared by every backend's ``init_worker_process_group`` (the base
    Megatron policy/ref workers), so the device-pinned init is
    not duplicated (and cannot drift) across backends.

    ``LOCAL_RANK`` is resolved upstream by ``resolve_pinned_local_rank`` (workers/worker.py) for
    all three CUDA_VISIBLE_DEVICES regimes, so it is the correct ``set_device`` index in each.

    Why ``device_id`` matters: without it, ProcessGroupNCCL falls back to guessing the device
    from the global rank at first-collective time. That guess is not reliably correct across our
    clusters — on cw-rno2a the FIRST collective (the weight-init barrier) DEADLOCKED: rank 0 raced
    ahead into weight-loading while the other ranks stalled at the barrier (observed on
    keep-6 / cw-rno2a, 2026-07-20). NOTE — an earlier hypothesis blamed a masked-vs-unmasked CVD
    difference between clusters (rno2a supposedly leaving all GPUs visible); the live ``[pg-init]``
    logs DISPROVED it: Ray masks CVD to a single device on BOTH cw-us-east-02a AND cw-rno2a
    (``visible_device_count=1``, ``LOCAL_RANK=0``). The surviving explanation is init ordering — the
    barrier firing before an explicit ``set_device`` on rno2a's bring-up. Pinning ``set_device`` AND
    passing ``device_id`` makes the device explicit and correct regardless of CVD masking or init
    ordering, which is what unblocked both clusters (validated 2026-07-20: both progress past the
    weight-init collective into vLLM engine bring-up).

    Idempotent: always pins the device; only creates the process group if not already initialized.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # Surface the resolved rank/device + the observed CVD masking (visible_device_count)
    # so the pinning can be validated per cluster from the logs.
    logger.info(
        "[pg-init] RANK={} LOCAL_RANK={} CUDA_VISIBLE_DEVICES={} visible_device_count={} -> set_device(cuda:{})",
        os.environ.get("RANK", "?"),
        local_rank,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        torch.cuda.device_count(),
        local_rank,
    )
    if not torch.distributed.is_initialized():
        # Default torch dist pg init timeout is 10 minutes (600 s); callers pass
        # SKYRL_WORKER_NCCL_TIMEOUT_IN_S.
        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=timeout_seconds),
            device_id=torch.device("cuda", local_rank),
        )


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # Determine the accepted kwarg by INSPECTING the installed _new_process_group_helper
    # signature rather than a lexicographic version-string compare. The old
    # `str(torch.__version__) >= "2.6"` test breaks at torch 2.10/2.11 because
    # "2.11.0+cu130" < "2.6" lexicographically -> it wrongly picked "pg_options",
    # which torch 2.11 _new_process_group_helper rejects (TypeError: unexpected
    # keyword argument 'pg_options'). Signature inspection is correct on every
    # torch (2.9 baked SIF picks backend_options exactly as before -> no regression).
    import inspect as _inspect

    _helper_params = _inspect.signature(_new_process_group_helper).parameters
    if "backend_options" in _helper_params:
        pg_options_param_name = "backend_options"
    elif "pg_options" in _helper_params:
        pg_options_param_name = "pg_options"
    else:
        raise RuntimeError(
            "_new_process_group_helper accepts neither 'backend_options' nor "
            f"'pg_options' (torch {torch.__version__}); signature: {_inspect.signature(_new_process_group_helper)}"
        )
    # Force the standalone store-based ncclUniqueId rendezvous path for THIS group, by
    # temporarily un-binding the caller's default process group's device across the helper call.
    #
    # This process group spans processes that live in SEPARATE torch.distributed worlds — the
    # trainer's world plus each inference engine's world — so it is NOT a subgroup of any existing
    # world. We create it by calling ``_new_process_group_helper`` directly with an empty
    # ``global_ranks_in_group`` ([]) so torch treats it as a "default-style" standalone group.
    #
    # ``_new_process_group_helper`` auto-enables an ``ncclCommSplit`` optimization whenever the
    # CALLER's own default process group is eager/device-bound (its ``bound_device_id`` is set):
    #
    #     if is_initialized() and _get_default_group().bound_device_id:
    #         split_from = _get_split_source(_get_default_group())
    #     ...
    #     if split_from:  # -> ProcessGroupNCCL created via ncclCommSplit(parent=default_comm)
    #
    # The megatron policy worker pins its default PG with an explicit ``device_id``
    # (``init_worker_process_group_with_device``). So on the TRAINER (rank 0) this makes the
    # weight-sync comm get built by SPLITTING from the trainer's default group — which contains
    # only the trainer's ranks, NOT the engines. The inference engines' default PG is not
    # device-bound, so THEY build the same weight-sync comm the plain way: a fresh
    # ``ncclCommInitRank`` whose ncclUniqueId is exchanged through the c10d store. The two
    # comm-creation paths are mutually incompatible for this cross-world group: rank 0 never
    # publishes (``store->set``) the uniqueId the engine ranks block on (``store->get('0')``), so
    # every rank deadlocks in ``broadcastUniqueNCCLID`` until the 30-minute store timeout fires and
    # the group dies before the first rollout (reproduced on cw-us-east-02a AND cw-rno2a,
    # 2026-07-20). Binding a ``device_id`` to THIS group did not help because the split is triggered
    # by the DEFAULT group's ``bound_device_id``, not this group's.
    #
    # Un-binding the default group's device for the duration of the helper call forces
    # ``split_from = None`` on every rank, so trainer and engines take the IDENTICAL fresh
    # store-uniqueId ``ncclCommInitRank`` path and rank 0 reliably ``store->set``s the uniqueId
    # before the engines ``store->get`` it. The default group is restored immediately afterward and
    # is otherwise unchanged; the new group is created device-unbound (as before), so no other path
    # is affected.
    default_pg = _get_default_group() if torch.distributed.is_initialized() else None
    saved_bound_device_id = default_pg.bound_device_id if default_pg is not None else None
    if default_pg is not None and saved_bound_device_id is not None:
        default_pg.bound_device_id = None
    try:
        pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name=group_name,
            **{pg_options_param_name: pg_options},
            timeout=timeout,
        )
    finally:
        if default_pg is not None and saved_bound_device_id is not None:
            default_pg.bound_device_id = saved_bound_device_id

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class CUDAIPCHandle:
    def __init__(
        self,
        tensor_type: type,
        size: tuple,
        stride: tuple,
        offset: int,
        storage_type: type,
        dtype: torch.dtype,
        device: torch.device,
        handle: bytes,
        storage_size_bytes: bytes,
        storage_offset_bytes: bytes,
        requires_grad: bool,
        ref_counter_handle: bytes,
        ref_counter_offset: bytes,
        event_handle: bytes,
        event_sync_required: bool,
    ):
        self.tensor_type = tensor_type
        self.size = size
        self.stride = stride
        self.offset = offset
        self.storage_type = storage_type
        self.dtype = dtype
        self.device = device
        self.handle = handle
        self.storage_size_bytes = storage_size_bytes
        self.storage_offset_bytes = storage_offset_bytes
        self.requires_grad = requires_grad
        self.ref_counter_handle = ref_counter_handle
        self.ref_counter_offset = ref_counter_offset
        self.event_handle = event_handle
        self.event_sync_required = event_sync_required

    def rebuild(self) -> torch.Tensor:
        # NOTE: Rebuild within the same process is not thread-safe and will
        #       likely crash (segfault core dump).
        return rebuild_cuda_tensor(
            self.tensor_type,
            self.size,
            self.stride,
            self.offset,
            self.storage_type,
            self.dtype,
            self.device,
            self.handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.requires_grad,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> CUDAIPCHandle:
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device to use CUDAIPC")
        tensor = tensor.share_memory_()
        storage = tensor._typed_storage()
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        return CUDAIPCHandle(
            tensor_type=type(tensor),
            size=tensor.size(),
            stride=tensor.stride(),
            # tensor offset in its storage
            offset=tensor_offset,
            storage_type=type(storage),
            dtype=tensor.dtype,
            device=device,
            # identifier which CUDA allocation is the storage in.
            handle=handle,
            # size(in bytes) of the storage
            storage_size_bytes=storage_size_bytes,
            # offset(in bytes) of the storage in the CUDA allocation
            storage_offset_bytes=storage_offset_bytes,
            requires_grad=tensor.requires_grad,
            ref_counter_handle=ref_counter_handle,
            ref_counter_offset=ref_counter_offset,
            event_handle=event_handle,
            event_sync_required=event_sync_required,
        )
