# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ProcessGroupNCCL with direct NCCL calls using CPU-only rendezvous.

Launch this worker with torchrun. The direct mode uses ncclCommInitRank and
ncclAllReduce through the loaded C library; it creates no NCCL process group or
split communicator. There is no Ray import, model, optimizer or publication.
"""

import argparse
import ctypes
import json
import os
import socket
import time
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import ClassVar

import torch
import torch.distributed as dist

from experiments.post_training.long_loop.nccl_runtime_receipts import runtime_receipt
from experiments.post_training.long_loop.replica_parameter_receipts import parameter_sha256

NCCL_OPERATIONS = {"SUM": 0, "MAX": 2, "MIN": 3}
DIST_OPERATIONS = {"SUM": dist.ReduceOp.SUM, "MAX": dist.ReduceOp.MAX, "MIN": dist.ReduceOp.MIN}


class _NcclUniqueId(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [("internal", ctypes.c_ubyte * 128)]


def check_nccl(library: ctypes.CDLL, status: int) -> None:
    if status != 0:
        raise RuntimeError(f"NCCL C API error {status}: {library.ncclGetErrorString(status).decode()}")


def direct_reduce(tensor: torch.Tensor, operation: str, *, library: ctypes.CDLL, communicator: ctypes.c_void_p) -> None:
    check_nccl(
        library,
        library.ncclAllReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            {torch.int32: 2, torch.float32: 7}[tensor.dtype],
            NCCL_OPERATIONS[operation],
            communicator,
            torch.cuda.current_stream(tensor.device).cuda_stream,
        ),
    )


@contextmanager
def direct_reducer(world: int, rank: int, library_path: str):
    library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
    library.ncclGetErrorString.argtypes = [ctypes.c_int]
    library.ncclGetErrorString.restype = ctypes.c_char_p
    library.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
    library.ncclGetUniqueId.restype = ctypes.c_int
    library.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
    library.ncclCommInitRank.restype = ctypes.c_int
    library.ncclAllReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.ncclAllReduce.restype = ctypes.c_int
    library.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    library.ncclCommDestroy.restype = ctypes.c_int
    unique_id = _NcclUniqueId()
    if rank == 0:
        check_nccl(library, library.ncclGetUniqueId(ctypes.byref(unique_id)))
    payload = [ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id)) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    unique_id = _NcclUniqueId.from_buffer_copy(payload[0])
    communicator = ctypes.c_void_p()
    check_nccl(library, library.ncclCommInitRank(ctypes.byref(communicator), world, unique_id, rank))
    try:
        yield partial(direct_reduce, library=library, communicator=communicator)
    finally:
        check_nccl(library, library.ncclCommDestroy(communicator))


def process_group_reduce(tensor: torch.Tensor, operation: str, *, group: dist.ProcessGroup) -> None:
    dist.all_reduce(tensor, op=DIST_OPERATIONS[operation], group=group)


def run(args) -> None:
    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[local_rank]
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.api}-{args.profile}-rank{rank}"
    os.environ.update(
        NCCL_DEBUG="TRACE",
        NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,TUNING,COLL,CALL,NET,NVLS",
        NCCL_DEBUG_FILE=str(out / f"{stem}.log"),
    )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    report = {
        "api": args.api,
        "profile": args.profile,
        "rank": rank,
        "host": socket.gethostname(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "cases": [],
        "completed": False,
    }
    try:
        report["runtime"] = runtime_receipt()
        libraries = report["runtime"]["loaded_nccl_libraries"]
        if len(libraries) != 1:
            raise RuntimeError("Direct library qualification requires exactly one loaded libnccl")
        if args.expected_library_sha256 and libraries[0]["sha256"] != args.expected_library_sha256:
            raise RuntimeError("The loaded NCCL library does not match the pinned test artifact")
        with ExitStack() as resources:
            if args.api == "direct-nccl":
                reduce = resources.enter_context(direct_reducer(world, rank, libraries[0]["path"]))
            else:
                group = dist.new_group(
                    list(range(world)), backend="nccl", timeout=timedelta(seconds=120), device_id=device
                )
                reduce = partial(process_group_reduce, group=group)
            for nbytes in args.message_bytes:
                for dtype in (torch.int32, torch.float32):
                    count = nbytes // 4
                    base = torch.arange(count, dtype=torch.int32, device=device).remainder_(128)
                    source = (base + rank).to(dtype)
                    before = parameter_sha256(source)
                    for operation in (("MIN", "MAX", "SUM") if dtype == torch.int32 else ("SUM",)):
                        result = source.clone()
                        started = time.monotonic()
                        reduce(result, operation)
                        torch.cuda.synchronize()
                        seconds = time.monotonic() - started
                        expected = (
                            base
                            if operation == "MIN"
                            else base + world - 1 if operation == "MAX" else base * world + world * (world - 1) // 2
                        ).to(dtype)
                        errors = int(result.ne(expected).sum().item())
                        indices = (
                            result.ne(expected).nonzero().flatten()[:16]
                            if errors
                            else torch.empty(0, dtype=torch.int64, device=device)
                        )
                        report["cases"].append(
                            {
                                "dtype": str(dtype),
                                "nbytes": nbytes,
                                "elements": count,
                                "operation": operation,
                                "seconds": seconds,
                                "input_sha256": before,
                                "input_after_sha256": parameter_sha256(source),
                                "output_sha256": parameter_sha256(result),
                                "expected_sha256": parameter_sha256(expected),
                                "members": list(range(world)),
                                "errors": errors,
                                "first_incorrect_elements": indices.tolist(),
                                "passed": errors == 0 and before == parameter_sha256(source),
                            }
                        )
                    (out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
            report["completed"] = True
            report["passed"] = all(case["passed"] for case in report["cases"])
            (out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
            dist.barrier()
    finally:
        (out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=("direct-nccl", "process-group"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--expected-library-sha256", default="")
    parser.add_argument("--message-bytes", nargs="+", type=int, default=(16 << 20, 32 << 20, 64 << 20))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
