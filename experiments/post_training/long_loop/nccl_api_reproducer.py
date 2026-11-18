# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ProcessGroupNCCL with direct NCCL calls using CPU-only rendezvous.

Launch this worker with torchrun. The direct mode uses ncclCommInitRank and
ncclAllReduce through torch.cuda.nccl; it creates no NCCL process group or
split communicator. There is no Ray import, model, optimizer or publication.
"""

import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

from experiments.post_training.long_loop.nccl_runtime_receipts import runtime_receipt
from experiments.post_training.long_loop.replica_parameter_receipts import parameter_sha256

NCCL_OPERATIONS = {"SUM": 0, "MAX": 2, "MIN": 3}
DIST_OPERATIONS = {"SUM": dist.ReduceOp.SUM, "MAX": dist.ReduceOp.MAX, "MIN": dist.ReduceOp.MIN}


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
        if args.api == "direct-nccl":
            payload = [nccl.unique_id() if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            communicator = nccl.init_rank(world, payload[0], rank)
            group = None
        else:
            group = dist.new_group(list(range(world)), backend="nccl", timeout=timedelta(seconds=120), device_id=device)
            communicator = None
        report["runtime"] = runtime_receipt()
        for nbytes in (16 << 20, 32 << 20, 64 << 20):
            for dtype in (torch.int32, torch.float32):
                count = nbytes // 4
                base = torch.arange(count, dtype=torch.int32, device=device).remainder_(128)
                source = (base + rank).to(dtype)
                before = parameter_sha256(source)
                for operation in ("MIN", "MAX", "SUM"):
                    result = source.clone()
                    started = time.monotonic()
                    if args.api == "direct-nccl":
                        nccl.all_reduce(
                            [result],
                            [result],
                            op=NCCL_OPERATIONS[operation],
                            streams=[torch.cuda.current_stream()],
                            comms=[communicator],
                        )
                    else:
                        dist.all_reduce(result, op=DIST_OPERATIONS[operation], group=group)
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
