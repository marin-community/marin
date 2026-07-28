"""Optimistic bulk HBM-to-HBM policy-transfer screen.

One self-contained PyTorch/NCCL harness. It moves a fixed number of unique
BF16 bytes (``S = 2 * params``) from source ranks to destination ranks over an
already-initialised NCCL communicator, and reports the receiver makespan.

It deliberately does not load a checkpoint, run model kernels, or reproduce
MarinSkyRL's per-parameter metadata RPC cadence. It measures the bulk data
plane only, so its results are an optimistic floor, never an end-to-end
refresh latency.

Modes
-----
p2p        one source rank -> one destination rank; unique bytes = S
broadcast  one source rank -> K destination ranks; unique bytes = S,
           logical receiver delivery = K * S (reported separately)
striped    N source ranks -> N destination ranks, pairwise disjoint shards;
           unique bytes = S, each pair carries S / N

Rank layout is static and explicit: ranks ``[0, num_source)`` are sources and
ranks ``[num_source, world_size)`` are destinations. Each side runs as its own
Iris job; ``--master-addr`` joins them. Within a job, this file is its own
per-node launcher: it re-executes itself once per local GPU.
"""

import argparse
import datetime
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

BYTES_PER_BF16 = 2
DEFAULT_PARAMS = 359.6e9
CHILD_MARKER = "PTB_LOCAL_RANK"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["p2p", "broadcast", "striped"], required=True)
    p.add_argument("--params", type=float, default=DEFAULT_PARAMS, help="parameter count P; payload S = 2P bytes")
    p.add_argument("--payload-fraction", type=float, default=1.0, help="run this fraction of S (calibration only)")
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--num-source", type=int, required=True, help="ranks [0, num_source) are sources")
    p.add_argument("--rank-base", type=int, required=True, help="global rank of this job's task 0, local rank 0")
    p.add_argument("--local-ranks", type=int, default=1, help="processes (GPUs) each task of this job launches")
    p.add_argument("--active-source", type=int, default=0, help="active source ranks (0 = all sources)")
    p.add_argument("--active-dest", type=int, default=0, help="active destination ranks (0 = all destinations)")
    p.add_argument("--master-addr", required=True)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--chunk-bytes", type=int, default=1 << 30, help="reusable per-rank CUDA buffer size")
    p.add_argument("--inflight", type=int, default=2, help="reusable buffers kept in flight (fixed max allocation)")
    p.add_argument("--reps", type=int, default=5, help="warm measured repetitions of the full stream")
    p.add_argument("--init-timeout", type=int, default=3600, help="process-group init/collective timeout (seconds)")
    p.add_argument("--tag", default="", help="free-form label recorded in the report")
    p.add_argument("--report-json", default="", help="rank 0 writes the result object here")
    return p.parse_args()


def task_index() -> int:
    """Iris task index, parsed from IRIS_TASK_ID (``<job path>/<index>:<attempt>``)."""
    raw = os.environ.get("IRIS_TASK_ID", "")
    if not raw:
        return 0
    return int(raw.rsplit("/", 1)[-1].split(":", 1)[0])


def supervise(local_ranks: int) -> int:
    """Spawn one child per local GPU and mirror the group's lifecycle."""
    children = []
    lock = threading.Lock()

    def pump(rank: int, stream) -> None:
        for line in iter(stream.readline, ""):
            with lock:
                sys.stdout.write(f"[local{rank}] {line}")
                sys.stdout.flush()
        stream.close()

    for local_rank in range(local_ranks):
        env = {**os.environ, CHILD_MARKER: str(local_rank)}
        child = subprocess.Popen(
            [sys.executable, *sys.argv], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        children.append(child)
        threading.Thread(target=pump, args=(local_rank, child.stdout), daemon=True).start()

    def forward(signum, _frame):
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward)

    first_failure = 0
    for local_rank, child in enumerate(children):
        code = child.wait()
        print(f"[supervisor] local rank {local_rank} exited {code}", flush=True)
        if code != 0 and first_failure == 0:
            first_failure = code
            for peer in children:
                if peer.poll() is None:
                    peer.send_signal(signal.SIGTERM)
    return first_failure


def env_probe(torch) -> dict:
    """Facts that pin the runtime and the physical placement of this process."""
    out = {
        "hostname": socket.gethostname(),
        "machine": os.uname().machine,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "device_count": torch.cuda.device_count(),
        "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
        "nccl_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")},
    }
    try:
        out["ib_devices"] = sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        out["ib_devices"] = []
    return out


def make_buffers(torch, count: int, chunk_bytes: int, device) -> list:
    """Bounded reusable BF16 HBM buffers, initialised so nothing is a zero page."""
    numel = chunk_bytes // BYTES_PER_BF16
    bufs = []
    for i in range(count):
        t = torch.empty(numel, dtype=torch.bfloat16, device=device)
        t.normal_(mean=0.0, std=0.02)
        t[0] = float(i + 1)
        bufs.append(t)
    return bufs


def run_p2p(dist, bufs, n_chunks, rank, src, dst, group) -> None:
    inflight = []
    for i in range(n_chunks):
        if len(inflight) == len(bufs):
            inflight.pop(0).wait()
        buf = bufs[i % len(bufs)]
        inflight.append(dist.isend(buf, dst, group=group) if rank == src else dist.irecv(buf, src, group=group))
    for w in inflight:
        w.wait()


def run_broadcast(dist, bufs, n_chunks, src, group) -> None:
    for i in range(n_chunks):
        dist.broadcast(bufs[i % len(bufs)], src=src, group=group)


def run_striped(dist, bufs, n_chunks, peer, is_source, group) -> None:
    inflight = []
    for i in range(n_chunks):
        if len(inflight) == len(bufs):
            inflight.pop(0).wait()
        buf = bufs[i % len(bufs)]
        op = dist.P2POp(dist.isend if is_source else dist.irecv, buf, peer, group=group)
        inflight.append(dist.batch_isend_irecv([op])[0])
    for w in inflight:
        w.wait()


def main() -> int:
    args = parse_args()
    if CHILD_MARKER not in os.environ and args.local_ranks > 1:
        return supervise(args.local_ranks)

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get(CHILD_MARKER, "0"))
    rank = args.rank_base + task_index() * args.local_ranks + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    probe = env_probe(torch)
    probe.update({"rank": rank, "local_rank": local_rank, "gpu_name": torch.cuda.get_device_name(local_rank)})
    print("[bench] env " + json.dumps(probe), flush=True)

    t0 = time.monotonic()
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=args.world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=args.init_timeout),
    )
    setup_seconds = time.monotonic() - t0
    print(f"[bench] rank={rank} process group up in {setup_seconds:.2f}s", flush=True)

    n_src = args.active_source or args.num_source
    n_dst = args.active_dest or (args.world_size - args.num_source)
    if args.mode == "p2p":
        n_src = n_dst = 1
    if args.mode == "striped" and n_src != n_dst:
        raise ValueError(f"striped needs equal widths, got {n_src} sources and {n_dst} destinations")

    src_ranks = list(range(n_src))
    dst_ranks = list(range(args.num_source, args.num_source + n_dst))
    members = src_ranks + dst_ranks
    # Every rank in the world must call new_group; only members get a usable handle.
    group = dist.new_group(ranks=members, backend="nccl")
    active = rank in members
    is_source = rank in src_ranks

    total_bytes = int(args.params * BYTES_PER_BF16 * args.payload_fraction)
    pairs = n_src if args.mode == "striped" else 1
    n_chunks = max(1, -(-(total_bytes // pairs) // args.chunk_bytes))
    per_pair_bytes = n_chunks * args.chunk_bytes
    unique_bytes = per_pair_bytes * pairs

    if not active:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    bufs = make_buffers(torch, args.inflight, args.chunk_bytes, device)

    # First collective on this communicator, timed apart from the measured stream.
    t0 = time.monotonic()
    dist.all_reduce(torch.ones(1, dtype=torch.float32, device=device), group=group)
    torch.cuda.synchronize()
    first_collective_seconds = time.monotonic() - t0
    print(f"[bench] rank={rank} first collective {first_collective_seconds:.3f}s", flush=True)

    peer = None
    if args.mode == "striped":
        idx = src_ranks.index(rank) if is_source else dst_ranks.index(rank)
        peer = dst_ranks[idx] if is_source else src_ranks[idx]

    samples = []
    for rep in range(args.reps):
        torch.cuda.synchronize()
        dist.barrier(group=group)
        t0 = time.monotonic()
        if args.mode == "p2p":
            run_p2p(dist, bufs, n_chunks, rank, src_ranks[0], dst_ranks[0], group)
        elif args.mode == "broadcast":
            run_broadcast(dist, bufs, n_chunks, src_ranks[0], group)
        else:
            run_striped(dist, bufs, n_chunks, peer, is_source, group)
        torch.cuda.synchronize()
        local_seconds = time.monotonic() - t0

        # Makespan is the slowest required receiver, not a host-clock difference.
        recv = torch.tensor([0.0 if is_source else local_seconds], dtype=torch.float64, device=device)
        dist.all_reduce(recv, op=dist.ReduceOp.MAX, group=group)
        makespan = float(recv.item())
        samples.append(makespan)
        print(
            f"[bench] rank={rank} rep={rep} local={local_seconds:.3f}s makespan={makespan:.3f}s "
            f"unique_rate={unique_bytes / makespan / 1e9:.3f} GB/s",
            flush=True,
        )

    if rank == 0:
        median = statistics.median(samples)
        result = {
            "tag": args.tag,
            "mode": args.mode,
            "params": args.params,
            "payload_fraction": args.payload_fraction,
            "requested_unique_bytes": total_bytes,
            "actual_unique_bytes": unique_bytes,
            "per_pair_bytes": per_pair_bytes,
            "chunk_bytes": args.chunk_bytes,
            "chunks_per_pair": n_chunks,
            "inflight_buffers": args.inflight,
            "world_size": args.world_size,
            "num_source_ranks_in_world": args.num_source,
            "active_source_ranks": src_ranks,
            "active_dest_ranks": dst_ranks,
            "setup_seconds": setup_seconds,
            "first_collective_seconds": first_collective_seconds,
            "makespan_samples_seconds": samples,
            "makespan_median_seconds": median,
            "makespan_min_seconds": min(samples),
            "makespan_max_seconds": max(samples),
            "unique_rate_median_GBps": unique_bytes / median / 1e9,
            "unique_rate_best_GBps": unique_bytes / min(samples) / 1e9,
            "logical_delivery_rate_median_GBps": (
                unique_bytes * len(dst_ranks) / median / 1e9 if args.mode == "broadcast" else None
            ),
            "env": probe,
        }
        print("[bench] RESULT " + json.dumps(result), flush=True)
        if args.report_json:
            with open(args.report_json, "w") as fh:
                json.dump(result, fh, indent=2)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
