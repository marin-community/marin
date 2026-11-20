# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare known-byte NCCL MIN/MAX under Ray and torchrun, without weights."""

import argparse
import json
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import ray
import torch
import torch.distributed as dist
from cloud.iris.ray_ports import RayPort, allocated_ray_port, resolve_ray_ports
from cloud.iris.ray_storage import RaySpillBackend, resolve_ray_spill_target
from cloud.iris.task_runtime import (
    RayLogSyncSession,
    pin_socket_ifname,
    ray_start_head,
    ray_stop,
    run_worker,
    training_driver_env,
    wait_for_nodes,
    write_head_result,
    write_rendezvous,
)
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from skyrl_train.weight_sync.expert_block.verify_weights import compare_replicas

from experiments.post_training.long_loop.replica_parameter_receipts import parameter_sha256
from experiments.post_training.nccl_transport_probe import allocate_port_block, wait_for_rendezvous

WORLD_SIZE = 8
GPUS_PER_NODE = 4
COLLECTIVE_TIMEOUT = 90
MODULE = "experiments.post_training.long_loop.comparator_differential"


def write_json(uri: str, payload) -> None:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "w") as stream:
        json.dump(payload, stream, indent=2)


def variant_name(launcher: str, visibility: str, buffer: str) -> str:
    return f"{launcher}-{visibility}-buffer-{buffer}"


def known_bytes(count: int, offset: int, device: str) -> torch.Tensor:
    return ((torch.arange(count, dtype=torch.int64, device=device) + offset) % 251).to(torch.uint8)


def exercise(rank: int, device_index: int, master: str, port: int, args) -> dict:
    """Check exact collective outputs and the unchanged comparator on known bytes."""
    if args.buffer == "default":
        os.environ.pop("NCCL_BUFFSIZE", None)
    else:
        os.environ["NCCL_BUFFSIZE"] = args.buffer
    torch.cuda.set_device(device_index)
    dist.init_process_group(
        "nccl",
        init_method="env://" if args.launcher == "torchrun" else f"tcp://{master}:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=COLLECTIVE_TIMEOUT),
        device_id=torch.device("cuda", device_index),
    )
    subgroups = [dist.new_group(list(range(offset, WORLD_SIZE, 2))) for offset in range(2)]
    groups = {"dense": dist.group.WORLD, "expert": subgroups[rank % 2]}
    report = {
        "rank": rank,
        "host": socket.gethostname(),
        "visibility": args.visibility,
        "launcher": args.launcher,
        "nccl_buffer": os.environ.get("NCCL_BUFFSIZE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_index": device_index,
        "visible_devices": torch.cuda.device_count(),
        "gpu_uuid": str(torch.cuda.get_device_properties(device_index).uuid),
        "nccl_version": torch.cuda.nccl.version(),
        "groups": {name: dist.get_process_group_ranks(group) for name, group in groups.items()},
        "collectives": [],
        "comparisons": [],
    }
    try:
        for group_name, group in groups.items():
            members = report["groups"][group_name]
            for size in [1024, 262145, 4194305]:
                base = torch.arange(size, device="cuda", dtype=torch.int32).remainder_(128)
                source = base + rank
                before = parameter_sha256(source)
                low, high = source.clone(), source.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
                dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
                low_errors = int(low.ne(base + min(members)).sum().item())
                high_errors = int(high.ne(base + max(members)).sum().item())
                report["collectives"].append(
                    {
                        "group": group_name,
                        "members": members,
                        "elements": size,
                        "input_sha256": before,
                        "input_after_sha256": parameter_sha256(source),
                        "min_sha256": parameter_sha256(low),
                        "max_sha256": parameter_sha256(high),
                        "min_errors": low_errors,
                        "max_errors": high_errors,
                    }
                )
        sources = {
            "dense": known_bytes(262146, 5, "cuda").view(torch.bfloat16),
            "expert": known_bytes((16 << 20) + 2, rank % 2, "cuda").view(torch.bfloat16),
        }
        for divergent in [False, True]:
            if divergent and rank == 0:
                for source in sources.values():
                    source.view(torch.uint8)[0] ^= 1
            before = {name: parameter_sha256(source) for name, source in sources.items()}
            comparison = compare_replicas(sources, groups, rank, int(divergent))
            expected = (1 + int(rank % 2 == 0)) if divergent else 0
            report["comparisons"].append(
                {
                    "case": "one_byte_divergence" if divergent else "identical_replicas",
                    "input_hashes": before,
                    "input_after_hashes": {name: parameter_sha256(source) for name, source in sources.items()},
                    "expected_mismatched_bytes": expected,
                    "actual": asdict(comparison),
                    "passed": comparison.mismatched_bytes == expected,
                }
            )
        report["passed"] = all(
            row["min_errors"] == row["max_errors"] == 0 and row["input_sha256"] == row["input_after_sha256"]
            for row in report["collectives"]
        ) and all(row["passed"] and row["input_hashes"] == row["input_after_hashes"] for row in report["comparisons"])
        name = variant_name(args.launcher, args.visibility, args.buffer)
        write_json(prefix_join(args.output_prefix, f"{name}-rank{rank}.json"), report)
        dist.barrier()
        return report
    finally:
        dist.destroy_process_group()


class CollectiveActor:
    def endpoint(self) -> tuple[str, int]:
        return ray.util.get_node_ip_address(), allocate_port_block(1)

    def run(self, rank: int, master: str, port: int, args: dict) -> dict:
        config = SimpleNamespace(**args)
        assigned = int(ray.get_gpu_ids()[0])
        if config.visibility == "unmasked":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPUS_PER_NODE))
        return exercise(rank, 0 if config.visibility == "masked" else assigned, master, port, config)


def ray_driver(args) -> None:
    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={
            "worker_process_setup_hook": "skyrl_train.worker_setup.configure_worker_process",
        },
    )
    pg = placement_group([{"GPU": 4, "CPU": 8}] * 2, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=90)
    outcomes = []
    try:
        for visibility in ["masked", "unmasked"]:
            for buffer in ["default", "262144"]:
                actor_type = ray.remote(num_gpus=1, num_cpus=1)(CollectiveActor)
                actors = [
                    actor_type.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // GPUS_PER_NODE,
                        )
                    ).remote()
                    for rank in range(WORLD_SIZE)
                ]
                try:
                    # Actor placement can reorder hosts. Ask actor zero for its own address.
                    master, port = ray.get(actors[0].endpoint.remote())
                    config = {
                        "launcher": "ray",
                        "visibility": visibility,
                        "buffer": buffer,
                        "output_prefix": args.output_prefix,
                    }
                    rows = ray.get(
                        [actor.run.remote(rank, master, port, config) for rank, actor in enumerate(actors)], timeout=150
                    )
                    if len({row["gpu_uuid"] for row in rows}) != WORLD_SIZE:
                        raise RuntimeError("Ray variants must own eight distinct physical GPUs")
                    outcomes.extend(rows)
                finally:
                    for actor in actors:
                        ray.kill(actor)
        for visibility in ["masked", "unmasked"]:
            for buffer in ["default", "262144"]:
                name = variant_name("torchrun", visibility, buffer)
                for rank in range(WORLD_SIZE):
                    filesystem, path = filesystem_for(prefix_join(args.output_prefix, f"{name}-rank{rank}.json"))
                    with filesystem.open(path) as stream:
                        outcomes.append(json.load(stream))
        for launcher in ["torchrun", "ray"]:
            for visibility in ["masked", "unmasked"]:
                for buffer in [None, "262144"]:
                    rows = [
                        r
                        for r in outcomes
                        if r["launcher"] == launcher and r["visibility"] == visibility and r["nccl_buffer"] == buffer
                    ]
                    if (
                        sorted(r["rank"] for r in rows) != list(range(WORLD_SIZE))
                        or len({r["gpu_uuid"] for r in rows}) != WORLD_SIZE
                    ):
                        raise RuntimeError("Each variant must report eight ranks on eight distinct physical GPUs")
        write_json(
            prefix_join(args.output_prefix, "receipt.json"),
            {
                "completed": True,
                "passed": all(row["passed"] for row in outcomes),
                "variants": 8,
                "rank_reports": len(outcomes),
                "ranks_per_variant": WORLD_SIZE,
                "failed_cases": [row for row in outcomes if not row["passed"]],
            },
        )
        if not all(row["passed"] for row in outcomes):
            raise RuntimeError("Known-input collective/comparator correctness failure; see pinned rank reports")
    finally:
        remove_placement_group(pg)
        ray.shutdown()


def task(args) -> None:
    info = get_job_info()
    if info is None or info.num_tasks != 2 or torch.cuda.device_count() != GPUS_PER_NODE:
        raise ValueError("This diagnostic requires two four-GPU tasks in one NVLink domain")
    derived_ifname = pin_socket_ifname()
    ctx = iris_ctx()
    for visibility in ["masked", "unmasked"]:
        for buffer in ["default", "262144"]:
            name = variant_name("torchrun", visibility, buffer)
            endpoint = f"comparator-{info.attempt_id}-{name}"
            endpoint_id = None
            if info.task_index == 0:
                master, port = info.advertise_host, allocate_port_block(1)
                endpoint_id = ctx.registry.register(endpoint, f"{master}:{port}")
            else:
                master, port = wait_for_rendezvous(ctx.resolver, endpoint, timeout=120)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--nnodes=2",
                        "--nproc-per-node=4",
                        f"--node-rank={info.task_index}",
                        f"--master-addr={master}",
                        f"--master-port={port}",
                        "-m",
                        MODULE,
                        "--mode=worker",
                        "--launcher=torchrun",
                        f"--visibility={visibility}",
                        f"--buffer={buffer}",
                        f"--output-prefix={args.output_prefix}",
                    ],
                    check=True,
                    timeout=150,
                )
            finally:
                if endpoint_id is not None:
                    ctx.registry.unregister(endpoint_id)
    run_ray_control(args.output_prefix, info, derived_ifname, MODULE, driver_timeout=360)


def run_ray_control(output_prefix: str, info, derived_ifname: str, driver_module: str, driver_timeout: int) -> None:
    """Run a bounded diagnostic driver on the native Ray task bootstrap."""
    for port_name in RayPort:
        os.environ[f"IRIS_PORT_{port_name.value.upper()}"] = "0"
    resolve_ray_ports()
    rendezvous = prefix_join(output_prefix, "rendezvous")
    node_args = SimpleNamespace(
        rendezvous_dir=rendezvous,
        ray_log_dir=prefix_join(output_prefix, "ray-logs"),
        ray_spill_backend=RaySpillBackend.LOCAL,
        ray_spill_dir="/tmp/comparator-ray-spill",
        ray_labels={},
        rendezvous_timeout=150,
        cluster_join_timeout=150,
    )
    if info.task_index != 0:
        raise SystemExit(run_worker(node_args))
    epoch = uuid.uuid4().hex
    logs = RayLogSyncSession(node_args.ray_log_dir, f"rank0-{socket.gethostname()}")
    stop_sync = None
    try:
        ray_start_head(
            info.advertise_host,
            resolve_ray_spill_target(rendezvous, node_args.ray_spill_backend, node_args.ray_spill_dir),
        )
        stop_sync = logs.start_periodic(rendezvous)
        ray_port = allocated_ray_port(RayPort.HEAD)
        address = f"{info.advertise_host}:{ray_port}"
        write_rendezvous(rendezvous, info.advertise_host, ray_port, epoch)
        wait_for_nodes(
            address,
            info.num_tasks,
            150,
            rewrite_cb=lambda: write_rendezvous(rendezvous, info.advertise_host, ray_port, epoch),
        )
        env = training_driver_env(derived_ifname)
        env["RAY_ADDRESS"] = address
        subprocess.run(
            [
                sys.executable,
                "-m",
                driver_module,
                "--mode=driver",
                f"--output-prefix={output_prefix}",
                f"--nodes={info.num_tasks}",
            ],
            env=env,
            check=True,
            timeout=driver_timeout,
        )
        write_head_result(rendezvous, epoch)
    finally:
        if stop_sync is not None:
            stop_sync.set()
        logs.sync_bounded("Comparator differential teardown")
        ray_stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["task", "driver", "worker"], default="task")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--launcher", default="torchrun")
    parser.add_argument("--visibility", choices=["masked", "unmasked"], default="masked")
    parser.add_argument("--buffer", choices=["default", "262144"], default="default")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.mode == "task":
        task(args)
    elif args.mode == "driver":
        ray_driver(args)
    else:
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        if args.visibility == "masked":
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[local_rank]
        exercise(
            rank,
            0 if args.visibility == "masked" else local_rank,
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            args,
        )


if __name__ == "__main__":
    main()
