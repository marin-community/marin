# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bisect NCCL message sizes and groups without loading model weights."""

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import ray
import torch
import torch.distributed as dist
from cloud.iris.task_runtime import pin_socket_ifname
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.long_loop.comparator_differential import run_ray_control, write_json
from experiments.post_training.long_loop.nccl_runtime_receipts import runtime_receipt
from experiments.post_training.long_loop.replica_parameter_receipts import parameter_sha256
from experiments.post_training.nccl_transport_probe import allocate_port_block, wait_for_rendezvous

MODULE = "experiments.post_training.long_loop.nccl_mechanism_probe"
GPUS_PER_NODE = 4
LOG_LIMIT = 8 << 20
PROFILE_KEYS = (
    "NCCL_CUMEM_ENABLE",
    "NCCL_BUFFSIZE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "NCCL_ALGO",
    "NCCL_NVLSTREE_MAX_CHUNKSIZE",
)
PRODUCTION_ENV = {"NCCL_CUMEM_ENABLE": "0", "NCCL_BUFFSIZE": "262144", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}
PROFILE_ENVIRONMENTS = {
    "production": PRODUCTION_ENV,
    "production-trace": PRODUCTION_ENV,
    "plain": {},
    "cumem-only": {"NCCL_CUMEM_ENABLE": "0"},
    "buffer-only": {"NCCL_BUFFSIZE": "262144"},
    "connections-only": {"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
    "without-cumem": {key: value for key, value in PRODUCTION_ENV.items() if key != "NCCL_CUMEM_ENABLE"},
    "without-buffer": {key: value for key, value in PRODUCTION_ENV.items() if key != "NCCL_BUFFSIZE"},
    "without-connections": {key: value for key, value in PRODUCTION_ENV.items() if key != "CUDA_DEVICE_MAX_CONNECTIONS"},
    "nvls-off": {**PRODUCTION_ENV, "NCCL_NVLS_ENABLE": "0"},
    "ring": {**PRODUCTION_ENV, "NCCL_ALGO": "Ring"},
    "tree": {**PRODUCTION_ENV, "NCCL_ALGO": "Tree"},
    "plain-nvls-tree": {"NCCL_ALGO": "NVLSTree"},
}
# Diagnostic-only FIFO capacity controls; production/comparator settings are unchanged.
for buffer_bytes in (262144, 524288, 1048576, 4194304):
    PROFILE_ENVIRONMENTS[f"fifo-{buffer_bytes}"] = {
        **PRODUCTION_ENV,
        "NCCL_BUFFSIZE": str(buffer_bytes),
        "NCCL_ALGO": "NVLSTree",
    }
for maximum_chunk in (32768, 65536):
    PROFILE_ENVIRONMENTS[f"fifo-262144-chunk-{maximum_chunk}"] = {
        **PRODUCTION_ENV,
        "NCCL_ALGO": "NVLSTree",
        "NCCL_NVLSTREE_MAX_CHUNKSIZE": str(maximum_chunk),
    }
SUITES = {
    "launcher": ("production", "production-trace", "plain"),
    "fifo": (
        "fifo-262144",
        "fifo-524288",
        "fifo-1048576",
        "fifo-4194304",
        "fifo-262144-chunk-32768",
        "fifo-262144-chunk-65536",
    ),
    "environment": (
        "production",
        "plain",
        "cumem-only",
        "buffer-only",
        "connections-only",
        "without-cumem",
        "without-buffer",
        "without-connections",
        "nvls-off",
        "ring",
        "tree",
        "plain-nvls-tree",
    ),
}
SIZES = (1024, 4194304, 4194305, 8388608, 16777215, 16777216, 16777217)


def profile_environment(profile: str) -> dict[str, str]:
    return dict(PROFILE_ENVIRONMENTS[profile])


def configured_environment(profile: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in PROFILE_KEYS:
        env.pop(key, None)
    env.update(profile_environment(profile))
    return env


def group_layouts(world: int) -> list[tuple[str, list[int]]]:
    """Describe equal-size groups that span different numbers of four-GPU hosts."""
    groups = [(f"child{world}", list(range(world)))]
    for size in [16, 8, 4, 2]:
        if size > world:
            continue
        step = world // size
        for offset in range(step):
            groups.append((f"strided{size}", list(range(offset, world, step))))
    for start in range(0, world if world >= 8 else 0, 8):
        groups.append(("contiguous8", list(range(start, start + 8))))
    return groups


def probe_groups(world: int, rank: int) -> list[tuple[str, dist.ProcessGroup]]:
    """Create a full child group first, then strided and contiguous controls."""
    groups = []
    for label, members in group_layouts(world):
        group = dist.new_group(members)
        if rank in members:
            groups.append((label, group))
    return groups


def mismatch_examples(actual: torch.Tensor, expected: torch.Tensor) -> list[dict]:
    """Retain the first sixteen incorrect values with their absolute element offsets."""
    indices = actual.ne(expected).nonzero().flatten()[:16]
    return [
        {"element": index, "actual": value if math.isfinite(value) else str(value), "expected": reference}
        for index, value, reference in zip(
            indices.tolist(), actual[indices].tolist(), expected[indices].tolist(), strict=True
        )
    ]


def integer_case(rank: int, label: str, group: dist.ProcessGroup, count: int, device: torch.device) -> dict:
    members = dist.get_process_group_ranks(group)
    base = torch.arange(count, dtype=torch.int32, device=device).remainder_(128)
    source = base + rank
    before = parameter_sha256(source)
    low, high, total = source.clone(), source.clone(), source.clone()
    started = time.monotonic()
    # Preserve the back-to-back MIN/MAX order of the failing comparator.
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    errors = {
        "MIN": int(low.ne(base + min(members)).sum().item()),
        "MAX": int(high.ne(base + max(members)).sum().item()),
        "SUM": int(total.ne(base * len(members) + sum(members)).sum().item()),
    }
    after = parameter_sha256(source)
    return {
        "group": label,
        "group_name": group.group_name,
        "members": members,
        "dtype": "int32",
        "elements": count,
        "nbytes": count * 4,
        "seconds": seconds,
        "input_sha256": before,
        "input_after_sha256": after,
        "outputs": {name: parameter_sha256(value) for name, value in [("MIN", low), ("MAX", high), ("SUM", total)]},
        "mismatch_examples": {
            name: mismatch_examples(value, expected) if errors[name] else []
            for name, value, expected in [
                ("MIN", low, base + min(members)),
                ("MAX", high, base + max(members)),
                ("SUM", total, base * len(members) + sum(members)),
            ]
        },
        "errors": errors,
        "passed": not any(errors.values()) and before == after,
    }


def training_reduction_case(
    group: dist.ProcessGroup, dtype: torch.dtype, device: torch.device, nbytes: int, operation: str
) -> dict:
    members = dist.get_process_group_ranks(group)
    count = nbytes // torch.tensor([], dtype=dtype).element_size()
    # BF16 inputs and every partial sum stay integral and below256: no reduction-order rounding ambiguity.
    modulus = 4 if dtype == torch.bfloat16 else 128
    offset = dist.get_rank() % 4 if dtype == torch.bfloat16 else dist.get_rank()
    offset_sum = sum(rank % 4 for rank in members) if dtype == torch.bfloat16 else sum(members)
    base = torch.arange(count, dtype=torch.int32, device=device).remainder_(modulus)
    source = (base + offset).to(dtype)
    before = parameter_sha256(source)
    result = source.clone()
    started = time.monotonic()
    dist.all_reduce(result, op={"SUM": dist.ReduceOp.SUM, "AVG": dist.ReduceOp.AVG}[operation], group=group)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    expected = (base * len(members) + offset_sum).to(dtype)
    if operation == "AVG":
        expected.div_(len(members))
    errors = int(result.ne(expected).sum().item())
    after = parameter_sha256(source)
    return {
        "group": f"child{len(members)}",
        "group_name": group.group_name,
        "members": members,
        "dtype": str(dtype),
        "elements": count,
        "nbytes": nbytes,
        "seconds": seconds,
        "input_sha256": before,
        "input_after_sha256": after,
        "output_sha256": parameter_sha256(result),
        "pattern_modulus": modulus,
        "rank_offset": offset,
        "offset_sum": offset_sum,
        "expected_sha256": parameter_sha256(expected),
        "errors": {operation: errors},
        "mismatch_examples": mismatch_examples(result, expected) if errors else [],
        "passed": errors == 0 and before == after,
    }


def exercise(rank: int, local_device: int, master: str, port: int, args) -> dict:
    for key in PROFILE_KEYS:
        os.environ.pop(key, None)
    os.environ.update(profile_environment(args.profile))
    log_path = Path(f"/tmp/nccl-mechanism-{args.launcher}-{args.profile}-rank{rank}-{os.getpid()}.log")
    os.environ.update(
        NCCL_DEBUG=("WARN" if args.profile == "production" else "TRACE"),
        NCCL_DEBUG_SUBSYS=("INIT,ENV,GRAPH,TUNING,COLL,CALL,NET,NVLS" if args.profile != "production" else "INIT,NET"),
        NCCL_DEBUG_FILE=str(log_path),
    )
    torch.cuda.set_device(local_device)
    report = {
        "rank": rank,
        "host": socket.gethostname(),
        "launcher": args.launcher,
        "profile": args.profile,
        "cases": [],
        "completed": False,
        "gpu_uuid": str(torch.cuda.get_device_properties(local_device).uuid),
    }
    name = f"{args.launcher}-{args.profile}-rank{rank}"
    try:
        dist.init_process_group(
            "nccl",
            init_method="env://" if args.launcher == "torchrun" else f"tcp://{master}:{port}",
            rank=rank,
            world_size=args.nodes * GPUS_PER_NODE,
            timeout=timedelta(seconds=120),
            device_id=torch.device("cuda", local_device),
        )
        report["runtime"] = runtime_receipt()
        write_json(f"{args.output_prefix}/{name}.json", report)
        groups = (
            probe_groups(args.nodes * GPUS_PER_NODE, rank)
            if args.suite == "launcher"
            else [(f"child{args.nodes * GPUS_PER_NODE}", dist.new_group(list(range(args.nodes * GPUS_PER_NODE))))]
        )
        device = torch.device("cuda", local_device)
        for label, group in groups:
            for count in (
                (SIZES if args.suite == "launcher" else (4194304, 8388608, 16777216))
                if label.startswith("child")
                else [16777216]
            ):
                report["cases"].append(integer_case(rank, label, group, count, device))
            write_json(f"{args.output_prefix}/{name}.json", report)
        for dtype in [torch.float32, torch.bfloat16]:
            for operation in (
                ["SUM"] if profile_environment(args.profile).get("NCCL_ALGO") == "NVLSTree" else ["SUM", "AVG"]
            ):
                report["cases"].append(training_reduction_case(groups[0][1], dtype, device, 64 << 20, operation))
        report["passed"] = all(row["passed"] for row in report["cases"])
        report["completed"] = True
        write_json(f"{args.output_prefix}/{name}.json", report)
        dist.barrier()
        return report
    finally:
        write_json(f"{args.output_prefix}/{name}.json", report)
        if dist.is_initialized():
            dist.destroy_process_group()
        if log_path.exists():
            raw = log_path.read_bytes()
            size = len(raw)
            if size > LOG_LIMIT:
                raw = raw[: LOG_LIMIT // 2] + b"\n[DIAGNOSTIC LOG CLIPPED]\n" + raw[-LOG_LIMIT // 2 :]
            filesystem, path = filesystem_for(f"{args.output_prefix}/nccl-logs/{name}.log")
            with filesystem.open(path, "wb") as stream:
                stream.write(raw)
            report["nccl_log"] = {
                "bytes": size,
                "copied_bytes": len(raw),
                "clipped": size > LOG_LIMIT,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            write_json(f"{args.output_prefix}/{name}.json", report)


class MechanismActor:
    def endpoint(self) -> tuple[str, int]:
        return ray.util.get_node_ip_address(), allocate_port_block(1)

    def run(self, rank: int, master: str, port: int, config: dict) -> dict:
        # Ray allocates one GPU per actor; retain its native one-GPU visibility mask.
        return exercise(rank, 0, master, port, SimpleNamespace(**config))


def driver(args) -> None:
    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={"worker_process_setup_hook": "skyrl_train.worker_setup.configure_worker_process"},
    )
    pg = placement_group([{"GPU": 4, "CPU": 8}] * args.nodes, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    outcomes = []
    try:
        for profile in SUITES[args.suite]:
            actor_type = ray.remote(num_gpus=1, num_cpus=1)(MechanismActor)
            actors = [
                actor_type.options(
                    runtime_env={"env_vars": profile_environment(profile)},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=rank // GPUS_PER_NODE
                    ),
                ).remote()
                for rank in range(args.nodes * GPUS_PER_NODE)
            ]
            try:
                master, port = ray.get(actors[0].endpoint.remote())
                config = {
                    "nodes": args.nodes,
                    "launcher": "ray",
                    "profile": profile,
                    "suite": args.suite,
                    "output_prefix": args.output_prefix,
                }
                outcomes.extend(
                    ray.get(
                        [actor.run.remote(rank, master, port, config) for rank, actor in enumerate(actors)], timeout=240
                    )
                )
            finally:
                for actor in actors:
                    ray.kill(actor)
        for profile in SUITES[args.suite]:
            for rank in range(args.nodes * GPUS_PER_NODE):
                filesystem, path = filesystem_for(f"{args.output_prefix}/torchrun-{profile}-rank{rank}.json")
                with filesystem.open(path) as stream:
                    outcomes.append(json.load(stream))
        for launcher in ["ray", "torchrun"]:
            for profile in SUITES[args.suite]:
                rows = [r for r in outcomes if r["launcher"] == launcher and r["profile"] == profile]
                if (
                    sorted(r["rank"] for r in rows) != list(range(args.nodes * GPUS_PER_NODE))
                    or len({r["gpu_uuid"] for r in rows}) != args.nodes * GPUS_PER_NODE
                ):
                    raise RuntimeError("Missing or physically duplicated ranks in differential")
        write_json(
            f"{args.output_prefix}/receipt.json",
            {
                "completed": True,
                "all_correct": all(r["passed"] for r in outcomes),
                "rank_reports": len(outcomes),
                "failed_case_count": sum(not c["passed"] for r in outcomes for c in r["cases"]),
            },
        )
        if not all(r["passed"] for r in outcomes):
            raise RuntimeError("Completed root-cause matrix reproduces collective corruption; inspect rank receipts")
    finally:
        remove_placement_group(pg)
        ray.shutdown()


def task(args) -> None:
    info = get_job_info()
    if info is None or info.num_tasks != args.nodes or torch.cuda.device_count() != GPUS_PER_NODE:
        raise ValueError("This probe requires the configured number of four-GB200 tasks in one NVLink domain")
    derived_ifname = pin_socket_ifname()
    ctx = iris_ctx()
    for profile in SUITES[args.suite]:
        name = f"nccl-mechanism-{info.attempt_id}-{profile}"
        endpoint = None
        if info.task_index == 0:
            master, port = info.advertise_host, allocate_port_block(1)
            endpoint = ctx.registry.register(name, f"{master}:{port}")
        else:
            master, port = wait_for_rendezvous(ctx.resolver, name, timeout=180)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nnodes={info.num_tasks}",
                    "--nproc-per-node=4",
                    f"--node-rank={info.task_index}",
                    f"--master-addr={master}",
                    f"--master-port={port}",
                    "-m",
                    MODULE,
                    "--mode=worker",
                    "--launcher=torchrun",
                    f"--profile={profile}",
                    f"--suite={args.suite}",
                    f"--nodes={info.num_tasks}",
                    f"--output-prefix={args.output_prefix}",
                ],
                env=configured_environment(profile),
                check=True,
                timeout=240,
            )
        finally:
            if endpoint is not None:
                ctx.registry.unregister(endpoint)
    if args.suite == "launcher":
        run_ray_control(args.output_prefix, info, derived_ifname, MODULE, driver_timeout=600)
    elif info.task_index == 0:
        outcomes = []
        for profile in SUITES[args.suite]:
            for rank in range(info.num_tasks * GPUS_PER_NODE):
                filesystem, path = filesystem_for(f"{args.output_prefix}/torchrun-{profile}-rank{rank}.json")
                with filesystem.open(path) as stream:
                    outcomes.append(json.load(stream))
        write_json(
            f"{args.output_prefix}/receipt.json",
            {
                "completed": all(row["completed"] for row in outcomes),
                "rank_reports": len(outcomes),
                "all_correct": all(row["passed"] for row in outcomes),
                "failed_case_count": sum(not case["passed"] for row in outcomes for case in row["cases"]),
            },
        )
        if not all(row["passed"] for row in outcomes):
            raise RuntimeError("Completed environment matrix reproduces collective corruption; inspect rank receipts")


def task_library_ab(args) -> None:
    """Compare the packaged, source baseline and patched NCCL libraries."""
    info = get_job_info()
    if info is None or info.num_tasks != args.nodes or torch.cuda.device_count() != GPUS_PER_NODE:
        raise ValueError("Library A/B requires the configured four-GB200 tasks")
    pin_socket_ifname()
    filesystem, path = filesystem_for(args.library_receipt)
    with filesystem.open(path, "rb") as stream:
        raw_receipt = stream.read()
    if hashlib.sha256(raw_receipt).hexdigest() != args.library_receipt_sha256:
        raise ValueError("The source-library receipt does not match its launch pin")
    receipt = json.loads(raw_receipt)
    if receipt["status"] != "built;not_qualified":
        raise ValueError("The library pair has not completed its verified build")
    libraries = {}
    for role in ("baseline", "patched"):
        artifact = receipt["libraries"][role]
        local = Path(f"/tmp/nccl-ab-{info.attempt_id}/{role}/libnccl.so.2")
        local.parent.mkdir(parents=True, exist_ok=True)
        filesystem, path = filesystem_for(artifact["uri"])
        with filesystem.open(path, "rb") as source, local.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
        with local.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != artifact["sha256"] or local.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Downloaded {role} NCCL artifact does not match the build receipt")
        libraries[role] = {"path": str(local), "sha256": actual}
    libraries["packaged"] = {"path": "", "sha256": "fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0"}
    variants = [
        ("packaged", "direct-nccl", "fifo-262144"),
        ("baseline", "process-group", "fifo-262144"),
        ("baseline", "direct-nccl", "fifo-262144"),
        ("patched", "process-group", "fifo-262144"),
        ("patched", "direct-nccl", "fifo-262144"),
        ("patched", "direct-nccl", "fifo-1048576"),
        ("baseline", "direct-nccl", "fifo-4194304"),
        ("patched", "direct-nccl", "fifo-4194304"),
        ("packaged", "direct-nccl", "fifo-4194304"),
        ("baseline", "direct-nccl", "fifo-1048576"),
        ("packaged", "direct-nccl", "fifo-1048576"),
    ]
    ctx = iris_ctx()
    for role, api, profile in variants:
        label = f"{role}-{profile}"
        name = f"nccl-ab-{info.attempt_id}-{api}-{label}"
        endpoint = None
        if info.task_index == 0:
            master, port = info.advertise_host, allocate_port_block(1)
            endpoint = ctx.registry.register(name, f"{master}:{port}")
        else:
            master, port = wait_for_rendezvous(ctx.resolver, name, timeout=180)
        directory = Path(f"/tmp/{name}")
        sizes = (
            (16, 32, 64)
            if role in ("baseline", "packaged") and profile == "fifo-262144"
            else (16, 32, 64, 128, 256, 512)
        )
        environment = configured_environment(profile)
        if libraries[role]["path"]:
            environment["LD_PRELOAD"] = libraries[role]["path"] + (
                ":" + environment["LD_PRELOAD"] if environment.get("LD_PRELOAD") else ""
            )
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nnodes={info.num_tasks}",
                    "--nproc-per-node=4",
                    f"--node-rank={info.task_index}",
                    f"--master-addr={master}",
                    f"--master-port={port}",
                    "-m",
                    "experiments.post_training.long_loop.nccl_api_reproducer",
                    f"--api={api}",
                    f"--profile={label}",
                    f"--output-directory={directory}",
                    f"--expected-library-sha256={libraries[role]['sha256']}",
                    "--message-bytes",
                    *[str(n << 20) for n in sizes],
                ],
                env=environment,
                check=True,
                timeout=240,
            )
        finally:
            for rank in range(info.task_index * GPUS_PER_NODE, (info.task_index + 1) * GPUS_PER_NODE):
                stem = f"{api}-{label}-rank{rank}"
                report_path = directory / f"{stem}.json"
                if not report_path.is_file():
                    continue
                report = json.loads(report_path.read_text())
                log_path = directory / f"{stem}.log"
                if log_path.is_file():
                    raw = log_path.read_bytes()
                    report["nccl_log"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    filesystem, path = filesystem_for(f"{args.output_prefix}/nccl-logs/{stem}.log")
                    with filesystem.open(path, "wb") as stream:
                        stream.write(raw)
                report["library_role"] = role
                write_json(f"{args.output_prefix}/{stem}.json", report)
            if endpoint is not None:
                ctx.registry.unregister(endpoint)
    name = f"nccl-ab-final-{info.attempt_id}"
    endpoint = None
    if info.task_index == 0:
        master, port = info.advertise_host, allocate_port_block(1)
        endpoint = ctx.registry.register(name, f"{master}:{port}")
    else:
        master, port = wait_for_rendezvous(ctx.resolver, name, timeout=180)
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{master}:{port}",
            rank=info.task_index,
            world_size=info.num_tasks,
            timeout=timedelta(seconds=120),
        )
        dist.barrier()
        if info.task_index == 0:
            outcomes = []
            for role, api, profile in variants:
                for rank in range(info.num_tasks * GPUS_PER_NODE):
                    filesystem, path = filesystem_for(f"{args.output_prefix}/{api}-{role}-{profile}-rank{rank}.json")
                    with filesystem.open(path) as stream:
                        outcomes.append(json.load(stream))
            completed = all(row["completed"] for row in outcomes)
            positive = [
                r
                for r in outcomes
                if r["library_role"] in ("baseline", "packaged") and r["profile"].endswith("fifo-262144")
            ]
            corrected = [r for r in outcomes if r["library_role"] == "patched"]
            defaults = [r for r in outcomes if r["profile"].endswith("fifo-4194304")]
            qualified = (
                completed and all(not r["passed"] for r in positive) and all(r["passed"] for r in corrected + defaults)
            )
            write_json(
                f"{args.output_prefix}/receipt.json",
                {
                    "completed": completed,
                    "qualification_passed": qualified,
                    "rank_reports": len(outcomes),
                    "all_correct": all(r["passed"] for r in outcomes),
                    "positive_control_reports": len(positive),
                    "patched_reports": len(corrected),
                    "library_receipt_sha256": args.library_receipt_sha256,
                },
            )
            if not qualified:
                raise RuntimeError("Library A/B did not meet its baseline-reproduction/patched-correctness criteria")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if endpoint is not None:
            ctx.registry.unregister(endpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["task", "driver", "worker", "library-task"], default="task")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--launcher", choices=["torchrun", "ray"], default="torchrun")
    parser.add_argument("--profile", choices=tuple(PROFILE_ENVIRONMENTS), default="production")
    parser.add_argument("--suite", choices=tuple(SUITES), default="launcher")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--library-receipt", default="")
    parser.add_argument("--library-receipt-sha256", default="")
    args = parser.parse_args()
    if args.mode == "library-task":
        task_library_ab(args)
    elif args.mode == "task":
        task(args)
    elif args.mode == "driver":
        driver(args)
    else:
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[local_rank]
        exercise(rank, 0, os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), args)


if __name__ == "__main__":
    main()
