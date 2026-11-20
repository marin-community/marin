# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the complete native policy-worker initializer through production Ray actors.

This one-rack control stops after initialization and replica comparison. It
constructs the original optimizer, router replay and weight extractor, but runs
no optimizer update, forward/backward, inference or publication.
"""

import argparse
import hashlib
import json
import os
import signal
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
from cloud.iris.hf_model_cache import stage_artifact_model_metadata
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
from iris.cluster.client.job_info import get_job_info
from marinskyrl.environment_contract import NCCL_BUFFER_SIZE_ENV_VAR
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.workers.worker import PPORayActorGroup

from experiments.post_training.long_loop.snowball_replica_probe import (
    GPUS_PER_TASK,
    NUM_TASKS,
    PROBE_TOPOLOGY,
    WORLD_SIZE,
    ReplicaProbeWorker,
    write_receipt,
)


class RayReplicaProbeWorker(ReplicaProbeWorker):
    """Preserve native initialization while attaching phase receipts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_prefix = self.cfg.replica_probe.output_prefix
        self.phase_differences = {}

    def init_worker_process_group(self, master_addr: str, master_port: int):
        super().init_worker_process_group(master_addr, master_port)
        self.receipt_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))

    def _normalize_mini_batch_size(self):
        # The native initializer calls this immediately after creating its optimizer.
        self.record_parameters(self.actor_module, "post_optimizer_init")
        return super()._normalize_mini_batch_size()

    def init_model(self, model_path, num_training_steps):
        super().init_model(model_path, num_training_steps=num_training_steps)
        self.record_parameters(self.actor_module, "end_worker_init")
        self.replica_sender = self._expert_block_sender
        self.replica_sender.inventory()
        self.compare_parameter_bytes("end_worker_init", version=0)
        report = {
            "rank": self._rank,
            "distributed_rank": dist.get_rank(),
            "mesh_rank": asdict(self.mesh_rank),
            "host": socket.gethostname(),
            "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "local_rank": os.environ["LOCAL_RANK"],
            "gloo_interface": os.environ["GLOO_SOCKET_IFNAME"],
            "full_init_completed": True,
            "router_replay_installed": self.model.router_replay is not None,
            "weight_extractor_type": type(self.weight_extractor).__name__,
        }
        write_receipt(prefix_join(self.output_prefix, f"full-init-rank{self._rank}.json"), report)
        if self._rank == 0:
            write_receipt(
                prefix_join(self.output_prefix, "receipt.json"),
                {
                    "completed": True,
                    "scope": "production PPORayActorGroup and complete policy init_model; no inference or publication",
                    "world_size": WORLD_SIZE,
                    "topology": PROBE_TOPOLOGY,
                    "phase_mismatches": {key: len(value) for key, value in self.phase_differences.items()},
                    "optimizer_constructed": True,
                    "optimizer_steps_applied": 0,
                    "config_sha256": self.cfg.replica_probe.config_sha256,
                },
            )
        return report


def compiled_config(args: argparse.Namespace):
    filesystem, path = filesystem_for(args.config_uri)
    with filesystem.open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError("Frozen production configuration SHA mismatch")
    compiled = OmegaConf.create(raw.decode())
    cfg = compiled.skyrl
    model = cfg.trainer.policy.model
    model.path = f"/tmp/snowball-ray-metadata-{args.config_sha256}"
    model.tokenizer_path = compiled.inputs.model.tokenizer_uri
    model.tokenizer_revision = compiled.inputs.model.tokenizer_revision
    cfg.trainer.run_name = args.output_prefix.rstrip("/").split("/")[-2]
    cfg.trainer.native_telemetry_root = prefix_join(args.output_prefix, "telemetry")
    cfg.replica_probe = {"output_prefix": args.output_prefix, "config_sha256": args.config_sha256}
    if cfg.trainer.placement.policy_num_nodes != NUM_TASKS:
        raise ValueError("The source policy node count must match the one-rack control")
    if cfg.trainer.placement.policy_num_gpus_per_node != GPUS_PER_TASK:
        raise ValueError("The source policy GPU count must match the one-rack control")
    return cfg


def driver(args: argparse.Namespace) -> None:
    cfg = compiled_config(args)
    initialize_ray(cfg)
    actor_type = ray.remote(num_gpus=1)(RayReplicaProbeWorker)
    try:
        actors = PPORayActorGroup(
            cfg,
            NUM_TASKS,
            GPUS_PER_TASK,
            actor_type,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            pin_to_ray_gpu_id=False,
            force_cvd_mask=False,
            actor_env_vars={NCCL_BUFFER_SIZE_ENV_VAR: str(cfg.trainer.policy.nccl_buffer_size_bytes)},
        )
        rows = ray.get(actors.async_init_model(cfg.trainer.policy.model.path, num_training_steps=cfg.trainer.max_steps))
        write_receipt(prefix_join(args.output_prefix, "actor-identities.json"), {"actors": rows})
        if sorted(row["rank"] for row in rows) != list(range(WORLD_SIZE)):
            raise RuntimeError("Not all policy ranks completed full initialization")
        if any(row["rank"] != row["distributed_rank"] for row in rows):
            raise RuntimeError("Ray policy rank differs from the initialized distributed rank")
        if len({row["gpu_uuid"] for row in rows}) != WORLD_SIZE:
            raise RuntimeError("Production Ray policy actors share physical GPUs")
        print(json.dumps({"full_policy_init_and_comparison_passed": True, "ranks": len(rows)}), flush=True)
    finally:
        ray.shutdown()


def task(args: argparse.Namespace) -> None:
    info = get_job_info()
    if info is None or info.num_tasks != NUM_TASKS or torch.cuda.device_count() != GPUS_PER_TASK:
        raise ValueError("This control requires sixteen tasks, each owning four GB200 GPUs")
    cfg = compiled_config(args)
    model = cfg.trainer.policy.model
    stage_artifact_model_metadata(model.source_uri, model.source_identity, model.path)
    # CLI diagnostic tasks use the same zero-port resolution as Kubernetes tasks.
    for port in RayPort:
        os.environ[f"IRIS_PORT_{port.value.upper()}"] = "0"
    resolve_ray_ports()
    derived_ifname = pin_socket_ifname()
    rendezvous = prefix_join(args.output_prefix, "rendezvous")
    node_args = SimpleNamespace(
        rendezvous_dir=rendezvous,
        ray_log_dir=prefix_join(args.output_prefix, "ray-logs"),
        ray_spill_backend=RaySpillBackend.LOCAL,
        ray_spill_dir="/tmp/skyrl-ray-spill",
        ray_labels={},
        rendezvous_timeout=300,
        cluster_join_timeout=300,
    )
    if info.task_index != 0:
        raise SystemExit(run_worker(node_args))
    epoch = uuid.uuid4().hex
    logs = RayLogSyncSession(node_args.ray_log_dir, f"rank0-{socket.gethostname()}")
    stop_sync = None

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        spill = resolve_ray_spill_target(rendezvous, node_args.ray_spill_backend, node_args.ray_spill_dir)
        ray_start_head(info.advertise_host, spill)
        stop_sync = logs.start_periodic(rendezvous)
        ray_port = allocated_ray_port(RayPort.HEAD)
        write_rendezvous(rendezvous, info.advertise_host, ray_port, epoch)
        address = f"{info.advertise_host}:{ray_port}"
        wait_for_nodes(
            address,
            NUM_TASKS,
            300,
            rewrite_cb=lambda: write_rendezvous(rendezvous, info.advertise_host, ray_port, epoch),
        )
        env = training_driver_env(derived_ifname)
        env["RAY_ADDRESS"] = address
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.post_training.long_loop.snowball_ray_replica_probe",
                "--driver",
                "--config-uri",
                args.config_uri,
                "--config-sha256",
                args.config_sha256,
                "--output-prefix",
                args.output_prefix,
            ],
            env=env,
            check=True,
        )
        write_head_result(rendezvous, epoch)
    finally:
        if stop_sync is not None:
            stop_sync.set()
        logs.sync_bounded("Ray replica control head teardown")
        ray_stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--config-uri", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.driver:
        driver(args)
    else:
        task(args)


if __name__ == "__main__":
    main()
