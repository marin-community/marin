# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load Snowball at DP8 and record imported and DDP-wrapped replica bytes.

This diagnostic uses the production Megatron worker's model-only initialization;
it constructs no optimizer, inference engine or training batch. The configuration
is a pinned compiled launch document from the failed run. PP is reduced to one
to preserve its DP8/CP4/EP4 replica groups with 32 GPUs.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist
from cloud.iris.hf_model_cache import stage_artifact_model_metadata
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from omegaconf import OmegaConf
from rigging.filesystem.buckets import filesystem_for
from rigging.network import interface_for_ipv4
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.weight_sync.expert_block.megatron_source import local_source_slices
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase, _MegatronInitMode

from experiments.post_training.long_loop.replica_parameter_receipts import (
    ParameterReceipt,
    parameter_sha256,
    replica_differences,
)
from experiments.post_training.nccl_transport_probe import allocate_port_block, wait_for_rendezvous


def write_receipt(uri: str, value: dict) -> None:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "w") as stream:
        json.dump(value, stream, indent=2)


class ReplicaProbeWorker(MegatronPolicyWorkerBase):
    """Attach read-only receipts to Bridge's existing import and DDP hooks."""

    def init_configs(self, *args, **kwargs):
        super().init_configs(*args, **kwargs)
        self.provider.register_pre_wrap_hook(lambda models: self.record_parameters(models, "imported"))
        self.provider.register_post_wrap_hook(lambda models: self.record_parameters(models, "post_ddp"))

    def record_parameters(self, models, phase: str):
        started = time.monotonic()
        tasks = self.bridge.get_conversion_tasks(models)
        local = local_source_slices(tasks, self.provider, pp=mpu.get_pipeline_model_parallel_rank())
        expert_names = {row.source_key for row in local.experts}
        hf_names = {}
        for task in tasks:
            names = task.mapping.hf_param
            hf_names[task.global_param_name] = tuple(sorted(names.values())) if isinstance(names, dict) else (names,)
        rows = []
        for name, parameter in sorted(local.sources.items()):
            group = (
                mpu.get_expert_data_parallel_group()
                if name in expert_names
                else mpu.get_data_parallel_group(with_context_parallel=True)
            )
            rows.append(
                ParameterReceipt(
                    dist.get_rank(),
                    name,
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    tuple(sorted(dist.get_process_group_ranks(group))),
                    hf_names[name],
                    parameter.numel() * parameter.element_size(),
                    parameter_sha256(parameter),
                )
            )
        payload = {
            "phase": phase,
            "rank": dist.get_rank(),
            "host": socket.gethostname(),
            "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "gloo_interface": os.environ["GLOO_SOCKET_IFNAME"],
            "seconds": time.monotonic() - started,
            "parameters": [asdict(row) for row in rows],
        }
        write_receipt(f"{self.output_prefix}/{phase}-rank{dist.get_rank()}.json", payload)
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(rows, gathered, dst=0, group=self.receipt_group)
        if dist.get_rank() == 0:
            differences = replica_differences([row for rank_rows in gathered for row in rank_rows])
            self.phase_differences[phase] = differences
            write_receipt(
                f"{self.output_prefix}/{phase}-differences.json",
                {
                    "phase": phase,
                    "differences": differences,
                    "compared_parameter_receipts": sum(len(rank_rows) for rank_rows in gathered),
                },
            )
            print(json.dumps({"phase": phase, "mismatching_parameter_groups": len(differences)}), flush=True)
        dist.barrier(group=self.receipt_group)
        return models


def worker(args: argparse.Namespace) -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    if dist.get_world_size() != 32:
        raise ValueError("This diagnostic requires 32 ranks: TP1/PP1/CP4/EP4/DP8")
    filesystem, path = filesystem_for(args.config_uri)
    with filesystem.open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError("Diagnostic configuration SHA mismatch")
    compiled = OmegaConf.create(raw.decode())
    cfg = compiled["skyrl"]
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    model = cfg.trainer.policy.model
    metadata_dir = f"/tmp/snowball-replica-metadata-{args.config_sha256}"
    if int(os.environ["LOCAL_RANK"]) == 0:
        stage_artifact_model_metadata(model.source_uri, model.source_identity, metadata_dir)
    # Each node's local rank zero has a different global rank; synchronize all
    # nodes before any local rank reads the node-local metadata directory.
    dist.barrier()
    model.path = metadata_dir
    model.tokenizer_path = compiled.inputs.model.tokenizer_uri
    model.tokenizer_revision = compiled.inputs.model.tokenizer_revision
    probe = object.__new__(ReplicaProbeWorker)
    probe.cfg = cfg
    probe._rank = dist.get_rank()
    probe._local_rank = int(os.environ["LOCAL_RANK"])
    probe.output_prefix = args.output_prefix
    probe.phase_differences = {}
    probe.receipt_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    probe.strategy = MegatronStrategy(cfg.trainer.policy.megatron_config, seed=cfg.trainer.seed)
    probe.strategy.setup_distributed(timeout=timedelta(seconds=180))
    probe._initialize_policy_modules(metadata_dir, mode=_MegatronInitMode.TRAINING)
    if dist.get_rank() == 0:
        write_receipt(
            f"{args.output_prefix}/receipt.json",
            {
                "completed": True,
                "scope": "load-only; SHA256 of exact stored parameter bytes",
                "config_sha256": args.config_sha256,
                "world_size": dist.get_world_size(),
                "topology": {"tp": 1, "pp": 1, "cp": 4, "ep": 4, "dp": 8},
                "source_uri": model.source_uri,
                "phase_mismatches": {key: len(value) for key, value in probe.phase_differences.items()},
                "optimizer_constructed": False,
                "inference_constructed": False,
            },
        )
    dist.barrier(group=probe.receipt_group)
    dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    info = get_job_info()
    if info is None or info.num_tasks != 8 or torch.cuda.device_count() != 4:
        raise ValueError("Submit an eight-task gang with four GB200 GPUs per task")
    # Resolve on each node before torchrun inherits the environment. Hostname
    # resolution in these pods otherwise makes Gloo advertise loopback to peers.
    os.environ["GLOO_SOCKET_IFNAME"] = interface_for_ipv4(info.advertise_host)
    ctx = iris_ctx()
    name = f"snowball-replica-probe-{info.attempt_id}"
    endpoint_id = None
    if info.task_index == 0:
        host, port = info.advertise_host, allocate_port_block(1)
        endpoint_id = ctx.registry.register(name, f"{host}:{port}")
    else:
        host, port = wait_for_rendezvous(ctx.resolver, name)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=8",
        "--nproc-per-node=4",
        f"--node-rank={info.task_index}",
        f"--master-addr={host}",
        f"--master-port={port}",
        str(Path(__file__).resolve()),
        "--worker",
        "--config-uri",
        args.config_uri,
        "--config-sha256",
        args.config_sha256,
        "--output-prefix",
        args.output_prefix,
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        if endpoint_id is not None:
            ctx.registry.unregister(endpoint_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config-uri", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
