# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace PP2 Snowball replicas through load and one native optimizer update.

The single-rack diagnostic uses the frozen production model and optimizer recipe,
with deterministic post-reduction gradients instead of a forward/backward batch.
It runs the unchanged production byte comparator before publication, constructs
no inference engine, and exports no weights.
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
from rigging.filesystem.storage_path import prefix_join
from rigging.network import interface_for_ipv4
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.weight_sync.expert_block.megatron_source import local_source_slices
from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.verify_weights import compare_replicas
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase, _MegatronInitMode

from experiments.post_training.long_loop.replica_parameter_receipts import (
    ParameterReceipt,
    comparator_snapshot,
    known_group_minmax,
    parameter_sha256,
    replica_differences,
    replica_iteration_differences,
)
from experiments.post_training.nccl_transport_probe import allocate_port_block, wait_for_rendezvous

PROBE_TOPOLOGY = {"tp": 1, "pp": 2, "cp": 4, "ep": 4, "dp": 8}
GPUS_PER_TASK = 4
WORLD_SIZE = PROBE_TOPOLOGY["tp"] * PROBE_TOPOLOGY["pp"] * PROBE_TOPOLOGY["cp"] * PROBE_TOPOLOGY["dp"]
NUM_TASKS = WORLD_SIZE // GPUS_PER_TASK
SYNTHETIC_GRADIENT = 1e-4


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
            "source_iteration_order": list(local.sources),
            "nonfinite_names": [name for name, value in local.sources.items() if not torch.isfinite(value).all()],
        }
        write_receipt(prefix_join(self.output_prefix, f"{phase}-rank{dist.get_rank()}.json"), payload)
        gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(
            {"rows": rows, "source_order": list(local.sources), "nonfinite_names": payload["nonfinite_names"]},
            gathered,
            dst=0,
            group=self.receipt_group,
        )
        verdict = [None]
        if dist.get_rank() == 0:
            all_rows = [row for rank_data in gathered for row in rank_data["rows"]]
            differences = replica_differences(all_rows)
            iteration_differences = replica_iteration_differences(
                all_rows, {rank: item["source_order"] for rank, item in enumerate(gathered)}
            )
            self.phase_differences[phase] = differences
            write_receipt(
                prefix_join(self.output_prefix, f"{phase}-differences.json"),
                {
                    "phase": phase,
                    "differences": differences,
                    "compared_parameter_receipts": len(all_rows),
                    "iteration_differences": iteration_differences,
                },
            )
            verdict[0] = {
                "order_errors": len(iteration_differences),
                "nonfinite": sum(len(item["nonfinite_names"]) for item in gathered),
            }
            print(json.dumps({"phase": phase, "mismatching_parameter_groups": len(differences)}), flush=True)
        dist.broadcast_object_list(verdict, src=0, group=self.receipt_group)
        if verdict[0]["order_errors"] or verdict[0]["nonfinite"]:
            raise RuntimeError(f"Invalid replica comparison inputs at {phase}: {verdict[0]}")
        return models

    def compare_parameter_bytes(self, phase: str, version: int) -> None:
        sender = self.replica_sender
        expert_names = {item.source_key for item in sender.expert_sources.values()}
        groups = {
            name: (
                mpu.get_expert_data_parallel_group()
                if name in expert_names
                else mpu.get_data_parallel_group(with_context_parallel=True)
            )
            for name in sender.sources
        }
        entry = comparator_snapshot(sender.sources, groups)
        entry.update(
            {
                "worker_rank": self._rank,
                "worker_local_rank": self._local_rank,
                "sender_trainer": asdict(sender.trainer),
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "current_cuda_device": torch.cuda.current_device(),
                "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                "nccl_buffer_size": os.environ.get("NCCL_BUFFSIZE"),
                "sender_parallel_state_is_probe_module": sender.parallel_state is mpu,
                "sender_dense_members": dist.get_process_group_ranks(
                    sender.parallel_state.get_data_parallel_group(with_context_parallel=True)
                ),
                "sender_expert_members": dist.get_process_group_ranks(
                    sender.parallel_state.get_expert_data_parallel_group()
                ),
            }
        )
        write_receipt(prefix_join(self.output_prefix, f"{phase}-comparator-entry-rank{dist.get_rank()}.json"), entry)
        known = known_group_minmax(groups, next(iter(sender.sources.values())).device)
        write_receipt(
            prefix_join(self.output_prefix, f"{phase}-known-minmax-rank{dist.get_rank()}.json"),
            {"rank": dist.get_rank(), "reports": known},
        )
        started = time.monotonic()
        report = compare_replicas(sender.sources, groups, dist.get_rank(), version)
        write_receipt(
            prefix_join(self.output_prefix, f"{phase}-byte-comparison-rank{dist.get_rank()}.json"),
            {
                "comparison": asdict(report),
                "seconds": time.monotonic() - started,
                "source_iteration_order": list(sender.sources),
            },
        )
        write_receipt(
            prefix_join(self.output_prefix, f"{phase}-comparator-exit-rank{dist.get_rank()}.json"),
            comparator_snapshot(sender.sources, groups),
        )
        failures = torch.tensor(int(report.mismatched_bytes != 0), device="cuda")
        dist.all_reduce(failures, op=dist.ReduceOp.MAX)
        if failures.item():
            raise RuntimeError(f"Production byte comparator found divergent replicas at {phase}")
        if not all(row["passed"] for row in known):
            raise RuntimeError(f"Known-input MIN/MAX failed on live replica groups at {phase}")


def worker(args: argparse.Namespace) -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    if dist.get_world_size() != WORLD_SIZE:
        raise ValueError(f"This diagnostic requires {WORLD_SIZE} ranks: {PROBE_TOPOLOGY}")
    filesystem, path = filesystem_for(args.config_uri)
    with filesystem.open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError("Diagnostic configuration SHA mismatch")
    compiled = OmegaConf.create(raw.decode())
    cfg = compiled["skyrl"]
    geometry = cfg.trainer.policy.megatron_config
    for name, size in {
        "tensor_model_parallel_size": PROBE_TOPOLOGY["tp"],
        "pipeline_model_parallel_size": PROBE_TOPOLOGY["pp"],
        "context_parallel_size": PROBE_TOPOLOGY["cp"],
        "expert_model_parallel_size": PROBE_TOPOLOGY["ep"],
        "expert_tensor_parallel_size": 1,
    }.items():
        if geometry[name] != size:
            raise ValueError(f"Pinned source geometry {name}={geometry[name]} differs from probe {size}")
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
    probe.strategy = MegatronStrategy(
        geometry, optimizer_config=cfg.trainer.policy.optimizer_config, seed=cfg.trainer.seed
    )
    probe.strategy.setup_distributed(timeout=timedelta(seconds=180))
    probe._initialize_policy_modules(metadata_dir, mode=_MegatronInitMode.TRAINING)
    optimizer_started = time.monotonic()
    probe.optimizer = get_megatron_optimizer(
        probe.actor_module, cfg.trainer.policy.optimizer_config, geometry.optimizer_config_kwargs
    )
    probe.scheduler = get_megatron_optimizer_param_scheduler(
        probe.optimizer, cfg.trainer.policy.optimizer_config, num_training_steps=cfg.trainer.max_steps
    )
    optimizer_init_seconds = time.monotonic() - optimizer_started
    probe.record_parameters(probe.actor_module, "post_optimizer_init")
    probe.replica_sender = ExpertBlockSender(probe, mpu)
    probe.replica_sender.inventory()
    probe.compare_parameter_bytes("post_optimizer_init", version=0)
    wrapper = MegatronModelWrapper(
        config=cfg,
        actor_module=probe.actor_module,
        actor_optimizer=probe.optimizer,
        vocabulary_size=probe.strategy.hf_config.vocab_size,
    )
    probe.optimizer.zero_grad()
    with torch.no_grad():
        for chunk in probe.actor_module:
            chunk.zero_grad_buffer()
            for parameter in chunk.parameters():
                if parameter.requires_grad:
                    parameter.main_grad.fill_(SYNTHETIC_GRADIENT)
    step_started = time.monotonic()
    step_result = probe.strategy.optimizer_step(
        probe.optimizer,
        wrapper,
        probe.scheduler,
        name="actor",
        max_consecutive_nonfinite_steps=cfg.trainer.policy.max_consecutive_nonfinite_steps,
    )
    torch.cuda.synchronize()
    optimizer_step_seconds = time.monotonic() - step_started
    write_receipt(
        prefix_join(args.output_prefix, f"optimizer-step-rank{dist.get_rank()}.json"),
        {
            "result": asdict(step_result),
            "seconds": optimizer_step_seconds,
            "synthetic_post_reduction_gradient": SYNTHETIC_GRADIENT,
        },
    )
    if not step_result.applied:
        raise RuntimeError("The diagnostic optimizer step was not applied")
    probe.record_parameters(probe.actor_module, "post_optimizer_step")
    probe.compare_parameter_bytes("post_optimizer_step", version=1)
    if dist.get_rank() == 0:
        write_receipt(
            prefix_join(args.output_prefix, "receipt.json"),
            {
                "completed": True,
                "scope": (
                    "load plus one native optimizer update on synthetic post-reduction gradients; "
                    "no forward/backward or inference"
                ),
                "config_sha256": args.config_sha256,
                "world_size": dist.get_world_size(),
                "topology": PROBE_TOPOLOGY,
                "source_uri": model.source_uri,
                "phase_mismatches": {key: len(value) for key, value in probe.phase_differences.items()},
                "optimizer_constructed": True,
                "optimizer_steps_applied": 1,
                "optimizer_init_seconds_rank0": optimizer_init_seconds,
                "optimizer_step_seconds_rank0": optimizer_step_seconds,
                "inference_constructed": False,
            },
        )
    dist.barrier(group=probe.receipt_group)
    dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    info = get_job_info()
    if info is None or info.num_tasks != NUM_TASKS or torch.cuda.device_count() != GPUS_PER_TASK:
        raise ValueError(f"Submit a {NUM_TASKS}-task gang with {GPUS_PER_TASK} GB200 GPUs per task")
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
        f"--nnodes={NUM_TASKS}",
        f"--nproc-per-node={GPUS_PER_TASK}",
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
