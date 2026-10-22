# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl SFT on Iris from the pinned ``marin-community/axolotl`` fork.

The fork pin, the transformers pin, and the supported architectures are explicit.
The pod-side entrypoint clones the fork at the pinned revision, re-pins transformers
(so the fork's dependency resolution cannot clobber it), writes one self-contained
config — the shared chat template rides inline as ``chat_template_jinja`` — launches
through ``torchrun`` with FSDP2 so a 30B MoE actually fits in 8 GPUs, verifies the
saved tokenizer embeds the configured template (the fork's ``template_integrity``
plugin, re-checked outside the save path), extracts the realized token counts, and
uploads the output and the manifest to object storage. Nothing under the pod's
scratch directory is treated as durable.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

import fsspec
import yaml
from fray.iris_backend import convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import ResourceConfig
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.timing import Duration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

AXOLOTL_FORK_REPO = "https://github.com/marin-community/axolotl"
# Fork main at the time of this launcher; MARIN_FORK.md records the fork point and
# the ported features (template_integrity, expert_parallel, mfu, supabase_register).
AXOLOTL_FORK_REVISION = "a208d55050da5794098c183f7ac474e89c3f6859"
AXOLOTL_TRANSFORMERS_PIN = "5.12.1"
# Architectures the fork's transformers pin must ship for the Phase 0 bases.
SUPPORTED_ARCHITECTURES = ("qwen3_moe", "qwen3_5_moe", "nemotron_h")

TEMPLATE_INTEGRITY_PLUGIN = "axolotl.integrations.template_integrity.TemplateIntegrityPlugin"
EXPERT_PARALLEL_PLUGIN = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
# Architectures the expert-parallel plugin is verified against; Nemotron-H support
# is unverified and must not be turned on by config accident.
EXPERT_PARALLEL_VERIFIED_ARCHITECTURES = ("qwen3_moe", "qwen3_5_moe")

# Pod-local scratch: checkpoints land here first and are uploaded after training.
# Anything written only to the pod is lost with it.
SCRATCH_ROOT = "/scratch/axolotl"


@dataclass(frozen=True)
class AxolotlOptimizer:
    """Explicit optimizer settings; every field is required."""

    learning_rate: str  # axolotl YAML form, e.g. "5e-6"
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    lr_scheduler: str
    warmup_steps: int
    max_grad_norm: float

    def __post_init__(self) -> None:
        # axolotl's YAML takes the learning rate as a string like "5e-6"; float() reads it back.
        if float(self.learning_rate) <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("adam betas must lie in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(frozen=True)
class AxolotlBaseModel:
    """A Phase 0 base checkpoint and its architecture."""

    name: str
    hf_id: str
    revision: str
    architecture: str


@dataclass(frozen=True)
class AxolotlSpec:
    """One Axolotl SFT run: pinned base, data, shared template, optimizer, seed."""

    name: str
    base: AxolotlBaseModel
    # Path (or glob) of the CL12 data manifest's jsonl files, readable from the pod.
    data_manifest: str
    optimizer: AxolotlOptimizer
    # The shared chat template for this family, from
    # chat_template.shared_template(FAMILY_TOKEN_MAPS[architecture]).
    chat_template: str
    template_sha256: str
    seed: int
    sequence_len: int = 32768
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    epochs: int = 1
    # One conversation per sequence: the Stage 0 budget is matched in examples, and
    # the #6839 masking canary is defined on unpacked sequences.
    sample_packing: bool = False
    # The fork's expert-parallel plugin size (1 = off).
    expert_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.base.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture {self.base.architecture!r} is outside the fork's supported set "
                f"{SUPPORTED_ARCHITECTURES}"
            )
        if not self.chat_template:
            raise ValueError("chat_template is required; build it with chat_template.shared_template(...)")
        if self.template_sha256 != hashlib.sha256(self.chat_template.encode()).hexdigest():
            raise ValueError("template_sha256 does not match the chat_template string")
        if not 0 <= self.seed < 2**31:
            raise ValueError(f"seed must be a non-negative int32, got {self.seed}")
        if self.sequence_len <= 0 or self.gradient_accumulation_steps <= 0 or self.per_device_batch_size <= 0:
            raise ValueError("sequence_len, per_device_batch_size and gradient_accumulation_steps must be positive")
        if self.expert_parallel_size > 1 and self.base.architecture not in EXPERT_PARALLEL_VERIFIED_ARCHITECTURES:
            raise ValueError(
                f"expert parallelism is not verified on {self.base.architecture!r}; verify the DeepEP "
                "plugin against this architecture first, then add it to EXPERT_PARALLEL_VERIFIED_ARCHITECTURES"
            )


def axolotl_config(spec: AxolotlSpec, *, scratch_output_dir: str) -> dict[str, Any]:
    """The fork's training YAML for ``spec``.

    The chat template rides inline (``chat_template: jinja`` +
    ``chat_template_jinja``), so the submitter's disk never has to agree with the
    pod's. ``template_integrity`` is enabled through ``plugins:`` —
    ``tokenizer_save_jinja_files: false`` alone does not enable it — and the
    saved checkpoint must embed the template. Qwen models train with their router
    aux loss on (``output_router_logits``); NemotronH's correction bias is a frozen
    buffer this config cannot update, and the manifest records that deviation.
    FSDP2 shards parameters, gradients and optimizer state, which is what makes a
    30B MoE with Adam state fit on 8 GPUs.
    """
    plugins = [TEMPLATE_INTEGRITY_PLUGIN]
    if spec.expert_parallel_size > 1:
        plugins.append(EXPERT_PARALLEL_PLUGIN)
    config: dict[str, Any] = {
        "base_model": spec.base.hf_id,
        "revision_of_model": spec.base.revision,
        "sequence_len": spec.sequence_len,
        "sample_packing": spec.sample_packing,
        "plugins": plugins,
        # template_integrity: the saved checkpoint embeds its chat template.
        "tokenizer_save_jinja_files": False,
        "tokenizer_use_fast": True,
        "train_on_inputs": False,
        "datasets": [{"path": spec.data_manifest, "type": "chat_template"}],
        "num_epochs": spec.epochs,
        "per_device_train_batch_size": spec.per_device_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "optimizer": "adamw_torch_fused",
        "learning_rate": spec.optimizer.learning_rate,
        "adam_beta1": spec.optimizer.adam_beta1,
        "adam_beta2": spec.optimizer.adam_beta2,
        "weight_decay": spec.optimizer.weight_decay,
        "lr_scheduler": spec.optimizer.lr_scheduler,
        "warmup_steps": spec.optimizer.warmup_steps,
        "max_grad_norm": spec.optimizer.max_grad_norm,
        "seed": spec.seed,
        "logging_steps": 1,
        # Train-equals-serve: the shared template, inline.
        "chat_template": "jinja",
        "chat_template_jinja": spec.chat_template,
        # Qwen's router aux loss is off by default; without it the router trains
        # only through the LM loss.
        **({"output_router_logits": True} if spec.base.architecture.startswith("qwen3") else {}),
        # FSDP2: parameters, gradients and optimizer state sharded across the node.
        "fsdp_version": 2,
        "fsdp": ["full_shard", "auto_wrap"],
        "fsdp_config": {
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_auto_wrap_policy": "SIZE_BASED_WRAP",
            "fsdp_sync_module_states": True,
        },
        "output_dir": scratch_output_dir,
    }
    if spec.expert_parallel_size > 1:
        config["expert_parallel_size"] = spec.expert_parallel_size
    return config


def training_command(config_path: str, *, nodes: int, gpus_per_node: int) -> list[str]:
    """The exact pod-side training command; the printed form and the subprocess
    call share this builder."""
    return [
        "torchrun",
        f"--nnodes={nodes}",
        f"--nproc-per-node={gpus_per_node}",
        "-m",
        "axolotl.cli.train",
        config_path,
    ]


def axolotl_launch_command(
    spec: AxolotlSpec,
    *,
    nodes: int = 1,
    gpus_per_node: int = 8,
    max_steps: int | None = None,
) -> str:
    """The exact shell command a launch runs; the 20-step smoke passes ``max_steps``."""
    config = axolotl_config(spec, scratch_output_dir=f"{SCRATCH_ROOT}/{spec.name}")
    if max_steps is not None:
        config["max_steps"] = max_steps
    config_path = f"{SCRATCH_ROOT}/{spec.name}/axolotl_config.yaml"
    return " && ".join(
        [
            f"git clone {AXOLOTL_FORK_REPO} && git -C axolotl checkout {AXOLOTL_FORK_REVISION}",
            shlex.join(training_command(config_path, nodes=nodes, gpus_per_node=gpus_per_node)),
        ]
    )


@dataclass(frozen=True)
class AxolotlLaunchPayload:
    """Everything the pod-side entrypoint needs, JSON-serializable."""

    config: dict[str, Any]
    fork_revision: str
    output_root: str
    manifest: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def run_axolotl_training(payload_json: str) -> None:
    """Pod-side entrypoint: clone the pinned fork, re-pin transformers, train.

    The fork is cloned at the pinned revision and installed into the pod
    environment; transformers is then re-pinned explicitly because the fork's
    dependency resolution may otherwise upgrade it. Training runs through torchrun
    with the config's FSDP2 settings. The saved tokenizer must embed the chat
    template byte-for-byte, the realized token counts are read from the trainer
    state, and output plus manifest are uploaded to ``output_root`` — the pod's
    scratch directory does not survive the job.
    """
    payload = AxolotlLaunchPayload(**json.loads(payload_json))
    workdir = os.environ.get("AXOLOTL_WORKDIR", os.getcwd())
    fork_dir = os.path.join(workdir, "axolotl")
    if not os.path.exists(os.path.join(fork_dir, ".git")):
        subprocess.run(["git", "clone", AXOLOTL_FORK_REPO, fork_dir], check=True)
    subprocess.run(["git", "checkout", payload.fork_revision], cwd=fork_dir, check=True)
    subprocess.run(["uv", "pip", "install", "--system", "-e", "."], cwd=fork_dir, check=True)
    # The fork's dependency resolution may upgrade transformers; put the pin back.
    subprocess.run(
        ["uv", "pip", "install", "--system", f"transformers=={AXOLOTL_TRANSFORMERS_PIN}"],
        cwd=fork_dir,
        check=True,
    )

    config_path = os.path.join(workdir, "axolotl_config.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(payload.config, handle, sort_keys=False)

    command = training_command(
        config_path,
        nodes=int(os.environ.get("AXOLOTL_NODES", "1")),
        gpus_per_node=int(os.environ.get("AXOLOTL_GPUS", "8")),
    )
    logger.info("running %s", shlex.join(command))
    subprocess.run(command, check=True, cwd=workdir)

    output_dir = payload.config["output_dir"]
    _verify_template_integrity(payload, output_dir)
    realized_tokens = _realized_token_counts(output_dir)

    manifest = dict(payload.manifest)
    manifest["realized_tokens"] = realized_tokens
    manifest_path = os.path.join(workdir, "axolotl_run_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    _upload(output_dir, manifest_path, payload.output_root)
    logger.info("uploaded output and manifest to %s", payload.output_root)


def _verify_template_integrity(payload: AxolotlLaunchPayload, output_dir: str) -> None:
    """The fork's template_integrity contract, re-checked outside the save path.

    The comparison is unconditional: every Stage 0 run carries the shared template.
    """
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    embedded = getattr(tokenizer, "chat_template", None)
    if embedded is None:
        raise ValueError(
            f"saved checkpoint at {output_dir} embeds no chat template; "
            "the serving stack would fail (the template_integrity footgun)"
        )
    configured = payload.config["chat_template_jinja"]
    if embedded != configured:
        raise ValueError("the embedded chat template differs from the configured one")


def _realized_token_counts(output_dir: str) -> dict[str, Any]:
    """The run's realized token counts from the trainer state.

    Axolotl logs token totals in ``trainer_state.json``'s ``log_history``; a run
    whose state carries none fails here rather than shipping a manifest that
    cannot be checked against the data budget.
    """
    state_paths = sorted(glob.glob(os.path.join(output_dir, "**", "trainer_state.json"), recursive=True))
    if not state_paths:
        raise ValueError(f"no trainer_state.json under {output_dir}; cannot record realized tokens")
    with open(state_paths[-1]) as handle:
        state = json.load(handle)
    totals: dict[str, int] = {}
    for entry in state.get("log_history", []):
        for key, value in entry.items():
            if "token" in key and isinstance(value, (int, float)) and key not in ("tokens_per_second_per_gpu",):
                totals[key] = max(totals.get(key, 0), int(value))
    if not totals:
        raise ValueError(
            f"trainer_state.json at {state_paths[-1]} carries no token counts; "
            "the manifest cannot be checked against the data budget"
        )
    return {"source": state_paths[-1], "counts": totals, "global_step": state.get("global_step")}


def _upload(output_dir: str, manifest_path: str, output_root: str) -> None:
    """Upload the checkpoint, the trainer state and the manifest to ``output_root``."""
    filesystem = fsspec.filesystem("s3")
    target = f"{output_root.rstrip('/')}/{os.path.basename(output_dir)}"
    filesystem.put(output_dir, target, recursive=True)
    filesystem.put(manifest_path, f"{target}/axolotl_run_manifest.json")


def launch_axolotl(
    spec: AxolotlSpec,
    *,
    endpoint_url: str,
    output_root: str,
    nodes: int = 1,
    gpus_per_node: int = 8,
    max_steps: int | None = None,
    priority_band: int = job_pb2.PRIORITY_BAND_INHERIT,
) -> str:
    """Submit one Axolotl SFT job to Iris and return the job id."""
    scratch_output_dir = f"{SCRATCH_ROOT}/{spec.name}"
    config = axolotl_config(spec, scratch_output_dir=scratch_output_dir)
    if max_steps is not None:
        config["max_steps"] = max_steps
    manifest = {
        "spec_name": spec.name,
        "base": {"hf_id": spec.base.hf_id, "revision": spec.base.revision, "arch": spec.base.architecture},
        "data_manifest": spec.data_manifest,
        "fork_revision": AXOLOTL_FORK_REVISION,
        "transformers_pin": AXOLOTL_TRANSFORMERS_PIN,
        "optimizer": {
            "learning_rate": spec.optimizer.learning_rate,
            "weight_decay": spec.optimizer.weight_decay,
            "adam_beta1": spec.optimizer.adam_beta1,
            "adam_beta2": spec.optimizer.adam_beta2,
            "lr_scheduler": spec.optimizer.lr_scheduler,
            "warmup_steps": spec.optimizer.warmup_steps,
        },
        "sequence_len": spec.sequence_len,
        "batch": {
            "per_device": spec.per_device_batch_size,
            "gradient_accumulation": spec.gradient_accumulation_steps,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "expert_parallel_size": spec.expert_parallel_size,
        },
        "seed": spec.seed,
        "epochs": spec.epochs,
        "sample_packing": spec.sample_packing,
        "template_sha256": spec.template_sha256,
        "router_rules": (
            {"output_router_logits": True}
            if spec.base.architecture.startswith("qwen3")
            else {
                "deviation": (
                    "NemotronH's correction bias is a frozen buffer; Axolotl does not update it "
                    "during training. Recorded here per the review; a bias-update callback in the "
                    "fork is the follow-up if the Stage 0 routing-shift measurement demands it."
                )
            }
        ),
        "chat_template_sha_source": "chat_template.template_sha256",
        "output_root": output_root,
        "scratch_output_dir": scratch_output_dir,
    }
    payload = AxolotlLaunchPayload(
        config=config,
        fork_revision=AXOLOTL_FORK_REVISION,
        output_root=output_root,
        manifest=manifest,
    )

    resources = ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_node,
        cpu=_GPU_HOST_CPU_PER_DEVICE * gpus_per_node,
        ram=f"{_GPU_HOST_RAM_GB_PER_DEVICE * gpus_per_node}g",
        disk=f"{_GPU_HOST_DISK_GB_PER_DEVICE * gpus_per_node}g",
        replicas=nodes,
    )
    native_resources = convert_resources(resources)
    command = Entrypoint(
        command=["python", "-m", "experiments.post_training.axolotl_sft.launcher", "--payload", "launch-payload.json"],
        workdir_files={"launch-payload.json": payload.to_json().encode()},
    )
    marin_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    environment = EnvironmentSpec(
        env_vars={
            "GIT_COMMIT": marin_commit,
            "AXOLOTL_WORKDIR": scratch_output_dir,
            "AXOLOTL_NODES": str(nodes),
            "AXOLOTL_GPUS": str(gpus_per_node),
        },
        extras=["gpu"],
    )
    converted = convert_resources(resources)
    with IrisClient.remote(endpoint_url) as client:
        job = client.submit(
            wrap_multiprocess(command, native_resources, processes_per_task=1),
            name=f"axolotl-{spec.name}",
            resources=native_resources,
            replicas=nodes,
            environment=environment,
            coscheduling=resolve_coscheduling(resources, nodes),
            scheduling_timeout=Duration.from_hours(24),
            timeout=Duration.from_hours(72),
            max_retries_failure=0,
            priority_band=priority_band,
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        )
    logger.info("submitted axolotl run %s as job %s", spec.name, job)
    del converted
    return str(job)


_GPU_HOST_CPU_PER_DEVICE = 16
_GPU_HOST_RAM_GB_PER_DEVICE = 220
_GPU_HOST_DISK_GB_PER_DEVICE = 512


def __main__() -> None:  # pragma: no cover - entrypoint
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="submit a 20-step smoke run of one spec")
    smoke.add_argument("--endpoint", required=True)
    smoke.add_argument("--output-root", required=True)
    smoke.add_argument("--spec-json", required=True, help="AxolotlSpec as JSON")
    run = subparsers.add_parser("run-payload", help="pod-side entrypoint")
    run.add_argument("--payload", required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        spec = AxolotlSpec(**json.loads(args.spec_json))
        job = launch_axolotl(spec, endpoint_url=args.endpoint, output_root=args.output_root, max_steps=20)
        print(job)
    elif args.command == "run-payload":
        with open(args.payload) as handle:
            run_axolotl_training(handle.read())


if __name__ == "__main__":
    __main__()
