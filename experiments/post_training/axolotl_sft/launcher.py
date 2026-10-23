# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl SFT on Iris from the pinned ``marin-community/axolotl`` fork.

The submitter side (this module, in Marin's environment) builds one self-contained
JSON payload per run: the fork's training YAML with the shared chat template inline,
the pins, and the object-storage locations of the CL12 data, the init checkpoint and
the output. The job runs in the pinned Axolotl task image, where
:mod:`experiments.post_training.axolotl_sft.pod_runner` (standard library only)
clones the fork, stages the inputs, trains through ``torchrun`` with FSDP2 on one
8-GPU node, checks the saved tokenizer embeds the template, and uploads the output.

Launch one stage (``--dry-run`` prints the Iris command without submitting)::

    python -m experiments.post_training.axolotl_sft.launcher submit \\
        --base qwen3-30b-a3b-base --stage 1 --data-root s3://.../stage0-cl12/2026.10.22 \\
        --output-root s3://.../stage0-axolotl --job-name opd-0b-qwen3-s1 [--max-steps 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

AXOLOTL_FORK_REPO = "https://github.com/marin-community/axolotl"
# Fork main at the time of this launcher; MARIN_FORK.md records the fork point and
# the ported features (template_integrity, expert_parallel, mfu, supabase_register).
AXOLOTL_FORK_REVISION = "a208d55050da5794098c183f7ac474e89c3f6859"
AXOLOTL_TRANSFORMERS_PIN = "5.12.1"
# The fork's published task image (upstream axolotl-uv base plus the fork's recipe
# layer, built by its build-marin-tinker-sft workflow at fork commit 79759fbe). The
# pod runner overlays the fork's source at AXOLOTL_FORK_REVISION on top of it.
AXOLOTL_TASK_IMAGE = (
    "ghcr.io/marin-community/axolotl@sha256:810adf6a83dc6976a680affd994989782128d15ba524b84a188312c8fc5d95ed"
)
# Architectures the fork's transformers pin must ship for the Phase 0 bases.
SUPPORTED_ARCHITECTURES = ("qwen3_moe", "qwen3_5_moe", "nemotron_h")
# FSDP2 wraps each decoder block.
DECODER_LAYER_CLASSES = {
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "qwen3_5_moe": "Qwen3_5MoeDecoderLayer",
    "nemotron_h": "NemotronHBlock",
}

TEMPLATE_INTEGRITY_PLUGIN = "axolotl.integrations.template_integrity.TemplateIntegrityPlugin"
# Shipped next to the pod runner; see fp32_master.py.
FP32_MASTER_PLUGIN = "fp32_master.Fp32MasterWeightsPlugin"
GPUS_PER_NODE = 8
POD_RUNNER = "experiments/post_training/axolotl_sft/pod_runner.py"
PAYLOAD_ENV = "AXOLOTL_PAYLOAD"


@dataclass(frozen=True)
class AxolotlOptimizer:
    """Explicit optimizer settings; every field is required."""

    learning_rate: str  # axolotl YAML form, e.g. "5e-6"
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
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
    """One Axolotl SFT run: pinned base, data, shared template, optimizer, seed.

    ``data_uri`` is one CL12 ``{source}.jsonl.gz`` in object storage; ``init_uri``,
    when set, is a previous stage's uploaded export, which initializes the weights
    instead of the pinned base (the base still proves tokenizer and template
    provenance). Training is unpacked, one conversation per sequence: the Stage 0
    budget is matched in examples, and the masking canary is defined on unpacked
    sequences.
    """

    name: str
    base: AxolotlBaseModel
    data_uri: str
    optimizer: AxolotlOptimizer
    # The shared chat template for this family and its end-of-turn token, from
    # chat_template.FAMILY_TOKEN_MAPS[architecture].
    chat_template: str
    template_sha256: str
    turn_close: str
    seed: int
    # The MoE router auxiliary loss: True trains the router through it (Qwen's
    # pretraining rule); False records a deviation in the manifest.
    output_router_logits: bool
    sequence_len: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    init_uri: str | None = None

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
        if not self.base.revision:
            raise ValueError("a base without a revision pin cannot be cited in the manifest")

    @property
    def global_batch(self) -> int:
        return self.per_device_batch_size * GPUS_PER_NODE * self.gradient_accumulation_steps


def axolotl_config(spec: AxolotlSpec, *, max_steps: int | None = None) -> dict[str, Any]:
    """The fork's training YAML for ``spec``; the pod runner fills in local paths.

    The chat template rides inline (``chat_template: jinja`` +
    ``chat_template_jinja``). Only assistant turns train, through their end-of-turn
    token. ``template_integrity`` makes the saved checkpoint embed the template, and
    the fp32-master plugin keeps fp32 parameters and Adam state under bf16 compute.
    FSDP2 shards parameters, gradients and optimizer state across the node.
    """
    config: dict[str, Any] = {
        # A chained stage's pod runner replaces this with the staged previous export.
        "base_model": spec.base.hf_id,
        "revision_of_model": spec.base.revision,
        "trust_remote_code": False,
        "strict": False,
        "plugins": [TEMPLATE_INTEGRITY_PLUGIN, FP32_MASTER_PLUGIN],
        "tokenizer_save_jinja_files": False,
        "chat_template": "jinja",
        "chat_template_jinja": spec.chat_template,
        # The family's end-of-turn token ends generation, as it ends every trained turn.
        "special_tokens": {"eos_token": spec.turn_close},
        "datasets": [
            {
                "path": spec.data_uri,
                "ds_type": "json",
                "split": "train",
                "type": "chat_template",
                "field_messages": "messages",
                "roles_to_train": ["assistant"],
                "train_on_eot": "turn",
                "eot_tokens": [spec.turn_close],
            }
        ],
        "train_on_inputs": False,
        "val_set_size": 0.0,
        "shuffle_merged_datasets": False,
        "dataset_processes": 32,
        "sequence_len": spec.sequence_len,
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "micro_batch_size": spec.per_device_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "num_epochs": spec.epochs,
        "optimizer": "adamw_torch_fused",
        "learning_rate": spec.optimizer.learning_rate,
        "adam_beta1": spec.optimizer.adam_beta1,
        "adam_beta2": spec.optimizer.adam_beta2,
        "adam_epsilon": spec.optimizer.adam_epsilon,
        "weight_decay": spec.optimizer.weight_decay,
        "lr_scheduler": spec.optimizer.lr_scheduler,
        "warmup_steps": spec.optimizer.warmup_steps,
        "max_grad_norm": spec.optimizer.max_grad_norm,
        "seed": spec.seed,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "attn_implementation": "sdpa",
        "experts_implementation": "grouped_mm",
        # The router aux loss is explicit on the spec: without it the router trains only
        # through the LM loss.
        "output_router_logits": spec.output_router_logits,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 1,
        "fsdp_config": {
            "fsdp_version": 2,
            "offload_params": False,
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": DECODER_LAYER_CLASSES[spec.base.architecture],
            "state_dict_type": "FULL_STATE_DICT",
            "reshard_after_forward": True,
        },
    }
    if max_steps is not None:
        config["max_steps"] = max_steps
        config["save_strategy"] = "no"
    return config


def run_manifest(spec: AxolotlSpec, *, max_steps: int | None) -> dict[str, Any]:
    """The provenance record uploaded beside the run's output."""
    return {
        "spec_name": spec.name,
        "base": {"hf_id": spec.base.hf_id, "revision": spec.base.revision, "arch": spec.base.architecture},
        "data_uri": spec.data_uri,
        "init_uri": spec.init_uri,
        "fork_revision": AXOLOTL_FORK_REVISION,
        "transformers_pin": AXOLOTL_TRANSFORMERS_PIN,
        "task_image": AXOLOTL_TASK_IMAGE,
        "optimizer": asdict(spec.optimizer),
        "sequence_len": spec.sequence_len,
        "batch": {
            "per_device": spec.per_device_batch_size,
            "gradient_accumulation": spec.gradient_accumulation_steps,
            "gpus": GPUS_PER_NODE,
            "global": spec.global_batch,
        },
        "seed": spec.seed,
        "epochs": spec.epochs,
        "max_steps": max_steps,
        "template_sha256": spec.template_sha256,
        "master_weights": "fp32 (bf16 compute)",
        "router_rules": (
            {"output_router_logits": True}
            if spec.output_router_logits
            else {
                "deviation": (
                    "NemotronH's correction bias is a frozen buffer; Axolotl does not update it "
                    "during training. A bias-update callback in the fork is the follow-up if the "
                    "Stage 0 routing-shift measurement demands it."
                )
            }
        ),
    }


def launch_payload(spec: AxolotlSpec, *, output_uri: str, max_steps: int | None = None) -> dict[str, Any]:
    """Everything the pod runner needs, JSON-serializable."""
    return {
        "config": axolotl_config(spec, max_steps=max_steps),
        "fork_revision": AXOLOTL_FORK_REVISION,
        "transformers_pin": AXOLOTL_TRANSFORMERS_PIN,
        "data_uri": spec.data_uri,
        "init_uri": spec.init_uri,
        "output_uri": output_uri,
        "gpus": GPUS_PER_NODE,
        "manifest": run_manifest(spec, max_steps=max_steps),
    }


def iris_command(payload: dict[str, Any], *, job_name: str, cluster: str, timeout_hours: int) -> list[str]:
    """The ``iris job run`` argv for one run: one 8-GPU node in the Axolotl image."""
    return [
        "uv",
        "run",
        "iris",
        "--config",
        "lib/iris/config/marin.yaml",
        "job",
        "run",
        "--no-wait",
        "--enable-extra-resources",
        "--target-cluster",
        cluster,
        "--priority",
        "interactive",
        "--gpu",
        f"H100x{GPUS_PER_NODE}",
        "--cpu",
        "96",
        "--memory",
        "1200GB",
        "--disk",
        "2000GB",
        "--timeout",
        str(timeout_hours * 3600),
        "--no-sync",
        "--task-image",
        AXOLOTL_TASK_IMAGE,
        "--job-name",
        job_name,
        "-e",
        PAYLOAD_ENV,
        json.dumps(payload, sort_keys=True),
        "-e",
        "HF_TOKEN",
        os.environ.get("HF_TOKEN", ""),
        "--",
        "python",
        POD_RUNNER,
    ]


def main() -> None:
    # Imported here: the stage chain imports this module for its dataclasses.
    from experiments.post_training.stage0.axolotl_stage0 import stage_output, stage_specs  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Submit one Axolotl Stage 0 stage to Iris.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--base", required=True)
    submit.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    submit.add_argument("--data-root", required=True, help="the CL12 artifact directory")
    submit.add_argument("--output-root", required=True)
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--max-steps", type=int, default=None, help="a smoke run's step cap")
    submit.add_argument("--cluster", default="cw-rno2a")
    submit.add_argument("--timeout-hours", type=int, default=96)
    submit.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = stage_specs(args.base, data_root=args.data_root, output_root=args.output_root)[args.stage - 1]
    output_uri = stage_output(spec, output_root=args.output_root)
    if args.max_steps is not None:
        output_uri = f"{output_uri}-smoke{args.max_steps}"
    payload = launch_payload(spec, output_uri=output_uri, max_steps=args.max_steps)
    command = iris_command(payload, job_name=args.job_name, cluster=args.cluster, timeout_hours=args.timeout_hours)
    if args.dry_run:
        redacted = [
            ("<HF_TOKEN>" if previous == "HF_TOKEN" else part)
            for previous, part in zip(["", *command[:-1]], command, strict=True)
        ]
        print(shlex.join(redacted))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
