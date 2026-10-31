# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl SFT on Iris from the pinned ``marin-community/axolotl`` fork.

The submitter side (this module, in Marin's environment) builds one self-contained
JSON payload per run: the fork's training YAML with the shared chat template inline,
the pins, and the object-storage locations of the CL12 data, the init checkpoint and
the output. The job runs in the pinned Axolotl task image, where
:mod:`experiments.post_training.axolotl_sft.pod_runner` (standard library only)
clones the fork, stages the inputs, trains through ``torchrun`` with FSDP2 on the
configured GPU nodes, checks the saved tokenizer embeds the template, and uploads the output.

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
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
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
# Shipped next to the pod runner; see router_aux_loss.py. Axolotl itself ignores
# `output_router_logits`, so the aux loss trains only through this plugin.
ROUTER_AUX_LOSS_PLUGIN = "router_aux_loss.RouterAuxLossPlugin"
# Packed conversations isolated in attention; see packed_attention.py.
PACKED_ATTENTION_PLUGIN = "packed_attention.PackedAttentionPlugin"
# The loss without [tokens x vocab] logits (the pod runner installs the fork's pinned build).
CUT_CROSS_ENTROPY_PLUGIN = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
# Per-architecture load-time checks shipped next to the pod runner (see nemotron_h_checks.py).
ARCHITECTURE_PLUGINS: dict[str, tuple[str, ...]] = {
    "qwen3_moe": (),
    "qwen3_5_moe": (),
    "nemotron_h": ("nemotron_h_checks.NemotronHChecksPlugin",),
}
GPUS_PER_NODE = 8
POD_RUNNER = "experiments/post_training/axolotl_sft/pod_runner.py"
PAYLOAD_ENV = "AXOLOTL_PAYLOAD"


class AttentionKernel(StrEnum):
    """Attention backends a spec may train with.

    Under sdpa, transformers isolates packed conversations with a materialized block-diagonal
    mask, so every layer computes attention over the whole 32k pack. The flash kernels take
    the packed segments as varlen sequences from the position-id resets instead.
    """

    SDPA = "sdpa"
    FLASH_ATTN3 = "kernels-community/flash-attn3"
    FLASH_ATTN2 = "kernels-community/flash-attn2"
    FLEX = "flex_attention"


class PromptStrategy(StrEnum):
    CHAT_TEMPLATE = "chat_template"
    GENERATION_MASK = "experiments.post_training.axolotl_sft.generation_mask"


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
    """One Axolotl SFT run: pinned base, data, explicit template, optimizer, seed.

    ``data_uri`` is one CL12 ``{source}.jsonl.gz`` in object storage; ``init_uri``,
    when set, is a previous stage's uploaded export, which initializes the weights
    instead of the pinned base (the base still proves tokenizer and template
    provenance). Conversations pack into ``sequence_len``-token sequences, one packed
    sequence per device micro-batch, as Snowball's stages pack.
    """

    name: str
    base: AxolotlBaseModel
    data_uri: str
    optimizer: AxolotlOptimizer
    # The complete training template and its end-of-turn token.
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
    nodes: int
    init_uri: str | None = None
    prompt_strategy: PromptStrategy = PromptStrategy.CHAT_TEMPLATE
    attn_implementation: AttentionKernel = AttentionKernel.SDPA

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
        return self.per_device_batch_size * self.nodes * GPUS_PER_NODE * self.gradient_accumulation_steps


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
        "plugins": [
            TEMPLATE_INTEGRITY_PLUGIN,
            FP32_MASTER_PLUGIN,
            CUT_CROSS_ENTROPY_PLUGIN,
            PACKED_ATTENTION_PLUGIN,
            *ARCHITECTURE_PLUGINS[spec.base.architecture],
            *([ROUTER_AUX_LOSS_PLUGIN] if spec.output_router_logits else []),
        ],
        "tokenizer_save_jinja_files": False,
        "chat_template": "jinja",
        "chat_template_jinja": spec.chat_template,
        # Multimodal shells (Qwen3.5-MoE) also load and save a processor, whose chat_template.jinja
        # transformers reads ahead of tokenizer_config.json; give it the same template.
        "processor_kwargs": {"chat_template": spec.chat_template},
        # The family's end-of-turn token ends generation, as it ends every trained turn.
        "special_tokens": {"eos_token": spec.turn_close},
        "datasets": [
            {
                "path": spec.data_uri,
                "ds_type": "json",
                "split": "train",
                "type": spec.prompt_strategy.value,
                "field_messages": "messages",
                "roles_to_train": ["assistant"],
                "train_on_eot": "turn",
            }
        ],
        # Axolotl reads eot_tokens from the top level, not from the dataset entry.
        "eot_tokens": [spec.turn_close],
        "train_on_inputs": False,
        "val_set_size": 0.0,
        "shuffle_merged_datasets": False,
        "dataset_processes": 32,
        "sequence_len": spec.sequence_len,
        # One packed epoch per stage, as Snowball trains: conversations pack into sequence_len
        # tokens, isolated from each other by the packed-attention plugin.
        "sample_packing": True,
        "pad_to_sequence_len": True,
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
        "attn_implementation": spec.attn_implementation.value,
        "experts_implementation": "grouped_mm",
        # Reduce-scatter gradients after every micro-batch. Skipping the sync during accumulation
        # keeps the whole model's unsharded bf16 gradients on every GPU (~61 GB for Qwen3-30B).
        "accelerator_config": {"gradient_accumulation_kwargs": {"sync_each_batch": True}},
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
            # FSDP's own per-block checkpointing; transformers' gradient_checkpointing under
            # FSDP2 adds an all-gather in backward and ran the Qwen3 smoke out of memory.
            "activation_checkpointing": True,
        },
    }
    if max_steps is not None:
        config["max_steps"] = max_steps
        # A smoke saves only the final model; Axolotl rejects save_steps without step saves.
        config["save_strategy"] = "no"
        del config["save_steps"], config["save_total_limit"]
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
            "gpus_per_node": GPUS_PER_NODE,
            "nodes": spec.nodes,
            "global": spec.global_batch,
        },
        "seed": spec.seed,
        "epochs": spec.epochs,
        "max_steps": max_steps,
        "template_sha256": spec.template_sha256,
        "prompt_strategy": spec.prompt_strategy.value,
        "attn_implementation": spec.attn_implementation.value,
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
        "nodes": spec.nodes,
        "manifest": run_manifest(spec, max_steps=max_steps),
    }


def iris_command(
    payload: dict[str, Any], *, job_name: str, cluster: str, timeout_hours: int, preprocess_only: bool = False
) -> list[str]:
    """The ``iris job run`` argv for one run: a gang of 8-GPU nodes in the Axolotl image.

    ``preprocess_only`` runs Axolotl's config validation and tokenization on a CPU node instead.
    """
    if preprocess_only:
        hardware = ["--cpu", "32", "--memory", "256GB", "--disk", "1000GB"]
    else:
        hardware = [
            "--gpu",
            f"H100x{GPUS_PER_NODE}",
            "--replicas",
            str(payload["nodes"]),
            "--cpu",
            "96",
            "--memory",
            "1200GB",
            "--disk",
            "2000GB",
        ]
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
        *hardware,
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
        *(["-e", "AXOLOTL_PREPROCESS_ONLY", "1"] if preprocess_only else []),
        "--",
        "python",
        POD_RUNNER,
    ]


def main() -> None:
    # Imported here: the stage chain imports this module for its dataclasses.
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415
        QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT,
        qwen_native_curriculum_specs,
        qwen_native_s3_spec,
        stage_output,
        stage_specs,
    )

    parser = argparse.ArgumentParser(description="Submit one Axolotl Stage 0 stage to Iris.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--base", required=True)
    submit.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    submit.add_argument(
        "--recipe",
        choices=("shared-curriculum", "qwen-native-s3", "qwen-native-curriculum"),
        default="shared-curriculum",
    )
    submit.add_argument("--data-root", required=True, help="the CL12 artifact directory")
    submit.add_argument("--output-root", required=True)
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--max-steps", type=int, default=None, help="a smoke run's step cap")
    submit.add_argument("--cluster", default="cw-rno2a")
    submit.add_argument("--timeout-hours", type=int, default=96)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--preprocess-only", action="store_true", help="validate and tokenize on CPU only")
    submit.add_argument("--init-uri", default=None, help="override the stage's init checkpoint (reload tests)")
    args = parser.parse_args()

    if args.recipe == "qwen-native-s3":
        if args.stage != 3 or args.init_uri is not None:
            parser.error("qwen-native-s3 requires stage 3 initialized from the pinned base")
        spec = qwen_native_s3_spec(args.base, data_root=args.data_root)
    elif args.recipe == "qwen-native-curriculum":
        if args.output_root != QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT:
            parser.error(f"qwen-native-curriculum chains its stages under {QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT}")
        spec = qwen_native_curriculum_specs(args.base, data_root=args.data_root)[args.stage - 1]
    else:
        spec = stage_specs(args.base, data_root=args.data_root, output_root=args.output_root)[args.stage - 1]
    if args.init_uri is not None:
        spec = replace(spec, init_uri=args.init_uri)
    output_uri = stage_output(spec, output_root=args.output_root)
    if args.max_steps is not None:
        output_uri = f"{output_uri}-smoke{args.max_steps}"
    payload = launch_payload(spec, output_uri=output_uri, max_steps=args.max_steps)
    command = iris_command(
        payload,
        job_name=args.job_name,
        cluster=args.cluster,
        timeout_hours=args.timeout_hours,
        preprocess_only=args.preprocess_only,
    )
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
