# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl Stage 0 configs (CL12): shared curriculum and native Qwen terminal SFT.

The shared curriculum runs chat, thinking, and terminal SFT in order under the same
budget as the Snowball run: global batch 64 examples per step (1 per device x 16 GPUs on
two nodes x 4 accumulation steps), one epoch per source, fresh
optimizer per stage, with betas, epsilon, schedule, clip and warmup identical to the
Snowball config — both are built from :mod:`.budget`, never from trainer defaults.

The original curriculum uses each family's instance of the shared template.
``qwen_native_curriculum_specs`` runs the same three stages on a Qwen base's pinned
native template, retaining native history truncation; ``qwen_native_s3_spec`` instead
initializes terminal SFT from the untrained Qwen base on that template.

Router policy, kept consistent with the Snowball arm: the Qwen bases carry
``output_router_logits=True`` so the MoE auxiliary loss trains, as Grug's does.
Nemotron-H's router correction bias is a frozen buffer under this fork, so its run sets
``output_router_logits=False`` — a recorded deviation from the other arms, not an
omission.

Stage N+1 initializes from stage N's uploaded export, derived from the previous spec's
own run name through :func:`stage_output`, so the chain and the launcher's output layout
cannot drift apart (the review's doubled-path bug).
"""

import hashlib
from dataclasses import replace

from experiments.post_training.axolotl_sft.bases import AXOLOTL_BASES, AxolotlBaseModel
from experiments.post_training.axolotl_sft.launcher import (
    ROUTER_AUX_LOSS_PLUGIN,
    AttentionKernel,
    AxolotlOptimizer,
    AxolotlSpec,
    PromptStrategy,
    axolotl_config,
)
from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template, template_sha256
from experiments.post_training.stage0.budget import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    AXOLOTL_NODES,
    EPOCHS_PER_STAGE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LR_SCHEDULE,
    MAX_GRAD_NORM,
    PER_DEVICE_BATCH,
    SEQUENCE_LEN,
    STAGE0_SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from experiments.post_training.stage0.convert import STAGE_ORDER, source_filename
from experiments.post_training.stage0.qwen3_native_template import CHAT_TEMPLATE as QWEN3_NATIVE_TEMPLATE
from experiments.post_training.stage0.qwen35_native_template import CHAT_TEMPLATE as QWEN35_NATIVE_TEMPLATE

# Router aux loss per architecture: on for the Qwen MoEs (as on Grug), off — the recorded
# deviation — for Nemotron-H's frozen correction bias.
ROUTER_AUX_LOSS: dict[str, bool] = {
    "qwen3_moe": True,
    "qwen3_5_moe": True,
    "nemotron_h": False,
}
QWEN_NATIVE_TEMPLATES = {"qwen3_moe": QWEN3_NATIVE_TEMPLATE, "qwen3_5_moe": QWEN35_NATIVE_TEMPLATE}


def require_router_aux_loss(spec: AxolotlSpec, config: dict) -> None:
    """Fail at construction when an architecture that trains its router aux loss would not.

    Axolotl ignores ``output_router_logits`` in its config; only the router-aux plugin turns the
    loss on. A Qwen stage once trained with it silently off, so the spec, the config and the
    plugin list must all agree with ``ROUTER_AUX_LOSS``.
    """
    expected = ROUTER_AUX_LOSS[spec.base.architecture]
    if spec.output_router_logits != expected:
        raise ValueError(f"{spec.name}: output_router_logits={spec.output_router_logits}, the recipe says {expected}")
    if expected and ROUTER_AUX_LOSS_PLUGIN not in config["plugins"]:
        raise ValueError(f"{spec.name}: the router aux loss is required but {ROUTER_AUX_LOSS_PLUGIN} is not loaded")


def stage_optimizer() -> AxolotlOptimizer:
    """The budget's optimizer, explicit in every field; identical to the Snowball AdamConfig."""
    return AxolotlOptimizer(
        # axolotl's YAML takes the learning rate as a string; :g round-trips the float.
        learning_rate=f"{LEARNING_RATE:g}",
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        adam_epsilon=ADAM_EPSILON,
        lr_scheduler=LR_SCHEDULE,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
    )


def stage_output(spec: AxolotlSpec, *, output_root: str) -> str:
    """Where the pod runner uploads ``spec``'s output: ``{output_root}/{spec.name}``.

    The stage chain derives each init path from the previous spec's own name through
    this one function, so renaming a stage breaks the chain loudly at config time
    instead of silently loading a stale checkpoint.
    """
    return f"{output_root}/{spec.name}"


def stage_specs(base_name: str, *, data_root: str, output_root: str) -> list[AxolotlSpec]:
    """The three chained stage specs for one base, in curriculum order.

    ``data_root`` is the CL12 conversion's output directory (``stage0_convert``'s resolved
    path), so both trainer stacks read the same converted files. Stage 1 trains from the
    pinned base; stages 2 and 3 initialize from the previous stage's output directory.
    """
    base: AxolotlBaseModel = AXOLOTL_BASES[base_name]
    tokens = FAMILY_TOKEN_MAPS[base.architecture]
    chat_template = shared_template(tokens)
    optimizer = stage_optimizer()
    router_aux_loss = ROUTER_AUX_LOSS[base.architecture]
    specs: list[AxolotlSpec] = []
    for stage_index, source in enumerate(STAGE_ORDER):
        previous = specs[-1] if specs else None
        specs.append(
            AxolotlSpec(
                name=f"stage0-{base_name}/s{stage_index + 1}-{source}",
                base=base,
                data_uri=f"{data_root}/{source_filename(source)}",
                optimizer=optimizer,
                output_router_logits=router_aux_loss,
                sequence_len=SEQUENCE_LEN,
                per_device_batch_size=PER_DEVICE_BATCH,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                epochs=EPOCHS_PER_STAGE,
                nodes=AXOLOTL_NODES,
                chat_template=chat_template,
                template_sha256=template_sha256(tokens),
                turn_close=tokens.turn_close,
                seed=STAGE0_SEED,
                init_uri=None if previous is None else stage_output(previous, output_root=output_root),
            )
        )
        require_router_aux_loss(specs[-1], axolotl_config(specs[-1]))
    return specs


QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT = "s3://marin-rno2a/marin/users/benfeuer/opd-0b/stage0-axolotl-native"
# Varlen flash attention over each packed conversation, a throughput change only: sdpa's
# block-diagonal mask made every Qwen3 layer attend over the whole 32k pack (~85 s/step).
# packing_check.py: packed and alone logits are bit-identical under this kernel.
QWEN_NATIVE_CURRICULUM_ATTENTION = AttentionKernel.FLASH_ATTN3


def qwen_native_spec(spec: AxolotlSpec) -> AxolotlSpec:
    """``spec`` rendered with its base's pinned native Qwen template and final-render labels."""
    template = QWEN_NATIVE_TEMPLATES[spec.base.architecture]
    native = replace(
        spec,
        chat_template=template,
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        prompt_strategy=PromptStrategy.GENERATION_MASK,
    )
    require_router_aux_loss(native, axolotl_config(native))
    return native


def qwen_native_curriculum_specs(base_name: str, *, data_root: str) -> list[AxolotlSpec]:
    """The shared three-stage curriculum on a Qwen base's native template.

    Only the template, the label strategy it needs and the attention kernel differ from
    :func:`stage_specs`; the stages chain under :data:`QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT`,
    apart from the shared run.
    """
    shared = stage_specs(base_name, data_root=data_root, output_root=QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT)
    return [replace(qwen_native_spec(spec), attn_implementation=QWEN_NATIVE_CURRICULUM_ATTENTION) for spec in shared]


def qwen_native_s3_spec(base_name: str, *, data_root: str) -> AxolotlSpec:
    """Train terminal SFT directly from a pinned Qwen base with its native template."""
    terminal = stage_specs(base_name, data_root=data_root, output_root="")[2]
    return qwen_native_spec(replace(terminal, name=f"stage0-{base_name}/s3-native-terminal", init_uri=None))
