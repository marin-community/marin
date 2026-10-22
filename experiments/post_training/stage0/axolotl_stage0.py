# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl Stage 0 configs (CL12): the same three stages on the three MoE bases.

Each base runs the #8225 curriculum in order — chat, thinking, terminal — under the same
budget as the Snowball run: global batch 64 examples per step (8 per device x the
launcher's 8-GPU torchrun width, accumulation 1), one epoch per source, fresh optimizer
per stage, with betas, schedule, clip and warmup identical to the Snowball config — both
are built from :mod:`.budget`, never from trainer defaults.

Every base trains on its family's instance of the shared template,
``shared_template(FAMILY_TOKEN_MAPS[arch])``, baked into the spec. No published template
is fetched and no family needs a template source override; reasoning renders from
``reasoning_content`` in the converted rows on every architecture.

Router policy, kept consistent with the Snowball arm: the Qwen bases carry
``output_router_logits=True`` so the MoE auxiliary loss trains, as Grug's does.
Nemotron-H's router correction bias is a frozen buffer under this fork, so its run sets
``output_router_logits=False`` — a recorded deviation from the other arms, not an
omission.

Stage N+1 initializes from stage N's exported checkpoint directory, derived from the
previous spec's own run name, so the chain and the launcher's output layout cannot drift
apart (the review's doubled-path bug).
"""

from experiments.post_training.axolotl_sft.bases import AXOLOTL_BASES, AxolotlBaseModel
from experiments.post_training.axolotl_sft.launcher import AxolotlOptimizer, AxolotlSpec
from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template, template_sha256
from experiments.post_training.stage0.budget import (
    ADAM_BETA1,
    ADAM_BETA2,
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

# Router aux loss per architecture: on for the Qwen MoEs (as on Grug), off — the recorded
# deviation — for Nemotron-H's frozen correction bias.
ROUTER_AUX_LOSS: dict[str, bool] = {
    "qwen3_moe": True,
    "qwen3_5_moe": True,
    "nemotron_h": False,
}


def stage_optimizer() -> AxolotlOptimizer:
    """The budget's optimizer, explicit in every field; identical to the Snowball AdamConfig."""
    return AxolotlOptimizer(
        # axolotl's YAML takes the learning rate as a string; :g round-trips the float.
        learning_rate=f"{LEARNING_RATE:g}",
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        lr_scheduler=LR_SCHEDULE,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
    )


def stage_output(spec: AxolotlSpec, *, output_root: str) -> str:
    """Where the launcher leaves ``spec``'s exported checkpoint: the run's own directory.

    ``launch_axolotl`` addresses a run at ``{output_root}/{spec.name}`` (its
    ``AXOLOTL_WORKDIR``), and the pod-side entrypoint trains in scratch and uploads the
    run directory there. The stage chain derives each init path from the previous spec's
    own name through this one function, so renaming a stage breaks the chain loudly at
    config time instead of silently loading a stale checkpoint.
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
                data_manifest=f"{data_root}/{source_filename(source)}",
                optimizer=optimizer,
                output_router_logits=router_aux_loss,
                sequence_len=SEQUENCE_LEN,
                per_device_batch_size=PER_DEVICE_BATCH,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                epochs=EPOCHS_PER_STAGE,
                chat_template=chat_template,
                template_sha256=template_sha256(tokens),
                seed=STAGE0_SEED,
                sample_packing=False,
                init_checkpoint=None if previous is None else stage_output(previous, output_root=output_root),
            )
        )
    return specs
