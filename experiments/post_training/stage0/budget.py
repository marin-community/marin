# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #8225 Stage 0 training budget — one definition for both trainers.

The Snowball run (Levanter) and the three Axolotl bases must train under identical
hyperparameters for Stage 0 to be a shared control, so every value lives here exactly
once and both config modules construct their trainer-specific settings from these
constants. #8225: Adam at 5e-6 (betas 0.9/0.95, no warmup, gradient clip 1.0) on a
cosine schedule, global batch 64 sequences of 32,768 tokens, one epoch per source,
on 64 GPUs (8 nodes x 8 devices).

Warmup is expressed in steps here; the Levanter config takes a fraction, so the
Snowball module maps ``WARMUP_STEPS == 0`` to ``warmup=0.0`` — the zero agrees across
both semantics.
"""

LEARNING_RATE = 5e-6
# One Stage 0 seed for every architecture: the run is a shared control, not a
# seed replicate (Phase 1a varies seeds).
STAGE0_SEED = 0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0

SEQUENCE_LEN = 32768
GLOBAL_BATCH = 64
EPOCHS_PER_STAGE = 1

# Axolotl side: one conversation per sequence, 8 per device on the launcher's default
# 8-GPU torchrun width with accumulation 1 — 8 x 8 x 1 = 64 examples per step.
PER_DEVICE_BATCH = 8
GRADIENT_ACCUMULATION_STEPS = 1

# Snowball side: the mesh's context axis spans the DCN, one axis unit per node.
TRAIN_NODES = 8
GPUS_PER_NODE = 8

assert GLOBAL_BATCH == PER_DEVICE_BATCH * GPUS_PER_NODE * GRADIENT_ACCUMULATION_STEPS
