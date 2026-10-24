# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #8225 Stage 0 training budget — one definition for both trainers.

The Snowball run (Levanter) and the three Axolotl bases must train under identical
hyperparameters for Stage 0 to be a shared control, so every value lives here exactly
once and both config modules construct their trainer-specific settings from these
constants. #8225: Adam at 5e-6 (betas 0.9/0.95, no warmup, gradient clip 1.0) on a
cosine schedule, global batch 64 packed sequences of 32,768 tokens, one packed epoch
per source. Both trainers pack: Snowball on 64 GPUs (8 nodes x 8 devices), each Axolotl
base on 16 (2 nodes x 8) with gradient accumulation.

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
ADAM_EPSILON = 1e-8
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0

SEQUENCE_LEN = 32768
GLOBAL_BATCH = 64
EPOCHS_PER_STAGE = 1

# Axolotl side: one packed 32k-token sequence per device micro-batch, on two 8-GPU nodes
# (fp32 masters and Adam state for a 30B MoE do not fit one node's HBM) with
# accumulation 4 — 1 x 16 x 4 = 64 packed sequences per step.
PER_DEVICE_BATCH = 1
AXOLOTL_NODES = 2
GRADIENT_ACCUMULATION_STEPS = 4

# Snowball side: the mesh's context axis spans the DCN, one axis unit per node.
TRAIN_NODES = 8
GPUS_PER_NODE = 8

assert GLOBAL_BATCH == PER_DEVICE_BATCH * AXOLOTL_NODES * GPUS_PER_NODE * GRADIENT_ACCUMULATION_STEPS
