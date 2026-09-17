# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token budgets and transitions shared by the hero recipe and cooldown search."""

PRETRAIN_TOKENS = 15_000_000_000_000
COOLDOWN_TOKENS = 3_750_000_000_000
TOTAL_TOKENS = PRETRAIN_TOKENS + COOLDOWN_TOKENS
MIXTURE_SWITCH_FRACTION = 108_000 / 390_251
MAX_COMPONENT_EPOCHS = 8.0
