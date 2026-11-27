# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trainer rows that establish eligibility for RL run, job and execution pickers."""

RL_PRODUCER_ROW_NAMES = ("lifecycle", "policy_step", "rollout_call")
RL_PRODUCER_ROW_NAMES_SQL = ", ".join(f"'{name}'" for name in RL_PRODUCER_ROW_NAMES)
