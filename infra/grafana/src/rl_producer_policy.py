# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Producer-row identity shared by the RL picker and recent-run queries."""

from vllm_observability import sql_string

# The MarinSkyRL row names whose records count as RL producers; the run, job,
# execution and recent-run eligibility filters all import this list.
RL_PRODUCER_ROW_NAMES = ("lifecycle", "policy_step", "rollout_call")

RL_PRODUCER_ROW_NAMES_SQL = ", ".join(sql_string(name) for name in RL_PRODUCER_ROW_NAMES)
