# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Producer-row identity shared by the RL picker, recent-run and engine queries."""

from vllm_observability import sql_string

# The MarinSkyRL row names whose records count as RL producers. The engine
# panels and every picker share this single definition.
RL_PRODUCER_ROW_NAMES = ("lifecycle", "policy_step", "rollout_call")

RL_PRODUCER_ROW_NAMES_SQL = ", ".join(sql_string(name) for name in RL_PRODUCER_ROW_NAMES)
