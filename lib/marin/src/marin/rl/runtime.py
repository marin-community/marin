# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime handles for RL workers.

Separate from orchestration.py to avoid circular imports
(rl_job → rollout_worker → runtime vs orchestration → rl_job).
"""

from dataclasses import dataclass

from fray import ActorHandle


@dataclass(frozen=True)
class WeightTransferRuntime:
    """Runtime handles for weight transfer coordination.

    Separate from WeightTransferConfig (which stays pure config, serializable, hashable).
    """

    arrow_flight_coordinator: ActorHandle | None = None


@dataclass(frozen=True)
class RLRuntimeHandles:
    """Runtime handles passed to RL workers.

    Created by the coordinator, passed to workers explicitly.
    Workers never call get_default_job_ctx() or discover actors by name.
    """

    curriculum: ActorHandle
    run_state: ActorHandle
    weight_transfer: WeightTransferRuntime
