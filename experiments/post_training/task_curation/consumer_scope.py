# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select task requirements that need a bespoke RL-consumer check."""

from finestore.rl.curation import ConsumerNeed
from pydantic import BaseModel, ConfigDict, Field
from taskcompendium.models import ExecutionRequirements, TaskSpec, judge_spec

CONSUMER_RUNTIME_COMMIT = "b937327803bd3656305b070f2b56c5176c7a6c9d"


class ConsumerCapacity(BaseModel):
    """Limits of the configured consumer's sandbox and phase deadlines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpus: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    storage_mb: int = Field(gt=0)
    agent_timeout: float | None = Field(gt=0)
    verifier_timeout: float | None = Field(gt=0)

    def supports(self, requirements: ExecutionRequirements) -> bool:
        return (
            requirements.cpus <= self.cpus
            and requirements.memory_mb <= self.memory_mb
            and requirements.storage_mb <= self.storage_mb
            and (self.agent_timeout is None or requirements.agent_timeout_sec <= self.agent_timeout)
            and (self.verifier_timeout is None or requirements.verifier_timeout_sec <= self.verifier_timeout)
        )


def consumer_needs(task: TaskSpec, capacity: ConsumerCapacity) -> frozenset[ConsumerNeed]:
    """Select bespoke needs; standard graders and satisfiable allocations are exempt."""
    needs = set()
    providers = task.environment_requirements.tool_providers
    if providers:
        needs.add(ConsumerNeed.TOOL_PROVIDER)
    if any(provider.network_access == "service_only" for provider in providers.values()):
        needs.add(ConsumerNeed.SERVICE)
    if judge_spec(task.grader) is not None:
        needs.add(ConsumerNeed.CUSTOM_JUDGE)
    if task.execution_requirements is not None and not capacity.supports(task.execution_requirements):
        needs.add(ConsumerNeed.EXECUTION)
    verifier = task.verifier_execution_requirements
    if verifier is not None and verifier != task.execution_requirements:
        needs.add(ConsumerNeed.VERIFIER_ALLOCATION)
    if verifier is not None and not capacity.supports(verifier):
        needs.add(ConsumerNeed.EXECUTION)
    return frozenset(needs)
