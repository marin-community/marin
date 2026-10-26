# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, SolverMachine
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    OpenEndedProposalSource,
    document_template,
)
from experiments.post_training.capability_driven_envs.open_ended_runner import (
    OpenEndedRuntime,
    SlotRunStatus,
    run_open_ended_bundle,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_open_ended import bundle
from experiments.post_training.capability_driven_envs.tests.test_open_ended_pipeline import coordinator
from experiments.post_training.capability_driven_envs.tests.test_open_ended_proposal_source import plan, slot

CONFIG = OpenEndedProposalConfig(Environment.SHELLSIM, SolverMachine.SHELLSIM)


def document(number: int) -> str:
    return document_template(bundle(), number, CONFIG).replace(
        "<...>",
        "Use the supplied donor and cell-type counts for a reproducible donor-aware analysis. Compare both available "
        "denominators, explain missing calls, and keep conclusions calibrated to the evidence. The private author "
        "builds a frozen weighted rubric and validates positive, partial, severe-negative, verbosity, and "
        "evaluator-injection controls before accepting the task.",
    )


async def test_bundle_runner_persists_proposals_runs_workpieces_and_is_idempotent(tmp_path):
    model = ScriptedModel()
    model.reply(plan(slot(1, status="null"), slot(2)))
    model.reply(document(2))
    source = OpenEndedProposalSource(model, LLMPolicy(), CONFIG)
    runtime = OpenEndedRuntime(source, lambda hydrated: coordinator(tmp_path, hydrated))

    first = await run_open_ended_bundle(bundle(), runtime, slot_count=2, root=tmp_path)
    request_count = len(model.requests)
    second = await run_open_ended_bundle(bundle(), runtime, slot_count=2, root=tmp_path)

    assert second == first
    assert len(model.requests) == request_count
    assert [result.status for result in first.slots] == [SlotRunStatus.NULL, SlotRunStatus.WORKPIECE]
    assert first.slots[1].terminal is Terminal.ACCEPTED
    assert first.slots[1].workpiece_id is not None
    assert list((tmp_path / "proposal-batches" / first.bundle_digest).glob("*.json"))
