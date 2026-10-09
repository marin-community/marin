# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A proposal source over catalog capabilities that calls no model.

``RecordSource`` writes each slot's proposal from the capability's catalog record: a ShellSim task with
a simple (single-value) grade, its Task section the capability's outcome and sample tasks. It stands
in for a GLM-driven capability source.
"""

from taskforge.proposal.model import (
    BuildItem,
    Environment,
    Grounding,
    ProposalHeader,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
)
from taskforge.proposal.source import ProposalBatch, SlotProposal

from experiments.post_training.capability_driven_envs.catalog import CapabilityIdea

BODY = """\
## Task
{outcome}

Sample tasks from the catalog:
{samples}

## Realism and workflow
A kitchen scales a recipe from its stated yield to the portions an order needs, working from the recipe
file in the workspace.

## Research plan
None: every quantity is in the recipe file.

## Build plan
Write the recipe file with its yield and ingredient weights, and ask for one scaled weight.

## Grader design and controls
A numeric grader with zero tolerance on the scaled weight in grams. Controls: the scaled weight passes;
the unscaled weight and the per-portion weight fail; an empty reply fails.

## Risks and null conditions
None: the answer is a whole number of grams.
"""


def proposal_id(idea: CapabilityIdea, slot: int) -> str:
    return f"{idea.capability_id}/{slot}"


def source_ref(idea: CapabilityIdea) -> SourceRef:
    return SourceRef(kind=SourceKind.CAPABILITY, ref=idea.capability_id, hash=idea.capability_hash)


def record_proposal(idea: CapabilityIdea, slot: int) -> TaskProposal:
    """Slot ``slot``'s proposal for ``idea``, written from its catalog record."""
    samples = idea.capability.get("sample_tasks", [])
    assert isinstance(samples, list)
    return TaskProposal(
        header=ProposalHeader(
            id=proposal_id(idea, slot),
            source=source_ref(idea),
            environment=Environment.SHELLSIM,
            verification=Verification.SIMPLE,
            grounding=Grounding.UNVERIFIED,
            research=(),
            build=(BuildItem("recipe", "the recipe file the solver scales"),),
            resources=("workspace/recipe.txt",),
            null_reason=None,
        ),
        body=BODY.format(
            outcome=idea.capability["outcome"], samples="\n".join(f"- {sample['instruction']}" for sample in samples)
        ),
    )


class RecordSource:
    """A ``ProposalSource[CapabilityIdea]`` that writes every slot from the catalog record, without a model."""

    async def propose(self, idea: CapabilityIdea, n: int) -> ProposalBatch:
        slots = tuple(SlotProposal(slot, record_proposal(idea, slot), (), (), None) for slot in range(n))
        return ProposalBatch(planning_request=(), planning=(), slots=slots)
