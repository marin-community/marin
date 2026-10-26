# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import TaskSpec
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, SolverMachine
from verifyit.spec import JudgeSpec

from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    document_template,
)
from experiments.post_training.capability_driven_envs.open_ended_task_design import load_task_designs
from experiments.post_training.capability_driven_envs.open_ended_zephyr import run_bundle_lifecycle
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_open_ended_pipeline import SERVICES
from experiments.post_training.capability_driven_envs.tests.test_open_ended_task_design import designed_bundle

CONFIG = OpenEndedProposalConfig(Environment.SHELLSIM, SolverMachine.SHELLSIM)


async def test_complete_lifecycle_port_returns_an_accepted_taskspec(tmp_path):
    hydrated = designed_bundle()
    designs = load_task_designs(hydrated, "private/task_designs.json")
    design = designs.designs[0]
    model = ScriptedModel()
    model.reply(
        json.dumps(
            {
                "coverage_rationale": "one manually supported question",
                "slots": [
                    {
                        "slot": 1,
                        "title": "MAIT receptor support",
                        "scientific_question": "Does TRAV1-2 usage support the supplied MAIT annotation?",
                        "analysis_obligations": ["compute a donor-aware comparison"],
                        "distinctive_challenge": "separate pooled and donor-level evidence",
                        "status": "propose",
                        "reason": None,
                    }
                ],
            }
        )
    )
    document = document_template(hydrated, 1, CONFIG).replace(
        "<...>",
        "Use the supplied sufficient counts for a donor-aware receptor analysis. Preserve both denominators, "
        "missing-call semantics, provenance, and cautious biological interpretation. The private author applies "
        "the frozen weighted rubric and reference, then validates positive, partial, severe-negative, verbosity, "
        "and evaluator-injection controls without exposing any private evidence to the solver.",
    )
    model.reply(document)
    reward_by_answer = {
        control.answer: (control.minimum_reward + control.maximum_reward) / 2 for control in design.controls
    }

    async def score(judge: JudgeSpec, answer: str) -> float:
        assert judge.model == "fixture-judge"
        return reward_by_answer[answer]

    result = await run_bundle_lifecycle(
        hydrated,
        model=model,
        policy=LLMPolicy(),
        services=SERVICES,
        judge_model="fixture-judge",
        control_scorer=score,
        root=tmp_path,
    )

    assert result["bundle_digest"] == hydrated.content_digest
    slots = result["slots"]
    assert isinstance(slots, list) and len(slots) == 1
    assert slots[0]["terminal"] == "accepted"
    assert slots[0]["validation"]["calibration"]["passed"] is True
    task = TaskSpec.model_validate_json(slots[0]["task_json"])
    assert task.resources.worker == hydrated.solver_resources
    assert task.context.events[-1].content == design.instruction
