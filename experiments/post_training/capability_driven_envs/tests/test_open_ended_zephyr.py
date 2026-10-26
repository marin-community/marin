# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import TaskSpec
from taskforge.content_hash import canonical_json
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, SolverMachine
from verifyit.spec import JudgeSpec

from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    document_template,
)
from experiments.post_training.capability_driven_envs.open_ended_zephyr import run_bundle_lifecycle
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_open_ended import bundle
from experiments.post_training.capability_driven_envs.tests.test_open_ended_pipeline import SERVICES
from experiments.post_training.capability_driven_envs.tests.test_open_ended_task_design import design

CONFIG = OpenEndedProposalConfig(Environment.SHELLSIM, SolverMachine.SHELLSIM)


async def test_complete_lifecycle_port_returns_an_accepted_taskspec(tmp_path):
    hydrated = bundle()
    authored = design()
    model = ScriptedModel()
    model.reply(
        json.dumps(
            {
                "coverage_rationale": "one supported question",
                "slots": [
                    {
                        "slot": 1,
                        "title": "Receptor support",
                        "scientific_question": "Does receptor usage support the supplied annotation?",
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
        "Use the supplied sufficient counts for a donor-aware analysis. Preserve both denominators, missing-call "
        "semantics, provenance, and cautious interpretation. The generic author derives a weighted rubric, private "
        "reference, and calibration controls from the curator input without exposing private evidence to the solver.",
    )
    model.reply(document)
    model.reply(canonical_json(authored))
    reward_by_answer = {
        control.answer: (control.minimum_reward + control.maximum_reward) / 2 for control in authored.controls
    }

    async def score(judge: JudgeSpec, answer: str) -> float:
        assert judge.model == "scripted"
        return reward_by_answer[answer]

    result = await run_bundle_lifecycle(
        hydrated,
        model=model,
        policy=LLMPolicy(),
        services=SERVICES,
        control_scorer=score,
        root=tmp_path,
    )

    assert result["bundle_digest"] == hydrated.content_digest
    assert result["release_status"] == "unreviewed_do_not_publish"
    slots = result["slots"]
    assert isinstance(slots, list) and len(slots) == 1
    assert slots[0]["terminal"] == "accepted"
    assert slots[0]["validation"]["calibration"]["passed"] is True
    assert slots[0]["design"]["reference"] == authored.reference
    task = TaskSpec.model_validate_json(slots[0]["task_json"])
    assert task.resources.worker == hydrated.solver_resources
    assert task.context.events[-1].content == authored.instruction
