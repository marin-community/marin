# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generator box finds the Reasoning Gym type and entry of each evaluation task that an idea cites."""

import pytest

pytest.importorskip("taskforge")

import base64
import json

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskforge.capability import CapabilityIdea

from experiments.post_training.rsi.generator_box import cited_types, examples_by_type, typed_idea

GENERATED_ENTRY = {"question": "Sort: b, A", "answer": "A, b", "metadata": {"source_dataset": "word_sorting"}}
NEMOTRON_TERMS = {
    "dataset": "ultra_sft_step3200_reasoning_gym",
    "question": "Spell 'cat' backward",
    "answer": "tac",
    "metadata": {"source_dataset": "spell_backward", "source_index": 3},
    "pass_rate": 0.5,
}


def task(task_id: str, config: dict | None) -> TaskSpec:
    verifier = []
    if config is not None:
        content = base64.b64encode(json.dumps(config).encode()).decode()
        verifier = [{"path": "config.json", "source": {"kind": "inline_file", "content_base64": content}}]
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=task_id),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="test"),
        source=Source(dataset="test", revision="1", row=task_id, importer_revision="1"),
    )
    return TaskSpec.model_validate({**spec.model_dump(), "resources": {"verifier": verifier}})


EVALUATION = [
    task("generated-0", {"contract": {"task": "word_sorting", "entry": GENERATED_ENTRY}}),
    task("generated-1", {"contract": {"task": "word_sorting", "entry": GENERATED_ENTRY}}),
    task("nemotron-0", {"contract": NEMOTRON_TERMS}),
    task("calendar-0", {"contract": {"dataset": "calendar_v2", "question": "When?", "answer": "9:00"}}),
    task("coding-0", None),
]


def idea(evidence: list[str]) -> CapabilityIdea:
    return CapabilityIdea(
        capability_id="strings.ordering",
        subject_id="failed-runs",
        subject_name="Failed runs",
        catalog_version="test",
        capability={"id": "strings.ordering", "evidence_tasks": evidence},
        prerequisite_edges=(),
        capability_hash="0",
    )


def test_both_reasoning_gym_row_forms_give_their_type_and_entry():
    examples = examples_by_type(EVALUATION)
    assert examples == {
        "word_sorting": {"generated-0": GENERATED_ENTRY, "generated-1": GENERATED_ENTRY},
        "spell_backward": {
            "nemotron-0": {"question": "Spell 'cat' backward", "answer": "tac", "metadata": NEMOTRON_TERMS["metadata"]}
        },
    }


def test_an_idea_gets_the_types_it_cites_most_cited_first_with_all_their_examples():
    examples = examples_by_type(EVALUATION)
    cited = idea(["nemotron-0", "generated-0", "generated-1", "calendar-0"])
    assert cited_types(cited, examples) == ["word_sorting", "spell_backward"]
    assert cited_types(idea(["calendar-0", "coding-0"]), examples) == []
    # The program sees every evaluation task of the type, also the ones that the idea does not cite.
    typed = typed_idea(idea(["generated-0"]), examples["word_sorting"])
    assert typed.capability["example_entries"] == [GENERATED_ENTRY, GENERATED_ENTRY]
    assert typed.capability["example_tasks"] == ["Sort: b, A", "Sort: b, A"]
