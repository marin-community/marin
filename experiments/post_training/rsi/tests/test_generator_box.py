# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generator box finds the Reasoning Gym type and entry of each evaluation task that an idea cites."""

import pytest

pytest.importorskip("taskforge")

import asyncio
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

from experiments.post_training.rsi.generator_box import cited_types, examples_by_type, in_turns, typed_idea

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
    assert list(cited_types(cited, examples).items()) == [("word_sorting", 2), ("spell_backward", 1)]
    assert cited_types(idea(["calendar-0", "coding-0"]), examples) == {}
    # The program sees every evaluation task of the type, also the ones that the idea does not cite.
    typed = typed_idea(idea(["generated-0"]), examples["word_sorting"])
    assert typed.capability["example_entries"] == [GENERATED_ENTRY, GENERATED_ENTRY]
    assert typed.capability["example_tasks"] == ["Sort: b, A", "Sort: b, A"]


def test_a_later_batch_gets_turns_in_proportion_to_its_weight():
    async def run() -> list[str]:
        batches: asyncio.Queue[tuple[int, list[str]] | None] = asyncio.Queue()
        batches.put_nowait((1, ["a0", "a1", "a2", "a3", "a4", "a5"]))
        stream = in_turns(batches)
        taken = [await anext(stream)]
        # The gap idea's batch, with three times the cited failures, arrives while the first batch could still fill
        # the whole share.
        batches.put_nowait((3, ["b0", "b1", "b2", "b3", "b4", "b5"]))
        batches.put_nowait((5, []))
        batches.put_nowait(None)
        return taken + [item async for item in stream]

    items = asyncio.run(run())
    assert items[0] == "a0"
    # Three turns of the later batch for each turn of the first, then the rest of the first batch.
    assert [item[0] for item in items[1:9]].count("b") == 6
    assert items[9:] == ["a3", "a4", "a5"]
    assert sorted(items) == sorted([f"a{k}" for k in range(6)] + [f"b{k}" for k in range(6)])
