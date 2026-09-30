# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer verifier wire contracts resolve and grade in a fresh process."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError
from tasktrove_verify.spec import MathType

from taskcompendium.grading import exact_answer, numeric_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.verifiers.mathematical import mathematical_answer
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


@pytest.mark.parametrize(
    "verifier,kind,old_kind,correct,wrong",
    [
        (exact_answer(("yes",)), "exact", "exact_answer", "YES", "no"),
        (numeric_answer(0.5, 0.0, 0.0), "numeric", "numeric_answer", "0.5", "2"),
        (multiple_choice_answer("B", 4), "mcq", "mcq_answer", "B", "C"),
        (mathematical_answer(r"\sqrt{2}", MathType.SCALAR), "math", "mathematical_answer", r"2/\sqrt{2}", "2"),
    ],
)
def test_serialized_answer_verifier_grades_in_fresh_process(tmp_path, verifier, kind, old_kind, correct, wrong):
    task = TaskSpec(
        id="serialized-answer",
        context=ConversationInput(events=(TextMessage(role="user", content="Solve the task."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=Source(dataset="test", revision="1", row=kind, importer_revision="1"),
    )
    payload = json.loads(task.model_dump_json())
    assert payload["verifier"]["kind"] == kind
    with pytest.raises(ValidationError):
        VerifierSpec.model_validate_json(json.dumps({**payload["verifier"], "kind": old_kind}))
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(payload))
    script = """
import asyncio
import sys
from pathlib import Path
from taskcompendium.models import TaskSpec, ConversationTrace, TextMessage
from taskcompendium.submission import GradingAttempt, PlainText
from taskcompendium.verifier_registry import grade_answer

specification = TaskSpec.model_validate_json(Path(sys.argv[1]).read_text())
async def run():
    for answer in sys.argv[2:]:
        trace = ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=answer)))
        result = await grade_answer(specification, PlainText(id="plain"), GradingAttempt(trace, object()))
        print(result.status, result.reward)
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), correct, wrong], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == ["graded 1.0", "graded 0.0"]
