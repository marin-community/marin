# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean structured-output imports and Harbor coverage."""

import io
import json
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import CsvColumnsSpec, XmlElementsSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove import import_task as import_tasktrove
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.structured_outputs import CSV_MODE, XML_MODE
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial

SOURCE = "synthetic-structured-output"
RELEASE_URI = "fixture://tasktrove"
RELEASE_REVISION = "test-revision"


@dataclass(frozen=True)
class StructuredTaskFixture:
    path: str
    mode: str
    tags: tuple[str, ...]
    wrapper: str
    tail: str
    contract: str
    request: str
    kind: VerifierKind
    answers: tuple[str, str]
    contract_type: type[XmlElementsSpec] | type[CsvColumnsSpec]


TASKS = {
    "xml": StructuredTaskFixture(
        path="synthetic-xml.tar.gz",
        mode=XML_MODE,
        tags=("structured-outputs", XML_MODE, "synthetic", "xml"),
        wrapper=(
            """You will produce a structured response. Write your final answer to `/app/answer.txt`.
Emit a single well-formed XML document. The verifier checks the output structure.

---

"""
        ),
        tail=(
            """\n## Submitting your answer (IMPORTANT)
Write your XML document to `/app/answer.txt`.
An empty or missing `/app/answer.txt` scores 0.
"""
        ),
        contract='mode = "xml-elements"\nrequired = ["root"]\nany_of = []\n',
        request="Create one well-formed XML document that follows the provided JSON Schema.",
        kind=VerifierKind.XML_ELEMENTS,
        answers=("<root><item>value</item></root>", "<other />"),
        contract_type=XmlElementsSpec,
    ),
    "csv": StructuredTaskFixture(
        path="synthetic-csv.tar.gz",
        mode=CSV_MODE,
        tags=("structured-outputs", CSV_MODE, "synthetic", "csv"),
        wrapper=(
            """You will produce a structured response. Write your final answer to `/app/answer.txt`.
Emit CSV data conforming to the schema. The verifier checks the output structure.

---

"""
        ),
        tail=(
            """\n## Submitting your answer (IMPORTANT)
Write your CSV data to `/app/answer.txt`.
An empty or missing `/app/answer.txt` scores 0.
"""
        ),
        contract='mode = "csv-columns"\nrequired = ["id", "label"]\nany_of = []\n',
        request="Create CSV data following the provided schema",
        kind=VerifierKind.CSV_COLUMNS,
        answers=("id,label\n1,ready\n", "id,other\n1,ready\n"),
        contract_type=CsvColumnsSpec,
    ),
}


def _archive(mode: str):
    task = TASKS[mode]
    metadata = f"""[metadata]
tasktrove_source = "{SOURCE}"
tasktrove_path = "{task.path}"
family = "other"
converter = "nemotron_structured_outputs"
    mode = "{task.mode}"
tags = {json.dumps(task.tags)}
"""
    prompt = "".join(
        (
            task.wrapper,
            'JSON Schema: {"required": ["root"]}\n\nSynthetic prompt text.\n',
            task.tail,
        )
    )
    members = {
        "task.toml": metadata.encode(),
        "instruction.md": prompt.encode(),
        "tests/verifier.toml": task.contract.encode(),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return read_archive(buffer.getvalue(), SOURCE, task.path, RELEASE_URI, RELEASE_REVISION)


def test_structured_output_imports_keep_ordered_tags_and_generic_verifier_modes():
    for mode, task in TASKS.items():
        specification = import_tasktrove(_archive(mode))
        assert specification.source.dataset == RELEASE_URI
        assert specification.source.revision == RELEASE_REVISION
        assert specification.source.row == f"{SOURCE}:{task.path}"
        assert specification.tags == task.tags
        assert specification.verifier.kind is task.kind


@pytest.mark.parametrize("mode", ("xml", "csv"))
async def test_structured_output_import_matches_source_checker(mode: str, tmp_path: Path):
    task = TASKS[mode]
    archive = _archive(mode)
    specification = import_tasktrove(archive)
    source_contract = parse_spec(archive.files["tests/verifier.toml"].decode())
    assert isinstance(source_contract, task.contract_type)
    convention = PlainText(id="plain")

    for answer, expected_reward in zip(task.answers, (1.0, 0.0), strict=True):
        output = tmp_path / "answer.txt"
        output.write_text(answer)
        source_result = source_grade(replace(source_contract, output=str(output)), tmp_path, tmp_path)
        result = await grade_answer(
            specification,
            convention,
            GradingAttempt(
                ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=answer))),
                {},
            ),
        )
        assert source_result.reward == expected_reward
        assert (result.status, result.reward) == (Outcome.GRADED, expected_reward)


@pytest.mark.parametrize("mode", ("xml", "csv"))
@pytest.mark.parametrize("valid", (True, False), ids=("valid", "invalid"))
async def test_structured_output_import_runs_through_harbor(mode: str, valid: bool, tmp_path: Path):
    task = TASKS[mode]
    harbor_task = lower_to_harbor(
        import_tasktrove(_archive(mode)),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    answer = task.answers[0 if valid else 1]
    trial = await run_replay_trial(harbor_task, {"role": "assistant", "content": answer}, tmp_path / "trials", mode)

    outcome = json.loads((tmp_path / f"trials/{mode}/verifier/taskcompendium-result.json").read_text())
    assert trial.exception_info is None, trial.exception_info
    assert outcome == {"status": "graded", "reward": float(valid), "error": None}


def test_structured_output_prompt_hides_only_source_submission_scaffolding():
    for mode, task in TASKS.items():
        specification = import_tasktrove(_archive(mode))
        prompt = render_instruction(specification, PlainText(id="plain"))
        assert "/app/answer.txt" not in prompt
        assert "The verifier checks" not in prompt
        assert task.request.split(" that")[0] in prompt
        assert "Synthetic prompt text." in prompt
