# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded NeMo Gym source-row importer contracts."""

import hashlib
import json
from pathlib import Path

import msgspec
import pytest
from tasktrove_verify.modes.grade_ifeval import resolve_checks
from tasktrove_verify.spec import Constraint, Mode

from taskcompendium.importers.nemo import (
    CODE_REVISION,
    IFEVAL_REVISION,
    import_code_answer,
    import_hub_code_row,
    import_hub_instruction_row,
    import_instruction_following,
    load_code_sample,
    load_instruction_sample,
)
from taskcompendium.models import (
    AssistantFinal,
    CodeAnswerVerifier,
    ConstraintVerifier,
    Rejected,
    Rendering,
    ResourceRole,
)
from taskcompendium.rendering import render_task

FIXTURES = Path(__file__).parent / "fixtures/nemo"
IMAGE = "python@sha256:" + "a" * 64


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _attempts() -> dict[str, dict[str, str]]:
    return json.loads(_fixture("attempts.json"))


def test_source_fixture_hashes_match_attested_provenance():
    provenance = json.loads(_fixture("provenance.json"))
    for source in ("instruction_following", "code_answer"):
        record = provenance[source]
        data = _fixture(record["fixture"])
        assert hashlib.sha256(data).hexdigest() == record["raw_sha256"]
        assert json.loads(data)[record["selector"]["field"]] == record["selector"]["value"]


def test_instruction_import_keeps_constraints_and_raw_source_private():
    specification = load_instruction_sample(FIXTURES / "instruction-following-17616.json")

    assert not isinstance(specification, Rejected)
    step = specification.steps[0]
    assert specification.id == "nemo/ifeval/17616/binary"
    assert specification.metadata.source.revision == IFEVAL_REVISION
    assert isinstance(step.verifier, ConstraintVerifier)
    assert [constraint.name for constraint in step.verifier.constraints] == [
        "length_constraints:nth_paragraph_first_word",
        "last_word:last_word_answer",
    ]
    assert {resource.path for resource in specification.resources if ResourceRole.VERIFIER in resource.roles} == {
        "source-row.json"
    }

    task = render_task(specification, (Rendering("answer", AssistantFinal()),))
    public = msgspec.json.encode(task).decode()
    assert "instruction_constraints" not in public
    assert "source-row.json" not in public
    assert "last_word:last_word_answer" not in public


def test_instruction_sample_preserves_binary_and_fractional_semantics():
    binary = import_instruction_following(_fixture("instruction-following-17616.json"), aggregation="binary")
    fractional = load_instruction_sample(FIXTURES / "instruction-following-17616.json", aggregation="fraction")

    assert not isinstance(binary, Rejected)
    assert not isinstance(fractional, Rejected)
    assert binary.id == "nemo/ifeval/17616/binary"
    assert fractional.id == "nemo/ifeval/17616/fraction"
    assert binary.steps[0].verifier != fractional.steps[0].verifier
    assert isinstance(binary.steps[0].verifier, ConstraintVerifier)
    assert isinstance(fractional.steps[0].verifier, ConstraintVerifier)

    checks = resolve_checks(
        tuple(Constraint(constraint.name, constraint.params) for constraint in binary.steps[0].verifier.constraints)
    )
    attempts = _attempts()["instruction_following"]
    assert [check(attempts["good"], constraint.params)[0] for constraint, check in checks] == [True, True]
    assert [check(attempts["one_constraint_wrong"], constraint.params)[0] for constraint, check in checks] == [
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("offset", "identifier", "constraints_count"),
    ((0, 17616, 2), (1, 44654, 3)),
)
def test_collection_instruction_rows_preserve_hub_source_identity(offset, identifier, constraints_count):
    data = _fixture(f"collection/instruction-following-{offset}.json")
    provenance = json.loads(_fixture(f"collection/instruction-following-{offset}.provenance.json"))

    specification = import_hub_instruction_row(data, split=provenance["split"], offset=provenance["offset"])

    assert not isinstance(specification, Rejected)
    assert specification.id == f"nemo/ifeval/{identifier}/binary"
    assert specification.metadata.source.dataset == provenance["dataset"]
    assert specification.metadata.source.revision == provenance["revision"]
    assert specification.metadata.source.row == str(provenance["offset"])
    assert isinstance(specification.steps[0].verifier, ConstraintVerifier)
    assert len(specification.steps[0].verifier.constraints) == constraints_count
    private = {
        resource.path: resource for resource in specification.resources if ResourceRole.VERIFIER in resource.roles
    }
    assert json.loads(private["source-provenance.json"].content.data) == {
        "dataset": provenance["dataset"],
        "revision": provenance["revision"],
        "split": provenance["split"],
        "offset": str(provenance["offset"]),
    }

    task = render_task(specification, (Rendering("answer", AssistantFinal()),))
    public = msgspec.json.encode(task).decode()
    assert "instruction_id_list" not in public
    assert "source-row.json" not in public


def test_code_import_is_answer_only_and_retains_hidden_test_bundle():
    specification = load_code_sample(
        FIXTURES / "code-answer-c69268d8bdb4da0685d7b187c88296c1.json", verifier_image=IMAGE
    )

    assert not isinstance(specification, Rejected)
    step = specification.steps[0]
    assert specification.requirements.capabilities == ()
    assert specification.metadata.source.revision == CODE_REVISION
    assert isinstance(step.verifier, CodeAnswerVerifier)
    assert step.verifier.verifier.mode is Mode.SCRIPT
    assert step.verifier.output_path == "answer.txt"
    private = {
        resource.path: resource for resource in specification.resources if ResourceRole.VERIFIER in resource.roles
    }
    assert private["source-row.json"].content.data == _fixture("code-answer-c69268d8bdb4da0685d7b187c88296c1.json")
    assert "packed" not in private["source-row.json"].content.data.decode()
    assert "base64.b64decode" in private["check_code_answer.py"].content.data.decode()
    assert "code_from_response" not in private["check_code_answer.py"].content.data.decode()
    assert "without Markdown fences" not in step.instructions

    task = render_task(specification, (Rendering("answer", AssistantFinal()),))
    public = msgspec.json.encode(task).decode()
    assert "unit_tests" not in public
    assert "check_code_answer.py" not in public
    assert "source-row.json" not in public


def test_code_import_defers_packed_bundle_decoding_to_the_private_checker():
    row = json.loads(_fixture("code-answer-c69268d8bdb4da0685d7b187c88296c1.json"))
    row["verifier_metadata"]["unit_tests"] = {"packed": "not-yet-decoded"}

    specification = import_code_answer(json.dumps(row, separators=(",", ":")).encode(), verifier_image=IMAGE)

    assert not isinstance(specification, Rejected)
    raw = next(resource for resource in specification.resources if resource.path == "source-row.json")
    assert b"not-yet-decoded" in raw.content.data


def test_code_import_rejects_function_call_cases_and_mutable_runtime():
    row = json.loads(_fixture("code-answer-c69268d8bdb4da0685d7b187c88296c1.json"))
    row["verifier_metadata"]["unit_tests"]["fn_name"] = "solve"
    result = import_code_answer(json.dumps(row).encode(), verifier_image=IMAGE)
    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"

    result = import_code_answer(_fixture("code-answer-c69268d8bdb4da0685d7b187c88296c1.json"), verifier_image="python:3")
    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"


def test_code_import_requires_an_isolated_runtime():
    result = import_code_answer(_fixture("code-answer-c69268d8bdb4da0685d7b187c88296c1.json"))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"


@pytest.mark.parametrize("offset", (5135, 13176))
def test_collection_code_rows_preserve_hub_offsets_and_private_test_data(offset):
    data = _fixture(f"collection/coding-{offset}.json")
    specification = import_hub_code_row(data, split="train", offset=offset, verifier_image=IMAGE)

    assert not isinstance(specification, Rejected)
    assert specification.metadata.source.row == str(offset)
    private = {
        resource.path: resource for resource in specification.resources if ResourceRole.VERIFIER in resource.roles
    }
    private_paths = set(private)
    assert private_paths == {"check_code_answer.py", "source-provenance.json", "source-row.json"}
    provenance = json.loads(private["source-provenance.json"].content.data)
    assert provenance == {
        "dataset": "nvidia/Nemotron-RL-coding-competitive_coding",
        "revision": CODE_REVISION,
        "split": "train",
        "offset": str(offset),
    }

    task = render_task(specification, (Rendering("answer", AssistantFinal()),))
    public = msgspec.json.encode(task).decode()
    assert "unit_tests" not in public
    assert "source-provenance.json" not in public
