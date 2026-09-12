# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic contracts survive storage, rendering, and submission changes."""

import json

import msgspec
import pyarrow.parquet as pq
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.extraction import ExtractionError, extract
from taskcompendium.grading import grade_attempt
from taskcompendium.lowering import export_task, validate_lowering
from taskcompendium.models import (
    AnswerRequirements,
    AssistantFinal,
    BoxedLatex,
    Chat,
    ChatWithTools,
    Embedded,
    ExecutionConfig,
    FileSubmission,
    JsonPath,
    NoEnvironment,
    Outcome,
    PlainText,
    Protocol,
    PythonRuntime,
    Resource,
    ResourceRef,
    ResourceRole,
    ShellSimEnvironment,
    Source,
    TaskMetadata,
    TaskSpecification,
    VerifierSpec,
    XmlPath,
)
from taskcompendium.resources import materialize
from taskcompendium.serialization import from_json, read_parquet, specification_hash, to_json, write_parquet


@pytest.fixture
def math_task():
    return TaskSpecification(
        id="unit/math",
        instructions="Compute 1/2 + 1/4.",
        environment=NoEnvironment(),
        resources=(),
        verifier=VerifierSpec(Mode.MATH, {"expected": "3/4"}),
        verifier_runtime=PythonRuntime(),
        metadata=TaskMetadata(Source("unit", "v1", "0", "v1")),
    )


@pytest.mark.parametrize(
    "extractor,good,bad",
    [
        (PlainText(), "3/4", "4/3"),
        (BoxedLatex(), r"Reasoning. \boxed{\frac{3}{4}}", r"\boxed{\frac{4}{3}}"),
        (JsonPath(), '{"answer":"3/4"}', '{"answer":"4/3"}'),
        (XmlPath(), "<answer>3/4</answer>", "<answer>4/3</answer>"),
    ],
)
def test_equivalent_submissions_keep_same_semantic_grading(math_task, tmp_path, extractor, good, bad):
    protocol = Protocol("answer", Chat(), AssistantFinal(extractor))
    success = grade_attempt(math_task, protocol, good, tmp_path)
    failure = grade_attempt(math_task, protocol, bad, tmp_path)
    empty = grade_attempt(math_task, protocol, "", tmp_path)
    assert (success.status, success.reward) == (Outcome.GRADED, 1.0)
    assert (failure.status, failure.reward) == (Outcome.GRADED, 0.0)
    assert (empty.status, empty.reward) == (Outcome.EXTRACTION_ERROR, None)


def test_file_submission_reuses_math_verifier(math_task, tmp_path):
    protocol = Protocol("file", ChatWithTools(), FileSubmission("/app/answer.txt", JsonPath()))
    (tmp_path / "answer.txt").write_text('{"answer":"3/4"}')
    assert grade_attempt(math_task, protocol, None, tmp_path).reward == 1.0
    (tmp_path / "answer.txt").write_text('{"answer":"4/3"}')
    assert grade_attempt(math_task, protocol, None, tmp_path).reward == 0.0


@pytest.mark.parametrize(
    "extractor,text",
    [
        (JsonPath(), '{"answer":"A", "answer":"B"}'),
        (JsonPath(), '{"answer":null}'),
        (JsonPath(), '{"answer":NaN}'),
        (BoxedLatex(), r"\boxed{A} then \boxed{B}"),
        (BoxedLatex(), r"\boxed{\frac{1}{2}"),
        (XmlPath("/root/answer"), "<root><answer>A</answer><answer>B</answer></root>"),
        (XmlPath(), '<!DOCTYPE answer [<!ENTITY gold "A">]><answer>&gold;</answer>'),
    ],
)
def test_ambiguous_and_malformed_submissions_do_not_recover_answers(extractor, text):
    with pytest.raises(ExtractionError):
        extract(text, extractor)


def test_mixed_task_records_roundtrip_across_parquet_batches(math_task, tmp_path):
    reference = tmp_path / "input.txt"
    reference.write_text("public input")
    other = msgspec.structs.replace(
        math_task,
        id="unit/structured",
        environment=ShellSimEnvironment(),
        resources=(
            Resource("input.txt", (ResourceRole.AGENT,), Embedded(b"public input")),
            Resource("schema.json", (ResourceRole.VERIFIER,), Embedded(b'{"type":"object"}')),
        ),
        verifier=VerifierSpec(Mode.JSON_SCHEMA, {"schema": "schema.json"}),
        answer_requirements=AnswerRequirements("json"),
    )
    uri = str(tmp_path / "dataset.parquet")
    assert write_parquet([math_task, other], uri, batch_size=1) == 2
    restored = list(read_parquet(uri))
    assert restored == [math_task, other]
    assert [specification_hash(s) for s in restored] == [specification_hash(s) for s in (math_task, other)]
    assert from_json(to_json(other)) == other
    table = pq.read_table(uri, columns=["id", "environment"])
    assert [r["environment"]["kind"] for r in table.to_pylist()] == ["none", "shellsim"]


def test_format_is_semantic_and_cannot_be_replaced_by_wrapper(math_task):
    spec = msgspec.structs.replace(math_task, answer_requirements=AnswerRequirements("json"))
    with pytest.raises(ValueError, match="Intrinsic"):
        validate_lowering(
            spec, Protocol("xml", Chat(), AssistantFinal(XmlPath())), ExecutionConfig("chat", NoEnvironment())
        )
    with pytest.raises(ValueError, match="Chat supports"):
        validate_lowering(
            math_task, Protocol("file", Chat(), FileSubmission("/app/a")), ExecutionConfig("chat", NoEnvironment())
        )


def test_export_materializes_only_agent_projection(math_task, tmp_path):
    spec = msgspec.structs.replace(
        math_task,
        resources=(
            Resource("visible.txt", (ResourceRole.AGENT,), Embedded(b"public")),
            Resource("gold.txt", (ResourceRole.VERIFIER,), Embedded(b"private gold")),
            Resource("solve.sh", (ResourceRole.ORACLE,), Embedded(b"secret oracle")),
        ),
    )
    task = export_task(
        spec,
        Protocol("file", ChatWithTools(), FileSubmission("/app/answer.txt")),
        ExecutionConfig("replay", ShellSimEnvironment()),
        tmp_path / "export",
    )
    assert (task / "environment/inputs/visible.txt").read_bytes() == b"public"
    assert list((task / "environment/inputs").iterdir()) == [task / "environment/inputs/visible.txt"]
    assert "private gold" not in (task / "instruction.md").read_text()
    manifest = json.loads((task / "manifest.json").read_text())
    assert manifest["specification_sha256"] == specification_hash(spec)
    assert from_json((task / "specification.json").read_bytes()) == spec


def test_materialization_rejects_symlink_escape_and_modified_reference(math_task, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    spec = msgspec.structs.replace(
        math_task, resources=(Resource("escape/gold", (ResourceRole.VERIFIER,), Embedded(b"secret")),)
    )
    with pytest.raises(ValueError, match="escapes"):
        materialize(spec, ResourceRole.VERIFIER, root)
    assert not (outside / "gold").exists()
    original = tmp_path / "reference"
    original.write_text("changed")
    spec = msgspec.structs.replace(
        math_task, resources=(Resource("r", (ResourceRole.VERIFIER,), ResourceRef(str(original), "0" * 64)),)
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        materialize(spec, ResourceRole.VERIFIER, root)


@pytest.mark.parametrize("candidate", [r"wrong } \boxed{3/4}", r"\boxed{4/3} then \boxed{3/4}"])
def test_plain_math_does_not_reextract_an_embedded_answer(math_task, tmp_path, candidate):
    result = grade_attempt(math_task, Protocol("plain", Chat(), AssistantFinal()), candidate, tmp_path)
    assert result.status == Outcome.EXTRACTION_ERROR
    assert result.reward is None
