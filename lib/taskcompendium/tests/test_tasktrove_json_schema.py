# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove JSON-schema imports with direct-chat Harbor coverage."""

import json

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import JsonSchemaSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.json_schema import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import AnswerCall, GradingAttempt, JsonAnswer, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial
from .submission_helpers import answer_attempt
from .tasktrove_json_schema_fixtures import json_schema_archive


def _attempt(specification, response: str):
    return GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=response))),
        object(),
    )


async def _assert_source_direct_and_harbor_parity(archive, valid: str, invalid: str, tmp_path):
    specification = import_task(archive)
    contract = parse_spec(archive.files["tests/verifier.toml"].decode())
    assert isinstance(contract, JsonSchemaSpec)
    schema_dir = tmp_path / "tests"
    schema_dir.mkdir()
    (schema_dir / "schema.json").write_bytes(archive.files["tests/schema.json"])
    for response, expected_reward in ((valid, 1.0), (invalid, 0.0)):
        (tmp_path / "answer.txt").write_text(response)
        source_result = source_grade(contract, schema_dir, tmp_path)
        direct_result = await grade_answer(specification, PlainText(id="plain"), _attempt(specification, response))
        assert source_result.reward == expected_reward
        assert (direct_result.status, direct_result.reward) == (Outcome.GRADED, expected_reward)

    task_dir = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "harbor-task",
    )
    for response, name, expected_reward in ((valid, "valid", 1.0), (invalid, "invalid", 0.0)):
        harbor_result = await run_replay_trial(
            task_dir,
            {"role": "assistant", "content": response},
            tmp_path / "trials",
            name,
        )
        result = json.loads((tmp_path / f"trials/{name}/verifier/taskcompendium-result.json").read_text())
        assert harbor_result.exception_info is None, harbor_result.exception_info
        assert result == {"status": "graded", "reward": expected_reward, "error": None}
    return specification


async def test_json_schema_yaml_import_matches_source_grader_and_harbor(tmp_path):
    archive = json_schema_archive(
        "yaml",
        "nemotron_structured_outputs",
        "other",
        "laion__nemotron-gym-structured-outputs-v4",
        "synthetic-structured-yaml",
    )
    valid = "answer: any response accepted by the schema\n"
    invalid = "not-an-object\n"
    specification = await _assert_source_direct_and_harbor_parity(archive, valid, invalid, tmp_path)
    assert specification.verifier.kind is VerifierKind.JSON_SCHEMA
    assert specification.tags == ("test", "structured-output", "yaml")
    instruction = render_instruction(specification, PlainText(id="plain"))
    assert "/app/answer.txt" not in instruction
    assert "JSON Schema" in instruction


async def test_json_schema_json_converter_import_matches_source_grader_and_harbor(tmp_path):
    archive = json_schema_archive(
        "json",
        "nemotron_if_structured",
        "instruction-following",
        "laion__nemotron-gym-instruction-following-structured-v3",
        "synthetic-structured-json",
    )
    valid = '{"answer":"any response accepted by the schema"}'
    invalid = "{}"
    specification = await _assert_source_direct_and_harbor_parity(archive, valid, invalid, tmp_path)
    assert specification.tags == ("test", "structured-output", "json")


@pytest.mark.parametrize("convention", [PlainText(id="plain"), JsonAnswer(id="json"), AnswerCall(id="call")])
async def test_json_schema_preserves_document_requirements_without_harness_scaffold(convention):
    archive = json_schema_archive(
        "json",
        "nemotron_if_structured",
        "instruction-following",
        "laion__nemotron-gym-instruction-following-structured-v3",
        "synthetic-structured-json",
    )
    archive.files["instruction.md"] = (
        "# Evaluation contract\n\n"
        "If the source text does not specify a value, choose any value satisfying the schema.\n\n"
        "You will produce a JSON document. Write your final JSON to `/app/answer.txt`. "
        "The verifier parses your answer (optionally unwrapping a ```json fence) and validates "
        "against the schema with `jsonschema` Draft 2020-12.\n\n"
        "Provide an unindented JSON object whose answer begins with yard.\n"
        "\n## Submitting your answer (IMPORTANT)\n"
        "You are a terminal agent. Your chat reply is NOT graded — the grader only reads the file "
        "`/app/answer.txt` inside the sandbox. You MUST write your JSON document to `/app/answer.txt` "
        "by RUNNING A SHELL COMMAND, e.g. a heredoc:\n\n"
        "    cat > /app/answer.txt <<'EOF'\n    <your JSON document here>\n    EOF\n\n"
        "Then confirm it with `cat /app/answer.txt`. An empty or missing `/app/answer.txt` scores 0 "
        "regardless of what you wrote in your reply.\n"
    ).encode()
    specification = import_task(archive)
    prompt = render_instruction(specification, convention)
    assert "unindented JSON object whose answer begins with yard" in prompt
    assert "choose any value satisfying the schema" in prompt
    assert "/app/" not in prompt
    assert "verifier" not in prompt.lower() and "graded" not in prompt.lower() and "grader" not in prompt.lower()
    valid = '{"answer":"yard example"}'
    result = await grade_answer(specification, convention, answer_attempt(specification, convention, valid))
    invalid = await grade_answer(specification, convention, answer_attempt(specification, convention, "{}"))
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert (invalid.status, invalid.reward) == (Outcome.GRADED, 0.0)


def test_json_schema_rejects_substantive_instructions_after_submission_heading():
    archive = json_schema_archive(
        "json",
        "nemotron_if_structured",
        "instruction-following",
        "laion__nemotron-gym-instruction-following-structured-v3",
        "synthetic-structured-json",
    )
    archive.files[
        "instruction.md"
    ] += b"\n## Submitting your answer (IMPORTANT)\nThe answer must contain two paragraphs.\n"
    with pytest.raises(ValueError, match="Unsupported JSON-schema submission footer"):
        import_task(archive)
