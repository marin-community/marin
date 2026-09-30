# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove JSON-schema imports with direct-chat Harbor coverage."""

import json

from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import JsonSchemaSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.json_schema import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial
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
