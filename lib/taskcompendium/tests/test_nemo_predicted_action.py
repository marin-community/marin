# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NeMo next-action conversion and source-comparator regression coverage."""

import json
from pathlib import Path

import msgspec
import pytest

from taskcompendium.execution import Chat, HarborTaskBinding, NoEnvironment
from taskcompendium.grading import grade_attempt
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row, rendering
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    ExpectedFunctionCall,
    ExpectedFunctionCallBatch,
    ExpectedMessage,
    FinalActionSubmission,
    FunctionCall,
    Outcome,
    PredictedActionVerifier,
    Rejected,
    Rendering,
    ResourceRole,
    ToolCallComparatorConfig,
)
from taskcompendium.predicted_action import compare
from taskcompendium.rendering import render_task

ROW = {
    "responses_create_params": {
        "input": [
            {"type": "message", "role": "system", "content": "Use the available account tools when needed."},
            {"type": "message", "role": "user", "content": "Please look up order 17."},
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "integer"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "transfer_to_human",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ],
    },
    "expected_action": {"type": "function_call", "name": "lookup_order", "arguments": '{"order_id": 17}'},
}
SHA256 = canonical_sha256(ROW)
FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _transcript(*calls: tuple[str, str]) -> tuple[dict, ...]:
    return (
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": str(index), "type": "function", "function": {"name": name, "arguments": arguments}}
                for index, (name, arguments) in enumerate(calls)
            ],
        },
    )


def test_predicted_action_keeps_native_public_contract_and_private_expected_action(tmp_path):
    spec = import_row(ROW, SHA256)
    assert not isinstance(spec, Rejected)
    protocol = rendering(ROW, SHA256)
    assert isinstance(protocol.submission, FinalActionSubmission)
    public = msgspec.to_builtins(render_task(spec, (protocol,)))
    serialized = json.dumps(public)
    assert "lookup_order" in serialized
    assert '{\\"order_id\\": 17}' not in serialized
    assert "expected_action" not in serialized
    assert spec.requirements.capabilities == ()
    assert grade_attempt(spec, protocol, None, tmp_path, _transcript(("lookup_order", '{"order_id": 17}'))).reward == 1.0


def test_pinned_nemo_fixture_matches_catalog_provenance_and_source_scoring_without_network_access(tmp_path):
    row = json.loads((FIXTURES / "predicted-action.json").read_text())
    provenance = json.loads((FIXTURES / "predicted-action.provenance.json").read_text())
    assert canonical_sha256(row) == provenance["canonical_json_sha256"]
    specification = import_row(row, provenance["canonical_json_sha256"])
    assert not isinstance(specification, Rejected)
    assert specification.metadata.source.dataset == provenance["dataset"]
    private = {resource.path: resource for resource in specification.resources}
    assert set(private) == {"source-row.json", "source-provenance.json"}
    assert all(ResourceRole.VERIFIER in resource.roles for resource in private.values())
    assert json.loads(private["source-provenance.json"].content.data) == {
        "blob": provenance["github_blob_sha1"],
        "canonical_json_sha256": provenance["canonical_json_sha256"],
        "path": provenance["path"],
        "repository": provenance["repository"],
        "revision": provenance["repository_revision"],
    }
    protocol = rendering(row, provenance["canonical_json_sha256"])
    assert isinstance(protocol.submission, FinalActionSubmission)
    assert len(protocol.submission.functions) == 14
    expected = row["expected_action"]
    good = grade_attempt(
        specification,
        protocol,
        None,
        tmp_path,
        _transcript((expected["name"], expected["arguments"])),
    )
    wrong = grade_attempt(
        specification, protocol, None, tmp_path, _transcript(("get_event_details", '{"event_id":"x"}'))
    )
    assert (good.status, good.reward) == (Outcome.GRADED, 1.0)
    assert (wrong.status, wrong.reward) == (Outcome.GRADED, 0.0)


def test_predicted_action_preserves_source_wrong_and_extra_call_behavior(tmp_path):
    spec = import_row(ROW, SHA256)
    protocol = rendering(ROW, SHA256)
    wrong_name = grade_attempt(spec, protocol, None, tmp_path, _transcript(("transfer_to_human", "{}")))
    wrong_arguments = grade_attempt(spec, protocol, None, tmp_path, _transcript(("lookup_order", '{"order_id": 18}')))
    extra = grade_attempt(
        spec,
        protocol,
        None,
        tmp_path,
        _transcript(("lookup_order", '{"order_id": 17}'), ("transfer_to_human", "{}")),
    )
    empty = grade_attempt(spec, protocol, None, tmp_path, ())
    assert (wrong_name.status, wrong_name.reward, wrong_name.detail["category"]) == (
        Outcome.GRADED,
        0.0,
        "unexpected_tool",
    )
    assert (wrong_arguments.status, wrong_arguments.reward, wrong_arguments.detail["category"]) == (
        Outcome.GRADED,
        0.0,
        "argument_value_different",
    )
    assert (extra.status, extra.reward) == (Outcome.GRADED, 1.0)
    assert (empty.status, empty.reward, empty.detail["category"]) == (Outcome.GRADED, 0.0, "no_action_found")


def test_predicted_action_preserves_pinned_message_and_boolean_integer_limitations():
    config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)

    assert compare(ExpectedMessage("expected message"), ExpectedMessage("different message"), config) == (
        1.0,
        "expected_chat_message_found",
    )
    assert compare(
        ExpectedFunctionCall("lookup_order", '{"order_id": 1}'),
        ExpectedFunctionCall("lookup_order", '{"order_id": true}'),
        config,
    ) == (1.0, "expected_tool_call")


def test_predicted_action_batch_uses_explicit_cardinality_and_fractional_source_configuration(tmp_path):
    single = import_row(ROW, SHA256)
    assert not isinstance(single, Rejected)
    batch = ExpectedFunctionCallBatch(
        (FunctionCall("lookup_order", '{"order_id": 17}'), FunctionCall("transfer_to_human", "{}"))
    )
    verifier = PredictedActionVerifier(
        batch,
        ToolCallComparatorConfig(
            word_count_similarity_threshold=0.1,
            parallel_tool_call_rewarding=True,
            allow_subset=True,
            parallel_tool_call_reward_mode="fractional",
        ),
        single.metadata.source.revision,
    )
    spec = msgspec.structs.replace(single, steps=(msgspec.structs.replace(single.steps[0], verifier=verifier),))
    protocol = rendering(ROW, SHA256)
    unordered = grade_attempt(
        spec,
        protocol,
        None,
        tmp_path,
        _transcript(("transfer_to_human", "{}"), ("lookup_order", '{"order_id": 17}')),
    )
    subset = grade_attempt(spec, protocol, None, tmp_path, _transcript(("lookup_order", '{"order_id": 17}')))
    duplicate = grade_attempt(
        spec,
        protocol,
        None,
        tmp_path,
        _transcript(("lookup_order", '{"order_id": 17}'), ("lookup_order", '{"order_id": 17}')),
    )
    assert (unordered.status, unordered.reward) == (Outcome.GRADED, 1.0)
    assert (subset.status, subset.reward) == (Outcome.GRADED, 1.0)
    assert (duplicate.status, duplicate.reward) == (Outcome.GRADED, 0.5)


def test_private_expected_action_never_changes_the_public_contract():
    first = import_row(ROW, SHA256)
    assert not isinstance(first, Rejected)
    changed = dict(ROW)
    changed["expected_action"] = {"type": "function_call", "name": "transfer_to_human", "arguments": "{}"}
    second = import_row(changed, canonical_sha256(changed))
    assert not isinstance(second, Rejected)
    second = msgspec.structs.replace(second, metadata=first.metadata)
    first_public = msgspec.to_builtins(render_task(first, (rendering(ROW, SHA256),)))
    second_public = msgspec.to_builtins(render_task(second, (rendering(changed, canonical_sha256(changed)),)))
    first_public.pop("specification_sha256")
    second_public.pop("specification_sha256")
    assert first_public == second_public


def test_predicted_action_rejects_unsupported_source_history():
    unsupported = dict(ROW)
    request = dict(ROW["responses_create_params"])
    request["input"] = [
        *request["input"],
        {"type": "function_call", "name": "lookup_order", "arguments": "{}"},
    ]
    unsupported["responses_create_params"] = request

    result = import_row(unsupported, canonical_sha256(unsupported))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"


def test_predicted_action_rejects_mixed_step_output_contracts(tmp_path):
    specification = import_row(ROW, SHA256)
    assert not isinstance(specification, Rejected)
    specification = msgspec.structs.replace(specification, steps=(specification.steps[0], specification.steps[0]))

    with pytest.raises(ValueError, match="every step"):
        lower_to_harbor(
            specification,
            (rendering(ROW, SHA256), Rendering("answer", AssistantFinal())),
            HarborTaskBinding(NoEnvironment(), Chat()),
            tmp_path / "task",
        )
