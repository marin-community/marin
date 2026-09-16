# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for the pinned, stateful NeMo Workplace provider."""

import asyncio
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import msgspec
import pytest

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig, HarborTaskBinding
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.nemo_workplace import (
    FIXTURE_NAME,
    ProviderCall,
    build_sample,
    import_hub_row,
    provider_binding,
)
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import ActionInterface, AssistantFinal, Outcome, ProviderStateVerifier, Rejected, Rendering
from taskcompendium.providers.nemo_workplace.provider import NemoWorkplaceEnvironment
from taskcompendium.providers.nemo_workplace.tools import get_tools
from taskcompendium.rendering import render_instruction

FIXTURES = Path(__file__).parent / "fixtures/nemo"
REIMPORT_FIXTURES = FIXTURES / "reimport"


def test_workplace_rejects_binding_without_required_action_tools(tmp_path):
    sample = build_sample(FIXTURES)
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="action interfaces need explicit tool bindings"):
        lower_to_harbor(
            sample.specification, (sample.rendering,), HarborTaskBinding(sample.binding.environment), destination
        )
    assert not destination.exists()


def _environment() -> NemoWorkplaceEnvironment:
    """Construct the provider portion directly; Harbor covers its BaseEnvironment lifecycle below."""
    environment = object.__new__(NemoWorkplaceEnvironment)
    environment.tool_env = get_tools()
    environment.trace = []
    return environment


async def _dispatch(environment: NemoWorkplaceEnvironment, calls) -> list[str]:
    return [
        await environment.dispatch_action(call.name, call.arguments, f"call-{index}")
        for index, call in enumerate(calls, start=1)
    ]


async def test_workplace_provider_preserves_all_source_tools_and_stateful_outcomes():
    sample = build_sample(FIXTURES)
    good = _environment()
    assert len(await good.native_tool_definitions()) == 27
    assert await _dispatch(good, sample.known_good)
    parameters = sample.specification.steps[0].verifier.parameters
    assert (await good.grade_provider_state("nemo_workplace_v1", parameters)).reward == 1.0

    wrong = _environment()
    await _dispatch(wrong, sample.wrong_mutation)
    assert (await wrong.grade_provider_state("nemo_workplace_v1", parameters)).reward == 0.0

    noop = _environment()
    await _dispatch(noop, sample.noop)
    assert (await noop.grade_provider_state("nemo_workplace_v1", parameters)).reward == 0.0


@pytest.mark.parametrize("offset", (536, 1163))
async def test_hub_workplace_rows_preserve_source_identity_and_authoritative_state(offset):
    data = (REIMPORT_FIXTURES / f"workplace-{offset}.json").read_bytes()
    provenance = json.loads((REIMPORT_FIXTURES / f"workplace-{offset}.provenance.json").read_text())
    assert hashlib.sha256(data).hexdigest() == provenance["raw_sha256"]

    specification = import_hub_row(data, split=provenance["split"], offset=offset)

    assert not isinstance(specification, Rejected)
    assert specification.metadata.source.dataset == provenance["dataset"]
    assert specification.metadata.source.revision == provenance["revision"]
    assert specification.metadata.source.row == str(offset)
    verifier = specification.steps[0].verifier
    assert isinstance(verifier, ProviderStateVerifier)
    private = {resource.path: resource for resource in specification.resources}
    assert json.loads(private["source-provenance.json"].content.data) == {
        "dataset": provenance["dataset"],
        "revision": provenance["revision"],
        "split": provenance["split"],
        "offset": str(offset),
    }

    environment = _environment()
    source_tools = json.loads(data)["responses_create_params"]["tools"]
    assert {tool["name"] for tool in source_tools} == {
        tool["function"]["name"] for tool in await environment.native_tool_definitions()
    }
    calls = tuple(ProviderCall(action["name"], action["arguments"]) for action in verifier.parameters["ground_truth"])
    assert await _dispatch(environment, calls)
    assert (await environment.grade_provider_state(verifier.adapter, verifier.parameters)).reward == 1.0

    public = render_instruction(specification, Rendering("chat", AssistantFinal()))
    assert "ground_truth" not in public
    assert "source-provenance.json" not in public
    assert provider_binding().environment.interface == verifier.interface


def test_hub_workplace_import_rejects_a_tool_surface_the_shared_provider_cannot_supply():
    row = json.loads((REIMPORT_FIXTURES / "workplace-536.json").read_text())
    row["responses_create_params"]["tools"][0]["name"] = "unavailable_tool"

    result = import_hub_row(json.dumps(row).encode(), split="train", offset=536)

    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"


async def test_workplace_provider_returns_errors_and_recovers_with_correlated_trace():
    sample = build_sample(FIXTURES)
    environment = _environment()
    outputs = await _dispatch(environment, sample.recovery)

    assert "Error executing tool 'email_reply_email'" in outputs[0]
    assert [entry["call_id"] for entry in environment.trace] == ["call-1", "call-2"]
    assert [entry["output"] for entry in environment.trace] == outputs
    result = await environment.grade_provider_state(
        "nemo_workplace_v1", sample.specification.steps[0].verifier.parameters
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


async def test_workplace_sessions_are_fresh_and_concurrent():
    sample = build_sample(FIXTURES)
    first, second = _environment(), _environment()
    await asyncio.gather(_dispatch(first, sample.known_good), _dispatch(second, sample.noop))

    first_result, second_result = await asyncio.gather(
        first.grade_provider_state("nemo_workplace_v1", sample.specification.steps[0].verifier.parameters),
        second.grade_provider_state("nemo_workplace_v1", sample.specification.steps[0].verifier.parameters),
    )
    assert (first_result.reward, second_result.reward) == (1.0, 0.0)


def test_workplace_public_task_hides_private_seed_and_verifier_data():
    sample = build_sample(FIXTURES)
    public = render_instruction(sample.specification, sample.rendering)
    private = msgspec.json.encode(sample.specification).decode()

    assert "provider_state" not in public
    assert "ground_truth" not in public
    assert "source-row.json" not in public
    assert "ground_truth" in private
    assert FIXTURE_NAME not in public


def test_workplace_rejects_a_provider_verifier_for_another_seed(tmp_path):
    sample = build_sample(FIXTURES)
    step = sample.specification.steps[0]
    verifier = step.verifier
    assert isinstance(verifier, ProviderStateVerifier)
    incompatible = ActionInterface("workplace_assistant", "nemo-gym-v1", "f" * 64)
    altered = msgspec.structs.replace(
        sample.specification,
        steps=(
            msgspec.structs.replace(
                step,
                verifier=ProviderStateVerifier(incompatible, verifier.adapter, verifier.parameters),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Provider-state verifier"):
        lower_to_harbor(altered, (sample.rendering,), sample.binding, tmp_path / "invalid")


async def test_workplace_rejects_malformed_private_gold_as_invalid_task():
    environment = _environment()
    result = await environment.grade_provider_state(
        "nemo_workplace_v1", {"ground_truth": [{"name": "missing", "arguments": "{}"}]}
    )
    assert (result.status, result.reward) == (Outcome.INVALID_TASK, None)


def test_workplace_fixture_and_vendored_seed_match_pinned_raw_digests():
    provenance = json.loads((FIXTURES / "workplace-0.provenance.json").read_text())
    source = provenance["source_fixture"]
    assert hashlib.sha256((FIXTURES / source["fixture"]).read_bytes()).hexdigest() == source["fixture_raw_sha256"]
    root = Path(__file__).parents[1] / "src/taskcompendium/providers/nemo_workplace/vendor/csv_data"
    for relative, record in provenance["seed"]["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == record["raw_sha256"]
    tools = Path(__file__).parents[1] / "src/taskcompendium/providers/nemo_workplace/vendor/workplace_assistant_tools"
    for filename, record in provenance["server_contract"]["tools"].items():
        assert hashlib.sha256((tools / filename).read_bytes()).hexdigest() == record["raw_sha256"]


async def test_native_harbor_provider_trial_uses_scripted_http_and_preserves_call_ids(tmp_path, monkeypatch):
    sample = build_sample(FIXTURES)
    requests: list[dict] = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) == 1:
                call = sample.known_good[0]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-provider-1",
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Completed."}
            body = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        task = lower_to_harbor(
            sample.specification,
            (sample.rendering,),
            sample.binding,
            tmp_path / "task",
            reference_execution=HarborExecutionConfig(sample.binding, HarborLaunchConfig("provider_chat")),
            model_name="fixture",
            agent_kwargs={"api_base": f"http://127.0.0.1:{server.server_port}/v1", "max_turns": 3},
        )
        (task / "environment").rmdir()
        execution = json.loads((task / "reference-execution.json").read_text())
        result = await run_trial(task, execution, tmp_path / "trials", "provider")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 2
    assert len(requests[0]["tools"]) == 27
    assert "ground_truth" not in json.dumps(requests)
    assert "abcfd3d4727c66b6dfc145b59f720b819ac9de1b65df285cd30bc80bc10b3b8b" not in json.dumps(requests)
    transcript = json.loads((tmp_path / "trials/provider/agent/transcript.json").read_text())
    assert transcript[2]["tool_call_id"] == "call-provider-1"
