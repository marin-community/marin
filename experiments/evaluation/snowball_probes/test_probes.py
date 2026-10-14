# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import gzip
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import HfApi, ModelInfo
from iris.client.client import IrisClient
from iris.rpc import controller_pb2
from jsonschema import ValidationError, validate

from experiments.evaluation.snowball_probes.client import WorkerApi
from experiments.evaluation.snowball_probes.config import load_candidates
from experiments.evaluation.snowball_probes.protocol import (
    infrastructure_review,
    public_history,
    scenario_schema,
    validated_review,
)
from experiments.evaluation.snowball_probes.serve import model_config, verify_weight_snapshot
from experiments.evaluation.snowball_probes.store import Store, TokenCapReached
from experiments.evaluation.snowball_probes.verification import JsonSchemaCheck, ShellCheck, run_checks
from experiments.evaluation.snowball_probes.worker import (
    INFERENCE_TIMEOUT,
    SHELLSIM_CPU_LIMIT,
    Endpoints,
    RoundEnd,
    Worker,
)


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state.sqlite", 100_000)
    yield value
    value.close()


@pytest.mark.parametrize("changed_file", ["chat_template.jinja", "config.json", "model.safetensors"])
def test_cached_weights_require_matching_runtime_artifacts(monkeypatch, changed_file):
    names = ["chat_template.jinja", "config.json", "model.safetensors"]
    old = ModelInfo(id="test/model", siblings=[{"rfilename": name, "size": 10, "blobId": "old"} for name in names])
    new = ModelInfo(
        id="test/model",
        siblings=[
            {"rfilename": name, "size": 10, "blobId": "changed" if name == changed_file else "old"} for name in names
        ],
    )
    api = Mock(spec=HfApi)
    api.model_info.side_effect = [old, new]
    monkeypatch.setattr("experiments.evaluation.snowball_probes.serve.HfApi", lambda: api)
    candidate = replace(load_candidates(Path(__file__).with_name("merge_patched_grug.json"))[0], weights_revision="old")
    if changed_file != "chat_template.jinja":
        with pytest.raises(ValueError, match="runtime artifacts differ"):
            verify_weight_snapshot(candidate)
        return
    verify_weight_snapshot(candidate)
    serving = model_config(candidate)
    assert serving.revision == "old"
    assert serving.tokenizer_revision == candidate.revision


def test_chat_scenario_constrained_generation_rejects_workspace_checks():
    task = {
        "title": "Exact answer",
        "category": "formatting",
        "user_message": "Reply with exactly ready.",
        "files": {},
        "objective": "Return the requested text",
        "expected_checks": ["Exact output"],
        "checks": [{"kind": "exact_text", "name": "Requested text", "expected": "ready"}],
    }
    validate(task, scenario_schema("chat"))
    task["checks"] = [{"kind": "shell", "name": "Read artifact", "command": "cat result.txt"}]
    with pytest.raises(ValidationError):
        validate(task, scenario_schema("chat"))
    validate(task, scenario_schema("agentic"))


def test_unparsed_native_tool_call_excludes_both_scores_and_preserves_original_judgment():
    review = {"assessments": [{"model": "snowball-dr-doom", "score": 0}, {"model": "reference", "score": 1}]}
    trace = {
        "message_id": "unparsed-call",
        "model": "snowball-dr-doom",
        "payload": {
            "request": {"tools": [{"type": "function"}], "tool_choice": "auto"},
            "content": '<tool_call>\n{"name":"shell","arguments":{"command":"cat NOTES.md"}}\n</tool_call>',
            "tool_calls": [],
            "finish_reason": "stop",
        },
    }
    corrected = infrastructure_review(review, [trace])
    assert [item["score"] for item in corrected["assessments"]] == [None, None]
    assert corrected["original_review"] == review
    assert corrected["breakages"][0]["category"] == "infrastructure"
    assert corrected["breakages"][0]["evidence"][0]["message_id"] == "unparsed-call"
    assert infrastructure_review(corrected, [trace]) == corrected

    # Discussing a literal XML example is an answer, rather than an intended tool invocation.
    trace["payload"]["content"] = "For example, use " + trace["payload"]["content"]
    assert infrastructure_review(review, [trace]) == review


def test_reservations_bound_concurrent_generation_and_survive_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    ledger = Store(path, 100)
    ledger.reserve("snowball", 60)
    ledger.reserve("qwen", 40)
    with pytest.raises(TokenCapReached):
        ledger.reserve("glm", 1)
    ledger.settle("snowball", 12)
    ledger.reserve("glm", 30)
    ledger.close()

    recovered = Store(path, 100)
    recovered.recover()
    assert recovered.totals() == (82, 0)
    recovered.recover()
    assert recovered.totals() == (82, 0)
    recovered.reserve("next", 18)
    with pytest.raises(TokenCapReached):
        recovered.reserve("over-cap", 1)
    recovered.close()


def test_upload_acknowledgment_does_not_drop_newer_stream_snapshot(store):
    store.put("/traces", "message", {"content": "first"})
    path, key, _, generation = store.pending()[0]
    store.put("/traces", "message", {"content": "first second"})
    store.acknowledge(path, key, generation)
    assert json.loads(store.pending()[0][2]) == {"content": "first second"}


def test_preempted_model_blocks_admission_until_its_endpoint_returns(monkeypatch):
    clock = [10_000.0]
    monkeypatch.setattr("experiments.evaluation.snowball_probes.worker.time.monotonic", lambda: clock[0])
    client = Mock(spec=IrisClient)
    minted = []

    def mint(name, **kwargs):
        minted.append(name)
        return controller_pb2.Controller.MintEndpointTokenResponse(capability_url="/scoped" + name)

    client.mint_endpoint_token.side_effect = mint
    client.list_endpoint_instances.return_value = []
    endpoints = Endpoints(client, "http://iris", "/owned/serve", "/owned/gateway/ortet")

    async def run():
        candidate = {"key": "small", "state": "ready", "endpoint": "/serve/small", "revision": "pinned"}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"candidates": [candidate]}))
        ) as http:
            assert not await endpoints.refresh(http)
            assert endpoints.metadata == [{"key": "small", "revision": "pinned"}]
            assert minted == ["/owned/serve/probes"]
            client.list_endpoint_instances.return_value = [controller_pb2.Controller.Endpoint()]
            clock[0] += 16
            assert await endpoints.refresh(http)
            assert set(endpoints.urls) == {"small", "glm"}
            assert minted == ["/owned/serve/probes", "/owned/gateway/ortet", "/serve/small"]

    asyncio.run(run())


def test_transient_readiness_read_error_waits_and_recovers_without_failing_worker():
    client = Mock(spec=IrisClient)
    client.mint_endpoint_token.return_value = controller_pb2.Controller.MintEndpointTokenResponse(
        capability_url="/scoped"
    )
    client.list_endpoint_instances.return_value = [controller_pb2.Controller.Endpoint()]
    endpoints = Endpoints(client, "http://localhost", "/owned/serve", "/owned/gateway/ortet")
    endpoints.status_url = "http://localhost/status"
    endpoints.status_refreshed = float("inf")

    async def run():
        def interrupted(request):
            raise httpx.ReadError("Connection lost", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(interrupted)) as http:
            assert not await endpoints.refresh(http)
        assert endpoints.readiness_error == "Coordinator status transport failed: ReadError"
        endpoints.checked = 0
        candidate = {"key": "small", "endpoint": "/serve/small", "state": "ready", "revision": "pinned"}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"candidates": [candidate]}))
        ) as http:
            assert await endpoints.refresh(http)
        assert endpoints.readiness_error is None

    asyncio.run(run())


def make_worker(tmp_path: Path, store: Store) -> Worker:
    args = argparse.Namespace(
        run_id="test",
        state_dir=tmp_path,
        token_cap=100_000,
        concurrency=2,
        shellsim_cpu_limit=SHELLSIM_CPU_LIMIT,
        inference_timeout=INFERENCE_TIMEOUT,
        worker_job=None,
        applet="test",
        marina_url="http://localhost",
        coordinator="/test/serve",
    )
    endpoints = Endpoints(Mock(spec=IrisClient), "http://localhost", "/test/serve", "/test/gateway/ortet")
    endpoints.urls = {"small": "http://small/v1/chat/completions", "reference": "http://reference/v1/chat/completions"}
    endpoints.metadata = [
        {"key": "small", "reasoning_parser": "grug_optional", "reasoning_format": "api_fields"},
        {"key": "reference", "reasoning_parser": "qwen3", "reasoning_format": "api_fields"},
    ]
    api = Mock(spec=WorkerApi)
    api.claim.return_value = {"idea": None}
    return Worker(args, store, endpoints, [{"text": "inspiration", "row": 1}], "test-key", api)


def test_native_grug_call_executes_while_open_thinking_is_preserved_only_in_model_history(tmp_path, store):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.endpoints.metadata[0].update(reasoning_parser=None, reasoning_format="grug_inline")
        await worker.http.aclose()
        command = "python3 -c 'print(6+7+8)'"
        second_command = "python3 -c 'print(1+3)'"
        native = "<|start_think|>PRIVATE: use Python to verify."
        count = 0

        def respond(request):
            nonlocal count
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10, "max_model_len": 65536})
            body = json.loads(request.content)
            assert body["skip_special_tokens"] is False
            if count == 0:
                delta = {
                    "content": native,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "shell-call",
                            "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                        },
                        {
                            "index": 1,
                            "id": "second-call",
                            "function": {"name": "shell", "arguments": json.dumps({"command": second_command})},
                        },
                    ],
                }
                finish = "tool_calls"
            else:
                assistant = next(message for message in body["messages"] if message["role"] == "assistant")
                assert assistant["content"] == native
                assert assistant["tool_calls"][0]["function"]["name"] == "shell"
                assert json.loads(body["messages"][-1]["content"])["stdout"] == "4\n"
                delta = {"content": "<|start_think|>Verified.<|end_think|>21"}
                finish = "stop"
            count += 1
            chunks = [
                {"choices": [{"delta": delta, "finish_reason": finish}]},
                {"choices": [], "usage": {"completion_tokens": 20}},
            ]
            return httpx.Response(
                200, content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            )

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        history = [{"role": "user", "content": "Calculate 6+7+8 using shell."}]
        commands = []
        try:
            outcome = await worker.candidate_round(
                {"case_id": "native", "mode": "agentic"}, "small", history, {}, commands, 0
            )
            assert outcome == RoundEnd.ANSWER
            assert commands == [command, second_command]
            traces = store.case_traces("native")
            opening = next(trace["payload"] for trace in traces if trace["role"] == "assistant")
            assert opening["content"] == ""
            assert opening["reasoning"] == "PRIVATE: use Python to verify."
            assert opening["native_content"] == native
            tools = [trace for trace in traces if trace["role"] == "tool"]
            assert [trace["payload"]["tool_call_id"] for trace in tools] == ["shell-call", "second-call"]
            assert tools[0]["sequence"] < tools[1]["sequence"]
            public = public_history(history, True)
            assert "PRIVATE" not in json.dumps(public)
            assert public[-1]["content"] == "21"
            assert public[1]["tool_calls"][0]["function"]["name"] == "shell"
        finally:
            await worker.http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("reasoning_format", ["grug_inline", "api_fields"])
def test_native_grug_chat_preserves_thinking_markers_without_tools(tmp_path, store, reasoning_format):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.endpoints.metadata[0].update(
            reasoning_parser="grug" if reasoning_format == "api_fields" else None, reasoning_format=reasoning_format
        )
        await worker.http.aclose()
        native = '<|start_think|>PRIVATE: sum is 21.<|end_think|>{"answer":21}'

        def respond(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10, "max_model_len": 65536})
            body = json.loads(request.content)
            assert "tools" not in body
            if reasoning_format == "api_fields":
                delta = {"reasoning_content": "PRIVATE: sum is 21.", "content": '{"answer":21}'}
            else:
                delta = {"content": native}
            if reasoning_format == "grug_inline" and body.get("skip_special_tokens", True):
                delta["content"] = native.replace("<|start_think|>", "").replace("<|end_think|>", "")
            chunk = {
                "choices": [{"delta": delta, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 20},
            }
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        history = [{"role": "user", "content": "Reply only with JSON containing answer 21."}]
        try:
            await worker.candidate_round({"case_id": "native-chat", "mode": "chat"}, "small", history, {}, [], 0)
            answer = next(t["payload"] for t in store.case_traces("native-chat") if t["role"] == "assistant")
            assert answer["reasoning"] == "PRIVATE: sum is 21."
            assert json.loads(answer["content"]) == {"answer": 21}
            if reasoning_format == "api_fields":
                assert history[-1]["reasoning_content"] == "PRIVATE: sum is 21."
            else:
                assert history[-1]["content"] == native
            public = public_history(history, reasoning_format == "grug_inline")
            assert public[-1]["content"] == '{"answer":21}'
            assert "PRIVATE" not in json.dumps(public)
        finally:
            await worker.http.aclose()

    asyncio.run(run())


def test_predeclared_schema_rejects_extra_keys_bad_enums_and_wrapped_json():
    check = JsonSchemaCheck(
        kind="json_schema",
        name="Exact closed object",
        schema_text=json.dumps(
            {
                "type": "object",
                "properties": {"state": {"enum": ["ready", "failed"]}},
                "required": ["state"],
                "additionalProperties": False,
            }
        ),
    )
    for answer, expected in [
        (' {"state":"ready"} ', 1),
        ('{"state":"ready","note":"extra"}', 0),
        ('{"state":"READY"}', 0),
        ('```json\n{"state":"ready"}\n```', 0),
    ]:
        result = run_checks([check], answer, {}, [], SHELLSIM_CPU_LIMIT, None)
        assert result["score"] == expected
    assert run_checks([check], '{"state":"ready"}', {}, [], SHELLSIM_CPU_LIMIT, "token_limit")["score"] is None


def test_artifact_checks_distinguish_missing_output_from_unsupported_verifier():
    check = ShellCheck(
        kind="shell",
        name="Saved result",
        command='python3 -c \'import json; assert json.load(open("result.json")) == {"answer":21}\'',
    )
    assert run_checks([check], "Done", {}, [], SHELLSIM_CPU_LIMIT, None)["score"] == 0
    assert run_checks([check], "Done", {"result.json": '{"answer":21}'}, [], SHELLSIM_CPU_LIMIT, None)["score"] == 1
    unsupported = ShellCheck(kind="shell", name="Unavailable verifier", command="python3 -c 'import threading'")
    result = run_checks([unsupported], "Done", {}, [], SHELLSIM_CPU_LIMIT, None)
    assert result["score"] is None
    assert result["checks"][0]["status"] == "inconclusive"


def test_simulator_failure_excludes_paired_scores_and_keeps_judgment():
    review = {"assessments": [{"model": "antidoom", "score": 0}, {"model": "qwen", "score": 1}]}
    trace = {
        "message_id": "unsupported",
        "model": "antidoom",
        "role": "tool",
        "payload": {"tool_result": {"stderr": "python: unsupported operation", "unsupported": ["threading"]}},
    }
    result = infrastructure_review(review, [trace])
    assert result["original_review"] == review
    assert [item["score"] for item in result["assessments"]] == [None, None]
    assert result["kind"] == "infrastructure"


def test_failed_connection_records_transport_cause_without_charging_generated_tokens(tmp_path, store):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.endpoints.urls["glm"] = "http://driver/v1/chat/completions?token=private"
        await worker.http.aclose()

        def failed(request):
            raise httpx.ConnectError("private URL", request=request) from OSError(111, "Connection refused")

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(failed))
        case = {"case_id": "connection-failure"}
        with pytest.raises(RuntimeError, match="ConnectError"):
            await worker.completion(case, "glm", [{"role": "user", "content": "Test"}], 0, 0, "scenario", 100)
        assert store.totals() == (0, 0)
        trace = store.case_traces(case["case_id"])[0]
        assert trace["state"] == "error"
        assert trace["payload"]["error"] == "ConnectError (ConnectionRefusedError, errno=111)"
        assert "private" not in json.dumps(trace)
        await worker.http.aclose()

    asyncio.run(run())


def test_http_rejection_keeps_status_without_persisting_echoed_credentials(tmp_path, store):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.endpoints.urls["glm"] = "http://driver/v1/chat/completions?token=private-capability"
        await worker.http.aclose()
        worker.http = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    403, text=f"Rejected {request.url}; x-ortet-api-token={request.headers['x-ortet-api-token']}"
                )
            )
        )
        try:
            with pytest.raises(RuntimeError) as failure:
                await worker.completion(
                    {"case_id": "http-rejection"}, "glm", [{"role": "user", "content": "Test"}], 0, 0, "scenario", 100
                )
            assert "403" in str(failure.value)
            saved = json.dumps(store.case_traces("http-rejection"))
            for credential in ("private-capability", "test-key"):
                assert credential not in str(failure.value)
                assert credential not in saved
            assert store.case_traces("http-rejection")[0]["state"] == "error"
            assert store.totals() == (0, 0)
        finally:
            await worker.http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["headers", "stream"])
def test_hung_inference_releases_reservation_and_preserves_partial_trace(tmp_path, store, stage):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.args.inference_timeout = 0.02
        worker.endpoints.urls["glm"] = "http://driver/v1/chat/completions"
        await worker.http.aclose()

        class KeepaliveStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                chunk = {"choices": [{"delta": {"reasoning_content": "Partial thinking"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                while True:
                    await asyncio.sleep(0)
                    yield b": keepalive\n\n"

        async def stalled(request):
            if stage == "headers":
                await asyncio.Event().wait()
            return httpx.Response(200, stream=KeepaliveStream())

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(stalled))
        try:
            with pytest.raises(RuntimeError, match="Inference request exceeded"):
                await asyncio.wait_for(
                    worker.completion({"case_id": "hung"}, "glm", [], 0, 0, "scenario", 100), timeout=1
                )
            trace = store.case_traces("hung")[0]
            assert trace["state"] == "error"
            assert trace["payload"]["kind"] == "inference_timeout"
            assert trace["payload"]["usage_is_conservative"]
            if stage == "stream":
                assert trace["payload"]["reasoning"] == "Partial thinking"
            assert store.totals() == (100, 0)
        finally:
            await worker.http.aclose()

        chunk = {
            "choices": [{"delta": {"content": "Recovered"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 12},
        }
        worker.http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\n"))
        )
        try:
            result = await worker.completion({"case_id": "next"}, "glm", [], 0, 0, "scenario", 100)
            assert result["content"] == "Recovered"
            assert store.totals() == (112, 0)
        finally:
            await worker.http.aclose()

    asyncio.run(run())


def test_cluster_retry_restores_accounting_and_marks_unfinished_chats_interrupted(tmp_path, store, monkeypatch):
    worker = make_worker(tmp_path, store)
    case = {"case_id": "abandoned", "run_id": "test", "state": "generating"}
    trace = {
        "message_id": "partial",
        "case_id": "abandoned",
        "model": "small",
        "role": "assistant",
        "round": 0,
        "sequence": 0,
        "state": "streaming",
        "payload": {"reasoning": "Saved thinking"},
    }

    def saved_cases(run_id, state, limit):
        return {"total": 0, "cases": []} if state == "reviewing" else {"total": 1, "cases": [case]}

    def saved_case(case_id, trace_state):
        if trace_state != "streaming":
            raise RuntimeError("Full trace history is too large")
        return {**case, "traces": [trace]}

    worker.api.run.return_value = {"generated_tokens": 70, "reserved_tokens": 30, "desired_state": "paused"}
    worker.api.cases.side_effect = saved_cases
    worker.api.case.side_effect = saved_case

    async def run():
        try:
            await worker.restore_from_applet()
            assert store.totals() == (100, 0)
            assert store.get("/cases", "abandoned")["state"] == "interrupted"
            saved_trace = store.case_traces("abandoned")[0]
            assert saved_trace["state"] == "interrupted"
            assert saved_trace["payload"]["reasoning"] == "Saved thinking"
            assert worker.desired_state == "paused"
            # The restored record fences repeated recovery of the same reservation.
            store.put("/runs", "test", {"run_id": "test"})
            await worker.restore_from_applet()
            assert store.totals() == (100, 0)
        finally:
            await worker.http.aclose()

    asyncio.run(run())


def test_driver_reviews_quoted_tool_markers_without_changing_the_source_text(tmp_path, store):
    source = {"content": '<tool_call>{"command":"echo café; test 1 < 2"}</tool_call>'}
    output = json.dumps({"summary": "The command was emitted as text.", "breakages": [], "assessments": []})

    async def run():
        worker = make_worker(tmp_path, store)
        worker.endpoints.urls["glm"] = "http://driver/v1/chat/completions"
        await worker.http.aclose()

        def server(request):
            body = json.loads(request.content)
            prompt = body["messages"][1]["content"]
            if "<tool_call>" in prompt:
                return httpx.Response(400, text="Unescaped tool marker activated parser")
            assert json.loads(prompt) == source
            chunk = {
                "choices": [{"delta": {"content": output}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 12},
            }
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(server))
        try:
            result = await worker.structured({"case_id": "quoted-marker"}, "Review this trace", source, 0, 999, "review")
            assert result == output
            trace = store.case_traces("quoted-marker")[0]
            assert trace["state"] == "complete"
            assert json.loads(trace["payload"]["request"]["messages"][1]["content"]) == source
        finally:
            await worker.http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "leaked_prefix", ["", "Inspect privately.<|end_think|>", "<|start_think|>Inspect privately.<|end_think|>"]
)
def test_stream_preserves_thinking_tool_fragments_and_counts_reasoning_once(tmp_path, store, leaked_prefix):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "check "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "the files",
                        "content": leaked_prefix,
                        "tool_calls": [
                            {"index": 0, "id": "call-1", "function": {"name": "shell", "arguments": '{"command":'}}
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"cat input.txt"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 12}}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    async def run():
        worker = make_worker(tmp_path, store)
        await worker.http.aclose()
        worker.http = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: (
                    httpx.Response(200, json={"count": 3, "max_model_len": 65536})
                    if request.url.path == "/tokenize"
                    else httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
                )
            )
        )
        try:
            payload = await worker.completion(
                {"case_id": "case"}, "small", [{"role": "user", "content": "inspect"}], 0, 10, "candidate", 50
            )
        finally:
            await worker.http.aclose()
        expected_reasoning = "check the files" + ("Inspect privately." if leaked_prefix else "")
        assert payload["reasoning"] == expected_reasoning
        assert payload["content"] == ""
        assert payload["tool_calls"][0]["function"] == {"name": "shell", "arguments": '{"command":"cat input.txt"}'}
        assert store.totals() == (20, 0)
        saved = store.case_traces("case")[0]["payload"]
        assert saved["reasoning"] == expected_reasoning
        if leaked_prefix:
            assert saved["reasoning_recovery"]["original_content"] == leaked_prefix
            assert saved["reasoning_recovery"]["original_reasoning"] == "check the files"
        with gzip.open(next((tmp_path / "chunks").glob("*.jsonl.gz")), "rt") as journal:
            assert len(journal.read().splitlines()) == 4

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["answer", "token_limit", "workspace_limit"])
def test_agentic_tools_remain_available_past_six_steps_and_limits_are_recorded(tmp_path, store, outcome):
    async def run():
        worker = make_worker(tmp_path, store)
        await worker.http.aclose()
        requests = []
        history = [{"role": "user", "content": "Write seven numbered lines to progress.txt, then verify them."}]
        commands = []

        def response(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3, "max_model_len": 65536})
            body = json.loads(request.content)
            requests.append(body)
            step = len([m for m in body["messages"] if m["role"] == "tool"]) + 1
            limit = 32 if outcome == "workspace_limit" else 7
            if not body.get("tools"):
                delta = {"content": '<tool_call>{"name":"shell","arguments":{"command":"echo lost"}}</tool_call>'}
                finish = "stop"
            elif step > limit and outcome != "workspace_limit":
                delta = {"content": "Seven lines written and verified."}
                finish = "stop"
            else:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-{step}",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps(
                                    {"command": f"echo step-{step} >> progress.txt; cat progress.txt"}
                                ),
                            },
                        }
                    ]
                }
                finish = "tool_calls"
            used = body["max_tokens"] if outcome == "token_limit" and step == 7 else 10
            chunk = {"choices": [{"delta": delta, "finish_reason": finish}], "usage": {"completion_tokens": used}}
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        try:
            result = await worker.candidate_round(
                {"case_id": "seven-tools", "mode": "agentic"}, "small", history, {}, commands, 0
            )
        finally:
            await worker.http.aclose()
        tool_results = [json.loads(m["content"]) for m in history if m["role"] == "tool"]
        assert "step-7\n" in tool_results[6]["stdout"]
        assert all(r.get("tools") and r.get("tool_choice") == "auto" for r in requests)
        limits = [t for t in store.case_traces("seven-tools") if t["role"] == "limit"]
        if outcome == "answer":
            assert result == RoundEnd.ANSWER
            assert history[-1]["content"] == "Seven lines written and verified."
            assert not limits
        else:
            assert result.value == outcome
            assert limits[-1]["payload"]["kind"] == outcome
            if outcome == "workspace_limit":
                assert tool_results[-1]["kind"] == outcome
                assert "step-32\n" in tool_results[-2]["stdout"]
                assert len(commands) == 32

    asyncio.run(run())


def test_cpu_exhaustion_fails_chat_without_retries_followups_or_pair_review(tmp_path, store, monkeypatch):
    async def run():
        worker = make_worker(tmp_path, store)
        worker.args.shellsim_cpu_limit = 10_000
        worker.endpoints.ready = True
        worker.endpoints.urls["glm"] = "http://glm/v1/chat/completions"
        await worker.http.aclose()
        captured = []

        def response(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3, "max_model_len": 65536})
            body = json.loads(request.content)
            captured.append((request.url.host, body))
            finish = "stop"
            if request.url.host == "small":
                finish = "tool_calls"
                delta = {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": f"call-{i}",
                            "type": "function",
                            "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                        }
                        for i, command in enumerate(["python3 work.py", "echo must-not-run"])
                    ]
                }
            elif request.url.host == "reference":
                delta = {"content": "Reference answer"}
            else:
                purpose = body["prompt_cache_key"].rsplit(":", 1)[1]
                if purpose == "scenario":
                    content = {
                        "title": "CPU exhaustion probe",
                        "category": "coding",
                        "user_message": "Run work.py",
                        "files": {"work.py": "n=0\nfor i in range(100000):\n n+=i\nprint(n)\n"},
                        "objective": "Run the computation",
                        "expected_checks": ["Report the result"],
                        "checks": [],
                    }
                elif purpose == "followup":
                    content = {"user_message": "", "done": True}
                else:
                    raise AssertionError("CPU-limited pairs must not be reviewed")
                delta = {"content": json.dumps(content)}
            chunk = {"choices": [{"delta": delta, "finish_reason": finish}], "usage": {"completion_tokens": 10}}
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        try:
            await worker.conversation(0)
        finally:
            await worker.http.aclose()
        case = store.objects("/cases")[0]
        assert case["state"] == "error"
        assert case["review"]["kind"] == "cpu_limit"
        assert case["review"]["model"] == "small"
        assert not case["review"].get("assessments")
        traces = store.case_traces(case["case_id"])
        results = [t["payload"]["tool_result"] for t in traces if t["role"] == "tool"]
        assert len(results) == 1
        assert results[0]["stop_reason"] == "cpu_exhausted"
        assert next(t for t in traces if t["model"] == "small" and t["role"] == "end")["payload"]["outcome"] == "failed"
        assert not any(t["payload"].get("target_model") == "small" for t in traces if t["model"] == "glm")
        assert len([r for host, r in captured if host == "small"]) == 1
        assert worker.endpoints.ready

    asyncio.run(run())


def test_tool_workspaces_are_isolated_between_candidates(tmp_path, store, monkeypatch):
    async def run():
        worker = make_worker(tmp_path, store)
        calls = {"small": 0, "reference": 0}

        async def completion(case, model, history, *args):
            calls[model] += 1
            if calls[model] > 1:
                return {"content": "done", "reasoning": "", "tool_calls": [], "usage": {"completion_tokens": 1}}
            command = "echo changed > input.txt; cat input.txt" if model == "small" else "cat input.txt"
            return {
                "content": "",
                "reasoning": "inspect",
                "usage": {"completion_tokens": 1},
                "tool_calls": [
                    {
                        "id": model,
                        "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                    }
                ],
            }

        monkeypatch.setattr(worker, "completion", completion)
        histories = {key: [{"role": "user", "content": "Read input.txt"}] for key in calls}
        try:
            for key in calls:
                await worker.candidate_round(
                    {"case_id": "case", "mode": "agentic"}, key, histories[key], {"input.txt": "original\n"}, [], 0
                )
        finally:
            await worker.http.aclose()
        assert json.loads(histories["small"][2]["content"])["stdout"].strip() == "changed"
        assert json.loads(histories["reference"][2]["content"])["stdout"].strip() == "original"
        assert "reasoning_content" not in histories["small"][1]

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_evidence",
    [
        {"message_id": "message", "quote": "invented"},
        {"message_id": "other-candidate", "quote": "wrong answer"},
        {"message_id": "message", "quote": "duplicate"},
    ],
)
def test_reviews_reject_invented_and_cross_candidate_evidence(bad_evidence):
    review = {
        "summary": "Incorrect answer",
        "breakages": [
            {
                "model": "A",
                "category": "incorrect_answer",
                "severity": "high",
                "confidence": 0.9,
                "explanation": "Answer contradicts the file",
                "evidence": [bad_evidence],
            }
        ],
        "assessments": [
            {"model": "A", "score": 0, "strengths": [], "weaknesses": ["Wrong answer"]},
            {"model": "B", "score": 1, "strengths": ["Correct answer"], "weaknesses": []},
        ],
    }
    traces = [
        {"message_id": "message", "model": "small", "payload": {"content": "wrong answer"}},
        {"message_id": "other-candidate", "model": "reference", "payload": {"content": "wrong answer"}},
        {"message_id": "duplicate-1", "model": "small", "payload": {"content": "duplicate"}},
        {"message_id": "duplicate-2", "model": "small", "payload": {"content": "duplicate"}},
    ]
    with pytest.raises(ValueError):
        validated_review(json.dumps(review), {"A": "small", "B": "reference"}, traces)


def test_review_command_citation_keeps_exact_multiline_unicode_text_and_records_unique_link_repair():
    command = "python3 - <<'PY'\nprint('你好')\nPY"
    review = {
        "summary": "The command prints a greeting rather than running the tests",
        "breakages": [
            {
                "model": "A",
                "category": "missed_instruction",
                "severity": "medium",
                "confidence": 0.9,
                "explanation": "It never ran the requested test suite",
                "evidence": [{"message_id": "planning", "quote": command}],
            }
        ],
        "assessments": [
            {"model": "A", "score": 0, "strengths": [], "weaknesses": ["Skipped tests"]},
            {"model": "B", "score": None, "strengths": [], "weaknesses": []},
        ],
    }
    traces = [
        {"message_id": "planning", "model": "small", "payload": {"reasoning": "I should run tests"}},
        {
            "message_id": "command",
            "model": "small",
            "payload": {"tool_calls": [{"function": {"arguments": json.dumps({"command": command})}}]},
        },
        {"message_id": "other", "model": "reference", "payload": {"content": command}},
    ]
    accepted = validated_review(json.dumps(review), {"A": "small", "B": "reference"}, traces)
    evidence = accepted["breakages"][0]["evidence"][0]
    assert evidence == {"message_id": "command", "quote": command}
    assert accepted["citation_repairs"] == [
        {"model": "small", "original_message_id": "planning", "message_id": "command", "quote": command}
    ]

    review["breakages"][0]["evidence"][0]["message_id"] = "command"
    accepted = validated_review(json.dumps(review), {"A": "small", "B": "reference"}, traces)
    assert accepted["breakages"][0]["evidence"][0] == evidence
    assert "citation_repairs" not in accepted


@pytest.mark.parametrize("kind", ["topic", "prompt"])
@pytest.mark.parametrize("leaked_thinking", [False, True])
def test_queued_task_branches_independently_and_followups_only_see_their_own_public_history(
    tmp_path, store, monkeypatch, kind, leaked_thinking
):
    captured: dict = {"followups": [], "candidate_requests": {"small": [], "reference": []}}
    idea = {
        "idea_id": "json-schema",
        "text": "Return JSON with a nested object and strict field types.",
        "kind": kind,
        "mode": "chat",
    }

    async def run():
        worker = make_worker(tmp_path, store)
        worker.case_number = 1

        def response(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3, "max_model_len": 65536})
            body = json.loads(request.content)
            if request.url.host in ("small", "reference"):
                candidate = request.url.host
                captured["candidate_requests"][candidate].append(body)
                content, reasoning = candidate + " public answer", "PRIVATE_THINKING"
                if leaked_thinking and candidate == "small":
                    content = "PRIVATE_THINKING<|end_think|>" + content
                    reasoning = ""
            else:
                purpose = body["prompt_cache_key"].rsplit(":", 1)[1]
                response_format = body["response_format"]
                assert response_format["type"] == "json_schema"
                schema = response_format["json_schema"]
                assert schema["strict"] is True
                required = {
                    "scenario": {"objective", "expected_checks"},
                    "followup": {"done"},
                    "review": {"assessments"},
                }
                assert required[purpose].issubset(schema["schema"]["required"])
                data = json.loads(body["messages"][1]["content"])
                if purpose == "followup":
                    captured["followups"].append(data)
                else:
                    captured[purpose] = data
                reasoning = ""
                if purpose == "scenario":
                    value = {
                        "title": "Check a claim",
                        "category": "formatting",
                        "user_message": "Is this correct?",
                        "files": (
                            {}
                            if schema["schema"]["properties"]["files"].get("additionalProperties") is False
                            else {"unexpected.txt": "The observed generator adds workspace files to a chat-only task."}
                        ),
                        "objective": "Assess a claim",
                        "expected_checks": ["Give an accurate answer"],
                        "checks": [],
                    }
                elif purpose == "followup":
                    history = data["public_conversation"]
                    answer = next(message["content"] for message in history if message["role"] == "assistant")
                    candidate = answer.split()[0]
                    rounds = sum(message["role"] == "user" for message in history)
                    value = {
                        "user_message": f"Verify your {candidate} answer.",
                        "done": rounds == (2 if candidate == "small" else 3),
                    }
                else:
                    value = {
                        "summary": "Both answered",
                        "breakages": [],
                        "assessments": [
                            {"model": label, "score": None, "strengths": [], "weaknesses": []} for label in ("A", "B")
                        ],
                    }
                content = json.dumps(value)
            chunk = {
                "choices": [{"delta": {"content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 20},
            }
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        def claim(run_id, case_id):
            captured["claim"] = {"case_id": case_id}
            return {"idea": idea}

        worker.api.claim.side_effect = claim
        await worker.http.aclose()
        worker.endpoints.urls["glm"] = "http://glm/v1/chat/completions"
        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        try:
            await worker.conversation(0)
        finally:
            await worker.http.aclose()

    asyncio.run(run())
    assert "PRIVATE_THINKING" not in json.dumps(captured["followups"])
    assert "PRIVATE_THINKING" in json.dumps(captured["review"])
    assert all("checkpoint" not in trace["payload"] for trace in captured["review"]["traces"])
    opening = idea["text"] if kind == "prompt" else "Is this correct?"
    for candidate, count in (("small", 2), ("reference", 3)):
        requests = captured["candidate_requests"][candidate]
        assert len(requests) == count
        users = [message["content"] for message in requests[-1]["messages"] if message["role"] == "user"]
        assert users == [opening] + [f"Verify your {candidate} answer."] * (count - 1)
    for followup in captured["followups"]:
        public = followup["public_conversation"]
        answers = [message["content"] for message in public if message["role"] == "assistant"]
        assert len(set(answers)) == 1
        assert public[1] == {"role": "user", "content": opening}
        assert followup["queued_idea"] == captured["scenario"]["queued_idea"] == idea
    case = store.objects("/cases")[0]
    assert case["state"] == "complete"
    assert case["seed"]["queued_idea"] == idea
    assert case["category"] == "formatting" and case["mode"] == "chat"
    assert captured["claim"]["case_id"] == case["case_id"]
    traces = store.case_traces(case["case_id"])
    users = [trace for trace in traces if trace["role"] == "user"]
    assert [(trace["model"], trace["payload"]["content"]) for trace in users] == [
        ("shared", opening),
        ("reference", "Verify your reference answer."),
        ("small", "Verify your small answer."),
        ("reference", "Verify your reference answer."),
    ]
    followups = [trace for trace in traces if trace["model"] == "glm" and trace["payload"].get("purpose") == "followup"]
    assert {trace["payload"]["target_model"] for trace in followups} == {"small", "reference"}


@pytest.mark.parametrize("citation_error", ["punctuation", "ambiguous_message", "missing_assessments"])
def test_review_repair_pairs_exact_quotes_with_their_candidate_messages(tmp_path, store, monkeypatch, citation_error):
    source_text = "small: web-slinger bonus = 15%, not 50%."
    altered_quote = "web\u2011slinger bonus = 15%"
    wrong_message = "small: I cannot verify the bonus."
    rejected_quote = altered_quote if citation_error == "punctuation" else "web-slinger bonus = 15%"

    async def run():
        worker = make_worker(tmp_path, store)
        worker.case_number = 1

        def response(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3, "max_model_len": 65536})
            body = json.loads(request.content)
            if request.url.host == "small":
                round_number = sum(message["role"] == "user" for message in body["messages"])
                content = wrong_message if citation_error == "ambiguous_message" and round_number == 3 else source_text
            elif request.url.host == "reference":
                # An exact quote from the other candidate must not repair this candidate's evidence.
                content = altered_quote
            else:
                purpose = body["prompt_cache_key"].rsplit(":", 1)[1]
                data = json.loads(body["messages"][1]["content"])
                if purpose == "scenario":
                    value = {
                        "title": "Check the bonus",
                        "category": "reasoning",
                        "user_message": "Is the bonus 50%?",
                        "files": {},
                        "objective": "Check a numerical claim",
                        "expected_checks": ["Explain the bonus"],
                        "checks": [],
                    }
                elif purpose == "followup":
                    rounds = (len(data["public_conversation"]) - 1) // 2
                    value = {
                        "user_message": "Can you verify that?",
                        "done": citation_error != "ambiguous_message" or rounds == 3,
                    }
                else:
                    source = next(trace for trace in data["traces"] if trace["payload"].get("content") == source_text)
                    cited = source
                    if citation_error == "ambiguous_message":
                        cited = next(
                            trace for trace in data["traces"] if trace["payload"].get("content") == wrong_message
                        )
                    evidence = {"message_id": cited["message_id"], "quote": rejected_quote}
                    if purpose == "review_repair":
                        schema = body["response_format"]["json_schema"]["schema"]
                        branches = schema["properties"]["breakages"]["items"].get("anyOf", [])
                        # Reproduce the native failure unless the schema constrains the quote and message together.
                        if branches:
                            branch = next(
                                item for item in branches if item["properties"]["model"]["const"] == source["model"]
                            )
                            options = branch["properties"]["evidence"]["items"]["anyOf"]
                            pairs = {
                                (item["properties"]["message_id"]["const"], item["properties"]["quote"]["const"])
                                for item in options
                            }
                            if citation_error != "missing_assessments":
                                assert (cited["message_id"], rejected_quote) not in pairs
                            assert all(item["model"] == source["model"] for item in data["evidence_sources"])
                            chosen = next(item for item in data["evidence_sources"] if "web-slinger" in item["quote"])
                            assert (chosen["message_id"], chosen["quote"]) in pairs
                            evidence = {"message_id": chosen["message_id"], "quote": chosen["quote"]}
                    value = {
                        "summary": "The stated bonus is 15%",
                        "breakages": [
                            {
                                "model": source["model"],
                                "category": "incorrect_answer",
                                "severity": "low",
                                "confidence": 0.5,
                                "explanation": "The bonus needs checking",
                                "evidence": [evidence],
                            }
                        ],
                        "assessments": [
                            {"model": label, "score": None, "strengths": [], "weaknesses": []} for label in ("A", "B")
                        ],
                    }
                    if citation_error == "missing_assessments":
                        schema = body["response_format"]["json_schema"]["schema"]
                        positions = schema["properties"]["assessments"].get("prefixItems", [])
                        # Replay the observed repair that silently dropped all candidate assessments.
                        if purpose == "review" or not positions:
                            value["assessments"] = []
                        else:
                            value["assessments"] = [
                                {
                                    "model": item["properties"]["model"]["const"],
                                    "score": None,
                                    "strengths": [],
                                    "weaknesses": [],
                                }
                                for item in positions
                            ]
                content = json.dumps(value)
            chunk = {
                "choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 20},
            }
            return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        await worker.http.aclose()
        worker.endpoints.urls["glm"] = "http://glm/v1/chat/completions"
        worker.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        try:
            await worker.conversation(0)
        finally:
            await worker.http.aclose()

    asyncio.run(run())
    case = store.objects("/cases")[0]
    assert case["state"] == "complete"
    finding = case["review"]["breakages"][0]
    if citation_error == "missing_assessments":
        assert case["review"]["previous_review_error"] == "Review must assess every candidate"
        assert {item["model"] for item in case["review"]["assessments"]} == {"small", "reference"}
    else:
        assert rejected_quote in case["review"]["previous_review_error"]
    assert finding["model"] == "small"
    evidence = finding["evidence"][0]
    assert evidence["quote"] in source_text and "web-slinger" in evidence["quote"]
    cited = store.get("/traces", evidence["message_id"])
    assert cited["model"] == "small" and evidence["quote"] in cited["payload"]["content"]
    original = next(
        trace
        for trace in store.case_traces(case["case_id"])
        if trace["model"] == "glm" and trace["payload"].get("purpose") == "review"
    )
    with pytest.raises(ValueError):
        validated_review(original["payload"]["content"], case["review"]["aliases"], store.case_traces(case["case_id"]))
