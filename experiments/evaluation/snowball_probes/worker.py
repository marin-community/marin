# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continuously run paired GLM-as-user probes and upload their streamed traces."""

import argparse
import asyncio
import dataclasses
import fcntl
import gzip
import hashlib
import json
import logging
import random
import signal
import time
import uuid
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from urllib.parse import urljoin

import httpx
from dotenv import dotenv_values
from huggingface_hub import HfApi
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from marin.inference.shell_workspace import (
    MAX_WORKSPACE_COMMANDS,
    MAX_WORKSPACE_HISTORY_BYTES,
    ShellWorkspaceRequest,
    ShellWorkspaceTerminated,
    shell_command_result,
)
from marina.client import marina_request
from pydantic import ValidationError
from rigging.connect import capability_path
from rigging.timing import Duration

from experiments.evaluation.snowball_probes.config import (
    CATEGORIES,
    CONCURRENT_CHATS,
    TOKEN_CAP,
    WILDCHAT_DATASET,
    WILDCHAT_REVISION,
    ReasoningFormat,
)
from experiments.evaluation.snowball_probes.protocol import (
    FOLLOWUP_INSTRUCTIONS,
    GLM_LANGUAGE_INSTRUCTIONS,
    REVIEW_REPAIR_INSTRUCTIONS,
    SHELLSIM_INSTRUCTIONS,
    Followup,
    Review,
    Scenario,
    infrastructure_review,
    json_object,
    public_history,
    recover_grug_reasoning,
    review_instructions,
    review_repair_schema,
    review_repair_sources,
    scenario_instructions,
    scenario_schema,
    separate_grug_reasoning,
    validated_review,
)
from experiments.evaluation.snowball_probes.store import Store, TokenCapReached
from experiments.evaluation.snowball_probes.verification import run_checks, validate_checks

logger = logging.getLogger(__name__)
GLM_MAX_TOKENS = 8192
CANDIDATE_MAX_TOKENS = 16384
SHELLSIM_CPU_LIMIT = 500_000_000
INFERENCE_TIMEOUT = 1200
MAX_ROUNDS = 4
TRACE_ROUND_STRIDE = 1_000_000
MAX_TOOL_OUTPUT_CHARS = 4000
SNAPSHOT_INTERVAL = 2
CAPABILITY_REFRESH = 3600
READINESS_REFRESH = 15
CONNECT_RETRIES = 3
UPLOAD_BATCH_BYTES = 4 * 1024 * 1024
GLM_RESPONSE_MODELS = {"scenario": Scenario, "followup": Followup, "review": Review, "review_repair": Review}
APPLET_ORIGIN = "https://applets.marina.oa.dev"


class RoundEnd(StrEnum):
    ANSWER = "answer"
    TOKEN_LIMIT = "token_limit"
    CONTEXT_LIMIT = "context_limit"
    WORKSPACE_LIMIT = "workspace_limit"


class ShellSimCpuExhausted(RuntimeError):
    def __init__(self, model: str, cpu_limit: int):
        self.model = model
        super().__init__(f"{model}: ShellSim exhausted its {cpu_limit:,} CPU-unit budget")


TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a command in the isolated persistent ShellSim workspace at /work. No network or installs.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class ContextLimitReached(Exception):
    pass


def transport_failure(error: httpx.HTTPError) -> str:
    """Describe connection causes without exposing credential-bearing URLs."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, OSError):
            return f"{type(error).__name__} ({type(cause).__name__}, errno={cause.errno})"
        cause = cause.__cause__ or cause.__context__
    return type(error).__name__


def accumulate_chunk(payload: dict, chunk: dict) -> None:
    """Keep public text, hidden reasoning, tool fragments, usage, and finish reasons separately."""
    if chunk.get("usage"):
        payload["usage"] = chunk["usage"]
    for choice in chunk.get("choices", []):
        if choice.get("index", 0) != 0:
            continue
        delta = choice.get("delta", {})
        payload["content"] += delta.get("content") or ""
        payload["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
        for fragment in delta.get("tool_calls", []):
            index = fragment["index"]
            while len(payload["tool_calls"]) <= index:
                payload["tool_calls"].append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            call = payload["tool_calls"][index]
            if fragment.get("id"):
                call["id"] = fragment["id"]
            function = fragment.get("function", {})
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""
        if choice.get("finish_reason"):
            payload["finish_reason"] = choice["finish_reason"]


def inspiration_seeds(path: Path) -> list[dict]:
    """Cache a small, provenance-bearing WildChat sample without account or IP metadata."""
    if path.exists():
        return json.loads(path.read_text())
    revision = HfApi().dataset_info(WILDCHAT_DATASET).sha
    if revision != WILDCHAT_REVISION:
        raise ValueError("WildChat changed; update its pinned revision before collecting new inspiration")
    seeds = []
    with httpx.Client(timeout=60) as client:
        for offset in (0, 1000, 10000, 25000, 50000, 75000, 100000, 200000):
            response = client.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": WILDCHAT_DATASET,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": 12,
                },
            )
            response.raise_for_status()
            for item in response.json()["rows"]:
                row = item["row"]
                text = next((turn["content"] for turn in row["conversation"] if turn["role"] == "user"), "")
                if len(text.strip()) < 40:
                    continue
                seeds.append(
                    {"dataset": WILDCHAT_DATASET, "revision": revision, "row": item["row_idx"], "text": text[:6000]}
                )
    if not seeds:
        raise ValueError("WildChat returned no usable inspiration")
    path.write_text(json.dumps(seeds, ensure_ascii=False))
    return seeds


class Endpoints:
    def __init__(self, client: IrisClient, origin: str, coordinator: str, glm_gateway: str):
        self.client = client
        self.origin = origin
        self.coordinator = coordinator
        self.glm_gateway = glm_gateway
        self.urls: dict[str, str] = {}
        self.metadata: list[dict] = []
        self.refreshed = 0.0
        self.status_refreshed = 0.0
        self.names: dict[str, str] = {}
        self.checked = 0.0
        self.ready = False
        self.readiness_error: str | None = None
        self.status_url = ""
        self.refresh_lock = asyncio.Lock()

    def capability(self, name: str) -> str:
        response = self.client.mint_endpoint_token(name, ttl=Duration.from_seconds(7 * 86400))
        path = response.capability_url or capability_path(name, response.token)
        return urljoin(self.origin + "/", path)

    async def refresh(self, http: httpx.AsyncClient) -> bool:
        async with self.refresh_lock:
            now = time.monotonic()
            if now - self.checked < READINESS_REFRESH:
                return self.ready
            self.ready = False
            if not self.status_url or now - self.status_refreshed >= CAPABILITY_REFRESH:
                root = await asyncio.to_thread(self.capability, self.coordinator + "/probes")
                self.status_url = root + "/status"
                self.status_refreshed = now
            try:
                response = await http.get(self.status_url)
            except httpx.HTTPError as error:
                self.checked = now
                self.readiness_error = f"Coordinator status transport failed: {transport_failure(error)}"
                logger.warning(self.readiness_error)
                return False
            if response.status_code in {502, 503, 504}:
                self.checked = now
                self.readiness_error = f"Coordinator status HTTP {response.status_code}"
                logger.warning(self.readiness_error)
                return False
            if response.status_code >= 400:
                raise RuntimeError(f"Coordinator status HTTP {response.status_code}")
            candidates = {candidate["key"]: candidate for candidate in response.json()["candidates"]}
            self.metadata = [
                {key: value for key, value in candidate.items() if key not in {"endpoint", "jobs", "state", "error"}}
                for candidate in candidates.values()
            ]
            self.checked = now
            if any(candidate["state"] == "failed" for candidate in candidates.values()):
                raise RuntimeError("A candidate serving job failed; inspect the coordinator job logs")
            if any(candidate["state"] != "ready" for candidate in candidates.values()):
                return False
            instances = await asyncio.gather(
                *[
                    asyncio.to_thread(self.client.list_endpoint_instances, candidate["endpoint"])
                    for candidate in candidates.values()
                ]
            )
            if any(not registered for registered in instances):
                return False
            names = {key: candidate["endpoint"] for key, candidate in candidates.items()}
            if names != self.names or now - self.refreshed >= CAPABILITY_REFRESH:
                gateway = await asyncio.to_thread(self.capability, self.glm_gateway)
                self.urls = {"glm": gateway + "/v1/chat/completions"}
                for key, candidate in candidates.items():
                    self.urls[key] = await asyncio.to_thread(self.capability, candidate["endpoint"])
                    self.urls[key] += "/v1/chat/completions"
                self.refreshed = now
                self.names = names
            self.ready = True
            self.readiness_error = None
            return True


class Worker:
    def __init__(self, args: argparse.Namespace, store: Store, endpoints: Endpoints, seeds: list[dict], glm_key: str):
        self.args = args
        self.store = store
        self.endpoints = endpoints
        self.seeds = seeds
        self.glm_key = glm_key
        self.desired_state = "running"
        self.active: set[str] = set()
        self.upload_error: str | None = None
        self.stop = asyncio.Event()
        self.cap_reached = False
        self.config: dict = {}
        self.case_number = store.count("/cases")
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=30), transport=httpx.AsyncHTTPTransport(retries=CONNECT_RETRIES)
        )

    def run_snapshot(self) -> None:
        generated, reserved = self.store.totals()
        state = self.desired_state
        if self.desired_state == "running" and not self.endpoints.ready:
            state = "waiting_for_servers"
        if self.stop.is_set():
            state = "stopped"
        self.store.put(
            "/runs",
            self.args.run_id,
            {
                "run_id": self.args.run_id,
                "config": self.config,
                "token_cap": self.args.token_cap,
                "concurrency": self.args.concurrency,
                "generated_tokens": generated,
                "reserved_tokens": reserved,
                "state": state,
                "active_chats": len(self.active),
                "error": self.upload_error or self.endpoints.readiness_error,
            },
        )

    async def upload(self, path: str, body: dict) -> dict:
        result = await asyncio.to_thread(
            marina_request,
            self.args.marina_url,
            "POST",
            f"/a/{self.args.applet}/api{path}",
            json_body=body,
        )
        if not isinstance(result, dict):
            raise ValueError("Applet upload returned a non-object response")
        return result

    async def restore_from_marina(self) -> None:
        """Recover accounting and interrupted chats when a cluster attempt loses its disk."""
        if self.store.get("/runs", self.args.run_id) is not None:
            return
        prefix = f"/a/{self.args.applet}/api"
        snapshot = await asyncio.to_thread(
            marina_request, self.args.marina_url, "GET", f"{prefix}/runs/{self.args.run_id}"
        )
        assert isinstance(snapshot, dict)
        self.store.restore_budget(snapshot["generated_tokens"] + snapshot["reserved_tokens"])
        for state in ("generating", "reviewing"):
            cases = await asyncio.to_thread(
                marina_request,
                self.args.marina_url,
                "GET",
                f"{prefix}/cases",
                params={"run_id": self.args.run_id, "state": state, "limit": 200},
            )
            assert isinstance(cases, dict)
            if cases["total"] > len(cases["cases"]):
                raise ValueError("Too many interrupted chats to recover in one cluster attempt")
            for case in cases["cases"]:
                detail = await asyncio.to_thread(
                    marina_request,
                    self.args.marina_url,
                    "GET",
                    f"{prefix}/cases/{case['case_id']}",
                    params={"trace_state": "streaming"},
                )
                assert isinstance(detail, dict)
                for trace in detail["traces"]:
                    if trace["state"] == "streaming":
                        trace["state"] = "interrupted"
                        trace["payload"]["error"] = "Worker restarted; reservation charged conservatively"
                        self.store.put("/traces", trace["message_id"], trace)
                case["state"] = "interrupted"
                case["review"] = {"summary": "Worker restarted; pair unscored", "kind": "infrastructure"}
                self.store.put("/cases", case["case_id"], case)
        cases = await asyncio.to_thread(
            marina_request,
            self.args.marina_url,
            "GET",
            f"{prefix}/cases",
            params={"run_id": self.args.run_id, "limit": 1},
        )
        assert isinstance(cases, dict)
        self.case_number = cases["total"]
        self.desired_state = snapshot["desired_state"]

    async def flush(self) -> None:
        pending = self.store.pending(256)
        for path in ("/runs", "/cases", "/traces"):
            records = [record for record in pending if record[0] == path]
            offset = 0
            while offset < len(records):
                batch = []
                size = 0
                limit = 1 if path == "/runs" else 128
                for record in records[offset : offset + limit]:
                    record_bytes = len(record[2].encode())
                    if batch and size + record_bytes > UPLOAD_BATCH_BYTES:
                        break
                    batch.append(record)
                    size += record_bytes
                offset += len(batch)
                bodies = [json.loads(record[2]) for record in batch]
                result = (
                    await self.upload(path, bodies[0])
                    if path == "/runs"
                    else await self.upload(path + "/batch", {path[1:]: bodies})
                )
                for record in batch:
                    self.store.acknowledge(path, record[1], record[3])
                if path == "/runs":
                    self.desired_state = result["desired_state"]
                    if self.desired_state == "stopped":
                        self.stop.set()

    async def heartbeat(self) -> None:
        while not self.stop.is_set():
            self.run_snapshot()
            try:
                await self.flush()
                self.upload_error = None
            except (RuntimeError, httpx.HTTPError, OSError) as error:
                self.upload_error = f"Trace upload failed: {type(error).__name__}"
                logger.exception("Trace upload failed; local outbox retained")
            await asyncio.sleep(3)

    def trace(
        self,
        case: dict,
        model: str,
        role: str,
        round_number: int,
        sequence: int,
        payload: dict,
        state: str = "complete",
        message_id: str | None = None,
    ) -> dict:
        if role == "assistant" and payload.get("purpose") == "candidate":
            payload = self.candidate_payload(model, payload)
        trace = {
            "message_id": message_id or uuid.uuid4().hex,
            "case_id": case["case_id"],
            "model": model,
            "role": role,
            "round": round_number,
            "sequence": sequence,
            "state": state,
            "payload": payload,
        }
        self.store.put("/traces", trace["message_id"], trace)
        return trace

    def candidate_payload(self, model: str, payload: dict) -> dict:
        candidate = next(item for item in self.endpoints.metadata if item["key"] == model)
        result = {**payload, "native_content": payload["content"], "reasoning_format": candidate["reasoning_format"]}
        if candidate["reasoning_format"] == ReasoningFormat.GRUG_INLINE:
            separate_grug_reasoning(result)
        elif candidate["reasoning_parser"] == "grug_optional":
            recover_grug_reasoning(result)
        return result

    async def completion(
        self,
        case: dict,
        model: str,
        messages: list[dict],
        round_number: int,
        sequence: int,
        purpose: str,
        maximum: int,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
        target_model: str | None = None,
    ) -> dict:
        input_count: int | None = None
        if model != "glm":
            tokenize_body = {
                "model": model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            if tools:
                tokenize_body["tools"] = tools
            tokenize_url = self.endpoints.urls[model].removesuffix("/v1/chat/completions") + "/tokenize"
            try:
                tokenized = await self.http.post(tokenize_url, json=tokenize_body)
            except httpx.HTTPError as error:
                raise RuntimeError(f"{model} tokenization transport failed: {transport_failure(error)}") from None
            if tokenized.status_code >= 400:
                raise RuntimeError(f"{model} tokenization HTTP {tokenized.status_code}")
            counts = tokenized.json()
            input_count = counts["count"]
            maximum = min(maximum, counts["max_model_len"] - input_count)
            if maximum <= 0:
                raise ContextLimitReached(f"Input has {input_count} tokens in a {counts['max_model_len']}-token context")
        request_id = uuid.uuid4().hex
        self.store.reserve(request_id, maximum)
        payload: dict = {
            "content": "",
            "reasoning": "",
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
            "purpose": purpose,
            "request": {"messages": messages, "max_tokens": maximum, "tools": tools},
            "started_at": time.time(),
            "input_tokens_counted": input_count,
            "checkpoint": "GLM-5.3 · Ortet" if model == "glm" else model,
        }
        body: dict = {
            "model": "glm-5.3" if model == "glm" else model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": maximum,
            "temperature": 0.8 if model == "glm" and purpose == "scenario" else 0.6,
            "chat_template_kwargs": {"reasoning_effort": "low"} if model == "glm" else {"enable_thinking": True},
        }
        if model != "glm":
            candidate = next(item for item in self.endpoints.metadata if item["key"] == model)
            if candidate["reasoning_format"] == ReasoningFormat.GRUG_INLINE:
                body["skip_special_tokens"] = False
        if target_model is not None:
            payload["target_model"] = target_model
        if model == "glm":
            body["prompt_cache_key"] = self.args.run_id + ":" + case["case_id"] + ":" + purpose
            body["tool_choice"] = "none"
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": purpose,
                    "schema": (
                        response_schema
                        if response_schema is not None
                        else GLM_RESPONSE_MODELS[purpose].model_json_schema()
                    ),
                    "strict": True,
                },
            }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        payload["request"] = body
        headers = {"x-ortet-api-token": self.glm_key} if model == "glm" else {}
        self.trace(case, model, "assistant", round_number, sequence, payload, "streaming", request_id)
        last_snapshot = time.monotonic()
        state = "complete"
        charge: int | None = None
        started = time.monotonic()
        try:
            async with (
                asyncio.timeout(self.args.inference_timeout),
                self.http.stream(
                    "POST",
                    self.endpoints.urls[model],
                    headers=headers,
                    json=body,
                    timeout=httpx.Timeout(self.args.inference_timeout, connect=30),
                ) as response,
            ):
                if response.status_code >= 400:
                    charge = 0
                    # Capability URLs and bearer credentials must never enter error messages or saved traces.
                    raise RuntimeError(f"{model} inference HTTP {response.status_code}")
                chunk_directory = self.args.state_dir / "chunks"
                chunk_directory.mkdir(exist_ok=True)
                with gzip.open(chunk_directory / f"{request_id}.jsonl.gz", "wt") as journal:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "error" in chunk:
                            raise RuntimeError(f"{model} stream error: {chunk['error']}")
                        journal.write(json.dumps({"message_id": request_id, "chunk": chunk}, ensure_ascii=False) + "\n")
                        if "first_token_latency" not in payload and chunk.get("choices"):
                            payload["first_token_latency"] = time.monotonic() - started
                        accumulate_chunk(payload, chunk)
                        if time.monotonic() - last_snapshot >= SNAPSHOT_INTERVAL:
                            journal.flush()
                            self.trace(
                                case, model, "assistant", round_number, sequence, payload, "streaming", request_id
                            )
                            last_snapshot = time.monotonic()
                if payload["finish_reason"] is None:
                    raise RuntimeError(f"{model} stream ended without a finish reason")
                charge = payload["usage"].get("completion_tokens")
        except BaseException as error:
            state = "error"
            if isinstance(error, TimeoutError):
                payload["error"] = f"Inference request exceeded {self.args.inference_timeout:g} seconds"
                payload["kind"] = "inference_timeout"
                raise RuntimeError(f"{model}: {payload['error']}") from None
            if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
                charge = 0
            payload["error"] = (
                f"{type(error).__name__}: {str(error)[:1000]}"
                if not isinstance(error, httpx.HTTPError)
                else transport_failure(error)
            )
            if isinstance(error, httpx.HTTPError):
                raise RuntimeError(f"{model} inference transport failed: {transport_failure(error)}") from None
            raise
        finally:
            payload["duration"] = time.monotonic() - started
            payload["charged_tokens"] = self.store.settle(request_id, charge)
            payload["usage_is_conservative"] = charge is None
            final_trace = self.trace(case, model, "assistant", round_number, sequence, payload, state, request_id)
        if model == "glm":
            return payload
        displayed = final_trace["payload"]
        candidate = next(item for item in self.endpoints.metadata if item["key"] == model)
        grug = displayed["reasoning_format"] == ReasoningFormat.GRUG_INLINE or candidate["reasoning_parser"] == "grug"
        reported_reasoning = displayed["usage"].get("completion_tokens_details") or {}
        if grug and displayed["reasoning"] and reported_reasoning.get("reasoning_tokens") is None:
            try:
                counted = await self.http.post(
                    self.endpoints.urls[model].removesuffix("/v1/chat/completions") + "/tokenize",
                    json={"model": model, "prompt": displayed["reasoning"], "add_special_tokens": False},
                )
            except httpx.HTTPError as error:
                displayed["reasoning_count_error"] = transport_failure(error)
            else:
                if counted.status_code == 200:
                    displayed["reasoning_tokens_counted"] = counted.json()["count"]
                    displayed["reasoning_count_source"] = "retokenized_text"
                else:
                    displayed["reasoning_count_error"] = f"Tokenization HTTP {counted.status_code}"
            self.store.put("/traces", request_id, final_trace)
        return displayed

    async def structured(
        self,
        case: dict,
        instruction: str,
        data: object,
        round_number: int,
        sequence: int,
        purpose: str,
        response_schema: dict | None = None,
        target_model: str | None = None,
    ) -> str:
        # Literal tool markers in quoted traces can activate Ortet's output parser during GLM thinking.
        # JSON escapes preserve the source text while keeping those markers out of the prompt's tokens.
        serialized_data = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": serialized_data},
        ]
        payload = await self.completion(
            case,
            "glm",
            messages,
            round_number,
            sequence,
            purpose,
            GLM_MAX_TOKENS,
            response_schema=response_schema,
            target_model=target_model,
        )
        return payload["content"]

    async def candidate_round(
        self, case: dict, model: str, history: list[dict], files: dict[str, str], commands: list[str], round_number: int
    ) -> RoundEnd:
        remaining = CANDIDATE_MAX_TOKENS
        sequence = round_number * TRACE_ROUND_STRIDE + 10
        while remaining > 0:
            try:
                payload = await self.completion(
                    case,
                    model,
                    history,
                    round_number,
                    sequence,
                    "candidate",
                    remaining,
                    [TOOL] if case["mode"] == "agentic" else None,
                )
            except ContextLimitReached as error:
                self.trace(
                    case, model, "limit", round_number, sequence, {"content": str(error), "kind": "context_limit"}
                )
                return RoundEnd.CONTEXT_LIMIT
            remaining -= payload["usage"].get("completion_tokens", remaining)
            candidate = next(item for item in self.endpoints.metadata if item["key"] == model)
            content = (
                payload["native_content"]
                if candidate["reasoning_format"] == ReasoningFormat.GRUG_INLINE
                else payload["content"]
            )
            assistant = {"role": "assistant", "content": content or None}
            if candidate["reasoning_parser"] == "grug" and payload["reasoning"]:
                assistant["reasoning_content"] = payload["reasoning"]
            if payload["tool_calls"]:
                assistant["tool_calls"] = payload["tool_calls"]
            history.append(assistant)
            round_end = RoundEnd.ANSWER
            for call in payload["tool_calls"]:
                try:
                    if call["function"]["name"] != "shell":
                        raise ValueError("Only the shell tool is available")
                    arguments = json_object(call["function"]["arguments"])
                    request = ShellWorkspaceRequest.from_payload(
                        {"files": files, "commits": [], "history": commands, "command": arguments["command"]}
                    )
                    history_bytes = sum(len(command.encode()) for command in commands) + len(request.command.encode())
                    if len(commands) >= MAX_WORKSPACE_COMMANDS or history_bytes > MAX_WORKSPACE_HISTORY_BYTES:
                        output = {"error": "Harness workspace history limit reached", "kind": "workspace_limit"}
                        round_end = RoundEnd.WORKSPACE_LIMIT
                    else:
                        result = await asyncio.to_thread(
                            shell_command_result, request, cpu_limit=self.args.shellsim_cpu_limit
                        )
                        output = dataclasses.asdict(result)
                        output["output_truncated"] = any(
                            len(output[field]) > MAX_TOOL_OUTPUT_CHARS for field in ("stdout", "stderr")
                        )
                        for field in ("stdout", "stderr"):
                            output[field] = output[field][:MAX_TOOL_OUTPUT_CHARS]
                        if result.stop_reason != "cpu_exhausted":
                            commands.append(request.command)
                except ShellWorkspaceTerminated as error:
                    if error.stop_reason != "cpu_exhausted":
                        raise
                    output = {"error": str(error), "stop_reason": error.stop_reason, "kind": "cpu_limit"}
                except (ValueError, KeyError, json.JSONDecodeError) as error:
                    output = {"error": str(error), "kind": "invalid_tool_call"}
                content = json.dumps(output, ensure_ascii=False)
                history.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                sequence += 1
                self.trace(
                    case,
                    model,
                    "tool",
                    round_number,
                    sequence,
                    {"content": content, "tool_call_id": call["id"], "tool_result": output},
                )
                if output.get("stop_reason") == "cpu_exhausted":
                    self.trace(
                        case,
                        model,
                        "limit",
                        round_number,
                        sequence + 1,
                        {
                            "content": "Chat failed: ShellSim CPU budget exhausted",
                            "kind": "cpu_limit",
                            "cpu_limit": self.args.shellsim_cpu_limit,
                        },
                    )
                    raise ShellSimCpuExhausted(model, self.args.shellsim_cpu_limit)
            if round_end == RoundEnd.WORKSPACE_LIMIT:
                self.trace(
                    case,
                    model,
                    "limit",
                    round_number,
                    sequence + 1,
                    {"content": "Harness stopped this branch: workspace history limit reached", "kind": round_end.value},
                )
                return round_end
            if remaining <= 0 or payload.get("finish_reason") == "length":
                self.trace(
                    case,
                    model,
                    "limit",
                    round_number,
                    sequence + 1,
                    {"content": "Harness stopped this turn: completion-token budget reached", "kind": "token_limit"},
                )
                return RoundEnd.TOKEN_LIMIT
            if not payload["tool_calls"]:
                return RoundEnd.ANSWER
            sequence += 1
        raise AssertionError("Token exhaustion must return a limit outcome")

    def case_traces(self, case: dict) -> list[dict]:
        return self.store.case_traces(case["case_id"])

    async def candidate_conversation(
        self, case: dict, scenario: Scenario, idea: dict | None, model: str, index: int, history: list[dict]
    ) -> None:
        """Continue one candidate using only its own public conversation and workspace."""
        commands: list[str] = []
        user_message = scenario.user_message
        for round_number in range(MAX_ROUNDS):
            if self.stop.is_set():
                return
            while self.desired_state == "paused" and not self.stop.is_set():
                await asyncio.sleep(1)
            if self.stop.is_set():
                return
            case["round"] = max(case["round"], round_number)
            self.store.put("/cases", case["case_id"], case)
            if round_number:
                self.trace(
                    case, model, "user", round_number, round_number * TRACE_ROUND_STRIDE, {"content": user_message}
                )
            history.append({"role": "user", "content": user_message})
            try:
                round_end = await self.candidate_round(case, model, history, scenario.files, commands, round_number)
            except ShellSimCpuExhausted:
                self.trace(
                    case,
                    model,
                    "end",
                    round_number,
                    (round_number + 1) * TRACE_ROUND_STRIDE - 5,
                    {"content": "Chat failed: ShellSim CPU budget exhausted", "outcome": "failed"},
                )
                raise
            if round_number == 0:
                answers = [
                    trace["payload"]["content"]
                    for trace in self.case_traces(case)
                    if trace["model"] == model and trace["role"] == "assistant"
                ]
                verification = await asyncio.to_thread(
                    run_checks,
                    scenario.checks,
                    answers[-1] if answers else "",
                    scenario.files,
                    commands,
                    self.args.shellsim_cpu_limit,
                    None if round_end == RoundEnd.ANSWER else round_end.value,
                )
                self.trace(case, model, "verification", 0, TRACE_ROUND_STRIDE - 100 + index, verification)
            if round_number + 1 == MAX_ROUNDS or round_end in {RoundEnd.CONTEXT_LIMIT, RoundEnd.WORKSPACE_LIMIT}:
                break
            followup = Followup.model_validate(
                json_object(
                    await self.structured(
                        case,
                        FOLLOWUP_INSTRUCTIONS,
                        {
                            "task": scenario.user_message,
                            "mode": case["mode"],
                            "queued_idea": idea,
                            "public_conversation": public_history(
                                history,
                                next(item for item in self.endpoints.metadata if item["key"] == model)[
                                    "reasoning_format"
                                ]
                                == ReasoningFormat.GRUG_INLINE,
                            ),
                            "last_turn_end": round_end.value,
                        },
                        round_number,
                        (round_number + 1) * TRACE_ROUND_STRIDE - 10 + index,
                        "followup",
                        target_model=model,
                    )
                )
            )
            if followup.done:
                break
            user_message = followup.user_message
        self.trace(
            case,
            model,
            "end",
            round_number,
            (round_number + 1) * TRACE_ROUND_STRIDE - 5,
            {"content": "Conversation finished", "outcome": round_end.value},
        )

    async def conversation(self, slot: int) -> None:
        number = self.case_number
        self.case_number += 1
        seed = random.choice(self.seeds)
        category = CATEGORIES[number % len(CATEGORIES)]
        mode = "agentic" if number % 2 == 0 else "chat"
        case: dict = {
            "case_id": uuid.uuid4().hex,
            "run_id": self.args.run_id,
            "title": f"Preparing {category} probe",
            "category": category,
            "mode": mode,
            "state": "generating",
            "slot": slot,
            "round": 0,
            "seed": {key: value for key, value in seed.items() if key != "text"},
            "review": {},
        }
        self.store.put("/cases", case["case_id"], case)
        self.active.add(case["case_id"])
        try:
            claimed = await self.upload(f"/runs/{self.args.run_id}/ideas/claim", {"case_id": case["case_id"]})
            idea = claimed["idea"]
            if idea:
                idea = {key: idea[key] for key in ("idea_id", "text", "kind", "mode")}
                case["seed"]["queued_idea"] = idea
                if idea["mode"] != "auto":
                    mode = idea["mode"]
                    case["mode"] = mode
                case["title"] = "Preparing queued idea: " + idea["text"][:100]
                self.store.put("/cases", case["case_id"], case)
            content = await self.structured(
                case,
                scenario_instructions(category, mode, seed["text"], idea),
                {"probe_number": number, "queued_idea": idea},
                0,
                -20,
                "scenario",
                response_schema=scenario_schema(mode),
            )
            scenario = Scenario.model_validate(json_object(content))
            if idea:
                if scenario.category not in CATEGORIES:
                    raise ValueError("Queued scenario must choose a valid task category")
                category = scenario.category
                if idea["kind"] == "prompt":
                    scenario.user_message = idea["text"]
            if mode == "chat" and scenario.files:
                raise ValueError("A chat-only scenario cannot require workspace files")
            validate_checks(scenario.checks, mode)
            ShellWorkspaceRequest.from_payload({"files": scenario.files, "commits": [], "history": [], "command": "pwd"})
            case.update(title=scenario.title, category=category)
            self.trace(case, "shared", "scenario", 0, -10, scenario.model_dump())
            candidates = [candidate["key"] for candidate in self.endpoints.metadata]
            random.shuffle(candidates)
            aliases = {chr(65 + i): key for i, key in enumerate(candidates)}
            histories = {
                key: [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            + (
                                SHELLSIM_INSTRUCTIONS + "\nUse the shell tool to inspect and modify your workspace. "
                                "Tools remain available throughout every user turn. "
                                f"Each turn has a {CANDIDATE_MAX_TOKENS}-token completion budget; "
                                f"your conversation's workspace supports {MAX_WORKSPACE_COMMANDS} shell commands "
                                f"and {MAX_WORKSPACE_HISTORY_BYTES} bytes of command history. "
                                f"ShellSim has {self.args.shellsim_cpu_limit:,} CPU units to reconstruct and execute "
                                "your workspace; CPU exhaustion terminates your chat. "
                                "Verify your result and give a public answer when the task is complete."
                                if mode == "agentic"
                                else "Answer the user's request carefully."
                            )
                        ),
                    }
                ]
                for key in candidates
            }
            self.trace(case, "shared", "user", 0, 0, {"content": scenario.user_message})
            results = await asyncio.gather(
                *[
                    self.candidate_conversation(case, scenario, idea, key, index, histories[key])
                    for index, key in enumerate(candidates)
                ],
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise errors[0]
            case["state"] = "reviewing"
            self.store.put("/cases", case["case_id"], case)
            traces = self.case_traces(case)
            reverse = {key: alias for alias, key in aliases.items()}
            review_data = {
                "scenario": scenario.model_dump(),
                "queued_idea": idea,
                "traces": [
                    {
                        "message_id": trace["message_id"],
                        "model": reverse.get(trace["model"], trace["model"]),
                        "role": trace["role"],
                        "round": trace["round"],
                        "payload": {
                            key: value
                            for key, value in trace["payload"].items()
                            if key not in {"request", "checkpoint", "native_content", "reasoning_format"}
                        },
                    }
                    for trace in traces
                    if trace["model"] != "glm"
                ],
            }
            review_text = await self.structured(
                case, review_instructions(aliases), review_data, case["round"], MAX_ROUNDS * TRACE_ROUND_STRIDE, "review"
            )
            for attempt in range(2):
                try:
                    accepted = {**validated_review(review_text, aliases, traces), "aliases": aliases}
                    if attempt:
                        accepted["previous_review_error"] = case["review"]["review_error"]
                    case["review"] = accepted
                    case["state"] = "complete"
                    break
                except (ValueError, ValidationError) as error:
                    case["review"] = {
                        "summary": "GLM review failed evidence validation",
                        "review_error": str(error),
                        "aliases": aliases,
                    }
                    case["state"] = "review_error"
                    if attempt == 0:
                        repair_data = {**review_data, "invalid_review": review_text, "validation_error": str(error)}
                        try:
                            parsed_review = Review.model_validate(json_object(review_text))
                        except ValueError:
                            sources = []
                        else:
                            sources = await asyncio.to_thread(review_repair_sources, parsed_review, aliases, traces)
                        if sources:
                            repair_data["evidence_sources"] = [source.model_dump() for source in sources]
                        review_text = await self.structured(
                            case,
                            review_instructions(aliases) + "\n" + REVIEW_REPAIR_INSTRUCTIONS,
                            repair_data,
                            case["round"],
                            MAX_ROUNDS * TRACE_ROUND_STRIDE + 1,
                            "review_repair",
                            response_schema=review_repair_schema(sources, list(aliases)),
                        )
        except TokenCapReached:
            case["state"] = "budget_limit"
            self.cap_reached = True
            self.stop.set()
        except asyncio.CancelledError:
            case["state"] = "interrupted"
            raise
        except ShellSimCpuExhausted as error:
            case["state"] = "error"
            case["review"] = {
                "summary": "Chat failed: ShellSim CPU budget exhausted; pair unscored",
                "error": str(error),
                "kind": "cpu_limit",
                "model": error.model,
            }
        except Exception as error:
            self.endpoints.checked = 0
            case["state"] = "error"
            case["review"] = {"summary": f"Probe failed: {type(error).__name__}", "error": str(error)[:1000]}
            logger.exception("Probe %s failed", case["case_id"])
            await asyncio.sleep(10)
        finally:
            if case["state"] in {"complete", "review_error"}:
                case["review"] = infrastructure_review(case["review"], self.case_traces(case))
                if "original_review" in case["review"]:
                    case["state"] = "error"
            verification = [
                {"model": trace["model"], **trace["payload"]}
                for trace in self.case_traces(case)
                if trace["role"] == "verification"
            ]
            case["review"]["verification"] = verification
            case["review"]["verification_eligible"] = (
                case["state"] != "error"
                and len(verification) == len(self.endpoints.metadata)
                and all(result["score"] is not None for result in verification)
            )
            self.active.discard(case["case_id"])
            self.store.put("/cases", case["case_id"], case)

    async def slot(self, slot: int) -> None:
        while not self.stop.is_set():
            if self.desired_state == "paused" or self.upload_error:
                await asyncio.sleep(3)
                continue
            if not await self.endpoints.refresh(self.http):
                await asyncio.sleep(READINESS_REFRESH)
                continue
            await self.conversation(slot)

    def configuration(self) -> dict:
        return {
            "candidates": self.endpoints.metadata,
            "coordinator": self.args.coordinator,
            "driver": {
                "model": "glm-5.3",
                "role": "user and independent reviewer",
                "cluster": "ortet",
                "gateway": self.endpoints.glm_gateway,
                "language_policy": GLM_LANGUAGE_INSTRUCTIONS,
                "response_format": "json_schema",
                "input_encoding": "JSON with angle brackets escaped",
                "review_repair": "quotes constrained to exact same-candidate trace excerpts",
                "shellsim_capabilities": SHELLSIM_INSTRUCTIONS,
            },
            "wildchat": {"dataset": WILDCHAT_DATASET, "revision": WILDCHAT_REVISION, "mode": "inspiration"},
            "max_rounds": MAX_ROUNDS,
            "candidate_max_tokens": CANDIDATE_MAX_TOKENS,
            "glm_max_tokens": GLM_MAX_TOKENS,
            "token_accounting": "completion tokens include reasoning; missing usage charged at request maximum",
            "token_cap": self.args.token_cap,
            "concurrency": self.args.concurrency,
            "agentic_environment": "ShellSim",
            "conversation_policy": "shared opening; independent followups per candidate",
            "tool_policy": "tools remain enabled throughout every agentic turn; no per-turn call cutoff",
            "workspace_max_commands": MAX_WORKSPACE_COMMANDS,
            "workspace_max_history_bytes": MAX_WORKSPACE_HISTORY_BYTES,
            "shellsim_cpu_limit": self.args.shellsim_cpu_limit,
            "shellsim_cpu_exhaustion": "fail chat; no followup or review; pair unscored",
            "inference_timeout": self.args.inference_timeout,
            "worker_job": self.args.worker_job,
            "reasoning_policy": (
                "Per-candidate parsing: native inline thinking or server reasoning fields; "
                "Grug thinking retained in candidate history and excluded from GLM user followups"
            ),
            "verification_policy": "checks declared before answers; shared opening only; inconclusive pairs excluded",
            "score_policy": "verified opening checks separate from exploratory GLM conversation judgments",
            "harness_revision": (
                hashlib.sha256(
                    b"".join(
                        Path(__file__).with_name(name).read_bytes()
                        for name in ("worker.py", "protocol.py", "verification.py")
                    )
                ).hexdigest()
            ),
            "version": 19,
        }

    async def run(self) -> None:
        if self.args.worker_job is not None:
            await self.restore_from_marina()
        logger.info("Waiting for pinned candidate servers")
        await self.endpoints.refresh(self.http)
        self.config = self.configuration()
        self.run_snapshot()
        await self.flush()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        heartbeat = asyncio.create_task(self.heartbeat())
        slots = [asyncio.create_task(self.slot(slot)) for slot in range(self.args.concurrency)]
        stop_waiter = asyncio.create_task(self.stop.wait())
        try:
            done, _ = await asyncio.wait([*slots, stop_waiter], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self.stop.set()
            terminal = self.cap_reached or self.desired_state == "stopped"
            if terminal:
                (self.args.state_dir / "stopped").write_text("token cap or viewer stop\n")
            for task in [heartbeat, *slots, stop_waiter]:
                task.cancel()
            await asyncio.gather(heartbeat, *slots, stop_waiter, return_exceptions=True)
            self.run_snapshot()
            try:
                while self.store.pending():
                    await self.flush()
            finally:
                await self.http.aclose()
                if terminal:
                    logger.info("Stopping the run's serving coordinator and gateway")
                    await asyncio.to_thread(self.endpoints.client.job(JobName.from_string(self.args.coordinator)).cancel)
                    gateway_job = JobName.from_string(self.endpoints.glm_gateway.rsplit("/", 1)[0])
                    await asyncio.to_thread(self.endpoints.client.job(gateway_job).cancel)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applet", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--glm-gateway", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--glm-env", required=True, type=Path)
    parser.add_argument("--token-cap", type=int, default=TOKEN_CAP)
    parser.add_argument("--concurrency", type=int, default=CONCURRENT_CHATS)
    parser.add_argument("--shellsim-cpu-limit", type=int, default=SHELLSIM_CPU_LIMIT)
    parser.add_argument("--inference-timeout", type=float, default=INFERENCE_TIMEOUT)
    parser.add_argument("--marina-url", default=APPLET_ORIGIN)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if (
        args.concurrency < 1
        or args.shellsim_cpu_limit < 1
        or args.inference_timeout <= 0
        or args.token_cap < CANDIDATE_MAX_TOKENS
    ):
        raise ValueError("Choose positive concurrency and ShellSim CPU limits and at least one candidate output budget")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if (args.state_dir / "stopped").exists():
        logger.info("Run is terminal; create a new run and state directory to start another comparison")
        return
    glm_key = dotenv_values(args.glm_env).get("GLM_API_TOKEN")
    if not glm_key:
        raise ValueError("GLM_API_TOKEN is missing from the supplied dotenv file")
    with (args.state_dir / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(args.state_dir / "state.sqlite", args.token_cap)
        store.recover()
        seeds = inspiration_seeds(args.state_dir / "wildchat.json")
        info = get_job_info()
        args.worker_job = str(info.job_id) if info is not None else None
        with ExitStack() as stack:
            if info is None:
                endpoint = stack.enter_context(connect_controller(cluster_name="marin"))
                client = IrisClient.remote(endpoint.url, credentials=endpoint.credentials)
                origin = endpoint.url
            else:
                client = iris_ctx().client
                assert client is not None and info.controller_address is not None
                origin = info.controller_address
            endpoints = Endpoints(client, origin, args.coordinator, args.glm_gateway)
            try:
                asyncio.run(Worker(args, store, endpoints, seeds, glm_key).run())
            finally:
                store.close()


if __name__ == "__main__":
    main()
