# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Google Gemini Batch API implementation of `RolloutBackend`.

Gemini's batch flow (50% discount, 24h SLA, often quicker):

  Inline mode:
    1. POST .../models/{model}:batchGenerateContent  → operation name
    2. GET  /v1beta/{operation_name}                 → operation status / response
    3. Operation response.inlinedResponses[*]         contains per-request results

  File mode:
    1. Upload JSONL via Files API
    2. POST .../models/{model}:batchGenerateContent with input_config.file_name
    3. GET /v1beta/{batch_name} until terminal
    4. GET /download/v1beta/{dest.fileName}:download?alt=media for JSONL results

Inline mode is convenient for 100-prompt smoke jobs. File mode is what the
full run needs and mirrors the public Gemini batch docs.

Reads `GEMINI_API_KEY` from os.environ. Never logs the key.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import os
import pathlib
from collections.abc import Iterable
from typing import Any

import httpx

from experiments.ifbench.rollout.backend import (
    BatchHandle,
    BatchStatus,
    Rollout,
    RolloutRequest,
    SamplingConfig,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_UPLOAD_BASE_URL = "https://generativelanguage.googleapis.com/upload/v1beta"
_DOWNLOAD_BASE_URL = "https://generativelanguage.googleapis.com/download/v1beta"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return key


def _client(timeout: float = 60.0) -> httpx.Client:
    # x-goog-api-key header keeps the key out of URL logs.
    return httpx.Client(
        base_url=_BASE_URL,
        headers={"x-goog-api-key": _api_key()},
        timeout=timeout,
    )


def _upload_client(timeout: float = 300.0) -> httpx.Client:
    return httpx.Client(
        base_url=_UPLOAD_BASE_URL,
        headers={"x-goog-api-key": _api_key()},
        timeout=timeout,
    )


def _download_client(timeout: float = 300.0) -> httpx.Client:
    return httpx.Client(
        base_url=_DOWNLOAD_BASE_URL,
        headers={"x-goog-api-key": _api_key()},
        timeout=timeout,
    )


def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert chat-format messages into Gemini's contents list.

    Gemini uses 'user' / 'model' roles (not 'assistant'). For our IFBench
    rollouts every prompt is single-turn (just user), so this is trivial.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        out.append({"role": role, "parts": [{"text": m.get("content", "")}]})
    return out


def _request_to_gemini_inline(req: RolloutRequest) -> dict[str, Any]:
    """One inline request entry for batchGenerateContent.

    Gemini's inline-batch format wraps a `GenerateContentRequest` in an
    `InlinedRequest` envelope: `{"request": ..., "metadata": ...}`.
    The `{"key": ...}` envelope shown in some docs is the FILE-based JSONL
    format only; inline submissions reject that field.

    We stuff our correlation id into `metadata` so the API echoes it back on
    each `InlinedResponse`.
    """
    generation_config: dict[str, Any] = {
        "temperature": req.sampling.temperature,
        "topP": req.sampling.top_p,
        "maxOutputTokens": req.sampling.max_new_tokens,
    }
    if req.sampling.thinking_level is not None:
        # Gemini 3.x uses `thinkingConfig.thinkingLevel` ("minimal"|"low"|"medium"|"high")
        generation_config["thinkingConfig"] = {"thinkingLevel": req.sampling.thinking_level}
    inner = {
        "contents": _messages_to_gemini_contents(req.messages),
        "generationConfig": generation_config,
    }
    return {
        "request": inner,
        "metadata": {"correlation_id": f"{req.prompt_id}:{req.seed}"},
    }


def _request_to_gemini_file_line(req: RolloutRequest) -> dict[str, Any]:
    """One JSONL row for Gemini file-batch mode.

    File-batch rows use the public docs' `{"key": ..., "request": ...}` shape,
    unlike inline mode's `{"request": ..., "metadata": ...}` envelope.

    As of 2026-04-27, Gemini file-batch rejects `thinkingConfig` inside a row
    with INVALID_ARGUMENT even though sync and inline generation accept it.
    Omit it here; Gemini 3 Flash then uses its default dynamic/high thinking.
    """
    inline = _request_to_gemini_inline(req)
    generation_config = dict(inline["request"].get("generationConfig", {}))
    generation_config.pop("thinkingConfig", None)
    request = dict(inline["request"])
    request["generationConfig"] = generation_config
    return {
        "key": f"{req.prompt_id}:{req.seed}",
        "request": request,
    }


def _upload_jsonl(path: pathlib.Path, display_name: str) -> str:
    """Upload a JSONL file to Gemini Files API and return `files/...` name."""
    size = path.stat().st_size
    mime_type = "application/jsonl"
    with _upload_client() as client:
        start = client.post(
            "/files",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": display_name}},
        )
        if start.status_code >= 400:
            logger.error("Gemini file upload start failed: %s %s", start.status_code, start.text[:500])
            start.raise_for_status()
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Gemini file upload start did not return x-goog-upload-url")

        finish = httpx.post(
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=path.read_bytes(),
            timeout=300.0,
        )
        if finish.status_code >= 400:
            logger.error("Gemini file upload finalize failed: %s %s", finish.status_code, finish.text[:500])
            finish.raise_for_status()
        payload = finish.json()
    file_name = payload.get("file", {}).get("name")
    if not file_name:
        raise RuntimeError(f"Gemini file upload returned no file.name: {payload}")
    return file_name


@dataclasses.dataclass
class GeminiBackend:
    """Gemini batch backend. Reads GEMINI_API_KEY from os.environ."""

    name: str = "gemini"
    use_file_batch: bool = False

    def submit_batch(
        self,
        model_id: str,
        requests: Iterable[RolloutRequest],
        jsonl_dir: pathlib.Path,
    ) -> BatchHandle:
        reqs = list(requests)
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha1(
            json.dumps([dataclasses.asdict(r) for r in reqs], sort_keys=True).encode()
        ).hexdigest()[:12]
        jsonl_path = jsonl_dir / f"gemini_{model_id.replace('/', '_')}_{fingerprint}.jsonl"
        with jsonl_path.open("w") as f:
            for r in reqs:
                row = _request_to_gemini_file_line(r) if self.use_file_batch else _request_to_gemini_inline(r)
                f.write(json.dumps(row) + "\n")

        if self.use_file_batch:
            display_name = f"ifbench-{model_id.replace('/', '-')}-{fingerprint}"
            file_name = _upload_jsonl(jsonl_path, display_name)
            logger.info("Uploaded Gemini batch file %s → %s (%d requests)", jsonl_path.name, file_name, len(reqs))
            with _client(timeout=120.0) as client:
                resp = client.post(
                    f"/models/{model_id}:batchGenerateContent",
                    json={
                        "batch": {
                            "display_name": display_name,
                            "input_config": {"file_name": file_name},
                        },
                    },
                )
                if resp.status_code >= 400:
                    logger.error("Gemini file batch submit failed: %s %s", resp.status_code, resp.text[:500])
                    resp.raise_for_status()
                payload = resp.json()
            batch_name = payload.get("name") or payload.get("metadata", {}).get("name")
            if not batch_name:
                raise RuntimeError(f"Gemini file batch submit returned no batch name: {payload}")
            logger.info("Submitted Gemini file batch %s (model=%s, n=%d)", batch_name, model_id, len(reqs))
            return BatchHandle(
                backend=self.name,
                batch_id=batch_name,
                model_id=model_id,
                submitted_at_iso=datetime.datetime.now(datetime.UTC).isoformat(),
                expected_request_count=len(reqs),
            )

        # Inline batch limit is 20MB; check.
        size = jsonl_path.stat().st_size
        if size > 19_000_000:
            raise RuntimeError(
                f"Gemini inline batch too large: {size} bytes > 19MB limit. "
                "Switch to file-based batch (not implemented yet)."
            )

        # Build the inline payload
        inline_requests = [_request_to_gemini_inline(r) for r in reqs]
        with _client(timeout=120.0) as client:
            resp = client.post(
                f"/models/{model_id}:batchGenerateContent",
                json={
                    "batch": {
                        "displayName": f"ifbench-smoke-{fingerprint}",
                        "inputConfig": {"requests": {"requests": inline_requests}},
                    },
                },
            )
            if resp.status_code >= 400:
                logger.error("Gemini batch submit failed: %s %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
        # Long-running operation: response includes a name like "batches/abc123".
        batch_name = payload.get("name") or payload.get("metadata", {}).get("name")
        if not batch_name:
            raise RuntimeError(f"Gemini submit returned no batch name: {payload}")
        logger.info("Submitted Gemini batch %s (model=%s, n=%d)", batch_name, model_id, len(reqs))

        return BatchHandle(
            backend=self.name,
            batch_id=batch_name,
            model_id=model_id,
            submitted_at_iso=datetime.datetime.now(datetime.UTC).isoformat(),
            expected_request_count=len(reqs),
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        with _client() as client:
            # batch_id is a full resource name like "batches/abc123" — query directly.
            resp = client.get(f"/{handle.batch_id}")
            resp.raise_for_status()
            payload = resp.json()
        # Long-running operation states.
        state = (payload.get("metadata", {}).get("state") or payload.get("state") or "").upper()
        done = payload.get("done") is True
        completed = payload.get("metadata", {}).get("completedRequestCount", "?")
        total = payload.get("metadata", {}).get("requestCount", "?")
        logger.info("Gemini batch %s: state=%s done=%s (%s/%s)", handle.batch_id, state, done, completed, total)
        if done:
            error = payload.get("error")
            if error:
                logger.warning("Gemini batch %s done with error: %s", handle.batch_id, error)
                return BatchStatus.FAILED
            return BatchStatus.COMPLETED
        if state in ("BATCH_STATE_FAILED", "FAILED"):
            return BatchStatus.FAILED
        if state in ("BATCH_STATE_CANCELLED", "CANCELLED"):
            return BatchStatus.CANCELLED
        if state in ("BATCH_STATE_EXPIRED", "EXPIRED"):
            return BatchStatus.EXPIRED
        return BatchStatus.PENDING

    def download(self, handle: BatchHandle) -> Iterable[Rollout]:
        with _client(timeout=300.0) as client:
            resp = client.get(f"/{handle.batch_id}")
            resp.raise_for_status()
            payload = resp.json()
        if not payload.get("done"):
            raise RuntimeError(f"Gemini batch {handle.batch_id} not done yet")

        response_obj = payload.get("response", {})
        inlined = (
            response_obj.get("inlinedResponses", {}).get("inlinedResponses")
            or response_obj.get("inlined_responses", {}).get("inlined_responses")
            or []
        )

        if not inlined:
            file_name = (
                payload.get("dest", {}).get("fileName")
                or payload.get("dest", {}).get("file_name")
                or response_obj.get("responsesFile")
                or response_obj.get("responses_file")
            )
            if file_name:
                yield from _download_file_batch_rollouts(file_name, handle)
                return

        seen = 0
        for entry in inlined:
            seen += 1
            metadata = entry.get("metadata", {}) or {}
            corr_id = metadata.get("correlation_id", "") or entry.get("key", "")
            if ":" in corr_id:
                prompt_id, _, _ = corr_id.partition(":")
            else:
                prompt_id = corr_id

            response = entry.get("response", {})
            error = entry.get("error")
            text = ""
            finish_reason: str | None = None
            input_tokens: int | None = None
            output_tokens: int | None = None
            thinking_tokens: int | None = None
            if response:
                candidates = response.get("candidates", [])
                if candidates:
                    cand = candidates[0]
                    finish_reason = cand.get("finishReason")
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts if "text" in p)
                usage = response.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount")
                output_tokens = usage.get("candidatesTokenCount")
                thinking_tokens = usage.get("thoughtsTokenCount")

            if error:
                logger.warning("Gemini row error for %s: %s", corr_id, error)

            yield Rollout(
                prompt_id=prompt_id,
                model_id=handle.model_id,
                backend=self.name,
                response_text=text,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                seed=None,  # Gemini doesn't echo a seed; sampling is non-deterministic
                sampling_config_hash="",
                raw_provider_metadata={"finish_reason": finish_reason, "error": error},
            )

        if seen < handle.expected_request_count:
            logger.warning(
                "Gemini batch %s under-delivered: got %d, expected %d",
                handle.batch_id,
                seen,
                handle.expected_request_count,
            )


def _download_file_batch_rollouts(file_name: str, handle: BatchHandle) -> Iterable[Rollout]:
    with _download_client(timeout=300.0) as client:
        resp = client.get(f"/{file_name}:download", params={"alt": "media"})
        if resp.status_code >= 400:
            logger.error("Gemini file batch download failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        text = resp.text

    seen = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen += 1
        corr_id = row.get("key", "")
        if ":" in corr_id:
            prompt_id, _, seed_str = corr_id.partition(":")
            seed: int | None = int(seed_str) if seed_str.isdigit() else None
        else:
            prompt_id = corr_id
            seed = None

        response = row.get("response", {})
        error = row.get("error")
        text_out = ""
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        thinking_tokens: int | None = None
        if response:
            candidates = response.get("candidates", [])
            if candidates:
                cand = candidates[0]
                finish_reason = cand.get("finishReason")
                parts = cand.get("content", {}).get("parts", [])
                text_out = "".join(p.get("text", "") for p in parts if "text" in p)
            usage = response.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount")
            output_tokens = usage.get("candidatesTokenCount")
            thinking_tokens = usage.get("thoughtsTokenCount")
        if error:
            logger.warning("Gemini file row error for %s: %s", corr_id, error)

        yield Rollout(
            prompt_id=prompt_id,
            model_id=handle.model_id,
            backend="gemini",
            response_text=text_out,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            seed=seed,
            sampling_config_hash="",
            raw_provider_metadata={"finish_reason": finish_reason, "error": error, "file_name": file_name},
        )

    if seen < handle.expected_request_count:
        logger.warning(
            "Gemini file batch %s under-delivered: got %d, expected %d",
            handle.batch_id,
            seen,
            handle.expected_request_count,
        )


def sync_generate(
    model_id: str, messages: list[dict[str, str]], *, sampling: SamplingConfig | None = None
) -> dict[str, Any]:
    """One-shot sync generation. For key-validity smoke tests only.

    Returns the raw response body.
    """
    sampling = sampling or SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=256)
    payload: dict[str, Any] = {
        "contents": _messages_to_gemini_contents(messages),
        "generationConfig": {
            "temperature": sampling.temperature,
            "topP": sampling.top_p,
            "maxOutputTokens": sampling.max_new_tokens,
        },
    }
    if sampling.thinking_level is not None:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": sampling.thinking_level}
    with _client() as client:
        resp = client.post(f"/models/{model_id}:generateContent", json=payload)
        if resp.status_code >= 400:
            logger.error("Gemini sync generate failed: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        return resp.json()
