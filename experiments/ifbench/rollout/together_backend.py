# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Together AI Batch API implementation of `RolloutBackend`.

Together's batch flow (50% discount, 24-hour async):

  1. Files API upload (3-step: metadata POST → PUT to signed url → callback POST).
     We delegate this to the official `together` SDK because the 3-step dance is
     non-trivial and security-sensitive (S3 redirect URLs).
  2. POST /v1/batches (file_id, endpoint=/v1/chat/completions) → batch_id
  3. GET  /v1/batches/{batch_id}                              → status
  4. GET  /v1/files/{output_file_id}/content                  → jsonl results

For (2)-(4) we use plain httpx since the surface is small.

Reads `TOGETHER_API_KEY` from os.environ. Never logs the key.
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
from together import Together  # type: ignore[import-untyped]

from experiments.ifbench.rollout.backend import (
    BatchHandle,
    BatchStatus,
    Rollout,
    RolloutRequest,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.together.xyz/v1"
_TOGETHER_TO_NORMALISED_STATUS: dict[str, BatchStatus] = {
    "VALIDATING": BatchStatus.PENDING,
    "IN_PROGRESS": BatchStatus.PENDING,
    "QUEUED": BatchStatus.PENDING,
    "FINALIZING": BatchStatus.PENDING,
    "COMPLETED": BatchStatus.COMPLETED,
    "FAILED": BatchStatus.FAILED,
    "CANCELLED": BatchStatus.CANCELLED,
    "EXPIRED": BatchStatus.EXPIRED,
}


def _api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set in environment")
    return key


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        base_url=_BASE_URL,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=timeout,
    )


def _sdk_client() -> Together:
    """Together SDK; pulls TOGETHER_API_KEY from env automatically."""
    # Set TOGETHER_DISABLE_TQDM so the upload doesn't dump progress bars to
    # stderr in non-interactive runs.
    os.environ.setdefault("TOGETHER_DISABLE_TQDM", "true")
    return Together()


def _request_to_jsonl_line(req: RolloutRequest) -> dict[str, Any]:
    """Format one RolloutRequest as a Together-batch jsonl line."""
    body = {
        "model": req.model_id,
        "messages": req.messages,
        "temperature": req.sampling.temperature,
        "top_p": req.sampling.top_p,
        "max_tokens": req.sampling.max_new_tokens,
        "seed": req.seed,
    }
    # Together's custom_id cap is 64 chars. Our natural id can be longer.
    custom_id = f"{req.prompt_id}:{req.seed}"
    if len(custom_id) > 64:
        custom_id = hashlib.sha1(custom_id.encode()).hexdigest()
    return {"custom_id": custom_id, "body": body}


@dataclasses.dataclass
class TogetherBackend:
    """Together batch backend. Stateless across calls; uses os.environ for auth."""

    name: str = "together"

    def submit_batch(
        self,
        model_id: str,
        requests: Iterable[RolloutRequest],
        jsonl_dir: pathlib.Path,
    ) -> BatchHandle:
        reqs = list(requests)
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        # Stable filename so reruns under the same content are easy to identify.
        fingerprint = hashlib.sha1(
            json.dumps([dataclasses.asdict(r) for r in reqs], sort_keys=True).encode()
        ).hexdigest()[:12]
        jsonl_path = jsonl_dir / f"together_{model_id.replace('/', '_')}_{fingerprint}.jsonl"
        with jsonl_path.open("w") as f:
            for r in reqs:
                f.write(json.dumps(_request_to_jsonl_line(r)) + "\n")

        # 1) Upload via SDK (handles the 3-step protocol).
        sdk = _sdk_client()
        upload_resp = sdk.files.upload(file=jsonl_path, purpose="batch-api", check=False)
        file_id = upload_resp.id
        logger.info("Uploaded %s → file_id=%s (%d requests)", jsonl_path.name, file_id, len(reqs))

        # 2) Create batch via plain httpx (small surface).
        with _client() as client:
            batch_resp = client.post(
                "/batches",
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            if batch_resp.status_code >= 400:
                logger.error("Together batch create failed: %s %s", batch_resp.status_code, batch_resp.text[:500])
                batch_resp.raise_for_status()
            batch = batch_resp.json()
            # Together's CREATE response wraps the batch as {"job": {...}}
            # while GET returns the flat form. Handle both.
            inner = batch.get("job") if isinstance(batch.get("job"), dict) else batch
            batch_id = inner.get("id") or inner.get("batch_id")
            if not batch_id:
                logger.error("Together batch response had no id field. Body: %s", batch)
                raise RuntimeError(f"could not extract batch id from response: {batch}")
            logger.info("Submitted batch %s (model=%s, n=%d)", batch_id, model_id, len(reqs))

        return BatchHandle(
            backend=self.name,
            batch_id=batch_id,
            model_id=model_id,
            submitted_at_iso=datetime.datetime.now(datetime.UTC).isoformat(),
            expected_request_count=len(reqs),
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        with _client() as client:
            resp = client.get(f"/batches/{handle.batch_id}")
            resp.raise_for_status()
            payload = resp.json()
        upstream_status = (payload.get("status") or "").upper()
        normalised = _TOGETHER_TO_NORMALISED_STATUS.get(upstream_status, BatchStatus.PENDING)
        if upstream_status not in _TOGETHER_TO_NORMALISED_STATUS:
            logger.warning("Unknown Together batch status %r, treating as PENDING", upstream_status)
        completed = payload.get("request_counts", {}).get("completed", "?")
        total = payload.get("request_counts", {}).get("total", "?")
        progress = payload.get("progress")
        if progress is not None:
            logger.info("Batch %s: %s (progress=%s%%)", handle.batch_id, upstream_status, progress)
        else:
            logger.info("Batch %s: %s (%s/%s done)", handle.batch_id, upstream_status, completed, total)
        return normalised

    def download(self, handle: BatchHandle) -> Iterable[Rollout]:
        with _client(timeout=300.0) as client:
            # Need to re-fetch batch to learn the output_file_id
            batch_resp = client.get(f"/batches/{handle.batch_id}")
            batch_resp.raise_for_status()
            batch = batch_resp.json()
            output_file_id = batch.get("output_file_id")
            error_file_id = batch.get("error_file_id")
            if not output_file_id:
                raise RuntimeError(
                    f"Together batch {handle.batch_id} has no output_file_id "
                    f"(status={batch.get('status')}, error_file_id={error_file_id})"
                )
            content_resp = client.get(f"/files/{output_file_id}/content")
            content_resp.raise_for_status()
            text = content_resp.text

        seen = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            custom_id = row.get("custom_id", "")
            # Custom_id format: "{prompt_id}:{seed}" or sha1 if it was too long.
            if ":" in custom_id and len(custom_id) <= 64:
                prompt_id, _, seed_str = custom_id.partition(":")
                seed: int | None = int(seed_str) if seed_str.isdigit() else None
            else:
                prompt_id = custom_id
                seed = None

            error = row.get("error")
            response = row.get("response", {})
            body = response.get("body", {}) if isinstance(response, dict) else {}
            choices = body.get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            finish_reason = choices[0].get("finish_reason") if choices else None
            usage = body.get("usage", {})

            if error:
                logger.warning("Together row error for %s: %s", custom_id, error)

            yield Rollout(
                prompt_id=prompt_id,
                model_id=handle.model_id,
                backend=self.name,
                response_text=content,
                finish_reason=finish_reason,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                thinking_tokens=None,  # Together batch on these models has no reasoning
                seed=seed,
                sampling_config_hash="",  # filled in by Stage-2 caller, not provider
                raw_provider_metadata={"finish_reason": finish_reason, "error": error},
            )

        if seen < handle.expected_request_count:
            logger.warning(
                "Together batch %s under-delivered: got %d, expected %d",
                handle.batch_id,
                seen,
                handle.expected_request_count,
            )


def sync_complete(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """One-shot sync chat completion. For key-validity smoke tests only.

    Returns the raw response body. Caller decides what to extract.
    """
    with _client() as client:
        resp = client.post(
            "/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
            },
        )
        resp.raise_for_status()
        return resp.json()
