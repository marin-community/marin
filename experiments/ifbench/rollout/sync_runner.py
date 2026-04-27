# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parallel sync rollout runner - fallback when batch queues stall.

Hits chat-completion / generate APIs directly with bounded concurrency.
~10x faster than waiting for slow batch queues; pays full sync price.

Saves results in the same `Rollout` jsonl schema that the batch backends
emit, so Stage-4 extraction code consumes them identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import pathlib
from collections.abc import Iterable
from typing import Any

import httpx

from experiments.ifbench.rollout.backend import Rollout, RolloutRequest, SamplingConfig

logger = logging.getLogger(__name__)


def _gemini_payload(messages: list[dict[str, str]], sampling: SamplingConfig) -> dict[str, Any]:
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m.get("content", "")}]} for m in messages
    ]
    config: dict[str, Any] = {
        "temperature": sampling.temperature,
        "topP": sampling.top_p,
        "maxOutputTokens": sampling.max_new_tokens,
    }
    if sampling.thinking_level is not None:
        config["thinkingConfig"] = {"thinkingLevel": sampling.thinking_level}
    return {"contents": contents, "generationConfig": config}


def _together_payload(
    model_id: str, messages: list[dict[str, str]], sampling: SamplingConfig, seed: int
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_new_tokens,
        "seed": seed,
    }


async def _gemini_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, req: RolloutRequest) -> Rollout:
    async with sem:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{req.model_id}:generateContent"
        try:
            resp = await client.post(url, json=_gemini_payload(req.messages, req.sampling), timeout=300)
            if resp.status_code >= 400:
                logger.warning("Gemini sync %s: %d %s", req.prompt_id, resp.status_code, resp.text[:200])
                return _empty_rollout(req, "gemini", "http_error")
            data = resp.json()
        except Exception as e:
            logger.warning("Gemini sync %s exception: %s", req.prompt_id, e)
            return _empty_rollout(req, "gemini", str(type(e).__name__))

        cands = data.get("candidates", [])
        text = ""
        finish = None
        if cands:
            cand = cands[0]
            finish = cand.get("finishReason")
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if "text" in p)
        usage = data.get("usageMetadata", {}) or {}
        return Rollout(
            prompt_id=req.prompt_id,
            model_id=req.model_id,
            backend="gemini",
            response_text=text,
            finish_reason=finish,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            thinking_tokens=usage.get("thoughtsTokenCount"),
            seed=None,
            sampling_config_hash=req.sampling.hash_short(),
            raw_provider_metadata={"finish_reason": finish, "via": "sync"},
        )


async def _together_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, req: RolloutRequest) -> Rollout:
    async with sem:
        try:
            resp = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                json=_together_payload(req.model_id, req.messages, req.sampling, req.seed),
                timeout=300,
            )
            if resp.status_code >= 400:
                logger.warning("Together sync %s: %d %s", req.prompt_id, resp.status_code, resp.text[:200])
                return _empty_rollout(req, "together", "http_error")
            data = resp.json()
        except Exception as e:
            logger.warning("Together sync %s exception: %s", req.prompt_id, e)
            return _empty_rollout(req, "together", str(type(e).__name__))

        choices = data.get("choices", [])
        text = ""
        finish = None
        if choices:
            text = choices[0].get("message", {}).get("content", "") or ""
            finish = choices[0].get("finish_reason")
        usage = data.get("usage", {}) or {}
        return Rollout(
            prompt_id=req.prompt_id,
            model_id=req.model_id,
            backend="together",
            response_text=text,
            finish_reason=finish,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            thinking_tokens=None,
            seed=req.seed,
            sampling_config_hash=req.sampling.hash_short(),
            raw_provider_metadata={"finish_reason": finish, "via": "sync"},
        )


def _empty_rollout(req: RolloutRequest, backend: str, finish: str) -> Rollout:
    return Rollout(
        prompt_id=req.prompt_id,
        model_id=req.model_id,
        backend=backend,
        response_text="",
        finish_reason=finish,
        input_tokens=None,
        output_tokens=None,
        thinking_tokens=None,
        seed=None,
        sampling_config_hash=req.sampling.hash_short(),
        raw_provider_metadata={"finish_reason": finish, "via": "sync"},
    )


async def run_sync_batch(requests: list[RolloutRequest], backend: str, concurrency: int = 10) -> list[Rollout]:
    """Run all requests in parallel against `backend` with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    if backend == "gemini":
        headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    elif backend == "together":
        headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}
    else:
        raise ValueError(f"unknown backend {backend}")

    async with httpx.AsyncClient(headers=headers, timeout=300) as client:
        worker = _gemini_one if backend == "gemini" else _together_one
        tasks = [asyncio.create_task(worker(client, sem, r)) for r in requests]
        rollouts: list[Rollout] = []
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            r = await t
            rollouts.append(r)
            if i % 25 == 0 or i == len(tasks):
                logger.info("%s sync progress: %d/%d", backend, i, len(tasks))
        return rollouts


def write_rollouts_jsonl(rollouts: Iterable[Rollout], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")
