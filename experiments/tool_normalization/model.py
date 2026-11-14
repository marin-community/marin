# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Receipt-bound structured requests with durable, explicitly allocated capacity."""

import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jsonschema
from transformers import PreTrainedTokenizerBase

from experiments.tool_normalization.review_encoding import ENCODING_INSTRUCTIONS, encode_payload

RETRY_BACKOFF_BASE_SECONDS = 2.0
RETRY_BACKOFF_CAP_SECONDS = 120.0
RETRY_JITTER_FRACTION = 0.2


class TransientRequestError(RuntimeError):
    """Relay-side or incomplete failure that a retry can resolve."""


class PermanentRequestError(RuntimeError):
    """Failure a retry cannot resolve (auth, missing endpoint, deterministic overflow)."""


class CapacityUnavailable(RuntimeError):
    """No reserved inference slot can be used without exceeding the campaign allocation."""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    temperature: float
    top_p: float
    thinking_token_budget: int
    generation: str
    api_key_file: Path
    context_tokens: int
    output_tokens: int
    request_timeout: int
    capacity_file: Path
    ledger: Path


def initialize_ledger(config: ModelConfig) -> None:
    """Create the admission ledger before concurrent workers start."""
    config.ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.ledger, timeout=60) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, generation TEXT, state TEXT, allocation_sha TEXT)"
        )


def reserve(config: ModelConfig, identity: str) -> None:
    allocation = json.loads(config.capacity_file.read_text())
    if allocation["expires_at_unix"] <= time.time():
        raise CapacityUnavailable("Capacity allocation expired")
    if allocation["generation"] != config.generation:
        raise ValueError("Capacity allocated to another endpoint generation")
    slots = allocation["conversion_slots"]
    if slots < 1 or slots + allocation["science_slots"] > 32:
        raise ValueError("Invalid campaign capacity partition")
    with sqlite3.connect(config.ledger, timeout=60) as db:
        db.execute("BEGIN IMMEDIATE")
        occupied = db.execute("SELECT count(*) FROM requests WHERE state IN ('active','uncertain')").fetchone()[0]
        if occupied >= slots:
            raise CapacityUnavailable("Allocated request capacity is occupied")
        db.execute(
            "INSERT INTO requests VALUES (?,?,?,?)",
            (identity, config.generation, "active", hashlib.sha256(config.capacity_file.read_bytes()).hexdigest()),
        )


def complete(
    config: ModelConfig,
    tokenizer: PreTrainedTokenizerBase,
    directory: Path,
    system: str,
    payload: dict,
    schema: dict,
    *,
    reasoning_effort: str,
) -> dict:
    """Make one receipt-bound request, retrying transient relay failures with capped backoff until success."""
    # Trace delimiters are data, not the serving model's own special control tokens.
    encoded_payload = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": encoded_payload},
    ]
    template_options = {"reasoning_effort": reasoning_effort, "clear_thinking": True}
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=True, **template_options
    )["input_ids"]
    if len(prompt_ids) + config.output_tokens > config.context_tokens:
        compact = json.dumps(encode_payload(payload), ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        messages = [
            {"role": "system", "content": system + ENCODING_INSTRUCTIONS},
            {"role": "user", "content": compact},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, add_generation_prompt=True, **template_options
        )["input_ids"]
    if len(prompt_ids) + config.output_tokens > config.context_tokens:
        raise ValueError("length_deferred: full request and output reserve exceed served context")
    body = {
        "model": config.model,
        "cache_salt": config.generation,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "thinking_token_budget": config.thinking_token_budget,
        "max_tokens": config.output_tokens,
        "stream": True,
        "tool_choice": "none",
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template_options,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "conversion", "strict": True, "schema": schema},
        },
    }
    directory.mkdir(parents=True, exist_ok=False)
    request_file = directory / "request.json"
    request_file.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    identity = str(directory.resolve())
    reserve(config, identity)
    receipt = {
        "generation": config.generation,
        "model": config.model,
        "base_url": config.base_url,
        "started_at_unix": time.time(),
        "native_prompt_tokens": len(prompt_ids),
        "request_sha256": hashlib.sha256(request_file.read_bytes()).hexdigest(),
        "finish_reason": None,
        "done": False,
        "content": "",
        "reasoning": "",
        "usage": None,
    }
    raw_file = directory / "response.sse"
    terminal = False
    backoff = RETRY_BACKOFF_BASE_SECONDS
    try:
        while True:
            # Reset per-attempt state; partial artifacts from a failed attempt must not survive.
            receipt.update(
                {
                    "http_status": None,
                    "finish_reason": None,
                    "done": False,
                    "content": "",
                    "reasoning": "",
                    "usage": None,
                }
            )
            raw_file.unlink(missing_ok=True)
            try:
                with raw_file.open("xb") as raw, httpx.Client(timeout=config.request_timeout, trust_env=False) as client:
                    (directory / "post-started.json").write_text(json.dumps({"at": time.time()}))
                    with client.stream(
                        "POST",
                        config.base_url.rstrip("/") + "/chat/completions",
                        json=body,
                        headers={"Authorization": "Bearer " + config.api_key_file.read_text().strip()},
                    ) as response:
                        receipt["http_status"] = response.status_code
                        if response.status_code == 429 or response.status_code >= 500:
                            raw.write(response.read())
                            raise TransientRequestError(f"relay returned HTTP {response.status_code}")
                        if response.status_code != 200:
                            raw.write(response.read())
                            terminal = True
                            raise PermanentRequestError(f"endpoint returned HTTP {response.status_code}")
                        for line in response.iter_lines():
                            raw.write((line + "\n").encode())
                            raw.flush()
                            if not line.startswith("data:"):
                                continue
                            data = line.removeprefix("data:").strip()
                            if data == "[DONE]":
                                receipt["done"] = True
                                continue
                            event = json.loads(data)
                            if event.get("usage"):
                                receipt["usage"] = event["usage"]
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                receipt["content"] += delta.get("content") or ""
                                receipt["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                                if choice.get("finish_reason"):
                                    receipt["finish_reason"] = choice["finish_reason"]
                                    terminal = True
                if not receipt["done"] or receipt["finish_reason"] != "stop":
                    raise TransientRequestError("Incomplete model result")
                value = json.loads(receipt["content"])
                jsonschema.validate(value, schema)
                (directory / "value.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
                return value
            except PermanentRequestError:
                raise
            except Exception:
                # Relay errors, upstream outages, timeouts, malformed streams, and incomplete
                # results: wait and retry. Only PermanentRequestError (auth, missing endpoint,
                # deterministic overflow) fails fast; everything else retries until success so a
                # transient fault can never leak an uncertain lease or defer a row.
                pause = backoff * (1.0 + random.uniform(-RETRY_JITTER_FRACTION, RETRY_JITTER_FRACTION))
                time.sleep(pause)
                backoff = min(RETRY_BACKOFF_CAP_SECONDS, backoff * 2)
    finally:
        receipt["completed_at_unix"] = time.time()
        receipt["raw_sha256"] = hashlib.sha256(raw_file.read_bytes()).hexdigest() if raw_file.exists() else None
        receipt["lease_state"] = "terminal" if terminal else "uncertain"
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        with sqlite3.connect(config.ledger, timeout=60) as db:
            db.execute("UPDATE requests SET state=? WHERE id=?", (receipt["lease_state"], identity))
