# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Receipt-bound structured requests with durable, explicitly allocated capacity."""

import hashlib
import json
import os
import random
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jsonschema
from transformers import PreTrainedTokenizerBase

from experiments.tool_normalization.pool_manifest import PoolManifestError, fetch_credential
from experiments.tool_normalization.relay import resolve_relay_base_url
from experiments.tool_normalization.review_encoding import ENCODING_INSTRUCTIONS, encode_payload

RETRY_BACKOFF_BASE_SECONDS = 2.0
RETRY_BACKOFF_CAP_SECONDS = 120.0
RETRY_JITTER_FRACTION = 0.2
POOL_KEY_REFRESH_LIMIT = 2


class TransientRequestError(RuntimeError):
    """Relay-side or incomplete failure that a retry can resolve."""


class PermanentRequestError(RuntimeError):
    """Failure a retry cannot resolve (auth, missing endpoint, deterministic overflow)."""


class LengthDeferred(ValueError):
    """The request exceeds the prompt or output token budget."""


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
    max_attempts: int
    stop_file: Path
    deadline_unix: float
    relay_job: str | None
    # Pool manifest entries as (base_url, api_key_file, api_key_ref) triples; empty on the
    # relay path. Requests round-robin across entries and rotate on per-entry registry
    # 404/connect/5xx (2026-11-24 serving handoff contract).
    pool_entries: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self):
        # Config JSON round-trips entries as lists; normalize to tuples once.
        if self.pool_entries and any(isinstance(entry, list) for entry in self.pool_entries):
            object.__setattr__(self, "pool_entries", tuple(tuple(entry) for entry in self.pool_entries))


_POOL_LOCK = threading.Lock()
_POOL_COUNTER = 0
_POOL_KEY_REFRESHES: dict[str, int] = {}
_LAST_ENTRY: tuple[str, str, str] | None = None


def _select_entry(config: ModelConfig) -> tuple[str, Path]:
    """Pick the next pool entry (round-robin with a deterministic per-pod offset).

    Each call advances the rotation so a retry after a transient failure lands on the
    next validated entry, per the 2026-11-24 serving handoff contract.
    """
    global _POOL_COUNTER, _LAST_ENTRY
    if not config.pool_entries:
        return config.base_url, config.api_key_file
    offset = int(os.environ.get("GLM_POOL_RR_OFFSET", "0") or 0)
    with _POOL_LOCK:
        index = (offset + _POOL_COUNTER) % len(config.pool_entries)
        _POOL_COUNTER += 1
        _LAST_ENTRY = config.pool_entries[index]
    base_url, key_file, _ = _LAST_ENTRY
    return base_url, Path(key_file)


def refresh_pool_key(config: ModelConfig) -> None:
    """Re-fetch the just-401'd entry's credential ref; fatal once the bound is spent."""
    if not config.pool_entries or _LAST_ENTRY is None:
        raise PermanentRequestError("401 without a pool credential to refresh")
    _, key_file, key_ref = _LAST_ENTRY
    used = _POOL_KEY_REFRESHES.get(key_ref, 0)
    if used >= POOL_KEY_REFRESH_LIMIT:
        raise PermanentRequestError(f"401 persists after {used} credential refreshes of {key_ref}")
    try:
        fetch_credential(key_ref, Path(key_file))
    except PoolManifestError as error:
        raise PermanentRequestError(str(error)) from error
    _POOL_KEY_REFRESHES[key_ref] = used + 1


def parse_model_config(values: dict) -> ModelConfig:
    """Normalize paths from a serialized model configuration."""
    values = values.copy()
    for key in ("api_key_file", "capacity_file", "ledger", "stop_file"):
        values[key] = Path(values[key])
    return ModelConfig(**values)


def check_allocation(config: ModelConfig, deadline: float) -> None:
    if config.stop_file.exists() or time.time() >= deadline:
        raise CapacityUnavailable("Request stopped or allocation deadline reached")


def retry_pause(config: ModelConfig, seconds: float) -> None:
    resume_at = time.time() + seconds
    while time.time() < resume_at:
        check_allocation(config, config.deadline_unix)
        time.sleep(max(0.0, min(1.0, resume_at - time.time())))
    check_allocation(config, config.deadline_unix)


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
    """Make a receipt-bound request with bounded retries and allocation cancellation."""
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    check_allocation(config, config.deadline_unix)
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
        raise LengthDeferred("length_deferred: full request and output reserve exceed served context")
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
        "attempts": [],
    }
    raw_file = directory / "response.sse"
    terminal = False
    backoff = RETRY_BACKOFF_BASE_SECONDS
    blank_retries = 0
    try:
        for attempt in range(config.max_attempts):
            check_allocation(config, config.deadline_unix)
            terminal = False
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
            raw_file = directory / f"response-{attempt + 1}.sse"
            if config.pool_entries:
                # Pool mode: rotate entries per attempt (retry lands on the next validated
                # entry). Relay mode: keep receipt["base_url"], which the retry path may
                # have re-resolved after a relay move.
                base_url, key_file = _select_entry(config)
                receipt["base_url"] = base_url
            else:
                base_url, key_file = receipt["base_url"], config.api_key_file
            attempt_receipt = {"base_url": receipt["base_url"], "started_at_unix": time.time()}
            receipt["attempts"].append(attempt_receipt)
            request_deadline = min(config.deadline_unix, time.time() + config.request_timeout)
            try:
                with (
                    raw_file.open("xb") as raw,
                    httpx.Client(timeout=min(30.0, request_deadline - time.time()), trust_env=False) as client,
                ):
                    (directory / "post-started.json").write_text(json.dumps({"at": time.time()}))
                    with client.stream(
                        "POST",
                        base_url.rstrip("/") + "/chat/completions",
                        json=body,
                        headers={"Authorization": "Bearer " + key_file.read_text().strip()},
                    ) as response:
                        receipt["http_status"] = response.status_code
                        if response.status_code == 429 or response.status_code >= 500:
                            raw.write(response.read())
                            raise TransientRequestError(f"relay returned HTTP {response.status_code}")
                        if response.status_code == 404 and config.pool_entries:
                            # A crashed/restarting replica unregisters its endpoint: the proxy
                            # answers 404 while others stay healthy. A JSON error body means the
                            # endpoint itself rejected the request (model-name contract break,
                            # fatal); a plain proxy 404 means the entry is gone (rotate).
                            body_bytes = response.read()
                            raw.write(body_bytes)
                            try:
                                error_payload = json.loads(body_bytes.decode())
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                error_payload = None
                            if isinstance(error_payload, dict) and "error" in error_payload:
                                terminal = True
                                raise PermanentRequestError(f"endpoint rejected request: {str(error_payload)[:200]}")
                            raise TransientRequestError("registry 404: entry unavailable; rotating")
                        if response.status_code == 401 and config.pool_entries:
                            refresh_pool_key(config)
                            raise TransientRequestError("401: credential refreshed; retrying")
                        if response.status_code != 200:
                            raw.write(response.read())
                            terminal = True
                            raise PermanentRequestError(f"endpoint returned HTTP {response.status_code}")
                        for line in response.iter_lines():
                            check_allocation(config, request_deadline)
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
                                    if receipt["finish_reason"] == "length":
                                        raise LengthDeferred("length_deferred: model exhausted the output token budget")
                if not receipt["done"] or receipt["finish_reason"] != "stop":
                    raise TransientRequestError("Incomplete model result")
                if config.pool_entries and not receipt["content"].strip() and receipt["reasoning"].strip():
                    usage = receipt.get("usage") or {}
                    if usage.get("completion_tokens", 0) > 0:
                        # Completed reasoning-only empty answer (confirmed pool failure class,
                        # 2026-11-24 full-32-r2): the stream finished with stop + positive
                        # completion usage but produced no content. Retry EXACTLY once with
                        # entry rotation; a second blank is fatal. Count-only metadata for
                        # every blank attempt stays in the terminal receipt so a recovered
                        # failure is never hidden.
                        receipt.setdefault("blank_attempts", []).append(
                            {
                                "at": time.time(),
                                "base_url": base_url,
                                "model": config.model,
                                "finish_reason": receipt["finish_reason"],
                                "content_chars": len(receipt["content"]),
                                "reasoning_chars": len(receipt["reasoning"]),
                                "usage": usage,
                            }
                        )
                        if blank_retries < 1:
                            blank_retries += 1
                            raise TransientRequestError("blank answer: reasoning-only; retrying once with rotation")
                        terminal = True
                        raise PermanentRequestError("blank answer persisted after one retry")
                try:
                    value = json.loads(receipt["content"])
                    jsonschema.validate(value, schema)
                except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                    if config.pool_entries:
                        # A 200 with a malformed payload breaks the serving contract
                        # (schema errors are fatal, no rotation): surface instead of retrying.
                        terminal = True
                        raise PermanentRequestError(f"schema violation: {error}") from error
                    raise
                (directory / "value.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
                return value
            except (PermanentRequestError, CapacityUnavailable, LengthDeferred):
                raise
            except (httpx.HTTPError, TransientRequestError, json.JSONDecodeError, jsonschema.ValidationError):
                if attempt + 1 >= config.max_attempts:
                    raise
                pause = backoff * (1.0 + random.uniform(-RETRY_JITTER_FRACTION, RETRY_JITTER_FRACTION))
                retry_pause(config, pause)
                if config.relay_job is not None:
                    receipt["base_url"] = resolve_relay_base_url(
                        config.api_key_file.read_text().strip(), config.relay_job
                    )
                backoff = min(RETRY_BACKOFF_CAP_SECONDS, backoff * 2)
            finally:
                attempt_receipt.update(
                    completed_at_unix=time.time(),
                    finish_reason=receipt["finish_reason"],
                    http_status=receipt["http_status"],
                    raw_file=raw_file.name,
                    raw_sha256=hashlib.sha256(raw_file.read_bytes()).hexdigest() if raw_file.exists() else None,
                )
    finally:
        if raw_file.exists():
            shutil.copyfile(raw_file, directory / "response.sse")
        receipt["completed_at_unix"] = time.time()
        receipt["raw_sha256"] = hashlib.sha256(raw_file.read_bytes()).hexdigest() if raw_file.exists() else None
        receipt["lease_state"] = "terminal" if terminal else "uncertain"
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        with sqlite3.connect(config.ledger, timeout=60) as db:
            db.execute("UPDATE requests SET state=? WHERE id=?", (receipt["lease_state"], identity))
