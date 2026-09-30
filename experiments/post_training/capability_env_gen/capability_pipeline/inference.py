"""GLM streaming transport, durable per-request artifacts and resumable stage cache."""

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


class InfrastructureUnavailable(RuntimeError):
    """Retryable service fault; never counts as a model quality failure."""


class GLMClient:
    def __init__(
        self,
        base_url=None,
        token=None,
        tier="interactive",
        timeout=900,
        hold_seconds=3600,
    ):
        self.base_url = (base_url or os.environ["GLM_BASE_URL"]).rstrip("/")
        self.base_url = self.base_url.removesuffix("/v1")
        self.token = token or os.environ["GLM_API_TOKEN"]
        self.tier = tier
        self.timeout = timeout
        self.hold_seconds = hold_seconds

    def headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if self.tier == "bulk":
            # Belt and braces.  The tier is bound by the bearer token, and for
            # a bulk run submit.sh already passes GLM_BULK_TOKEN (the shared
            # load_secrets selects it on GLM_TIER=bulk), so requests land on
            # the bulk pool without this.  The header only lowers a request's
            # band; it keeps this client bulk even if handed the wrong token.
            headers["x-priority"] = "bulk"
        return headers

    def ready(self):
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=20) as r:
                health = json.load(r)
            pool = "bulk" if self.tier == "bulk" else "high"
            return health.get("workers", {}).get(pool, 0) >= 1
        except (OSError, ValueError):
            return False

    def complete(self, request, events_path):
        deadline = time.monotonic() + self.hold_seconds
        attempt = 0
        while True:
            if time.monotonic() >= deadline:
                raise InfrastructureUnavailable(
                    "infrastructure hold expired; resume this work item later"
                )
            if not self.ready():
                time.sleep(15)
                continue
            attempt += 1
            try:
                return self._stream(request, events_path, attempt)
            except urllib.error.HTTPError as e:
                body = e.read(2000).decode(errors="replace")
                # A missing relay route is an infrastructure outage even with /health green.
                if e.code in (408, 429, 500, 502, 503, 504) or (
                    e.code == 404 and "route" in body.lower()
                ):
                    time.sleep(min(30, 2 ** min(attempt, 5)) + random.random())
                    continue
                raise RuntimeError(
                    f"GLM HTTP {e.code}; request rejected (body withheld)"
                ) from None
            except (OSError, EOFError) as e:
                # The old response/socket has closed before another request is sent.
                if time.monotonic() >= deadline:
                    raise InfrastructureUnavailable(type(e).__name__) from e
                time.sleep(min(30, 2 ** min(attempt, 5)))

    def _stream(self, request, events_path, attempt):
        payload = dict(request, stream=True, stream_options={"include_usage": True})
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self.headers(),
        )
        start = time.monotonic()
        started_unix = time.time()
        ttft = None
        content, reasoning = [], []
        usage = {}
        finish = None
        events_path = Path(events_path).with_suffix(f".attempt-{attempt}.jsonl")
        done = False
        with (
            urllib.request.urlopen(req, timeout=self.timeout) as response,
            events_path.open("w") as log,
        ):
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                log.write(canonical(event) + "\n")
                log.flush()
                if "error" in event:
                    raise EOFError("upstream SSE error")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if ttft is None and (content or reasoning):
                        ttft = time.monotonic() - start
                    finish = choice.get("finish_reason") or finish
        if not done or not finish:
            raise EOFError("incomplete SSE stream")
        return {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "usage": usage,
            "finish_reason": finish,
            "elapsed_seconds": time.monotonic() - start,
            "ttft_seconds": ttft,
            "transport_attempts": attempt,
            "started_unix": started_unix,
        }


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("model result must be an object")
    return value


class StageStore:
    def __init__(self, root, client):
        self.root = Path(root)
        self.client = client

    def generate(
        self,
        stage,
        identity,
        system,
        prompt,
        validator,
        max_tokens=32000,
        structural_retries=1,
        response_format=None,
        repair_max_tokens=64000,
    ):
        if max_tokens < 1 or repair_max_tokens < max_tokens:
            raise ValueError(
                "token limits require 1 <= max_tokens <= repair_max_tokens"
            )
        request = {
            "model": "glm-5.3",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }
        if response_format is not None:
            request["response_format"] = response_format
        key = digest(
            {
                "request": request,
                "stage": stage,
                "identity": identity,
                "tier": self.client.tier,
                "schema_version": 1,
            }
        )
        work = self.root / "items" / stage / key
        work.mkdir(parents=True, exist_ok=True)
        final = work / "result.json"
        if final.exists():
            try:
                record = json.loads(final.read_text())
                if record.get("request_hash") != key:
                    raise ValueError("cached request hash mismatch")
                validator(record["artifact"])
            except Exception as exc:  # noqa: BLE001 -- stale cache must be preserved and regenerated
                # Validator/schema changes must not leave a work item permanently
                # wedged on an artifact that can no longer pass. Preserve the bad
                # result for audit, remove it from the success path, and regenerate.
                final.replace(work / "rejected-result.json")
                atomic_json(
                    work / "cache-rejection.json",
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "rejected": time.time(),
                    },
                )
            else:
                return record["artifact"]
        atomic_json(
            work / "request.json",
            {
                "request_hash": key,
                "identity": identity,
                "tier": self.client.tier,
                "request": request,
            },
        )
        started = time.time()
        atomic_json(
            work / "status.json",
            {"state": "running", "started": started, "identity": identity},
        )
        try:
            response = self.client.complete(request, work / "events.jsonl")
            atomic_json(work / "response.json", response)
            repaired = False
            try:
                if response["finish_reason"] != "stop":
                    raise ValueError(
                        f"incomplete model output: finish_reason={response['finish_reason']}"
                    )
                artifact = parse_json(response["content"])
                validator(artifact)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                if structural_retries <= 0:
                    raise
                artifact = self.generate(
                    stage + "-format-repair",
                    identity,
                    system,
                    prompt
                    + "\nYour preceding response failed a mandatory structural gate: "
                    + str(e)
                    + "\nReturn a COMPLETE corrected JSON object. Prior response:\n"
                    + response["content"],
                    validator,
                    max_tokens=min(repair_max_tokens, max_tokens * 2),
                    structural_retries=structural_retries - 1,
                    response_format=response_format,
                    repair_max_tokens=repair_max_tokens,
                )
                repaired = True
            atomic_json(
                final,
                {
                    "request_hash": key,
                    "artifact": artifact,
                    "usage": response["usage"],
                    "structural_repair_used": repaired,
                },
            )
            atomic_json(
                work / "status.json",
                {
                    "state": "complete",
                    "started": started,
                    "finished": time.time(),
                    "identity": identity,
                },
            )
            return artifact
        except Exception as e:
            atomic_json(
                work / "status.json",
                {
                    "state": "infra_hold"
                    if isinstance(e, InfrastructureUnavailable)
                    else "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "started": started,
                    "finished": time.time(),
                    "identity": identity,
                },
            )
            raise


def parallel_map(jobs, concurrency):
    """Refill on each completion, never wait at wave barriers. Errors remain per-item."""
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {pool.submit(fn): key for key, fn in jobs}
        for future in as_completed(pending):
            key = pending[future]
            try:
                results[key] = future.result()
                print(canonical({"event": "complete", "item": key}), flush=True)
            except Exception as e:  # noqa: BLE001 -- persist individual failures without losing siblings
                failures[key] = {"error_type": type(e).__name__, "error": str(e)}
                print(
                    canonical({"event": "failed", "item": key, **failures[key]}),
                    flush=True,
                )
    return results, failures
