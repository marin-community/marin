# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay real conversion requests in-region; publish only validated pool manifests."""

import argparse
import asyncio
import hashlib
import json
import random
import tarfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import httpx
from iris.client.client import iris_ctx
from rigging.connect import proxy_path
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.glm_synthesis_serving import pool

SOURCE_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization"
SOURCE_FLEETS = (("ortet-24-20261112", 24), ("h100-eight-20261105", 8))
SOURCE_START = 1795463520  # Conversion-active window: 2026-11-23 19:52-22:52 UTC.
SOURCE_END = SOURCE_START + 10800
CAPABILITY_HOURS = 168
ENDPOINT_POLL_SECONDS = 15
CLIENT_REQUEST_TIMEOUT_SECONDS = 1800
PROOF_REQUEST_COUNT = 200


@dataclass(frozen=True)
class Target:
    endpoint: str
    base_url: str
    token: str = field(repr=False)
    expires_at: int


class RequestState(StrEnum):
    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    STREAM_ERROR = "stream_error"
    INVALID_OUTPUT = "invalid_output"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True)
class RequestOutcome:
    endpoint: str
    state: RequestState
    status: int | None = None
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.state is RequestState.SUCCESS


@dataclass(frozen=True)
class StageSummary:
    requests: int
    passed: int
    failures: int
    elapsed: float
    rows_per_hour: float
    prompt_tokens_per_second_per_gpu: float
    completion_tokens_per_second_per_gpu: float
    per_endpoint: dict[str, int]


def archived_requests(count: int) -> list[dict]:
    """Read distinct request bodies without writing trace text."""
    root = StoragePath(SOURCE_ROOT)
    fs, _ = filesystem_for(str(root))
    candidates = []
    for fleet, replicas in SOURCE_FLEETS:
        for replica in range(1, replicas + 1):
            _, prefix = filesystem_for(str(root / fleet / f"r{replica}" / "batches"))
            for item in fs.ls(prefix, detail=True):
                if item["name"].endswith(".tar.gz") and SOURCE_START <= item["LastModified"].timestamp() <= SOURCE_END:
                    candidates.append(item["name"])
    random.Random(1234).shuffle(candidates)
    requests = []
    seen = set()
    for path in candidates:
        with fs.open(path, "rb") as source, tarfile.open(fileobj=source, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith("/review/request.json"):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("Archived request has no content")
                raw = stream.read()
                identity = hashlib.sha256(raw).hexdigest()
                if identity in seen:
                    continue
                body = json.loads(raw)
                if not body.get("stream") or body.get("response_format", {}).get("type") != "json_schema":
                    raise ValueError("Archived request does not match the conversion contract")
                if not {"thinking_token_budget", "cache_salt"} <= body.keys():
                    raise ValueError("Archived request lacks consumer fields")
                body["model"] = pool.SERVED_MODEL_NAME
                requests.append(body)
                seen.add(identity)
                if len(requests) == count:
                    return requests
    raise RuntimeError(f"Only {len(requests)} distinct conversion requests found; need {count}")


def ready_targets(endpoints: list[str], wait_timeout: float) -> list[Target]:
    client = iris_ctx().client
    if client is None:
        raise RuntimeError(f"Run inside an Iris CPU job on {pool.CLUSTER}")
    deadline = time.monotonic() + wait_timeout
    targets = []
    for endpoint in endpoints:
        print(json.dumps({"waiting_for": endpoint}), flush=True)
        while not client.list_endpoint_instances(endpoint):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{endpoint} did not register")
            time.sleep(ENDPOINT_POLL_SECONDS)
        minted = client.mint_endpoint_token(endpoint, ttl=Duration.from_hours(CAPABILITY_HOURS))
        targets.append(
            Target(
                endpoint,
                f"{pool.CONTROLLER_PROXY_ORIGIN}{proxy_path(endpoint)}/v1",
                minted.token,
                minted.expires_at.epoch_ms,
            )
        )
    return targets


async def request_outcome(client: httpx.AsyncClient, target: Target, body: dict) -> RequestOutcome:
    """Require HTTP 200, complete SSE, valid JSON content, stop, and token usage."""
    content = []
    usage = {}
    done = False
    finish_reason = None
    headers = {"Authorization": f"Bearer {target.token}"}
    try:
        async with client.stream("POST", target.base_url + "/chat/completions", headers=headers, json=body) as response:
            if response.status_code != 200:
                return RequestOutcome(target.endpoint, RequestState.HTTP_ERROR, response.status_code)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    return RequestOutcome(target.endpoint, RequestState.STREAM_ERROR, 200)
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        content.append(text)
                    finish_reason = choice.get("finish_reason") or finish_reason
                usage = event.get("usage") or usage
        valid = done and bool(content) and usage.get("completion_tokens", 0) > 0 and finish_reason == "stop"
        if valid:
            json.loads("".join(content))
        return RequestOutcome(
            endpoint=target.endpoint,
            state=RequestState.SUCCESS if valid else RequestState.INVALID_OUTPUT,
            status=200,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    except httpx.HTTPError as error:
        # Exception text can contain response data or credentials.
        return RequestOutcome(target.endpoint, RequestState.TRANSPORT_ERROR, error=type(error).__name__)
    except ValueError as error:
        return RequestOutcome(target.endpoint, RequestState.INVALID_OUTPUT, error=type(error).__name__)


async def check_models(client: httpx.AsyncClient, targets: list[Target]) -> None:
    for target in targets:
        response = await client.get(target.base_url + "/models", headers={"Authorization": f"Bearer {target.token}"})
        if response.status_code != 200:
            raise RuntimeError(f"{target.endpoint}: /models returned {response.status_code}")
        if pool.SERVED_MODEL_NAME not in [item["id"] for item in response.json()["data"]]:
            raise RuntimeError(f"{target.endpoint}: /models lacks {pool.SERVED_MODEL_NAME}")


async def prove(targets: list[Target], bodies: list[dict], concurrency: int) -> list[RequestOutcome]:
    semaphore = asyncio.Semaphore(concurrency)
    outcomes = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=CLIENT_REQUEST_TIMEOUT_SECONDS, limits=limits) as client:
        await check_models(client, targets)

        async def run(index: int, body: dict) -> None:
            async with semaphore:
                outcome = await request_outcome(client, targets[index % len(targets)], body)
                outcomes.append(outcome)
                if len(outcomes) % 25 == 0:
                    print(
                        json.dumps({"completed": len(outcomes), "passed": sum(item.ok for item in outcomes)}), flush=True
                    )

        await asyncio.gather(*(run(index, body) for index, body in enumerate(bodies)))
    return outcomes


async def sustained_load(
    targets: list[Target], bodies: list[dict], concurrency_per_replica: int, duration: float
) -> list[RequestOutcome]:
    """Replay real bodies; each round gets a new salt to prevent whole-request cache reuse."""
    outcomes = []
    stopped = asyncio.Event()
    deadline = time.monotonic() + duration
    request_index = 0
    concurrency = len(targets) * concurrency_per_replica
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=CLIENT_REQUEST_TIMEOUT_SECONDS, limits=limits) as client:

        async def worker(target: Target) -> None:
            nonlocal request_index
            while time.monotonic() < deadline and not stopped.is_set():
                index = request_index
                request_index += 1
                original = bodies[index % len(bodies)]
                body = {**original, "cache_salt": f"{original['cache_salt']}-load-{index // len(bodies)}"}
                outcome = await request_outcome(client, target, body)
                outcomes.append(outcome)
                if not outcome.ok:
                    stopped.set()
                if len(outcomes) % 100 == 0:
                    print(json.dumps({"load_completed": len(outcomes)}), flush=True)

        await asyncio.gather(*(worker(target) for target in targets for _ in range(concurrency_per_replica)))
    return outcomes


def stage_summary(outcomes: list[RequestOutcome], elapsed: float, replicas: int) -> StageSummary:
    passed = [outcome for outcome in outcomes if outcome.ok]
    gpu_time = elapsed * replicas * pool.GPUS_PER_ENGINE
    return StageSummary(
        len(outcomes),
        len(passed),
        len(outcomes) - len(passed),
        elapsed,
        len(passed) * 3600 / elapsed,
        sum(outcome.prompt_tokens for outcome in passed) / gpu_time,
        sum(outcome.completion_tokens for outcome in passed) / gpu_time,
        dict(Counter(outcome.endpoint for outcome in passed)),
    )


def write_report(path: StoragePath, outcomes: list[RequestOutcome], elapsed: float, replicas: int) -> StageSummary:
    summary = stage_summary(outcomes, elapsed, replicas)
    write_text(path, json.dumps({**asdict(summary), "outcomes": [asdict(outcome) for outcome in outcomes]}))
    print(json.dumps(asdict(summary)), flush=True)
    return summary


def write_text(path: StoragePath, content: str) -> None:
    """Write through the declared bucket routing, including CoreWeave endpoints."""
    fs, resolved = filesystem_for(str(path))
    with fs.open(resolved, "w") as stream:
        stream.write(content)


def publish_manifest(output: StoragePath, targets: list[Target], concurrency_per_replica: int) -> None:
    manifest = []
    for target in targets:
        credential = output / "credentials" / (target.endpoint.rsplit("/", 1)[-1] + ".jwt")
        write_text(credential, target.token)
        manifest.append({"base_url": target.base_url, "api_key_ref": str(credential)})
    write_text(output / "credential-expiry.json", json.dumps({target.endpoint: target.expires_at for target in targets}))
    write_text(
        output / "status.json",
        json.dumps(
            {
                "model": pool.SERVED_MODEL_NAME,
                "cluster": pool.CLUSTER,
                "healthy_entry_count": len(targets),
                "per_engine_concurrency": concurrency_per_replica,
                "total_concurrency": len(targets) * concurrency_per_replica,
                "validated_at": int(time.time()),
                "proof_ref": str(output / "proof.json"),
                "sustained_ref": str(output / "sustained.json"),
            }
        ),
    )
    write_text(output / "manifest.json", json.dumps(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--requests", type=int, default=PROOF_REQUEST_COUNT)
    parser.add_argument(
        "--concurrency", type=int, default=24, help="Global proof concurrency; load concurrency per replica."
    )
    parser.add_argument("--duration", type=float, default=900)
    parser.add_argument("--wait-timeout", type=float, default=7200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests < PROOF_REQUEST_COUNT or args.requests < len(args.endpoint):
        raise ValueError(f"Proof requires at least {PROOF_REQUEST_COUNT} requests and one per replica")
    if args.concurrency < 1 or args.duration < 0 or args.wait_timeout < 0:
        raise ValueError("Concurrency must be positive and durations must not be negative")
    targets = ready_targets(args.endpoint, args.wait_timeout)
    bodies = archived_requests(args.requests)
    output = StoragePath(args.output)
    started = time.monotonic()
    outcomes = asyncio.run(prove(targets, bodies, args.concurrency))
    proof = write_report(output / "proof.json", outcomes, time.monotonic() - started, len(targets))
    if proof.failures:
        raise RuntimeError("Real-request proof failed; consumer manifest was not published")
    if args.duration <= 0:
        return
    started = time.monotonic()
    outcomes = asyncio.run(sustained_load(targets, bodies, args.concurrency, args.duration))
    load = write_report(output / "sustained.json", outcomes, time.monotonic() - started, len(targets))
    if load.failures:
        raise RuntimeError("Sustained real-request load failed; consumer manifest was not published")
    publish_manifest(output, targets, args.concurrency)


if __name__ == "__main__":
    main()
