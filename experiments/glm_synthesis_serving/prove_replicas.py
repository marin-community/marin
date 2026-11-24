# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay archived conversion requests in-region, recording aggregate facts only."""

import argparse
import asyncio
import hashlib
import json
import random
import tarfile
import time
from collections import Counter
from dataclasses import dataclass

import httpx
from iris.client.client import iris_ctx
from rigging.filesystem.buckets import filesystem_for
from rigging.timing import Duration

from experiments.glm_synthesis_serving import pool, warmup_bench

SOURCE_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization"
SOURCE_FLEETS = (("ortet-24-20261112", 24), ("h100-eight-20261105", 8))
SOURCE_START = 1795463520  # Conversion-active window: 2026-11-23 19:52–22:52 UTC.
SOURCE_END = SOURCE_START + 10800
CAPABILITY_HOURS = 168


@dataclass(frozen=True)
class Target:
    endpoint: str
    base_url: str
    token: str


def archived_requests(count: int) -> list[dict]:
    """Read distinct request bodies from regional archives without writing trace text."""
    fs, root = filesystem_for(SOURCE_ROOT)
    candidates = []
    for fleet, replicas in SOURCE_FLEETS:
        for replica in range(1, replicas + 1):
            for item in fs.ls(f"{root}/{fleet}/r{replica}/batches", detail=True):
                if not item["name"].endswith(".tar.gz"):
                    continue
                if SOURCE_START <= item["LastModified"].timestamp() <= SOURCE_END:
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
                if "thinking_token_budget" not in body:
                    raise ValueError("Archived request lacks the consumer's thinking_token_budget field")
                body["model"] = pool.SERVED_MODEL_NAME
                requests.append(body)
                seen.add(identity)
                if len(requests) == count:
                    return requests
    raise RuntimeError(f"Only {len(requests)} distinct conversion requests found; need {count}")


async def request_outcome(client: httpx.AsyncClient, target: Target, body: dict) -> dict:
    """Require HTTP 200, complete SSE, model content, valid JSON, and token usage."""
    started = time.monotonic()
    headers = {"Authorization": f"Bearer {target.token}"}
    content = []
    usage = None
    done = False
    finish_reason = None
    try:
        async with client.stream("POST", target.base_url + "/chat/completions", headers=headers, json=body) as response:
            if response.status_code != 200:
                return {"ok": False, "status": response.status_code, "endpoint": target.endpoint}
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    return {"ok": False, "error": "stream_error", "endpoint": target.endpoint}
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        content.append(text)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                usage = event.get("usage") or usage
        valid = (
            done
            and bool(content)
            and usage is not None
            and usage.get("completion_tokens", 0) > 0
            and finish_reason == "stop"
        )
        if valid:
            json.loads("".join(content))
        return {
            "endpoint": target.endpoint,
            "ok": valid,
            "status": 200,
            "done": done,
            "finish_reason": finish_reason,
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
            "completion_tokens": (usage or {}).get("completion_tokens", 0),
            "elapsed": time.monotonic() - started,
        }
    except (httpx.HTTPError, ValueError) as error:
        # Exception text can contain credential-bearing URLs or response content.
        return {"ok": False, "error": type(error).__name__, "endpoint": target.endpoint}


async def prove(targets: list[Target], bodies: list[dict], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=1800, limits=limits) as client:
        for target in targets:
            response = await client.get(target.base_url + "/models", headers={"Authorization": f"Bearer {target.token}"})
            if response.status_code != 200:
                raise RuntimeError(f"{target.endpoint}: /models returned {response.status_code}")
            if pool.SERVED_MODEL_NAME not in [item["id"] for item in response.json()["data"]]:
                raise RuntimeError(f"{target.endpoint}: /models lacks {pool.SERVED_MODEL_NAME}")

        async def run(index: int, body: dict) -> dict:
            async with semaphore:
                return await request_outcome(client, targets[index % len(targets)], body)

        return await asyncio.gather(*(run(index, body) for index, body in enumerate(bodies)))


def write_json(url: str, value: object) -> None:
    fs, path = filesystem_for(url)
    with fs.open(path, "w") as stream:
        json.dump(value, stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--duration", type=float, default=900)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests < 200 or args.requests < len(args.endpoint):
        raise ValueError("Proof requires at least 200 requests and one per replica")
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Run the proof inside an Iris CPU job on cw-us-east-08a")
    targets = [
        Target(
            endpoint=endpoint,
            base_url=f"http://iris-controller-svc:10000/proxy/{endpoint.lstrip('/').replace('/', '.')}/v1",
            token=client.mint_endpoint_token(endpoint, ttl=Duration.from_hours(CAPABILITY_HOURS)).token,
        )
        for endpoint in args.endpoint
    ]
    bodies = archived_requests(args.requests)
    started = time.monotonic()
    outcomes = asyncio.run(prove(targets, bodies, args.concurrency))
    elapsed = time.monotonic() - started
    passed = sum(result["ok"] for result in outcomes)
    report = {
        "requests": len(outcomes),
        "passed": passed,
        "failures": len(outcomes) - passed,
        "elapsed": elapsed,
        "rows_per_hour": passed * 3600 / elapsed,
        "per_endpoint": dict(Counter(result["endpoint"] for result in outcomes if result["ok"])),
        "outcomes": outcomes,
    }
    write_json(args.output + "/proof.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}), flush=True)
    if passed != len(outcomes):
        raise RuntimeError("Real-request proof failed; consumer manifest was not published")
    if args.duration <= 0:
        return

    async def sustain() -> list[dict]:
        async def bench(target: Target) -> dict:
            started = time.monotonic()
            stats = await warmup_bench.run_load(
                target.base_url, args.concurrency, args.duration, "high", target.token
            )
            return {"endpoint": target.endpoint, **stats.summary(time.monotonic() - started, pool.GPUS_PER_ENGINE)}

        return await asyncio.gather(*(bench(target) for target in targets))

    sustained = asyncio.run(sustain())
    write_json(args.output + "/sustained.json", sustained)
    print(json.dumps({"sustained": sustained}), flush=True)
    if any(result["failures"] for result in sustained):
        raise RuntimeError("Sustained load failed")
    manifest = []
    for target in targets:
        credential_ref = args.output + "/credentials/" + target.endpoint.rsplit("/", 1)[-1] + ".jwt"
        fs, path = filesystem_for(credential_ref)
        with fs.open(path, "w") as stream:
            stream.write(target.token)
        manifest.append({"base_url": target.base_url, "api_key_ref": credential_ref})
    write_json(args.output + "/manifest.json", manifest)


if __name__ == "__main__":
    main()
