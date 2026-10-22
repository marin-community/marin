# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect GLM traces for reconciled CruxEval requests without executing model output."""

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from marin.inference.openai_batch import jsonl_text
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.constants import DATA_REVISION, DATA_SHA256, GENERATION_TOKENS
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url

DATA_URL = (
    f"https://raw.githubusercontent.com/marin-community/evalchemy/{DATA_REVISION}"
    "/eval/chat_benchmarks/CruxEval/data/cruxeval.jsonl"
)
REQUEST_TIMEOUT = 1800


def complete_request(request: dict, endpoint: str, token: str, output: StoragePath) -> dict:
    """Save each direct bulk response for reuse when the same collection resumes."""
    filename = hashlib.sha256(request["custom_id"].encode()).hexdigest() + ".json"
    path = output / filename
    if path.exists():
        return json.loads(path.read_text())
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(request["body"]).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "x-priority": "bulk"},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=REQUEST_TIMEOUT) as response:
            result = {
                "custom_id": request["custom_id"],
                "response": {"status_code": response.status, "body": json.loads(response.read())},
            }
    except urllib.error.HTTPError as error:
        result = {
            "custom_id": request["custom_id"],
            "error": {
                "status_code": error.code,
                "message": error.read(4096).decode(errors="replace").replace(token, "[redacted]"),
            },
        }
    except (urllib.error.URLError, TimeoutError) as error:
        result = {"custom_id": request["custom_id"], "error": {"message": str(error).replace(token, "[redacted]")}}
    path.write_text(json.dumps(result) + "\n")
    return result


def build_requests(captured: list[dict], problems: list[dict]) -> list[dict]:
    """Identify each captured prompt by its exact final problem block, preserving all messages."""
    index = {}
    for problem in problems:
        for direction in ("input", "output"):
            assertion = (
                f"assert f(??) == {problem['output']}" if direction == "input" else f"assert f({problem['input']}) == ??"
            )
            suffix = f"{problem['code']}\n{assertion}\n[/PYTHON]\n[THOUGHT]\n"
            if suffix in index:
                raise ValueError("Ambiguous CruxEval problem block")
            index[suffix] = (direction, problem)
    requests = []
    seen = set()
    for original in captured:
        if set(original) != {"model", "messages", "max_tokens", "chat_template_kwargs"}:
            raise ValueError("Captured request settings differ from the reconciled baseline")
        if original["max_tokens"] != GENERATION_TOKENS or original["chat_template_kwargs"] != {"enable_thinking": True}:
            raise ValueError("Captured generation budget or thinking mode differs from baseline")
        messages = original["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Expected a single CruxEval user message")
        direction, problem = index[messages[0]["content"].rsplit("[PYTHON]\n", 1)[1]]
        identity = (direction, problem["id"])
        if identity in seen:
            raise ValueError(f"Repeated captured problem: {identity}")
        seen.add(identity)
        custom_id = f"CruxEval::{direction}::{problem['id']}"
        requests.append(
            {
                "custom_id": custom_id,
                "task": direction,
                "source_id": problem["id"],
                "source": problem,
                "b_request": original,
                "body": {
                    "model": GLM_MODEL,
                    "messages": messages,
                    "max_tokens": original["max_tokens"],
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "chat_template_kwargs": {"reasoning_effort": "high"},
                    "prompt_cache_key": custom_id,
                },
            }
        )
    if len(seen) != 2 * len(problems):
        raise ValueError("Captured requests do not cover both tasks for every problem")
    return requests


@click.command()
@click.option("--prompts", required=True)
@click.option("--output", required=True)
@click.option("--relay-job", required=True)
@click.option("--run", is_flag=True)
def main(prompts: str, output: str, relay_job: str, run: bool) -> None:
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("CruxEval dataset checksum mismatch")
    problems = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    captured_text = StoragePath(prompts).read_text()
    captured = [json.loads(line) for line in captured_text.splitlines() if line.strip()]
    requests = build_requests(captured, problems)
    if len(problems) != 800 or len(requests) != 1600:
        raise ValueError("Expected 800 problems and 1600 requests")
    print(json.dumps({"requests": len(requests), "tasks": dict(Counter(row["task"] for row in requests))}), flush=True)
    if not run:
        return
    destination = StoragePath(output)
    destination.mkdirs()
    recipe = jsonl_text(requests)
    recipe_path = destination / "requests.jsonl"
    if recipe_path.exists() and recipe_path.read_text() != recipe:
        raise ValueError("Output holds a different request recipe")
    recipe_path.write_text(recipe)
    responses_dir = destination / "responses"
    responses_dir.mkdirs()
    endpoint = resolve_glm_base_url(relay_job).removesuffix("/v1") + "/bulk/v1/chat/completions"
    token = os.environ[GLM_BULK_TOKEN_ENV]
    responses = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(complete_request, request, endpoint, token, responses_dir) for request in requests]
        for future in as_completed(futures):
            result = future.result()
            responses[result["custom_id"]] = result
            if len(responses) % 50 == 0:
                print(f"Collected {len(responses)}/1600", flush=True)
    ordered = [responses[request["custom_id"]] for request in requests]
    (destination / "raw-responses.jsonl").write_text(jsonl_text(ordered))
    failures = sum(bool(response.get("error")) for response in ordered)
    manifest = {
        "requests": len(requests),
        "responses": len(ordered),
        "failures": failures,
        "grading": "not run; requires isolated execution",
        "evalchemy_revision": DATA_REVISION,
        "source_sha256": DATA_SHA256,
        "prompts": prompts,
        "prompts_sha256": hashlib.sha256(captured_text.encode()).hexdigest(),
        "requests_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "relay_job": relay_job,
        "concurrency": 8,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": None},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
