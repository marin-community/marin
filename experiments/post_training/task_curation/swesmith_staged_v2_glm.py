# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draft unverified handoff notes using the existing GLM bulk HTTP transport."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import dotenv_values
from marin.inference.openai_chat import OpenAIChatClient

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation.swesmith_staged import sha256, write_json


def draft_note(parent_path: Path, client: OpenAIChatClient, max_tokens: int) -> dict:
    parent = json.loads(parent_path.read_text())
    destination = parent_path.parent / "glm-handoff.json"
    request = {
        "model": GLM_MODEL,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write a concise handoff note for a developer taking over unverified work. "
                    "Use only the issue and repair-file names supplied as data. Do not assert that any implementation, "
                    "test, command or repair has succeeded or failed. Do not invent completed work. "
                    "Explain what behavior to inspect and which edge cases matter. "
                    "The code may already be correct. Do not insist on source edits or adding tests. "
                    "Do not mention hidden tests, a gold solution, labels or task generation. "
                    "Return only JSON with a single string field note, between 40 and 180 words."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"issue": parent["problem"], "repair_files": parent["changed_source_paths"]}),
            },
        ],
    }
    key = sha256(json.dumps(request, sort_keys=True).encode())
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("request_sha256") == key and saved["status"] == "accepted":
            return saved
        write_json(parent_path.parent / "glm-history" / (str(time.time_ns()) + ".json"), saved)
    result = {
        "parent_id": parent["parent_id"],
        "status": "failed",
        "request_sha256": key,
        "model": GLM_MODEL,
        "pool": "bulk",
        "request": request,
        "attempts": [],
    }
    for attempt in range(3):
        try:
            response = client.complete(request)
            result["attempts"].append(response)
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Model returned no final text within the output budget")
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            note = json.loads(content)["note"]
            if not isinstance(note, str) or not 20 <= len(note.split()) <= 250:
                raise ValueError("Handoff note is outside bounded prose contract")
            result.update(status="accepted", note=note)
            break
        except (ConnectionError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            result["attempts"].append({"error_type": type(exc).__name__})
            if attempt < 2:
                time.sleep(2**attempt)
    write_json(destination, result)
    return result


def main() -> None:
    code_hash = sha256(Path(__file__).read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--secrets-env", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry-failed-only", action="store_true")
    args = parser.parse_args()
    token = dotenv_values(args.secrets_env)[GLM_BULK_TOKEN_ENV]
    if not token:
        raise ValueError("The requested bulk credential is unavailable")
    client = OpenAIChatClient(args.base_url, token, timeout=300)
    paths = sorted((args.output / "parents").glob("*/result.json"))
    if args.retry_failed_only:
        paths = [
            path
            for path in paths
            if (path.parent / "glm-handoff.json").exists()
            and json.loads((path.parent / "glm-handoff.json").read_text())["status"] == "failed"
        ]
    if args.limit is not None:
        paths = paths[: args.limit]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda path: draft_note(path, client, args.max_tokens), paths))
    summary = {
        "counts": dict(Counter(row["status"] for row in results)),
        "elapsed": time.monotonic() - started,
        "model": GLM_MODEL,
        "max_tokens": args.max_tokens,
        "pool": "bulk",
        "endpoint": args.base_url,
        "code_sha256": code_hash,
        "privacy": (
            "Requests contain issue text and source path names only; no hidden tests, gold patches or credentials"
        ),
    }
    summary_name = "glm-retry-summary.json" if args.retry_failed_only else "glm-summary.json"
    write_json(args.output / summary_name, summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
