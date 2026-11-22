# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned, public-prompt-only Mochi response-cap cohort."""

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict

import pyarrow.parquet as pq
from marin.inference.load_test import LoadRequest, request_digest
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.assistant_prefix import assistant_prefix_prompt
from rolloutengine.harness.shell import task_prompt
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.runtime.shell import ShellToolConfig, resolve_shell_tool
from transformers import AutoTokenizer

TRAIN_URI = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4/rl-pool-nonagentic/2026.11.18.71/train.parquet"
TRAIN_SHA256 = "4cb4fd8206a113509e6fcd9c0c4b1198c6d925eb964b1622c98d12ef36151402"
EVAL_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/swe-verified-20261112-v4/release/validation.parquet"
)
EVAL_SHA256 = "8601f26d9b16467efdfb022166367a1c413adfa4118f5110a76b12258e254838"
TOKENIZER = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
NEMOTRON = "nvidia/Nemotron-RL-Ultra-Training-Blends"
MAX_PROMPT_TOKENS = 49152


def source_rows(uri: str, expected_sha256: str, count: int | None) -> list[tuple[int, TaskSpec]]:
    """Verify the regional input and read the selected Parquet row groups."""
    fs, path = filesystem_for(uri)
    digest = hashlib.sha256()
    with fs.open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Source digest changed: {uri}")
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        selected = set(
            range(parquet.metadata.num_rows)
            if count is None
            else random.Random(17).sample(range(parquet.metadata.num_rows), count)
        )
        offset = 0
        result = []
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            indices = sorted(index for index in selected if offset <= index < offset + size)
            if indices:
                column = parquet.read_row_group(group, columns=["task_spec"]).column("task_spec")
                result.extend((index, TaskSpec.model_validate_json(column[index - offset].as_py())) for index in indices)
            offset += size
    return result


def public_request(task: TaskSpec) -> tuple[dict, str]:
    """Match the default Shellbox or assistant-prefix training first turn."""
    last = task.context.events[-1]
    prefix = last.content if isinstance(last, TextMessage) and last.role == "assistant" else ""
    prompt = (
        assistant_prefix_prompt(task) if prefix else task_prompt(task, resolve_shell_tool(task, ShellToolConfig()), ())
    )
    options = dict(prompt.options)
    extra = options.pop("extra_body", {})
    body = {"messages": list(prompt.messages), **options, **extra}
    body["chat_template_kwargs"] = {"enable_thinking": True}
    return body, prefix


def prepare(output_uri: str, repeats: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    sources = [
        (
            "eval",
            EVAL_URI,
            EVAL_SHA256,
            [(i, t) for i, t in source_rows(EVAL_URI, EVAL_SHA256, None) if t.source.dataset == NEMOTRON],
        ),
        ("train", TRAIN_URI, TRAIN_SHA256, source_rows(TRAIN_URI, TRAIN_SHA256, 300)),
    ]
    requests = []
    metadata = {}
    excluded = []
    for split, uri, digest, tasks in sources:
        for index, task in tasks:
            body, prefix = public_request(task)
            options = dict(body)
            messages = options.pop("messages")
            template_kwargs = options.pop("chat_template_kwargs")
            options.setdefault("add_generation_prompt", True)
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, **options, **template_kwargs)
            # Transformers 5 returns BatchEncoding; older tokenizer releases return IDs.
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
            length = len(tokens)
            identity = f"{split}-{index}"
            if length > MAX_PROMPT_TOKENS:
                excluded.append({"identity": identity, "prompt_tokens": length, "reason": "training_prompt_filter"})
                continue
            metadata[identity] = {
                "split": split,
                "source_uri": uri,
                "source_sha256": digest,
                "row_index": index,
                "source": task.source.model_dump(),
                "answer_format": task.answer_format.model_dump(mode="json"),
                "assistant_prefix": prefix,
                "prompt_tokens": length,
            }
            requests.append(asdict(LoadRequest(identity, "chat/completions", body)))
    if sum(row["split"] == "eval" for row in metadata.values()) != 80:
        raise ValueError("Expected the unchanged 80-task Nemotron validation slice")
    root = StoragePath(output_uri)
    fs, path = filesystem_for(output_uri)
    digest = request_digest({"requests": requests})
    fs.pipe_file(path + "/requests.jsonl", "".join(json.dumps(row) + "\n" for row in requests).encode())
    fs.pipe_file(path + "/metadata.json", json.dumps(metadata).encode())
    manifest = {
        "requests_uri": str(root / "requests.jsonl"),
        "requests_sha256": digest,
        "metadata_uri": str(root / "metadata.json"),
        "metadata_sha256": request_digest(metadata),
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "prompt_count": len(requests),
        "repeats": repeats,
        "attempt_count": len(requests) * repeats,
        "split_counts": dict(Counter(row["split"] for row in metadata.values())),
        "excluded": excluded,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in metadata.values()),
        "training_validation_unchanged": True,
    }
    fs.pipe_file(path + "/manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_uri, args.repeats), indent=2))


if __name__ == "__main__":
    main()
