# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare sample-zero or rejection-selected demonstrations with the teacher's token IDs."""

import argparse
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.cruxeval_transfer.constants import STUDENT_MODEL, STUDENT_REVISION

CONTEXT = 32768
PAD_TOKEN = 128009


def training_tokens(prompt_tokens: list[int], completion_tokens: list[int], context: int) -> dict:
    """Assign next-token loss only to actual generated tokens, including any emitted EOS."""
    if not prompt_tokens or not completion_tokens:
        raise ValueError("Empty prompt or completion cannot define an SFT example")
    tokens = prompt_tokens + completion_tokens
    if len(tokens) > context:
        raise ValueError("Refusing to truncate a demonstration")
    mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    padding = context - len(tokens)
    return {
        "input_ids": tokens + [PAD_TOKEN] * padding,
        "loss_weights": mask[1:] + [0] * (padding + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    source = StoragePath(args.source)
    output = StoragePath(args.output)
    manifest_bytes = (source / "manifest.json").read_text()
    manifest = json.loads(manifest_bytes)
    complete = json.loads((source / "complete.json").read_text())
    if manifest["requests"] != 12800 or complete["requests"] != 12800:
        raise ValueError("Expected completed eight-sample collection of all 1600 prompts")
    prompt_ids = manifest["prompt_ids"]
    if len(prompt_ids) != 1600 or len(set(prompt_ids)) != 1600:
        raise ValueError("Expected 1600 unique prompts")
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    selected = (
        selection["selected"]
        if selection
        else [{"custom_id": prompt + "::0", "prompt_id": prompt, "sample": 0} for prompt in prompt_ids]
    )
    if not selected or len({row["prompt_id"] for row in selected}) != len(selected):
        raise ValueError("Expected one selected response per covered prompt")
    if not {row["prompt_id"] for row in selected}.issubset(prompt_ids):
        raise ValueError("Selected prompts are outside the frozen universe")

    def read_record(selected_row: dict) -> tuple[dict, str]:
        identity = selected_row["custom_id"]
        digest = hashlib.sha256(identity.encode()).hexdigest()
        raw = (source / "responses" / (digest + ".json")).read_text()
        record = json.loads(raw)
        if record["custom_id"] != identity or record["sample"] != selected_row["sample"] or record["status_code"] != 200:
            raise ValueError(f"Invalid selected sample: {identity}")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if selection and digest != selected_row["response_sha256"]:
            raise ValueError(f"Graded response changed: {identity}")
        return record, digest

    audits = []
    probes = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        with ThreadPoolExecutor(16) as pool, gzip.open(directory / "train-prebuilt.jsonl.gz", "wt") as stream:
            for record, digest in pool.map(read_record, selected):
                request = record["request"]
                if request["chat_template_kwargs"] != {"enable_thinking": True}:
                    raise ValueError("Unexpected thinking mode")
                prompt_tokens = tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                response = record["response"]
                reported_prompt = response.get("prompt_token_ids")
                if reported_prompt is not None and reported_prompt != prompt_tokens:
                    raise ValueError(f"Server prompt differs from pinned template: {record['custom_id']}")
                if response["usage"]["prompt_tokens"] != len(prompt_tokens):
                    raise ValueError("Server and local prompt token counts differ")
                if len(response["choices"]) != 1:
                    raise ValueError("Expected one completion per request")
                choice = response["choices"][0]
                tokens = choice["token_ids"]
                if choice["finish_reason"] not in ("stop", "length"):
                    raise ValueError(f"Unsupported completion status: {choice['finish_reason']}")
                if len(tokens) != response["usage"]["completion_tokens"]:
                    raise ValueError("Returned token IDs do not cover the full completion")
                encoded = training_tokens(prompt_tokens, tokens, CONTEXT)
                stream.write(json.dumps(encoded, separators=(",", ":")) + "\n")
                audit = {
                    "id": record["custom_id"],
                    "task": record["task"],
                    "source_id": record["source_id"],
                    "prompt_tokens": len(prompt_tokens),
                    "supervised_tokens": len(tokens),
                    "tokens": len(prompt_tokens) + len(tokens),
                    "finish_reason": choice["finish_reason"],
                    "response_sha256": digest,
                }
                audits.append(audit)
                if sum(probe["task"] == record["task"] for probe in probes) < 2:
                    probes.append(
                        audit
                        | {
                            "messages": request["messages"],
                            "prompt_token_ids": prompt_tokens,
                            "completion_token_ids": tokens,
                            "completion_text": choice["message"].get("content"),
                            "input_ids": encoded["input_ids"][: audit["tokens"]],
                            "loss_weights": encoded["loss_weights"][: audit["tokens"]],
                        }
                    )
            padding_rows = (-len(audits)) % 16
            for _ in range(padding_rows):
                stream.write(json.dumps({"input_ids": [PAD_TOKEN] * CONTEXT, "loss_weights": [0] * CONTEXT}) + "\n")
        if selection is None and Counter(row["task"] for row in audits) != {"input": 800, "output": 800}:
            raise ValueError("Lost direction coverage")
        for name, rows in (("token-audit.jsonl", audits), ("row-probes.jsonl", probes)):
            (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = {
            "source": args.source,
            "source_manifest_sha256": hashlib.sha256(manifest_bytes.encode()).hexdigest(),
            "student": STUDENT_MODEL,
            "student_revision": STUDENT_REVISION,
            "teacher": manifest["teacher"],
            "selection": (
                selection["selection"]
                if selection
                else "sample0 for every prompt, no correctness or reasoning filter; length-capped samples retained"
            ),
            "selection_manifest": selection,
            "direction_counts": dict(Counter(row["task"] for row in audits)),
            "test_exposed": True,
            "training_rows": len(audits),
            "padding_rows": padding_rows,
            "batch_size": 16,
            "steps_per_epoch": (len(audits) + padding_rows) // 16,
            "max_sequence_tokens": CONTEXT,
            "supervised_tokens": sum(row["supervised_tokens"] for row in audits),
            "input_tokens": sum(row["tokens"] for row in audits),
            "longest_row": max(row["tokens"] for row in audits),
            "finish_reasons": dict(Counter(row["finish_reason"] for row in audits)),
            "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()},
        }
        (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        output.mkdirs()
        for path in sorted(directory.iterdir(), key=lambda path: path.name == "manifest.json"):
            (output / path.name).write_bytes(path.read_bytes())
        print("PLAIN_SFT_PREPARED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
