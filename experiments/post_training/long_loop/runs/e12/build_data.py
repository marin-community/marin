# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build self-contained Pi serialization exercises; no model or sandbox execution."""

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SEED = "e12-b-20261114-r1"
EOS = 128001
HEREDOC = (
    re.compile(r"^\s*cat\s*>\s*(\S+)\s*<<\s*(['\"])(\w+)\2\s*$"),
    re.compile(r"^\s*cat\s*<<\s*(['\"])(\w+)\1\s*>\s*(\S+)\s*$"),
)
EXTENSIONS = {
    ".py",
    ".c",
    ".cpp",
    ".h",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".sh",
    ".sql",
    ".lua",
    ".rb",
}
SECRET = re.compile(
    "api[_-]?key|secret[_-]?key|access[_-]?token|BEGIN .*PRIVATE KEY|gh[pousr]_[A-Za-"
    "z0-9]{20,}|sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}",
    re.I,
)
SPECIAL = re.compile(r"<\|[^>]+\|>|</?tool_call>|</?tool_response|</?tools>")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def commands(message: dict):
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if function.get("name") == "bash" and isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
            yield arguments["command"]


def source_bodies(path: Path) -> tuple[list[dict], dict]:
    candidates = {}
    counts = collections.Counter()
    for row_index, line in enumerate(path.open()):
        row = json.loads(line)
        if row["heldout"]:
            raise ValueError("The input must be the frozen heldout-safe TMax component")
        first_user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        for message_index, message in enumerate(row["messages"]):
            if message["role"] != "assistant":
                continue
            for command in commands(message):
                lines = command.splitlines(keepends=True)
                for start, header in enumerate(lines):
                    match = HEREDOC[0].fullmatch(header.rstrip("\n"))
                    if match:
                        filename, _, marker = match.groups()
                    else:
                        match = HEREDOC[1].fullmatch(header.rstrip("\n"))
                        if not match:
                            continue
                        _, marker, filename = match.groups()
                    filename = filename.strip("'\"")
                    extension = Path(filename).suffix
                    if extension not in EXTENSIONS:
                        continue
                    ends = [j for j in range(start + 1, len(lines)) if lines[j].rstrip("\r\n") == marker]
                    if not ends:
                        continue
                    text = "".join(lines[start + 1 : ends[0]])
                    if not 64 <= len(text.encode()) <= 32768 or len(text.splitlines()) >= 1900:
                        continue
                    counts["quoted_code_candidates"] += 1
                    if SECRET.search(text) or SPECIAL.search(text):
                        counts["secret_or_template_markers"] += 1
                        continue
                    sha = digest(text)
                    record = {
                        "task_id": row["task_id"],
                        "trial_id": row["trial_id"],
                        "expanded_source_row": row_index,
                        "source_row": row["source_row"],
                        "source_file": row["source_file"],
                        "source_row_sha256": row["source_row_sha256"],
                        "message_index": message_index,
                        "source_code_sha256": sha,
                        "code": text,
                        "extension": extension,
                        "parent_instruction": first_user,
                    }
                    bucket = int(digest(SEED + ":task:" + row["task_id"]), 16) % 10
                    record["split"] = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
                    if sha in candidates:
                        candidates[sha]["all_source_tasks"].add(row["task_id"])
                        candidates[sha]["all_source_splits"].add(record["split"])
                    else:
                        candidates[sha] = {
                            **record,
                            "all_source_tasks": {row["task_id"]},
                            "all_source_splits": {record["split"]},
                        }
    eligible = []
    for record in candidates.values():
        if len(record["all_source_splits"]) != 1:
            counts["cross_split_duplicate_code"] += 1
            continue
        record["all_source_tasks"] = sorted(record["all_source_tasks"])
        record["all_source_splits"] = sorted(record["all_source_splits"])
        eligible.append(record)
    counts["unique_code_before_split_gate"] = len(candidates)
    counts["eligible_source_bodies"] = len(eligible)
    eligible.sort(key=lambda r: digest(SEED + r["source_code_sha256"]))
    return eligible, dict(counts)


def make_exercise(source: dict, cardinality: int) -> dict | None:
    code = source["code"]
    lines = code.splitlines(keepends=True)
    if len(lines) < cardinality * 4:
        return None
    rng = random.Random(SEED + source["source_code_sha256"] + str(cardinality))
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    ranges = []
    for i in range(cardinality):
        lo, hi = i * len(lines) // cardinality, (i + 1) * len(lines) // cardinality
        width = min(rng.randint(4, 12), hi - lo - 1)
        if width < 2:
            return None
        start = lo + rng.randrange(max(1, hi - lo - width))
        new = "".join(lines[start : start + width])
        words = list(re.finditer(r"\b[A-Za-z_][A-Za-z_0-9]{2,}\b", new))
        if not words or not 50 <= len(new) <= 4000:
            return None
        word = words[rng.randrange(len(words))]
        old = new[: word.start()] + "e12_fix_" + digest(new + str(i))[:10] + new[word.end() :]
        ranges.append((offsets[start], offsets[start + width], old, new))
    before = code
    for start, end, old, _ in reversed(ranges):
        before = before[:start] + old + before[end:]
    edits = [{"oldText": old, "newText": new} for _, _, old, new in ranges]
    spans = []
    for edit in edits:
        old = edit["oldText"]
        if before.count(old) != 1:
            return None
        start = before.index(old)
        spans.append((start, start + len(old), edit["newText"]))
    spans.sort()
    if any(left[1] > right[0] for left, right in itertools.pairwise(spans)):
        return None
    after = before
    for start, end, new in reversed(spans):
        after = after[:start] + new + after[end:]
    if after != code:
        raise ValueError("Simultaneous edit did not restore the target")
    identity = digest(SEED + source["source_code_sha256"] + str(cardinality))
    path = "/workspace/e12-fixture-" + identity[:12] + source["extension"]
    arguments = {"path": path, "edits": edits}
    return {
        **source,
        "id": "e12-b-" + identity[:24],
        "before_text": before,
        "after_text": code,
        "before_sha256": digest(before),
        "after_sha256": digest(code),
        "arguments": arguments,
        "edit_count": cardinality,
    }


def render(exercise: dict, tokenizer, training_template: str, tools: list[dict]) -> dict:
    replacements = [{"region": i + 1, "replacement": e["newText"]} for i, e in enumerate(exercise["arguments"]["edits"])]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding assistant. Make the requested exact file corrections using the "
                "edit tool. Match disjoint edits against the original file. This is a synthetic f"
                "ile-edit exercise."
            ),
        },
        {
            "role": "user",
            "content": (
                "The file "
                + exercise["arguments"]["path"]
                + " contains corrupted regions. Restore each region to the exact replacement text b"
                "elow, leaving all other bytes unchanged. Use one edit call with one edits-array "
                "entry per region.\n" + json.dumps(replacements, ensure_ascii=False)
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "e12-read",
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": exercise["arguments"]["path"]}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "e12-read", "name": "read", "content": exercise["before_text"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "e12-edit", "type": "function", "function": {"name": "edit", "arguments": exercise["arguments"]}}
            ],
        },
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=True,
    )
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    runs = [i for i, value in enumerate(mask) if value and (i == 0 or not mask[i - 1])]
    if len(runs) != 2:
        raise ValueError("Expected read and final edit generation spans")
    target_start = runs[-1]
    mask = [int(value and i >= target_start) for i, value in enumerate(mask)]
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, chat_template=training_template, tokenize=False, enable_thinking=True
    )
    inference_rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=True)
    if rendered != inference_rendered:
        raise ValueError("10/05 training and inference template disagree with explicit thinking=True")
    target = tokenizer.decode(ids[target_start:], skip_special_tokens=False)
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", target, re.S)
    if len(matches) != 1 or json.loads(matches[0]) != {"name": "edit", "arguments": exercise["arguments"]}:
        raise ValueError("Rendered target did not preserve the exact Pi call")
    if not 128 <= sum(mask) <= 2048 or len(ids) >= 8192 or EOS in ids:
        return {}
    ids.append(EOS)
    mask.append(0)
    weights = [float(v) for v in mask[1:]] + [0.0]
    if len(ids) != len(weights) or weights[-1] or weights[target_start - 1] != 1:
        raise ValueError("Shifted target mask is invalid")
    if any(weights[: target_start - 1]):
        raise ValueError("Prompt/read-observation tokens carry loss")
    return {
        "messages_json": json.dumps(messages, ensure_ascii=False),
        "rendered_text": rendered,
        "input_ids": ids,
        "loss_weights": weights,
        "sequence_tokens": len(ids),
        "target_tokens": sum(mask),
        "target_start": target_start,
    }


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records, counts = source_bodies(args.source)
    (args.output / "source-candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    (args.output / "source-counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


def build(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template = args.template.read_text()
    tools = json.loads(args.tools.read_text())
    edit_schema = next(t["function"]["parameters"] for t in tools if t["function"]["name"] == "edit")
    allowed = set(json.loads(args.allowed.read_text()))
    sources = [json.loads(line) for line in (args.output / "source-candidates.jsonl").open()]
    counts = collections.Counter()
    selected = []
    task_counts = collections.Counter()
    quotas = {
        "train": {1: 333, 2: 333, 3: 167, 4: 167},
        "validation": {1: 34, 2: 33, 3: 17, 4: 16},
        "test": {1: 34, 2: 33, 3: 17, 4: 16},
    }
    chosen = collections.Counter()
    for source in sources:
        if source["source_code_sha256"] not in allowed:
            continue
        split = source["split"]
        if task_counts[(split, source["task_id"])] >= 4:
            continue
        for cardinality in sorted(quotas[split], key=lambda k: -(quotas[split][k] - chosen[(split, k)])):
            if chosen[(split, cardinality)] >= quotas[split][cardinality]:
                continue
            exercise = make_exercise(source, cardinality)
            if exercise is None:
                counts["fixture_unavailable"] += 1
                continue
            jsonschema.validate(exercise["arguments"], edit_schema)
            encoded = render(exercise, tokenizer, template, tools)
            if not encoded:
                counts["length_gate"] += 1
                continue
            selected.append({**exercise, **encoded})
            chosen[(split, cardinality)] += 1
            task_counts[(split, source["task_id"])] += 1
            break
    for split, quota in quotas.items():
        if any(chosen[(split, n)] != count for n, count in quota.items()):
            raise ValueError(f"Insufficient clean distinct sources for {split}: {chosen}")
    for split in quotas:
        records = [x for x in selected if x["split"] == split]
        records.sort(key=lambda r: digest(SEED + r["id"]))
        rows = [
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, set)) else v
                for k, v in row.items()
                if k not in ("code", "parent_instruction", "all_source_tasks", "all_source_splits")
            }
            for row in records
        ]
        table = pa.Table.from_pylist(rows)
        table = table.set_column(
            table.schema.get_field_index("input_ids"),
            "input_ids",
            pa.array([r["input_ids"] for r in rows], type=pa.list_(pa.int32())),
        )
        table = table.set_column(
            table.schema.get_field_index("loss_weights"),
            "loss_weights",
            pa.array([r["loss_weights"] for r in rows], type=pa.list_(pa.float32())),
        )
        pq.write_table(table, args.output / (split + ".parquet"), compression="zstd")
    (args.output / "selected-exercises.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected)
    )
    summary = {
        split: {
            "rows": sum(x["split"] == split for x in selected),
            "tasks": len({x["task_id"] for x in selected if x["split"] == split}),
            "sequence_tokens": sum(x["sequence_tokens"] for x in selected if x["split"] == split),
            "target_tokens": sum(x["target_tokens"] for x in selected if x["split"] == split),
            "cardinalities": {str(n): chosen[(split, n)] for n in quotas[split]},
        }
        for split in quotas
    }
    (args.output / "render-summary.json").write_text(
        json.dumps({"splits": summary, "rejections": counts}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "build"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--tools", type=Path)
    parser.add_argument("--allowed", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
