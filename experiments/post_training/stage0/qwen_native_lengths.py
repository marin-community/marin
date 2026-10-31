# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit every row of one CL12 source under one pinned native Qwen template on CPU."""

import argparse
import functools
import gzip
import hashlib
import json
import logging
import multiprocessing
import re
from dataclasses import asdict, dataclass

from rigging.filesystem.buckets import filesystem_for
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.stage0.axolotl_stage0 import QWEN_NATIVE_TEMPLATES
from experiments.post_training.stage0.convert import MAX_RENDERED_TOKENS

logger = logging.getLogger(__name__)
ASSISTANT_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)
EMPTY_THINK = re.compile(r"<think>\s*</think>")


@dataclass(frozen=True)
class RowAudit:
    row_id: str
    tokens: int
    reasoning_turns_dropped: int
    # The last assistant turn carries reasoning, and whether the template drops it.
    final_reasoning: bool
    final_reasoning_dropped: bool
    empty_think_blocks: int
    structured_tool_calls: int


@functools.cache
def native_tokenizer(family: str) -> PreTrainedTokenizerBase:
    repo, revision = FAMILY_TOKENIZER_REVISIONS[family]
    return AutoTokenizer.from_pretrained(repo, revision=revision)


def audit_row(family: str, row: dict) -> RowAudit:
    tokenizer = native_tokenizer(family)
    template = QWEN_NATIVE_TEMPLATES[family]
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False)
    tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    # Trace reasoning retention through the actual template, including its tool-response rules.
    traced = []
    markers = {}
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and (message.get("reasoning_content") or "").strip():
            markers[index] = f"MARIN_REASONING_{row['row_id']}_{index}"
            traced.append({**message, "reasoning_content": markers[index]})
        else:
            traced.append(message)
    trace = tokenizer.apply_chat_template(traced, chat_template=template, tokenize=False)
    final = max(index for index, message in enumerate(messages) if message["role"] == "assistant")
    empty = sum(len(EMPTY_THINK.findall(span)) for span in ASSISTANT_SPAN.findall(rendered))
    return RowAudit(
        row["row_id"],
        tokens,
        sum(marker not in trace for marker in markers.values()),
        final in markers,
        final in markers and markers[final] not in trace,
        empty,
        sum(len(message.get("tool_calls") or []) for message in messages),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(QWEN_NATIVE_TEMPLATES), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True, help="new artifact directory; source is never changed")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    source_fs, source_path = filesystem_for(args.source)
    output_fs, output_path = filesystem_for(args.out)
    if output_fs.exists(f"{output_path}/manifest.json"):
        raise FileExistsError(f"Audit already exists: {args.out}")
    output_fs.makedirs(output_path, exist_ok=True)
    repo, revision = FAMILY_TOKENIZER_REVISIONS[args.family]
    report = {
        "source": args.source,
        "family": args.family,
        "tokenizer": repo,
        "revision": revision,
        "template_sha256": hashlib.sha256(QWEN_NATIVE_TEMPLATES[args.family].encode()).hexdigest(),
        "max_tokens": MAX_RENDERED_TOKENS,
        "rows": 0,
        "retained_tokens": 0,
        "rows_losing_reasoning": 0,
        "reasoning_turns_dropped": 0,
        "rows_with_final_reasoning": 0,
        "rows_losing_final_reasoning": 0,
        "rows_with_empty_think": 0,
        "empty_think_blocks": 0,
        "structured_tool_call_rows": 0,
        "overlength_row_ids": [],
    }
    with (
        source_fs.open(source_path, "rb") as raw,
        gzip.open(raw, "rt") as source,
        output_fs.open(f"{output_path}/rows.jsonl", "wt") as results,
        multiprocessing.get_context("spawn").Pool(args.workers) as pool,
    ):
        rows = (json.loads(line) for line in source)
        for audit in pool.imap(functools.partial(audit_row, args.family), rows, chunksize=16):
            results.write(json.dumps(asdict(audit)) + "\n")
            report["rows"] += 1
            if audit.tokens > MAX_RENDERED_TOKENS:
                report["overlength_row_ids"].append(audit.row_id)
            else:
                report["retained_tokens"] += audit.tokens
            report["rows_losing_reasoning"] += bool(audit.reasoning_turns_dropped)
            report["reasoning_turns_dropped"] += audit.reasoning_turns_dropped
            report["rows_with_final_reasoning"] += audit.final_reasoning
            report["rows_losing_final_reasoning"] += audit.final_reasoning_dropped
            report["rows_with_empty_think"] += bool(audit.empty_think_blocks)
            report["empty_think_blocks"] += audit.empty_think_blocks
            report["structured_tool_call_rows"] += bool(audit.structured_tool_calls)
            if report["rows"] % 1000 == 0:
                logger.info("%s: audited %d rows", args.family, report["rows"])
    with output_fs.open(f"{output_path}/manifest.json", "wt") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Audit complete: %s", {k: v for k, v in report.items() if k != "overlength_row_ids"})


if __name__ == "__main__":
    main()
