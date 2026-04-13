#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=1.0",
#     "pydantic>=2.0",
# ]
# ///
"""
Agent-driven log scanner with two composable modes: grep and summarize.

Usage:
    logscan grep <logfile> <query>          — find matching lines (outputs line-numbered source)
    logscan summarize <logfile> <query>     — produce a markdown report

All modes support: [--chunk-tokens N] [--concurrency N] [--model M] [-v] [--stdin]

Pipe modes together:
    logscan grep log.txt "errors" | logscan summarize --stdin "summarize these errors"

Requires GEMINI_API_KEY environment variable.
"""

import argparse
import json
import math
import os
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, resp: types.GenerateContentResponse) -> None:
        meta = resp.usage_metadata
        if not meta:
            return
        with self._lock:
            self.input_tokens += meta.prompt_token_count or 0
            self.output_tokens += meta.candidates_token_count or 0

    def report(self) -> str:
        total = self.input_tokens + self.output_tokens
        return f"Tokens — input: {self.input_tokens:,}  " f"output: {self.output_tokens:,}  " f"total: {total:,}"


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class GrepResult(BaseModel):
    line_numbers: list[int]


class SummarizeResult(BaseModel):
    summary: str


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    index: int
    text: str
    start_line: int  # 1-based
    end_line: int  # inclusive


def split_into_chunks(lines: list[str], chunk_tokens: int) -> list[Chunk]:
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0

    while i < len(lines):
        buf: list[str] = []
        char_count = 0
        j = i
        while j < len(lines) and char_count < chunk_chars:
            buf.append(lines[j])
            char_count += len(lines[j])
            j += 1

        chunks.append(
            Chunk(
                index=chunk_idx,
                text="".join(buf),
                start_line=i + 1,
                end_line=j,
            )
        )
        chunk_idx += 1
        i = j

    return chunks


def number_lines(text: str, start: int) -> str:
    out: list[str] = []
    for i, line in enumerate(text.splitlines(keepends=True)):
        out.append(f"{start + i:>8} | {line}")
    return "".join(out)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GREP_SYSTEM = textwrap.dedent(
    """\
    You are a log line filter. You receive a section of a log file with line
    numbers and a query describing what to look for.

    Return ONLY the line numbers of lines that match the query. Be highly
    selective — only include lines that directly and specifically match what
    the query asks for. Do not include tangentially related lines.

    If nothing matches, return an empty line_numbers array.
"""
)

SUMMARIZE_SYSTEM = textwrap.dedent(
    """\
    You are a log summarizer. You receive a section of a log file (or a set of
    previous summaries) and a query describing what to focus on.

    Produce a concise summary (100-300 words) covering:
    - Key events and patterns relevant to the query
    - Counts and frequencies of recurring items
    - Timestamps and line ranges for important events
    - Anomalies or state changes

    Be specific: include numbers, timestamps, and identifiers. Do not pad with
    generic statements.
"""
)

SUMMARIZE_REDUCE_SYSTEM = textwrap.dedent(
    """\
    You are a log analysis report writer. You receive summaries from multiple
    sections of a log file and the original query.

    Produce a comprehensive markdown report that:
    - Synthesizes all summaries into a coherent narrative
    - Quantifies: counts, time ranges, rates, affected entities
    - Includes specific line numbers and timestamps for key events
    - Groups findings by theme or error type
    - Ends with actionable conclusions

    Use markdown formatting: headers, bullet points, bold for emphasis.
"""
)


# ---------------------------------------------------------------------------
# Map functions
# ---------------------------------------------------------------------------


def map_grep(
    client: genai.Client,
    model: str,
    chunk: Chunk,
    query: str,
    usage: Usage,
) -> list[int]:
    numbered = number_lines(chunk.text, chunk.start_line)
    prompt = f"Query: {query}\n\nLines {chunk.start_line}\u2013{chunk.end_line}:\n\n{numbered}"

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=GREP_SYSTEM,
            response_mime_type="application/json",
            response_schema=GrepResult,
            temperature=0.1,
        ),
    )
    usage.add(resp)
    return GrepResult.model_validate_json(resp.text).line_numbers


def map_summarize(
    client: genai.Client,
    model: str,
    chunk: Chunk,
    query: str,
    usage: Usage,
) -> str:
    numbered = number_lines(chunk.text, chunk.start_line)
    prompt = f"Query: {query}\n\nLines {chunk.start_line}\u2013{chunk.end_line}:\n\n{numbered}"

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SUMMARIZE_SYSTEM,
            response_mime_type="application/json",
            response_schema=SummarizeResult,
            temperature=0.2,
        ),
    )
    usage.add(resp)
    return SummarizeResult.model_validate_json(resp.text).summary


# ---------------------------------------------------------------------------
# Reduce: summarize (LLM-based hierarchical merge)
# ---------------------------------------------------------------------------


def reduce_summaries(
    client: genai.Client,
    model: str,
    summaries: list[str],
    query: str,
    usage: Usage,
    concurrency: int,
) -> str:
    if len(summaries) > 20:
        group_size = max(2, int(math.sqrt(len(summaries))))
        batches = [summaries[i : i + group_size] for i in range(0, len(summaries), group_size)]
        print(
            f"\nCombining {len(summaries)} summaries in {len(batches)} groups ...",
            file=sys.stderr,
        )

        combined: list[str | None] = [None] * len(batches)
        completed = 0

        with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as executor:
            future_to_idx = {
                executor.submit(
                    _merge_summary_batch,
                    client,
                    model,
                    batch,
                    query,
                    usage,
                ): bid
                for bid, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                bid = future_to_idx[future]
                combined[bid] = future.result()
                completed += 1
                print(f"  [{completed}/{len(batches)}] combine group {bid} done", file=sys.stderr)

        summaries = [s for s in combined if s is not None]

    return _merge_summary_batch(client, model, summaries, query, usage, final=True)


def _merge_summary_batch(
    client: genai.Client,
    model: str,
    summaries: list[str],
    query: str,
    usage: Usage,
    final: bool = False,
) -> str:
    numbered = "\n\n---\n\n".join(f"**Section {i+1}:**\n{s}" for i, s in enumerate(summaries))
    prompt = f"Original query: {query}\n\n" f"Summaries to merge:\n\n{numbered}"

    system = SUMMARIZE_REDUCE_SYSTEM if final else SUMMARIZE_SYSTEM

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=SummarizeResult,
            temperature=0.2,
        ),
    )
    usage.add(resp)
    return SummarizeResult.model_validate_json(resp.text).summary


# ---------------------------------------------------------------------------
# Parallel map executor
# ---------------------------------------------------------------------------


def run_map(
    client: genai.Client,
    model: str,
    chunks: list[Chunk],
    query: str,
    usage: Usage,
    concurrency: int,
    verbose: bool,
    map_fn: Any,
) -> list[Any]:
    total = len(chunks)
    results: list[Any | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=min(concurrency, total)) as executor:
        future_to_idx = {executor.submit(map_fn, client, model, chunk, query, usage): chunk.index for chunk in chunks}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            print(f"[{completed}/{total}] chunk {idx} done", file=sys.stderr)
            if verbose:
                r = results[idx]
                print(
                    json.dumps(r, indent=2) if not isinstance(r, str) else r,
                    file=sys.stderr,
                )

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_TOKENS = {"grep": 5_000, "summarize": 50_000}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent-driven log scanner: grep or summarize.",
    )
    parser.add_argument("mode", choices=["grep", "summarize"])
    parser.add_argument("logfile", nargs="?", type=Path)
    parser.add_argument("query")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--stdin", action="store_true", help="Read input from stdin")
    args = parser.parse_args()

    chunk_tokens = args.chunk_tokens or DEFAULT_CHUNK_TOKENS[args.mode]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    if args.stdin:
        text = sys.stdin.read()
    elif args.logfile:
        text = args.logfile.read_text()
    else:
        print("Error: provide a logfile or --stdin.", file=sys.stderr)
        sys.exit(1)

    lines = text.splitlines(keepends=True)
    if not lines:
        print("Empty input.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    usage = Usage()
    chunks = split_into_chunks(lines, chunk_tokens)
    print(
        f"{args.mode}: {len(lines)} lines, {len(chunks)} chunks "
        f"(~{chunk_tokens} tok/chunk, {min(args.concurrency, len(chunks))} workers)",
        file=sys.stderr,
    )

    if args.mode == "grep":
        all_line_nums: list[list[int]] = run_map(
            client,
            args.model,
            chunks,
            args.query,
            usage,
            args.concurrency,
            args.verbose,
            map_grep,
        )
        merged = sorted(set(n for batch in all_line_nums for n in batch))
        for n in merged:
            if 1 <= n <= len(lines):
                print(f"{n}: {lines[n - 1]}", end="")

    elif args.mode == "summarize":
        summaries: list[str] = run_map(
            client,
            args.model,
            chunks,
            args.query,
            usage,
            args.concurrency,
            args.verbose,
            map_summarize,
        )
        print("\nReducing ...", file=sys.stderr)
        report = reduce_summaries(
            client,
            args.model,
            summaries,
            args.query,
            usage,
            args.concurrency,
        )
        print(report)

    print(f"\n{usage.report()}", file=sys.stderr)


if __name__ == "__main__":
    main()
