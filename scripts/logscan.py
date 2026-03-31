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
Log scanner that uses Gemini to analyze large log files in overlapping chunks,
processing them in parallel and producing a merged summary.

Usage:
    uv run scripts/logscan.py <logfile> <query>
        [--chunk-tokens 50000] [--overlap 0.2]
        [--concurrency 16] [--model gemini-2.5-flash-lite]

Requires GEMINI_API_KEY environment variable.
"""

import argparse
import json
import os
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class HighlightedLine(BaseModel):
    line_number: int
    text: str
    reason: str


class ChunkResult(BaseModel):
    chunk_index: int
    summary: str
    highlighted_lines: list[HighlightedLine]


class MergedSummary(BaseModel):
    summary: str
    highlighted_lines: list[HighlightedLine]


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


def split_into_chunks(
    lines: list[str],
    chunk_tokens: int = 50_000,
    overlap_fraction: float = 0.2,
) -> list[Chunk]:
    """Split lines into overlapping windows of approximately chunk_tokens tokens."""
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = int(chunk_chars * overlap_fraction)
    stride_chars = chunk_chars - overlap_chars

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

        advance_chars = 0
        next_i = i

        while next_i < j and advance_chars < stride_chars:
            advance_chars += len(lines[next_i])
            next_i += 1

        if next_i == i:
            next_i += 1

        i = next_i

    return chunks


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHUNK_SYSTEM = textwrap.dedent(
    """\
You are a log analysis assistant. The user will provide a section of a log
file (with line numbers) and a query describing what to look for.

Your job is to be SELECTIVE. Do NOT highlight every line that mentions the
topic — only highlight lines that are genuinely significant:
- First occurrences of a new error or pattern
- Lines that show a state *change* (things getting worse, a new failure mode)
- The most severe instances (not every repeat of the same message)
- Lines that establish causality or timing between events

If a message repeats 50 times, highlight 1-2 representative instances and
note the count in your summary — do NOT list all 50.

Keep summaries concise (2-4 sentences). Focus on patterns and counts, not
individual line descriptions.

If nothing relevant is found, set summary to exactly "No matches." and
return an empty highlighted_lines list. Do not elaborate or explain why
nothing was found.
"""
)

MERGE_SYSTEM = textwrap.dedent(
    """\
You are a log analysis assistant. You will receive per-chunk summaries and
highlighted lines from scanning a large log file. Produce a single merged
report.

Rules:
- Deduplicate lines that appeared in overlapping chunks (same line_number).
- Order highlighted_lines by line_number ascending.
- The summary should be a coherent narrative covering: what went wrong,
  when it started, how it progressed, and what the impact was.
- Quantify: mention counts of recurring errors, affected servers, time ranges.
- Keep the summary under ~10 sentences.
- Keep highlighted_lines to the ~20 most important lines across the entire log.
"""
)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def number_lines(text: str, start: int) -> str:
    """Prefix each line with its original line number."""
    out: list[str] = []
    for i, line in enumerate(text.splitlines(keepends=True)):
        out.append(f"{start + i:>8} | {line}")
    return "".join(out)


def analyze_chunk(
    client: genai.Client,
    model: str,
    chunk: Chunk,
    query: str,
) -> ChunkResult:
    numbered = number_lines(chunk.text, chunk.start_line)

    prompt = (
        f"Query: {query}\n\n"
        f"Chunk index: {chunk.index}\n"
        f"Lines {chunk.start_line}-{chunk.end_line}:\n\n"
        f"{numbered}"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=CHUNK_SYSTEM,
            response_mime_type="application/json",
            response_schema=ChunkResult,
            temperature=0.1,
        ),
    )

    return ChunkResult.model_validate_json(resp.text)


def merge_results(
    client: genai.Client,
    model: str,
    chunk_results: list[ChunkResult],
    query: str,
) -> MergedSummary:
    results_json = json.dumps(
        [r.model_dump() for r in chunk_results],
        indent=2,
    )

    prompt = f"Original query: {query}\n\nPer-chunk results:\n{results_json}"

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=MERGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=MergedSummary,
            temperature=0.2,
        ),
    )

    return MergedSummary.model_validate_json(resp.text)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_chunk_md(result: ChunkResult) -> str:
    parts = [f"### Chunk {result.chunk_index}\n", f"{result.summary}\n"]

    if result.highlighted_lines:
        parts.append("")
        for hl in result.highlighted_lines:
            parts.append(f"- **L{hl.line_number}**: `{hl.text.rstrip()}` — {hl.reason}")
        parts.append("")

    return "\n".join(parts)


def format_merged_md(merged: MergedSummary) -> str:
    parts = ["---", "## Final Summary\n", merged.summary, ""]

    if merged.highlighted_lines:
        parts.append("### Key Lines\n")
        for hl in merged.highlighted_lines:
            parts.append(f"- **L{hl.line_number}**: `{hl.text.rstrip()}` — {hl.reason}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a log file with Gemini and surface relevant lines.",
    )

    parser.add_argument("logfile", type=Path, help="Path to the log file")
    parser.add_argument("query", help="What to look for in the logs")
    parser.add_argument("--chunk-tokens", type=int, default=50_000)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    lines = args.logfile.read_text().splitlines(keepends=True)

    if not lines:
        print("Empty log file.", file=sys.stderr)
        sys.exit(1)

    chunks = split_into_chunks(lines, args.chunk_tokens, args.overlap)

    total = len(chunks)
    concurrency = min(args.concurrency, total)

    print(
        f"Scanning {len(lines)} lines in {total} chunks "
        f"(~{args.chunk_tokens} tokens each, {int(args.overlap * 100)}% overlap, "
        f"{concurrency} workers)\n",
        file=sys.stderr,
    )

    chunk_results: list[ChunkResult | None] = [None] * total

    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(analyze_chunk, client, args.model, chunk, args.query): chunk.index for chunk in chunks
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            chunk_results[idx] = result

            completed += 1

            print(f"[{completed}/{total}] chunk {idx} done", file=sys.stderr)

            if result.highlighted_lines:
                print(format_chunk_md(result))

    results: list[ChunkResult] = [r for r in chunk_results if r is not None]

    if len(results) == 1:
        merged = MergedSummary(
            summary=results[0].summary,
            highlighted_lines=results[0].highlighted_lines,
        )
    else:
        print("\nMerging results ...", file=sys.stderr)
        merged = merge_results(client, args.model, results, args.query)

    print(format_merged_md(merged))


if __name__ == "__main__":
    main()
