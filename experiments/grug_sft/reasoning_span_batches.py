# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count and repair reasoning-span shapes in exported 10.05 SFT batches (marin-community/marin#10519).

Under ``Reasoning: /think`` the ``marin-v3`` renderer emits one ``<|start_think|>…<|end_think|>`` span per
analysis message and no span for an assistant turn without analysis. #10519 changes the target format so every
``/think`` assistant turn starts with exactly one span, empty when the turn has no reasoning. This module applies
that format to the token batches ``export_sft_batches.py`` wrote, so the first steps of the 10.05 SFT can be
replayed on repaired data without re-rendering the mix:

* a ``/think`` assistant turn that does not start with ``<|start_think|>`` gets an empty span inserted after its
  header newline;
* back-to-back spans (``<|end_think|><|start_think|>``) inside one turn are joined with one newline token, the
  way the renderer joins consecutive analysis messages.

Spans separated by other text, or starting after other text, are counted and left unchanged: the renderer fix
moves that analysis into the initial span, and #10519 leaves that reordering as an open question.

A conversation's mode comes from its system header. Documents the packer left-sliced start mid-conversation,
have no header, and are counted as ``unknown`` and left unchanged.

Rows keep their length: a longer row loses tokens from its tail, a shorter one is padded (segment id -1, loss
weight 0) the way the packer pads.

    uv run python -m experiments.grug_sft.reasoning_span_batches --input ROOT/batches --steps 50 --report scan.json
    uv run python -m experiments.grug_sft.reasoning_span_batches --input ROOT/batches --steps 50 \\
        --output ROOT/batches_10519 --report patch.json
"""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from rigging.log_setup import configure_logging
from transformers import PreTrainedTokenizerBase

from experiments.grug_sft.replay_data import StepBatch, read_step, write_step

logger = logging.getLogger(__name__)

PAD_SEGMENT = -1
THINK_MARKER = "Reasoning: /think"
NOTHINK_MARKER = "Reasoning: /nothink"


@dataclass(frozen=True)
class SpanTokens:
    """Token ids of the chat-format pieces the scan matches; sequences because a test tokenizer may split them."""

    bos: int
    start_think: int
    end_think: int
    newline: int
    pad: int
    eot: tuple[int, ...]
    start_header: tuple[int, ...]
    system_header: tuple[int, ...]
    assistant_header: tuple[int, ...]
    think_marker: tuple[int, ...]
    nothink_marker: tuple[int, ...]
    tool_call: tuple[int, ...]

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> "SpanTokens":
        def ids(text: str) -> tuple[int, ...]:
            return tuple(tokenizer.encode(text, add_special_tokens=False))

        def one(text: str) -> int:
            (token,) = ids(text)
            return token

        return cls(
            bos=tokenizer.bos_token_id,
            start_think=one("<|start_think|>"),
            end_think=one("<|end_think|>"),
            newline=one("\n"),
            pad=tokenizer.eos_token_id,
            eot=ids("<|eot_id|>"),
            start_header=ids("<|start_header_id|>"),
            system_header=ids("<|start_header_id|>system<|end_header_id|>"),
            assistant_header=ids("<|start_header_id|>assistant<|end_header_id|>\n"),
            think_marker=ids(THINK_MARKER),
            nothink_marker=ids(NOTHINK_MARKER),
            tool_call=ids("<tool_call>"),
        )


# ``marin-community/marin-tokenizer`` and the 10.05 checkpoint's tokenizer (Llama 3 vocabulary).
MARIN_TOKENS = SpanTokens(
    bos=128000,
    start_think=128002,
    end_think=128003,
    newline=198,
    pad=128001,
    eot=(128009,),
    start_header=(128006,),
    system_header=(128006, 9125, 128007),
    assistant_header=(128006, 78191, 128007, 198),
    think_marker=(26197, 287, 25, 611, 27963),
    nothink_marker=(26197, 287, 25, 611, 2201, 27963),
    tool_call=(128005,),
)


class Mode(StrEnum):
    THINK = "think"
    NOTHINK = "nothink"
    UNDEFINED = "undefined"
    UNKNOWN = "unknown"


class TurnShape(StrEnum):
    """Reasoning-span shape of one assistant turn."""

    ONE_SPAN = "one_span"
    ZERO_SPAN_TEXT = "zero_span_text"
    ZERO_SPAN_TOOL_CALL = "zero_span_tool_call"
    ADJACENT_SPANS = "adjacent_spans"
    SEPARATED_SPANS = "separated_spans"
    LATE_SPAN = "late_span"


ZERO_SPAN_SHAPES = frozenset({TurnShape.ZERO_SPAN_TEXT, TurnShape.ZERO_SPAN_TOOL_CALL})


def find_pattern(tokens: np.ndarray, pattern: tuple[int, ...]) -> np.ndarray:
    """Start positions of every occurrence of ``pattern`` in ``tokens``."""
    positions = np.flatnonzero(tokens[: len(tokens) - len(pattern) + 1] == pattern[0])
    for offset, token in enumerate(pattern[1:], start=1):
        positions = positions[tokens[positions + offset] == token]
    return positions


def starts_with(tokens: np.ndarray, pattern: tuple[int, ...], at: int = 0) -> bool:
    return len(tokens) >= at + len(pattern) and tuple(tokens[at : at + len(pattern)]) == pattern


def conversation_mode(tokens: np.ndarray, spans: SpanTokens) -> Mode:
    """Reasoning mode from the system header a rendered conversation starts with."""
    start = 1 if len(tokens) and tokens[0] == spans.bos else 0
    if not starts_with(tokens, spans.start_header, start):
        return Mode.UNKNOWN
    if not starts_with(tokens, spans.system_header, start):
        return Mode.UNDEFINED
    body = start + len(spans.system_header)
    if starts_with(tokens, spans.think_marker, body):
        return Mode.THINK
    if starts_with(tokens, spans.nothink_marker, body):
        return Mode.NOTHINK
    return Mode.UNDEFINED


def assistant_turns(tokens: np.ndarray, spans: SpanTokens) -> list[tuple[int, int]]:
    """``[content_start, content_end)`` per assistant turn; the end is its ``<|eot_id|>`` or the document end."""
    eots = find_pattern(tokens, spans.eot)
    turns = []
    for header in find_pattern(tokens, spans.assistant_header):
        content_start = int(header) + len(spans.assistant_header)
        following = np.searchsorted(eots, content_start)
        content_end = int(eots[following]) if following < len(eots) else len(tokens)
        turns.append((content_start, content_end))
    return turns


def turn_shape(content: np.ndarray, spans: SpanTokens) -> TurnShape:
    starts = np.flatnonzero(content == spans.start_think)
    if len(starts) == 0:
        return TurnShape.ZERO_SPAN_TOOL_CALL if starts_with(content, spans.tool_call) else TurnShape.ZERO_SPAN_TEXT
    if starts[0] != 0:
        return TurnShape.LATE_SPAN
    if len(starts) == 1:
        return TurnShape.ONE_SPAN
    if np.any((content[:-1] == spans.end_think) & (content[1:] == spans.start_think)):
        return TurnShape.ADJACENT_SPANS
    return TurnShape.SEPARATED_SPANS


@dataclass(frozen=True)
class DocumentPatch:
    tokens: np.ndarray
    loss_weight: np.ndarray
    shapes: Counter
    inserted_spans: int
    merged_pairs: int


def patch_document(tokens: np.ndarray, loss_weight: np.ndarray, spans: SpanTokens, *, repair: bool) -> DocumentPatch:
    """Classify each assistant turn of one rendered conversation and, for ``/think`` with ``repair``, fix its spans."""
    shapes: Counter = Counter()
    turns = assistant_turns(tokens, spans)
    if not repair or conversation_mode(tokens, spans) != Mode.THINK:
        for start, end in turns:
            shapes[turn_shape(tokens[start:end], spans)] += 1
        return DocumentPatch(tokens, loss_weight, shapes, 0, 0)

    pieces_tokens: list[np.ndarray] = []
    pieces_weight: list[np.ndarray] = []
    cursor = 0
    inserted = merged = 0
    for start, end in turns:
        content, weight = tokens[start:end], loss_weight[start:end]
        shape = turn_shape(content, spans)
        shapes[shape] += 1
        pieces_tokens.append(tokens[cursor:start])
        pieces_weight.append(loss_weight[cursor:start])
        # New tokens take the weight of the position that predicts them: the header newline for an empty span,
        # the replaced ``<|start_think|>`` for the joining newline.
        if shape in ZERO_SPAN_SHAPES:
            pieces_tokens.append(np.array([spans.start_think, spans.end_think], dtype=tokens.dtype))
            pieces_weight.append(np.full(2, loss_weight[start - 1], dtype=loss_weight.dtype))
            inserted += 1
        elif shape == TurnShape.ADJACENT_SPANS:
            pairs = np.flatnonzero((content[:-1] == spans.end_think) & (content[1:] == spans.start_think))
            keep = np.ones(len(content), dtype=bool)
            keep[pairs] = False
            content = content.copy()
            content[pairs + 1] = spans.newline
            content, weight = content[keep], weight[keep]
            merged += len(pairs)
        pieces_tokens.append(content)
        pieces_weight.append(weight)
        cursor = end
    pieces_tokens.append(tokens[cursor:])
    pieces_weight.append(loss_weight[cursor:])
    return DocumentPatch(np.concatenate(pieces_tokens), np.concatenate(pieces_weight), shapes, inserted, merged)


@dataclass(frozen=True)
class RowReport:
    documents: Counter
    """Documents by ``Mode``."""
    think_turns: Counter
    """Assistant turns of ``/think`` documents by ``TurnShape``."""
    other_turns: Counter
    """Assistant turns of every other document by ``TurnShape``."""
    inserted_spans: int
    merged_pairs: int
    length_delta: int
    """Tokens the row's documents grew by, before trimming or padding."""
    loss_tokens_before: int
    loss_tokens_after: int


def patch_row(
    tokens: np.ndarray,
    loss_weight: np.ndarray,
    segment_ids: np.ndarray,
    spans: SpanTokens,
    *,
    repair: bool,
    pad_token: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RowReport]:
    """Patch each document of one packed row and restore the row length."""
    documents: Counter = Counter()
    think_turns: Counter = Counter()
    other_turns: Counter = Counter()
    inserted = merged = 0
    out_tokens: list[np.ndarray] = []
    out_weight: list[np.ndarray] = []
    out_segments: list[np.ndarray] = []
    boundaries = np.flatnonzero(np.diff(segment_ids)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(tokens)]])
    for start, end in zip(starts, ends, strict=True):
        segment = int(segment_ids[start])
        if segment == PAD_SEGMENT:
            continue
        mode = conversation_mode(tokens[start:end], spans)
        documents[mode] += 1
        patch = patch_document(tokens[start:end], loss_weight[start:end], spans, repair=repair)
        (think_turns if mode == Mode.THINK else other_turns).update(patch.shapes)
        inserted += patch.inserted_spans
        merged += patch.merged_pairs
        out_tokens.append(patch.tokens)
        out_weight.append(patch.loss_weight)
        out_segments.append(np.full(len(patch.tokens), segment, dtype=segment_ids.dtype))
    new_tokens = np.concatenate(out_tokens)
    new_weight = np.concatenate(out_weight)
    new_segments = np.concatenate(out_segments)
    length = len(tokens)
    delta = len(new_tokens) - int(np.sum(segment_ids != PAD_SEGMENT))
    if len(new_tokens) > length:
        new_tokens, new_weight, new_segments = new_tokens[:length], new_weight[:length], new_segments[:length]
    elif len(new_tokens) < length:
        pad = length - len(new_tokens)
        new_tokens = np.concatenate([new_tokens, np.full(pad, pad_token, dtype=tokens.dtype)])
        new_weight = np.concatenate([new_weight, np.zeros(pad, dtype=loss_weight.dtype)])
        new_segments = np.concatenate([new_segments, np.full(pad, PAD_SEGMENT, dtype=segment_ids.dtype)])
        # The position before padding would predict a pad token.
        new_weight[length - pad - 1] = 0
    report = RowReport(
        documents,
        think_turns,
        other_turns,
        inserted,
        merged,
        delta,
        int(np.count_nonzero(loss_weight)),
        int(np.count_nonzero(new_weight)),
    )
    return new_tokens, new_weight, new_segments, report


def batch_pad_token(batch: StepBatch, spans: SpanTokens) -> int:
    """The token the packer padded with, or ``<|end_of_text|>`` when the batch has no padding."""
    padded = batch.tokens[batch.segment_ids == PAD_SEGMENT]
    return int(padded[0]) if len(padded) else spans.pad


def patch_batch(batch: StepBatch, spans: SpanTokens, *, repair: bool) -> tuple[StepBatch, dict]:
    pad_token = batch_pad_token(batch, spans)
    rows = [
        patch_row(
            batch.tokens[row], batch.loss_weight[row], batch.segment_ids[row], spans, repair=repair, pad_token=pad_token
        )
        for row in range(batch.tokens.shape[0])
    ]
    reports = [report for *_, report in rows]
    patched = StepBatch(
        tokens=np.stack([tokens for tokens, *_ in rows]),
        loss_weight=np.stack([weight for _, weight, *_ in rows]),
        segment_ids=np.stack([segments for _, _, segments, _ in rows]),
        sliding_window=batch.sliding_window,
    )
    summary = {
        "documents": dict(sum((r.documents for r in reports), Counter())),
        "think_turns": dict(sum((r.think_turns for r in reports), Counter())),
        "other_turns": dict(sum((r.other_turns for r in reports), Counter())),
        "inserted_spans": sum(r.inserted_spans for r in reports),
        "merged_pairs": sum(r.merged_pairs for r in reports),
        "rows_trimmed": sum(r.length_delta > 0 for r in reports),
        "rows_padded": sum(r.length_delta < 0 for r in reports),
        "max_length_delta": max(abs(r.length_delta) for r in reports),
        "loss_tokens_before": sum(r.loss_tokens_before for r in reports),
        "loss_tokens_after": sum(r.loss_tokens_after for r in reports),
    }
    return patched, summary


def process_step(input_root: str, output_root: str | None, step: int) -> dict:
    batch = read_step(input_root, step)
    patched, summary = patch_batch(batch, MARIN_TOKENS, repair=output_root is not None)
    if output_root is not None:
        write_step(output_root, step, patched)
    summary["step"] = step
    logger.info("step %d: %s", step, json.dumps(summary, sort_keys=True))
    return summary


def aggregate(summaries: list[dict]) -> dict:
    total: dict = {"steps": len(summaries)}
    for key in ("documents", "think_turns", "other_turns"):
        total[key] = dict(sum((Counter(s[key]) for s in summaries), Counter()))
    for key in (
        "inserted_spans",
        "merged_pairs",
        "rows_trimmed",
        "rows_padded",
        "loss_tokens_before",
        "loss_tokens_after",
    ):
        total[key] = sum(s[key] for s in summaries)
    total["max_length_delta"] = max(s["max_length_delta"] for s in summaries)
    return total


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory of exported step batches")
    parser.add_argument("--steps", type=int, required=True, help="Process steps [first, first + steps)")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--output", help="Write patched batches here; omitted, the run only counts")
    parser.add_argument(
        "--report", required=True, help="JSON file (local or object store) for per-step and total counts"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output is not None and args.output.rstrip("/") == args.input.rstrip("/"):
        raise ValueError("--output must differ from --input")

    steps = range(args.first, args.first + args.steps)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(process_step, [args.input] * len(steps), [args.output] * len(steps), steps))
    report = {"input": args.input, "output": args.output, "total": aggregate(summaries), "steps": summaries}
    fs, path = filesystem_for(args.report)
    fs.pipe(path, json.dumps(report, indent=1, sort_keys=True).encode())
    logger.info("total: %s", json.dumps(report["total"], sort_keys=True))


if __name__ == "__main__":
    main()
