# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop documents from a tokenized SFT store by reasoning length and give-up endings.

The filter reads an existing sharded Levanter store and writes a new one with the
same shard names. Kept documents are copied token for token, so a filtered store
differs from its source only in which documents it contains. Decisions use the
token ids alone: reasoning length is the longest ``<|start_think|>`` to
``<|end_think|>`` span, and give-up rules match text decoded from the end of the
last reasoning span and from the final answer.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, CacheMetadata, write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

START_THINK_ID = 128002
END_THINK_ID = 128003
READ_BATCH_ROWS = 256
WRITE_BATCH_ROWS = 256
# Decoded windows are cut from this many trailing tokens; both exceed the
# character windows below for this tokenizer.
REASONING_TAIL_TOKENS = 2048
ANSWER_HEAD_TOKENS = 4096
# Upper edges of the reasoning-length histogram reported per shard; the last
# bucket is everything longer.
REASONING_BUCKET_EDGES = (4_000, 8_000, 16_000, 32_000, 50_000, 64_000)
MATH_TAIL_CHARS = 1500
CODE_TAIL_CHARS = 4000

# Patterns from the reviewers' scripts for issue #11188
# (nemotron_math_abandon.py and nemotron_code_cp2.py), narrowed after the audit
# recorded in the research notebook.
MATH_GIVE_UP = re.compile(
    r"(?i)(given (the |that |my )?(limited |amount of )?time|time (is |are )?(limited|constraints?|up)\b|out of time"
    r"|time I have spent|spent (so much|too much|a lot of|enough|considerable) time"
    r"|\bI(?:'ll| will) guess|educated guess|best guess|my guess|as a guess|output the conjecture"
    r"|I(?:'ll| will) conjecture|cannot (produce|find|determine|derive) (a |the )?(correct|exact|closed|rigorous)"
    r"|I give up|ignore the .{0,30}issue|many might accept|hope(fully)? (it|this|that)('s| is)? (correct|right|accepted)"
    r"|without (a )?(full |rigorous |complete )?proof|I think I (must|have to|should) (conclude|stop|give))"
)
CODE_ANSWER_GIVE_UP = re.compile(
    r"(?i)placeholder|unable to (provide|produce|solve)"
    r"|not a (proper|correct|complete|real) (solution|answer|implementation)"
    r"|does NOT compute|dummy (answer|output|solution|implementation)|incomplete or incorrect"
    r"|may (TLE|time out|not pass)|hard-?coded? (the )?(sample|example|output)"
    r"|only (works|handles) (for )?small|not (fully )?(solved|implemented)"
)
CODE_REASONING_GIVE_UP = re.compile(
    r"(?i)hope (it|this)('s| is| will be)? (correct|right|accepted)|hope for the best|may not pass|might not pass"
    r"|(as|a|output) (a )?placeholder|cannot (solve|produce|devise)|can.t solve|I.m stuck|out of time|give up"
)


class GiveUpRule(StrEnum):
    NONE = "none"
    MATH = "math"
    CODE = "code"


class DropReason(StrEnum):
    REASONING_TOO_LONG = "reasoning_too_long"
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class FilterRules:
    """What to drop. ``max_reasoning_tokens=0`` disables the length cap."""

    max_reasoning_tokens: int
    give_up: GiveUpRule


@dataclass(frozen=True)
class Decision:
    row: int
    tokens: int
    reasoning_tokens: int
    reason: DropReason | None
    excerpt: str


def longest_reasoning_span(tokens: np.ndarray) -> tuple[int, int]:
    """Return ``(start, end)`` token offsets of the longest reasoning span, or ``(0, 0)``.

    ``start`` is the first reasoning token and ``end`` the position of its
    closing ``<|end_think|>``. A span without a close runs to the document end.
    """
    starts = np.flatnonzero(tokens == START_THINK_ID)
    if len(starts) == 0:
        return 0, 0
    ends = np.flatnonzero(tokens == END_THINK_ID)
    close_index = np.searchsorted(ends, starts)
    closes = np.where(close_index < len(ends), ends[np.minimum(close_index, len(ends) - 1)], len(tokens))
    longest = int(np.argmax(closes - starts))
    return int(starts[longest]) + 1, int(closes[longest])


def _match_excerpt(text: str, match: re.Match) -> str:
    return text[max(0, match.start() - 160) : match.end() + 160].replace("\n", " ")


def give_up_excerpt(tokens: np.ndarray, rule: GiveUpRule, tokenizer: Tokenizer) -> str | None:
    """Return the matched text when the last reasoning span or answer gives up, else None."""
    if rule == GiveUpRule.NONE:
        return None
    ends = np.flatnonzero(tokens == END_THINK_ID)
    if len(ends) == 0:
        return None
    last_end = int(ends[-1])
    starts = np.flatnonzero(tokens[:last_end] == START_THINK_ID)
    reasoning_start = int(starts[-1]) + 1 if len(starts) else 0
    tail_ids = tokens[max(reasoning_start, last_end - REASONING_TAIL_TOKENS) : last_end]
    tail = tokenizer.decode(tail_ids.tolist(), skip_special_tokens=False)
    if rule == GiveUpRule.MATH:
        tail = tail[-MATH_TAIL_CHARS:]
        match = MATH_GIVE_UP.search(tail)
        return _match_excerpt(tail, match) if match else None
    answer = tokenizer.decode(
        tokens[last_end + 1 : last_end + 1 + ANSWER_HEAD_TOKENS].tolist(), skip_special_tokens=False
    )
    match = CODE_ANSWER_GIVE_UP.search(answer)
    if match:
        return "ANSWER: " + _match_excerpt(answer, match)
    tail = tail[-CODE_TAIL_CHARS:]
    match = CODE_REASONING_GIVE_UP.search(tail)
    return "REASONING: " + _match_excerpt(tail, match) if match else None


def decide(row: int, tokens: np.ndarray, rules: FilterRules, tokenizer: Tokenizer) -> Decision:
    start, end = longest_reasoning_span(tokens)
    reasoning_tokens = end - start
    if rules.max_reasoning_tokens and reasoning_tokens > rules.max_reasoning_tokens:
        return Decision(row, len(tokens), reasoning_tokens, DropReason.REASONING_TOO_LONG, "")
    excerpt = give_up_excerpt(tokens, rules.give_up, tokenizer)
    if excerpt is not None:
        return Decision(row, len(tokens), reasoning_tokens, DropReason.GIVE_UP, excerpt)
    return Decision(row, len(tokens), reasoning_tokens, None, "")


def _shard_rows(shard: str) -> Iterator[tuple[int, np.ndarray]]:
    store = JaggedArrayStore.open(str(StoragePath(shard) / "input_ids"), mode="r", dtype=np.int32)
    count = len(store)
    for start in range(0, count, READ_BATCH_ROWS):
        indices = list(range(start, min(start + READ_BATCH_ROWS, count)))
        for row, tokens in zip(indices, store.get_batch_sync(indices), strict=True):
            yield row, np.asarray(tokens)


def filter_shard(source_shard: str, output_shard: str | None, rules: FilterRules, tokenizer_path: str) -> dict:
    """Decide every row of one shard and, when ``output_shard`` is set, write the kept rows."""
    tokenizer = Tokenizer.from_str(StoragePath(tokenizer_path).read_text())
    decisions: list[Decision] = []

    def kept_records() -> Iterator[dict[str, np.ndarray]]:
        for row, tokens in _shard_rows(source_shard):
            decision = decide(row, tokens, rules, tokenizer)
            decisions.append(decision)
            if decision.reason is None:
                yield {"input_ids": tokens}

    if output_shard is None:
        for _ in kept_records():
            pass
    else:
        write_levanter_cache(
            kept_records(), output_shard, metadata={}, batch_size=WRITE_BATCH_ROWS, max_pending_batches=4
        )

    kept = [d for d in decisions if d.reason is None]
    dropped = [d for d in decisions if d.reason is not None]
    rows_by_reason = Counter(d.reason.value for d in dropped)
    tokens_by_reason = Counter()
    for d in dropped:
        tokens_by_reason[d.reason.value] += d.tokens
    kept_rows = np.array([d.row for d in kept], dtype=np.int64)
    bucket = np.searchsorted(REASONING_BUCKET_EDGES, [d.reasoning_tokens for d in decisions], side="right")
    return {
        "reasoning_bucket_rows": np.bincount(bucket, minlength=len(REASONING_BUCKET_EDGES) + 1).tolist(),
        "reasoning_bucket_tokens": (
            np.bincount(bucket, weights=[d.tokens for d in decisions], minlength=len(REASONING_BUCKET_EDGES) + 1)
            .astype(np.int64)
            .tolist()
        ),
        "source_shard": source_shard,
        "rows": len(decisions),
        "tokens": sum(d.tokens for d in decisions),
        "kept_rows": len(kept),
        "kept_tokens": sum(d.tokens for d in kept),
        "kept_rows_sha256": hashlib.sha256(kept_rows.tobytes()).hexdigest(),
        "dropped_rows": dict(rows_by_reason),
        "dropped_tokens": dict(tokens_by_reason),
        "dropped": [dataclasses.asdict(d) for d in dropped],
    }


def _filter_shard_task(args: tuple[str, str | None, FilterRules, str]) -> dict:
    return filter_shard(*args)


def filter_store(source: str, output: str | None, rules: FilterRules, tokenizer_path: str, workers: int) -> dict:
    """Filter every shard of a sharded store; write a sharded ledger and ``selection.json``.

    With ``output=None`` nothing is written and the returned summary is a dry run.
    """
    source_ledger = CacheLedger.load(source)
    if source_ledger.layout != CACHE_LAYOUT_SHARDED:
        raise ValueError(f"{source} is not a sharded store")
    shards = source_ledger.finished_shards
    tasks = [
        (
            str(StoragePath(source) / shard),
            None if output is None else str(StoragePath(output) / shard),
            rules,
            tokenizer_path,
        )
        for shard in shards
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_filter_shard_task, tasks))

    for shard, result in zip(shards, results, strict=True):
        if result["rows"] != source_ledger.shard_rows[shard]:
            raise ValueError(f"{shard}: read {result['rows']} rows, ledger says {source_ledger.shard_rows[shard]}")
        if result["kept_rows"] == 0:
            raise ValueError(f"{shard}: every row was dropped")

    summary = {
        "source": source,
        "output": output,
        "rules": {"max_reasoning_tokens": rules.max_reasoning_tokens, "give_up": rules.give_up.value},
        "rows": sum(r["rows"] for r in results),
        "tokens": sum(r["tokens"] for r in results),
        "kept_rows": sum(r["kept_rows"] for r in results),
        "kept_tokens": sum(r["kept_tokens"] for r in results),
        "dropped_rows": dict(sum((Counter(r["dropped_rows"]) for r in results), Counter())),
        "dropped_tokens": dict(sum((Counter(r["dropped_tokens"]) for r in results), Counter())),
        "shards": results,
    }
    if output is None:
        return summary

    ledger = CacheLedger(
        total_num_rows=summary["kept_rows"],
        shard_rows={shard: r["kept_rows"] for shard, r in zip(shards, results, strict=True)},
        is_finished=True,
        finished_shards=list(shards),
        field_counts={"input_ids": summary["kept_tokens"]},
        field_counts_by_shard={shard: {"input_ids": r["kept_tokens"]} for shard, r in zip(shards, results, strict=True)},
        layout=CACHE_LAYOUT_SHARDED,
        metadata=CacheMetadata.empty(),
    )
    ledger._serialize_and_commit(output)
    (StoragePath(output) / "selection.json").write_text(json.dumps(summary) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Sharded store directory (the split, e.g. .../train)")
    parser.add_argument("--output", help="Output store directory; omit for a dry run")
    parser.add_argument("--summary", help="Where to write the summary JSON for a dry run")
    parser.add_argument("--tokenizer", required=True, help="Path or URL of tokenizer.json")
    parser.add_argument("--max-reasoning-tokens", type=int, required=True, help="0 disables the cap")
    parser.add_argument("--give-up", type=GiveUpRule, required=True, choices=list(GiveUpRule))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard", action="append", help="Dry-run only these shard names")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rules = FilterRules(max_reasoning_tokens=args.max_reasoning_tokens, give_up=args.give_up)
    if args.shard:
        if args.output:
            raise ValueError("--shard is for dry runs; a filtered store must contain every shard")
        results = [
            filter_shard(str(StoragePath(args.source) / shard), None, rules, args.tokenizer) for shard in args.shard
        ]
        summary = {"source": args.source, "shards": results}
    else:
        summary = filter_store(args.source, args.output, rules, args.tokenizer, args.workers)
    if args.summary:
        StoragePath(args.summary).write_text(json.dumps(summary) + "\n")
    for key in ("rows", "tokens", "kept_rows", "kept_tokens", "dropped_rows", "dropped_tokens"):
        if key in summary:
            logger.info("%s: %s", key, summary[key])


if __name__ == "__main__":
    main()
