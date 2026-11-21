# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace U+202F and U+2011 in the SFT token stores of a release cache catalog.

The gpt-oss teacher writes U+202F (narrow no-break space) and U+2011
(non-breaking hyphen) where other text uses a space or a hyphen. The characters
are usually one token in the release tokenizer but can split across tokens, so
occurrences are located on the row's bytes. The mechanism ran as a research
normalizer against an earlier mix; this module is its named, versioned form for
the release catalogs of the long-loop mixes, with the same per-row checks:

- a rewritten row decodes to the source row's text with only those two
  characters replaced,
- its special tokens are unchanged, and
- neither character remains.

The release catalogs add two requirements the research form did not have:

- Sources are opened read-only; every output store keeps the source's shard
  names and row order. A store with no target characters is never rewritten;
  its catalog entry keeps pointing at the source store.
- SFT stores carry ``loss_weights`` (prediction-indexed assistant-only
  supervision). A rewrite window never crosses a special token and supervision
  changes only at special tokens, so each window's source weight value is
  constant; rewritten rows splice that value through the window. The spliced
  mask must equal the mask the release format census derives from the tokens
  (assistant header spans), and teacher-only sources keep their per-span binary
  values. See :class:`CensusMaskRule` and :class:`TeacherMaskRule`.
- Each store carries a completion marker, ``normalization.json`` or
  ``unchanged.json``, so a restarted partition skips finished stores. Each
  marker is a receipt: rows read and rewritten, characters replaced, token
  delta, and SHA-256 digests of the input and output token (and, for rewritten
  stores, weight) streams. ``assemble_catalog`` writes the catalog-level
  receipt ``typography-receipt.json`` and the versioned catalog.

Replay entries of the catalog are never read.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
from levanter.store.cache import (
    CACHE_LAYOUT_SHARDED,
    LEDGER_FILE_NAME,
    CacheLedger,
    write_levanter_cache,
)
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.post_training.long_loop.source_loss import (
    SourceLossMetadata,
    SourceLossMode,
    source_loss_metadata,
)

logger = logging.getLogger(__name__)

TRANSFORM_NAME = "sft-typography-normalize"
TRANSFORM_VERSION = "v1"

READ_BATCH_ROWS = 256
WRITE_BATCH_ROWS = 256
FIRST_SPECIAL_ID = 128000
REPLACEMENTS = {"\u202f": " ", "\u2011": "-"}
SFT_PREFIX = "sft/"
NORMALIZED_MARKER = "normalization.json"
UNCHANGED_MARKER = "unchanged.json"
TARGET_BYTES = tuple(c.encode() for c in REPLACEMENTS)
COUNTED = ("nnbsp", "nbhyphen")

# Assistant-only supervision framing of this mix family, as verified by the
# release's format census: an assistant span opens with this header token
# sequence and runs to the token before the next header start. Rewritten rows
# must satisfy the census mask rule derived from these spans.
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"


class Occurrence(NamedTuple):
    """One target character inside a row, located on the row's bytes."""

    first_token: int
    last_token: int
    target_index: int


@dataclass(frozen=True)
class SourceLossPin:
    """Where the release's hash-pinned supervision sidecar lives."""

    uri: str
    sha256: str


@dataclass(frozen=True)
class ShardStats:
    """One shard's transform outcome and content digests.

    ``input_*_sha256`` digest the row token (and weight) bytes in row order;
    ``output_*_sha256`` digest the written stream, or ``None`` on a counting
    pass that writes nothing.
    """

    source_shard: str
    rows: int
    tokens: int
    out_tokens: int
    changed_rows: int
    count_nnbsp: int
    count_nbhyphen: int
    input_tokens_sha256: str
    input_weights_sha256: str
    output_tokens_sha256: str | None
    output_weights_sha256: str | None

    @property
    def token_delta(self) -> int:
        return self.out_tokens - self.tokens

    @property
    def characters_replaced(self) -> int:
        return self.count_nnbsp + self.count_nbhyphen


@dataclass(frozen=True)
class ShardTask:
    """One shard's rewrite job: read ``source_shard``, optionally write ``output_shard``."""

    source_shard: str
    output_shard: str | None
    tokenizer_path: str
    header_ids: np.ndarray
    mask_rule: "CensusMaskRule"


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE vocabularies."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("\xa1"), ord("\xac") + 1),
        *range(ord("\xae"), ord("\xff") + 1),
    ]
    byte_values = printable[:]
    codes = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            codes.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(byte_values, codes, strict=True)}


def normalize_text(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


class Rewriter:
    """Byte-located, token-level rewriting for one byte-level BPE tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        size = max(vocab.values()) + 1
        decoder = _byte_decoder()
        self.token_bytes: list[bytes] = [b""] * size
        for token, index in vocab.items():
            if index >= FIRST_SPECIAL_ID:
                self.token_bytes[index] = token.encode()
            else:
                self.token_bytes[index] = bytes(decoder[ch] for ch in token)
        # A token can hold a whole target or the start of one that the next token completes.
        prefixes = {t[:k] for t in TARGET_BYTES for k in range(1, len(t))}
        self.candidate = np.array(
            [any(t in b for t in TARGET_BYTES) or any(b.endswith(p) for p in prefixes) for b in self.token_bytes]
        )
        self.lengths = np.array([len(b) for b in self.token_bytes], dtype=np.int64)
        # A word starts at a special token or at a token whose text starts with whitespace.
        self.word_start = np.zeros(size, dtype=bool)
        for token, index in vocab.items():
            if index >= FIRST_SPECIAL_ID or token[:1] in ("Ġ", "Ċ", "ĉ", "č"):
                self.word_start[index] = True
        self.special = np.arange(size) >= FIRST_SPECIAL_ID

    def decode(self, ids: np.ndarray) -> str:
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    def occurrences(self, tokens: np.ndarray) -> list[Occurrence]:
        """Each target character in the row, located on the row's bytes."""
        if not self.candidate[tokens].any():
            return []
        data = b"".join(self.token_bytes[t] for t in tokens.tolist())
        ends = np.cumsum(self.lengths[tokens])
        found = []
        for target_index, target in enumerate(TARGET_BYTES):
            at = data.find(target)
            while at >= 0:
                first = int(np.searchsorted(ends, at, side="right"))
                last = int(np.searchsorted(ends, at + len(target) - 1, side="right"))
                found.append(Occurrence(first, last, target_index))
                at = data.find(target, at + len(target))
        return found

    def windows(self, tokens: np.ndarray, found: list[Occurrence]) -> list[tuple[int, int]]:
        """Merged token ranges covering each occurrence, its word and the following word."""
        starts = np.flatnonzero(self.word_start[tokens])
        spans = []
        for occurrence in found:
            first, last = occurrence.first_token, occurrence.last_token
            before = starts[starts <= first]
            begin = int(before[-1]) if len(before) else 0
            if self.special[tokens[begin]]:
                begin += 1
            # End after the next word, which may merge with the replacement space; stop at specials.
            after = starts[starts > last]
            end = int(after[1]) if len(after) > 1 else len(tokens)
            special_after = np.flatnonzero(self.special[tokens[last + 1 : end]])
            if len(special_after):
                end = last + 1 + int(special_after[0])
            spans.append((begin, end))
        merged: list[tuple[int, int]] = []
        for begin, end in sorted(spans):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        return merged

    def rewrite(self, tokens: np.ndarray, weights: np.ndarray, found: list[Occurrence]) -> tuple[np.ndarray, np.ndarray]:
        """Rewrite the row's tokens and splice its loss weights through each window.

        Each returned weight keeps its source value outside the rewrite windows
        and carries its window's constant source value inside. A window holding
        two different source values is rejected: windows never cross a special
        token and supervision changes only at special tokens, so mixed values
        mean the store violates its supervision contract.
        """
        token_pieces = []
        weight_pieces = []
        cursor = 0
        for begin, end in self.windows(tokens, found):
            window_weights = weights[begin:end]
            if not np.all(window_weights == window_weights[0]):
                raise ValueError("rewrite window spans a loss-weight boundary")
            text = normalize_text(self.decode(tokens[begin:end]))
            new_ids = np.asarray(self.tokenizer.encode(text, add_special_tokens=False).ids, dtype=tokens.dtype)
            token_pieces.append(tokens[cursor:begin])
            weight_pieces.append(weights[cursor:begin])
            token_pieces.append(new_ids)
            weight_pieces.append(np.full(len(new_ids), window_weights[0], dtype=weights.dtype))
            cursor = end
        token_pieces.append(tokens[cursor:])
        weight_pieces.append(weights[cursor:])
        out = np.concatenate(token_pieces)
        out_weights = np.concatenate(weight_pieces)
        expected = normalize_text(self.decode(tokens))
        if self.decode(out) != expected:
            raise ValueError("rewritten row does not decode to the normalized source text")
        if self.occurrences(out):
            raise ValueError("rewritten row still holds a target character")
        if not np.array_equal(out[self.special[out]], tokens[self.special[tokens]]):
            raise ValueError("rewritten row changed its special tokens")
        return out, out_weights


def assistant_spans(tokens: np.ndarray, header_ids: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The candidate-supervised regions of one row, as the release's format census defines them.

    Returns the ``(left, right)`` spans and the boolean mask the census derives
    from them: a span opens at a canonical ``header_ids`` sequence and runs to
    the token before the next header start (or the row end).
    """
    expected = np.zeros(len(tokens), dtype=bool)
    starts = np.flatnonzero(tokens == header_ids[0])
    spans = []
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(tokens)
        if start + 1 >= len(tokens) or tokens[start + 1] != header_ids[1]:
            continue
        if np.array_equal(tokens[start : start + len(header_ids)], header_ids):
            left, right = int(start + len(header_ids) - 1), end - 1
            expected[left:right] = True
            spans.append((left, right))
    return spans, expected


def _check_binary(weights: np.ndarray) -> None:
    if np.count_nonzero((weights != 0) & (weights != 1)):
        raise ValueError("row holds nonbinary loss weights")


class CensusMaskRule:
    """The mask contract of ordinary chat sources: the mask must equal the census spans.

    A row passes when its weights are binary and ``weights.astype(bool)`` equals
    the boolean mask the format census derives from the row's tokens.
    """

    teacher_only: ClassVar[bool] = False

    def _check_shape(self, tokens: np.ndarray, weights: np.ndarray) -> None:
        if len(weights) != len(tokens):
            raise ValueError("row's loss_weights length differs from its input_ids length")
        _check_binary(weights)

    def verify_row(self, tokens: np.ndarray, weights: np.ndarray, header_ids: np.ndarray) -> list[tuple[int, int]]:
        """Check a row against the contract and return its assistant spans."""
        self._check_shape(tokens, weights)
        spans, expected = assistant_spans(tokens, header_ids)
        if not np.array_equal(weights.astype(bool), expected):
            raise ValueError("row's loss_weights disagree with the census assistant-span rule")
        return spans

    def verify_rewritten(
        self,
        tokens: np.ndarray,
        weights: np.ndarray,
        old_spans: list[tuple[int, int]],
        old_span_values: list[float],
        header_ids: np.ndarray,
    ) -> None:
        """Check a rewritten row's spliced mask; the spans must survive the rewrite."""
        spans = self.verify_row(tokens, weights, header_ids)
        if len(spans) != len(old_spans):
            raise ValueError("rewritten row changed the number of assistant spans")


class TeacherMaskRule(CensusMaskRule):
    """The mask contract of teacher-only sources: loss only inside whole spans.

    A row passes when weights are zero outside the assistant spans, each span
    carries one constant value (weighted or unweighted), and the row never ends
    on a weighted token. A rewrite keeps each span's source value.
    """

    teacher_only: ClassVar[bool] = True

    def verify_row(self, tokens: np.ndarray, weights: np.ndarray, header_ids: np.ndarray) -> list[tuple[int, int]]:
        self._check_shape(tokens, weights)
        spans, expected = assistant_spans(tokens, header_ids)
        if np.count_nonzero(weights[~expected]):
            raise ValueError("row holds loss outside assistant spans")
        for left, right in spans:
            active = weights[left:right]
            if active.size and not np.all(active == active[0]):
                raise ValueError("assistant span is not binary")
        if len(weights) and weights[-1] != 0:
            raise ValueError("row ends with a nonzero loss weight")
        return spans

    def verify_rewritten(
        self,
        tokens: np.ndarray,
        weights: np.ndarray,
        old_spans: list[tuple[int, int]],
        old_span_values: list[float],
        header_ids: np.ndarray,
    ) -> None:
        spans = self.verify_row(tokens, weights, header_ids)
        if len(spans) != len(old_spans):
            raise ValueError("rewritten row changed the number of assistant spans")
        for (left, right), value in zip(spans, old_span_values, strict=True):
            active = weights[left:right]
            if active.size and not np.all(active == value):
                raise ValueError("rewritten span lost its source weight value")


def _row_bytes(tokens: np.ndarray) -> bytes:
    return np.asarray(tokens, dtype="<i4").tobytes()


def _weights_bytes(weights: np.ndarray) -> bytes:
    return np.asarray(weights, dtype="<f4").tobytes()


def _shard_rows_and_weights(shard: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    tokens = JaggedArrayStore.open(str(StoragePath(shard) / "input_ids"), mode="r", dtype=np.int32)
    weights = JaggedArrayStore.open(str(StoragePath(shard) / "loss_weights"), mode="r", dtype=np.float32)
    if len(tokens) != len(weights):
        raise ValueError(f"{shard}: input_ids and loss_weights row counts differ")
    for start in range(0, len(tokens), READ_BATCH_ROWS):
        indices = list(range(start, min(start + READ_BATCH_ROWS, len(tokens))))
        for row_tokens, row_weights in zip(tokens.get_batch_sync(indices), weights.get_batch_sync(indices), strict=True):
            yield np.asarray(row_tokens), np.asarray(row_weights)


def normalize_shard(task: ShardTask) -> ShardStats:
    """Rewrite every row of one shard; with ``output_shard=None`` only count and verify."""
    rewriter = Rewriter(Tokenizer.from_str(StoragePath(task.tokenizer_path).read_text()))
    rule = task.mask_rule
    input_tokens_digest = hashlib.sha256()
    input_weights_digest = hashlib.sha256()
    output_tokens_digest = None if task.output_shard is None else hashlib.sha256()
    output_weights_digest = None if task.output_shard is None else hashlib.sha256()
    rows = tokens_seen = out_tokens = changed_rows = count_nnbsp = count_nbhyphen = 0

    def records() -> Iterator[dict[str, np.ndarray]]:
        nonlocal rows, tokens_seen, out_tokens, changed_rows, count_nnbsp, count_nbhyphen
        for tokens, weights in _shard_rows_and_weights(task.source_shard):
            rows += 1
            tokens_seen += len(tokens)
            input_tokens_digest.update(_row_bytes(tokens))
            input_weights_digest.update(_weights_bytes(weights))
            old_spans = rule.verify_row(tokens, weights, task.header_ids)
            found = rewriter.occurrences(tokens)
            if found:
                for occurrence in found:
                    if occurrence.target_index == 0:
                        count_nnbsp += 1
                    else:
                        count_nbhyphen += 1
                changed_rows += 1
                old_span_values = [float(weights[left]) for left, _ in old_spans]
                tokens, weights = rewriter.rewrite(tokens, weights, found)
                rule.verify_rewritten(tokens, weights, old_spans, old_span_values, task.header_ids)
            out_tokens += len(tokens)
            if task.output_shard is not None:
                output_tokens_digest.update(_row_bytes(tokens))
                output_weights_digest.update(_weights_bytes(weights))
                yield {"input_ids": tokens, "loss_weights": weights}

    if task.output_shard is None:
        for _ in records():
            pass
    else:
        write_levanter_cache(
            records(), task.output_shard, metadata={}, batch_size=WRITE_BATCH_ROWS, max_pending_batches=4
        )
    return ShardStats(
        source_shard=task.source_shard,
        rows=rows,
        tokens=tokens_seen,
        out_tokens=out_tokens,
        changed_rows=changed_rows,
        count_nnbsp=count_nnbsp,
        count_nbhyphen=count_nbhyphen,
        input_tokens_sha256=input_tokens_digest.hexdigest(),
        input_weights_sha256=input_weights_digest.hexdigest(),
        output_tokens_sha256=output_tokens_digest.hexdigest() if output_tokens_digest is not None else None,
        output_weights_sha256=output_weights_digest.hexdigest() if output_weights_digest is not None else None,
    )


def _map_shards(tasks: list[ShardTask], workers: int) -> Iterator[ShardStats]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_normalize_shard_task, tasks)


def _normalize_shard_task(task: ShardTask) -> ShardStats:
    return normalize_shard(task)


def _combine_digests(digests: list[str]) -> str:
    """SHA-256 over the per-shard digests in the store's ledger shard order."""
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest()


@dataclasses.dataclass(frozen=True)
class StoreSummary:
    """One store's transform receipt, aggregated over its shards."""

    source: str
    output: str | None
    shards: list[ShardStats]

    @property
    def rows(self) -> int:
        return sum(s.rows for s in self.shards)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.shards)

    @property
    def out_tokens(self) -> int:
        return sum(s.out_tokens for s in self.shards)

    @property
    def changed_rows(self) -> int:
        return sum(s.changed_rows for s in self.shards)

    @property
    def count_nnbsp(self) -> int:
        return sum(s.count_nnbsp for s in self.shards)

    @property
    def count_nbhyphen(self) -> int:
        return sum(s.count_nbhyphen for s in self.shards)

    @property
    def characters_replaced(self) -> int:
        return self.count_nnbsp + self.count_nbhyphen

    @property
    def token_delta(self) -> int:
        return self.out_tokens - self.tokens

    def to_json(self) -> dict:
        """The receipt document, without per-shard detail."""
        return {
            "source": self.source,
            "output": self.output,
            "rows": self.rows,
            "tokens": self.tokens,
            "out_tokens": self.out_tokens,
            "changed_rows": self.changed_rows,
            "count_nnbsp": self.count_nnbsp,
            "count_nbhyphen": self.count_nbhyphen,
            "characters_replaced": self.characters_replaced,
            "token_delta": self.token_delta,
            "transform": {"name": TRANSFORM_NAME, "version": TRANSFORM_VERSION},
            "store_input_tokens_sha256": _combine_digests([s.input_tokens_sha256 for s in self.shards]),
            "store_input_weights_sha256": _combine_digests([s.input_weights_sha256 for s in self.shards]),
            # Unchanged stores are never rewritten; the catalog keeps pointing at
            # the source store, so the output stream is the input stream.
            "store_output_tokens_sha256": _combine_digests(
                [s.output_tokens_sha256 if s.output_tokens_sha256 else s.input_tokens_sha256 for s in self.shards]
            ),
            "store_output_weights_sha256": _combine_digests(
                [s.output_weights_sha256 if s.output_weights_sha256 else s.input_weights_sha256 for s in self.shards]
            ),
        }


def _check_rows(source_ledger: CacheLedger, results: list[ShardStats]) -> None:
    for shard, result in zip(source_ledger.finished_shards, results, strict=True):
        if result.rows != source_ledger.shard_rows[shard]:
            raise ValueError(f"{shard}: read {result.rows} rows, ledger says {source_ledger.shard_rows[shard]}")


def _commit_store(source_ledger: CacheLedger, output: str, summary: StoreSummary) -> None:
    """Write the output store's ledger, then ``normalization.json`` as its completion marker."""
    shards = source_ledger.finished_shards
    results = summary.shards
    ledger = CacheLedger(
        total_num_rows=summary.rows,
        shard_rows={shard: r.rows for shard, r in zip(shards, results, strict=True)},
        is_finished=True,
        finished_shards=list(shards),
        field_counts={"input_ids": summary.out_tokens, "loss_weights": summary.out_tokens},
        field_counts_by_shard={
            shard: {"input_ids": r.out_tokens, "loss_weights": r.out_tokens}
            for shard, r in zip(shards, results, strict=True)
        },
        layout=CACHE_LAYOUT_SHARDED,
        metadata=source_ledger.metadata,
    )
    ledger._serialize_and_commit(output)
    (StoragePath(output) / NORMALIZED_MARKER).write_text(json.dumps(summary.to_json()) + "\n")


def _entry_ledger(entry: dict) -> CacheLedger:
    """The catalog entry's embedded component ledger."""
    ledger = CacheLedger.from_json(json.dumps(entry["ledger"]))
    if ledger.layout != CACHE_LAYOUT_SHARDED:
        raise ValueError(f"{entry['cache_dir']} is not a sharded store")
    return ledger


def _entry_shard_paths(entry: dict) -> list[StoragePath]:
    """Absolute source paths of the entry's finished shards."""
    base = StoragePath(entry["cache_dir"])
    return [base / shard for shard in entry["ledger"]["finished_shards"]]


def _under(child: StoragePath, base: StoragePath) -> bool:
    try:
        child.relative_to(base)
        return True
    except ValueError:
        return False


def _sft_stores(catalog: dict, output_root: str) -> dict[str, tuple[dict, str]]:
    """``name -> (catalog entry, output dir)`` for each SFT entry of a catalog.

    The output root must not overlap any source shard: sources are read-only.
    """
    root = StoragePath(output_root)
    for split in catalog["splits"].values():
        for name, entry in split.items():
            for shard_path in _entry_shard_paths(entry):
                if _under(shard_path, root) or _under(root, shard_path):
                    raise ValueError(f"output root {output_root} overlaps source store {name}: {shard_path}")
    entries = catalog["splits"]["train"]
    return {
        name: (entries[name], str(StoragePath(output_root) / re.sub(r"[^A-Za-z0-9._-]+", "__", name) / "train"))
        for name in sorted(entries)
        if name.startswith(SFT_PREFIX)
    }


def _header_ids(tokenizer_path: str) -> np.ndarray:
    tokenizer = Tokenizer.from_str(StoragePath(tokenizer_path).read_text())
    return np.asarray(tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False).ids)


def supervision_of(source_loss: dict[str, SourceLossMetadata], name: str) -> CensusMaskRule:
    """The pinned mask rule of one catalog source, from the release's supervision sidecar."""
    source = source_loss.get(name)
    if source is None:
        raise ValueError(f"{name}: the pinned supervision sidecar has no entry for this source")
    if source.loss_mode != SourceLossMode.CHAT_ASSISTANT_ONLY:
        raise ValueError(f"{name}: only chat_assistant_only SFT sources are normalized, got {source.loss_mode}")
    if source.loss_weights_key != "loss_weights":
        raise ValueError(f"{name}: unexpected prediction-weight key {source.loss_weights_key}")
    return TeacherMaskRule() if source.teacher_only else CensusMaskRule()


def _store_tasks(
    source_dir: str,
    ledger: CacheLedger,
    output: str | None,
    tokenizer_path: str,
    header_ids: np.ndarray,
    mask_rule: CensusMaskRule,
    write: bool,
) -> list[ShardTask]:
    """One task per finished shard of a store."""
    return [
        ShardTask(
            source_shard=str(StoragePath(source_dir) / shard),
            output_shard=str(StoragePath(output) / shard) if write and output is not None else None,
            tokenizer_path=tokenizer_path,
            header_ids=header_ids,
            mask_rule=mask_rule,
        )
        for shard in ledger.finished_shards
    ]


def normalize_catalog_part(
    catalog: str,
    output_root: str,
    tokenizer_path: str,
    workers: int,
    part: int,
    parts: int,
    source_loss_pin: SourceLossPin,
) -> None:
    """Normalize the SFT stores of one catalog partition (every ``parts``-th store, starting at ``part``).

    All shards of the partition share one worker pool. A counting pass finds the
    stores with target characters and rehearses every rewrite; only stores with
    target characters are written. Each finished store gets a completion marker,
    ``normalization.json`` or ``unchanged.json``, so a restarted part skips it.
    Replay entries are never read.
    """
    source_loss = source_loss_metadata(source_loss_pin.uri, source_loss_pin.sha256)
    catalog_text = StoragePath(catalog).read_text()
    stores = list(_sft_stores(json.loads(catalog_text), output_root).items())[part::parts]
    mask_rules = {name: supervision_of(source_loss, name) for name, _ in stores}
    pending = [
        (name, entry, output)
        for name, (entry, output) in stores
        if not (StoragePath(output) / NORMALIZED_MARKER).exists()
        and not (StoragePath(output) / UNCHANGED_MARKER).exists()
    ]
    logger.info("part %d/%d: %d stores, %d pending", part, parts, len(stores), len(pending))
    ledgers = {name: _entry_ledger(entry) for name, entry, _ in pending}
    header_ids = _header_ids(tokenizer_path)

    def run(items: list[tuple[str, dict, str]], write: bool) -> dict[str, list[ShardStats]]:
        tasks = [
            task
            for name, entry, output in items
            for task in _store_tasks(
                entry["cache_dir"], ledgers[name], output, tokenizer_path, header_ids, mask_rules[name], write
            )
        ]
        flat = iter(list(_map_shards(tasks, workers)))
        grouped = {name: [next(flat) for _ in ledgers[name].finished_shards] for name, _, _ in items}
        for name, results in grouped.items():
            _check_rows(ledgers[name], results)
        return grouped

    changed = []
    counted = run(pending, write=False)
    for name, entry, output in pending:
        summary = StoreSummary(entry["cache_dir"], None, counted[name])
        if summary.changed_rows:
            changed.append((name, entry, output))
        else:
            (StoragePath(output) / UNCHANGED_MARKER).write_text(json.dumps(summary.to_json()) + "\n")
            logger.info("%s: no target characters", name)
    written = run(changed, write=True)
    for name, entry, output in changed:
        summary = StoreSummary(entry["cache_dir"], output, written[name])
        _commit_store(ledgers[name], output, summary)
        logger.info("%s: %s of %s rows rewritten", name, summary.changed_rows, summary.rows)


def assemble_catalog(catalog: str, output_root: str, catalog_output: str, source_loss_pin: SourceLossPin) -> None:
    """Write a catalog whose rewritten SFT entries point at their normalized stores.

    Every SFT store must carry a completion marker from ``normalize_catalog_part``.
    A store without target characters keeps its source entry, so the versioned
    catalog pins one consistent source for the whole mix. Also writes the
    catalog-level receipt ``typography-receipt.json``.
    """
    if catalog_output == catalog:
        raise ValueError("--catalog-output must differ from --catalog")
    source_loss = source_loss_metadata(source_loss_pin.uri, source_loss_pin.sha256)
    source = json.loads(StoragePath(catalog).read_text())
    entries = source["splits"]["train"]
    report = {}
    for name, (_, output) in _sft_stores(source, output_root).items():
        supervision_of(source_loss, name)
        if (StoragePath(output) / NORMALIZED_MARKER).exists():
            summary = json.loads((StoragePath(output) / NORMALIZED_MARKER).read_text())
            entries[name] = {
                "cache_dir": output,
                "ledger": json.loads((StoragePath(output) / LEDGER_FILE_NAME).read_text()),
            }
        elif (StoragePath(output) / UNCHANGED_MARKER).exists():
            summary = json.loads((StoragePath(output) / UNCHANGED_MARKER).read_text())
        else:
            raise ValueError(f"{name}: no completion marker under {output}")
        report[name] = summary
    StoragePath(catalog_output).write_text(json.dumps(source))
    totals = {
        "stores": len(report),
        "rows": sum(r["rows"] for r in report.values()),
        "changed_rows": sum(r["changed_rows"] for r in report.values()),
        "tokens": sum(r["tokens"] for r in report.values()),
        "out_tokens": sum(r["out_tokens"] for r in report.values()),
        "token_delta": sum(r["token_delta"] for r in report.values()),
        "characters_replaced": sum(r["characters_replaced"] for r in report.values()),
    }
    totals.update({f"count_{n}": sum(r[f"count_{n}"] for r in report.values()) for n in COUNTED})
    receipt = {
        "transform": {"name": TRANSFORM_NAME, "version": TRANSFORM_VERSION},
        "catalog": {"uri": catalog},
        "source_loss": dataclasses.asdict(source_loss_pin),
        "stores": report,
        "totals": totals,
        "complete": True,
    }
    (StoragePath(output_root) / "typography-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    logger.info("receipt: %s", totals)


def normalize_store(
    source: str, output: str | None, tokenizer_path: str, workers: int, mask_rule: CensusMaskRule
) -> dict:
    """Normalize every shard of one self-ledgered store; with ``output=None`` only count."""
    source_ledger = CacheLedger.load(source)
    if source_ledger.layout != CACHE_LAYOUT_SHARDED:
        raise ValueError(f"{source} is not a sharded store")
    header_ids = _header_ids(tokenizer_path)
    tasks = _store_tasks(source, source_ledger, output, tokenizer_path, header_ids, mask_rule, write=output is not None)
    results = list(_map_shards(tasks, workers))
    _check_rows(source_ledger, results)
    summary = StoreSummary(source, output, results)
    if output is not None:
        _commit_store(source_ledger, output, summary)
    return summary.to_json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", help="Cache catalog whose SFT stores are normalized")
    parser.add_argument("--output-root", help="With --catalog: directory for the normalized stores")
    parser.add_argument("--part", type=int, help="With --catalog: normalize this partition of the SFT stores")
    parser.add_argument("--parts", type=int, help="With --catalog: number of partitions")
    parser.add_argument("--catalog-output", help="With --catalog: assemble the patched catalog here instead")
    parser.add_argument("--source-loss", help="URI of the release's pinned source_loss.json supervision sidecar")
    parser.add_argument("--source-loss-sha256", help="SHA-256 of the supervision sidecar")
    parser.add_argument("--source", help="Sharded store directory (the split, e.g. .../train)")
    parser.add_argument("--output", help="Output store directory; omit to count only")
    parser.add_argument("--summary", help="Where to write the summary JSON")
    parser.add_argument("--tokenizer", help="Path or URL of tokenizer.json; required unless assembling")
    parser.add_argument("--workers", type=int, help="Required unless assembling")
    parser.add_argument("--shard", action="append", help="Count only these shard names")
    parser.add_argument("--teacher-only", action="store_true", help="With --source: the store is teacher-only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    if args.catalog and args.catalog_output:
        assemble_catalog(args.catalog, args.output_root, args.catalog_output, _require_source_loss_pin(args))
        return
    if args.tokenizer is None or args.workers is None:
        raise ValueError("--tokenizer and --workers are required")
    if args.catalog:
        if args.part is None or args.parts is None:
            raise ValueError("--part and --parts are required with --catalog")
        normalize_catalog_part(
            args.catalog,
            args.output_root,
            args.tokenizer,
            args.workers,
            args.part,
            args.parts,
            _require_source_loss_pin(args),
        )
        return
    if args.shard:
        if args.output:
            raise ValueError("--shard counts only; a normalized store must contain every shard")
        header_ids = _header_ids(args.tokenizer)
        rule = TeacherMaskRule() if args.teacher_only else CensusMaskRule()
        summary = {
            "shards": [
                dataclasses.asdict(
                    normalize_shard(
                        ShardTask(
                            source_shard=str(StoragePath(args.source) / s),
                            output_shard=None,
                            tokenizer_path=args.tokenizer,
                            header_ids=header_ids,
                            mask_rule=rule,
                        )
                    )
                )
                for s in args.shard
            ]
        }
    else:
        rule = TeacherMaskRule() if args.teacher_only else CensusMaskRule()
        summary = normalize_store(args.source, args.output, args.tokenizer, args.workers, rule)
    if args.summary:
        StoragePath(args.summary).write_text(json.dumps(summary) + "\n")
    logger.info({k: v for k, v in summary.items() if k != "shards"})


def _require_source_loss_pin(args: argparse.Namespace) -> SourceLossPin:
    if not args.source_loss or not args.source_loss_sha256:
        raise ValueError("--source-loss and --source-loss-sha256 are required")
    return SourceLossPin(uri=args.source_loss, sha256=args.source_loss_sha256)


if __name__ == "__main__":
    main()
