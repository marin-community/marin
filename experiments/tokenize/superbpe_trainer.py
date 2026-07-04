# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train plain BPE and SuperBPE (two-stage superword BPE) tokenizers from raw text.

SuperBPE (Liu, Hayase, Hofmann, Oh, Smith, Choi; arXiv:2503.13423) trains a standard,
whitespace-respecting BPE tokenizer up to a *transition point* vocabulary size ``t``, then
continues merging past ``t`` without the whitespace constraint, so later merges can span what
used to be word boundaries ("superwords", e.g. `` of the`` as one token). The authors' own
implementation (github.com/PythonNut/superbpe) depends on a custom fork of
`huggingface/tokenizers` (`alisawuffles/tokenizers-superbpe`) that patches the Rust BPE trainer
to resume from an existing vocab/merge table under a new pretokenizer — a native extension that
conflicts with the stock `tokenizers` package this repo already depends on everywhere else.

This module reimplements the *algorithm* (not the fork) on top of the stock `tokenizers` library:

- Stage 1 is exactly standard BPE: :func:`train_plain_bpe` with a word-respecting pretokenizer
  regex, using the stock Rust `BpeTrainer` (fast, no reimplementation needed).
- Stage 2 (:func:`extend_with_superwords`) re-pretokenizes the corpus with a permissive regex
  that only isolates digit runs, multi-character punctuation runs, and trailing whitespace —
  the same regex the SuperBPE repo uses for its extension stage — so a "training word" can span
  several stage-1 words. Each such segment is encoded with the *stage-1 model only* (its already
  learned merges, no further training) to get a starting token-id sequence, and a from-scratch
  greedy BPE merge learner (:func:`_learn_merges`) continues merging pairs of *stage-1 tokens*
  (not bytes) up to the final vocab size, exactly matching the paper's algorithm.

Practical cap: unlike stage-1 words, stage-2 segments are not bounded by whitespace, so in the
worst case (prose with sparse punctuation) a segment could span an entire multi-line document.
We additionally split on newlines before stage-2 pretokenization (the paper's regex does not
require this), both to keep segments varied enough for the frequency-dedup in
:func:`_stage2_flat_segments` to be effective and as a tractability cap for this
reimplementation: superword merges can span any number of words/spaces exactly as in the paper,
but never a hard newline. This is expected to leave a small amount of cross-line superword
fertility on the table.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

logger = logging.getLogger(__name__)

# The exact stage-1 (word-respecting) and stage-2 (superword-permissive) pretokenization
# regexes from the SuperBPE reference implementation (github.com/PythonNut/superbpe,
# scripts/train_tokenizer.sh and scripts/extend_tokenizer.sh), reused verbatim so a trained
# tokenizer's segmentation matches the published method and the off-the-shelf superbpe-128k/
# superbpe-180k arms it is compared against.
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# The Llama-3 / GPT-4 word-split regex (used by marin-128k and most production BPE tokenizers):
# contraction-aware, groups letters with a single leading non-letter, digit runs of 1-3. Unlike
# STAGE1_REGEX it does NOT split on internal case transitions ("camelCase" stays one word), which
# is the production behavior. Used as the stage-1 pretokenizer for the LLAMA3 variant.
LLAMA3_STAGE1_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Pretok(StrEnum):
    """Pretokenizer variant governing how raw text is split into training "words"."""

    SUPERBPE = "superbpe"  # SuperBPE reference regexes: case-aware words, digit runs of 1-3
    DIGITS = "digits"  # SuperBPE regexes + every digit isolated (never merged) — arithmetic-friendly
    LLAMA3 = "llama3"  # Llama-3 / GPT-4 word regex for stage 1 (contraction-aware, no case split)


@dataclass(frozen=True)
class PretokConfig:
    """Resolved stage-1/stage-2 regexes and digit handling for a :class:`Pretok` variant."""

    stage1_regex: str
    stage2_regex: str
    split_digits: bool


def pretok_config(pretok: Pretok) -> PretokConfig:
    """The (stage-1 regex, stage-2 regex, individual-digit) settings for a pretokenizer variant."""
    if pretok == Pretok.SUPERBPE:
        return PretokConfig(STAGE1_REGEX, STAGE2_REGEX, split_digits=False)
    if pretok == Pretok.DIGITS:
        # Keep the SuperBPE regexes but add an individual-digit split step in both stages, so no
        # token ever spans two digits (stage-1 vocab and stage-2 superwords alike).
        return PretokConfig(STAGE1_REGEX, STAGE2_REGEX, split_digits=True)
    if pretok == Pretok.LLAMA3:
        # Llama-3 word regex for stage 1; stage 2 keeps the permissive superword regex so the
        # superword-merge effect is preserved on top of Llama-3-style base segmentation.
        return PretokConfig(LLAMA3_STAGE1_REGEX, STAGE2_REGEX, split_digits=False)
    raise ValueError(f"unknown pretok variant {pretok}")


# A pair must recur at least this many times (corpus-wide, after dedup-by-segment weighting)
# to be worth a merge; matches common BPE-trainer practice of stopping once no pair repeats.
_MIN_MERGE_COUNT = 2

# Number of merges learned per global pair-recount round. A single vectorized recount
# (np.unique + bincount over every adjacent pair in the corpus) costs the same regardless of
# how many merges it produces, and dominates per-merge cost at realistic corpus sizes — see the
# module docstring's "recount batching" note. Taking many merges per recount amortizes that
# fixed cost; conflicts between two chosen pairs (sharing a token position) are resolved by one
# left-to-right sweep, and any occurrence that loses a conflict is simply picked up as the same
# pair's count in the next round, so batching does not lose or corrupt any merge — it only
# reorders same-round merges relative to the strictly-sequential algorithm.
_MERGE_BATCH_SIZE = 2000


def _byte_level_pretokenizer(regex_string: str, *, split_digits: bool = False) -> pre_tokenizers.PreTokenizer:
    steps: list[pre_tokenizers.PreTokenizer] = [
        pre_tokenizers.Split(pattern=Regex(regex_string), behavior="isolated", invert=False)
    ]
    if split_digits:
        # Applied after Split (subdivides each digit run into single digits) and before ByteLevel
        # (byte-mapping only, use_regex=False, so it never re-splits). Prevents any cross-digit
        # merge so numbers tokenize digit-by-digit — the Llama-3-style arithmetic-friendly encoding.
        steps.append(pre_tokenizers.Digits(individual_digits=True))
    steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False))
    return pre_tokenizers.Sequence(steps)


def _byte_level_decoder() -> decoders.Decoder:
    # Matches the off-the-shelf superbpe-128k/180k tokenizer.json decoder exactly, so a trained
    # tokenizer round-trips (encode -> decode) the same way those arms do.
    return decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)


def train_plain_bpe(
    texts: Sequence[str],
    vocab_size: int,
    *,
    regex_string: str = STAGE1_REGEX,
    split_digits: bool = False,
) -> Tokenizer:
    """Train a standard byte-level BPE tokenizer with the stock Rust trainer.

    ``regex_string`` isolates pretokenization "words" (default: the SuperBPE stage-1 regex,
    a superset of the usual GPT-2/Llama word/number/punctuation split); merges never cross a
    word boundary. ``split_digits`` isolates each digit so numbers never merge. This is also
    stage 1 of :func:`extend_with_superwords`.
    """
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = _byte_level_pretokenizer(regex_string, split_digits=split_digits)
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=False, special_tokens=[])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.decoder = _byte_level_decoder()
    return tokenizer


_SEGMENT_BOUNDARY = -1  # sentinel id between segments in the flattened arrays; never a real token


def _stage2_flat_segments(
    stage1_tokenizer: Tokenizer,
    texts: Sequence[str],
    stage2_regex: str,
    split_digits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten stage-2 segments into one token-id array plus a parallel corpus-frequency weight.

    Splits each text on newlines first (the tractability cap described in the module
    docstring), then by ``stage2_regex`` (isolating only digit/punctuation/trailing-space
    runs, so a segment may span several stage-1 words). Each segment is encoded with the
    stage-1 model's already-learned merges only (no further training) to get its starting
    token-id sequence. Identical segments are deduplicated (each appears once in the returned
    arrays, at its corpus frequency as the weight) so the arrays scale with unique-segment
    count, not raw corpus size — the same trick a word-frequency BPE trainer uses to avoid
    rescanning the raw corpus. Segments are separated by :data:`_SEGMENT_BOUNDARY` so
    :func:`_learn_merges` never merges across a segment.
    """
    stage2_pretok = _byte_level_pretokenizer(stage2_regex, split_digits=split_digits)
    model_only = Tokenizer(stage1_tokenizer.model)
    model_only.pre_tokenizer = None

    segment_strs: list[str] = []
    for text in texts:
        for line in text.split("\n"):
            if not line:
                continue
            segment_strs.extend(piece for piece, _ in stage2_pretok.pre_tokenize_str(line) if piece)

    segment_counts: Counter[tuple[int, ...]] = Counter()
    batch_size = 50_000
    for start in range(0, len(segment_strs), batch_size):
        batch = segment_strs[start : start + batch_size]
        for encoding in model_only.encode_batch(batch, add_special_tokens=False):
            if encoding.ids:
                segment_counts[tuple(encoding.ids)] += 1

    if not segment_counts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    boundary = np.array([_SEGMENT_BOUNDARY], dtype=np.int64)
    id_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for ids_tuple, count in segment_counts.items():
        ids_arr = np.asarray(ids_tuple, dtype=np.int64)
        id_chunks.append(ids_arr)
        id_chunks.append(boundary)
        weight_chunks.append(np.full(len(ids_arr), count, dtype=np.int64))
        weight_chunks.append(boundary)  # placeholder weight at the boundary; never read (invalid pair)
    return np.concatenate(id_chunks), np.concatenate(weight_chunks)


def _learn_merges(
    ids: np.ndarray,
    weights: np.ndarray,
    start_id: int,
    target_vocab: int,
) -> list[tuple[int, int, int]]:
    """Greedy BPE merge learning over a flattened, corpus-frequency-weighted token stream.

    Standard greedy BPE (Sennrich et al.), adapted to merge pairs of already-tokenized ids
    (not characters) and vectorized with numpy. Each round does one full vectorized pair
    recount (weighted by ``weights``, so a deduplicated segment counts at its true corpus
    frequency) and then merges up to :data:`_MERGE_BATCH_SIZE` of the most frequent *distinct*
    pairs at once: every occurrence of every chosen pair is found in one pass, and a single
    left-to-right sweep resolves position conflicts between them (an occurrence that loses a
    conflict — e.g. two chosen pairs sharing a token — is simply re-counted and retried next
    round, so nothing is lost). Pairs that straddle a :data:`_SEGMENT_BOUNDARY` sentinel are
    never counted or merged. Returns ``(a, b, new_id)`` triples in learned order; stops early
    (before ``target_vocab``) if no pair recurs at least :data:`_MIN_MERGE_COUNT` times.
    """
    arr = ids.copy()
    wts = weights.copy()
    merges: list[tuple[int, int, int]] = []
    next_id = start_id
    key_base = np.int64(1) << 32  # ids must stay well under 2**32; true at any vocab scale here

    while next_id < target_vocab and arr.size > 1:
        left = arr[:-1]
        right = arr[1:]
        valid = (left != _SEGMENT_BOUNDARY) & (right != _SEGMENT_BOUNDARY)
        if not valid.any():
            break

        keys_all = np.where(valid, left.astype(np.int64) * key_base + right.astype(np.int64), np.int64(-1))
        vweights = wts[:-1][valid].astype(np.float64)
        uniq_keys, inverse = np.unique(keys_all[valid], return_inverse=True)
        sums = np.bincount(inverse, weights=vweights)

        eligible = sums >= _MIN_MERGE_COUNT
        if not eligible.any():
            break
        eligible_keys = uniq_keys[eligible]
        eligible_sums = sums[eligible]
        n_take = min(_MERGE_BATCH_SIZE, target_vocab - next_id, eligible_keys.size)
        top_unsorted = np.argpartition(-eligible_sums, n_take - 1)[:n_take]
        top = top_unsorted[np.argsort(-eligible_sums[top_unsorted])]  # descending by count, for a stable learned order
        chosen_keys = eligible_keys[top]

        key_to_newid: dict[int, int] = {}
        for key in chosen_keys.tolist():
            a, b = key // int(key_base), key % int(key_base)
            merges.append((a, b, next_id))
            key_to_newid[key] = next_id
            next_id += 1

        match = valid & np.isin(keys_all, chosen_keys)
        match_idx = np.nonzero(match)[0]
        # Greedy leftmost non-overlapping selection across every chosen pair combined (e.g.
        # pair (x, x) over "x x x" merges positions 0-1, not 1-2; two different chosen pairs
        # sharing a token resolve the same way — whichever starts first wins).
        keep: list[int] = []
        last = -2
        for idx in match_idx.tolist():
            if idx > last + 1:
                keep.append(idx)
                last = idx
        keep_idx = np.asarray(keep, dtype=np.int64)

        new_ids_at_keep = np.fromiter(
            (key_to_newid[int(k)] for k in keys_all[keep_idx]), dtype=np.int64, count=keep_idx.size
        )
        arr[keep_idx] = new_ids_at_keep
        drop_mask = np.ones(arr.shape[0], dtype=bool)
        drop_mask[keep_idx + 1] = False
        arr = arr[drop_mask]
        wts = wts[drop_mask]

    if next_id < target_vocab:
        logger.warning(
            "stage-2 merge learning stopped early at vocab %d/%d (no pair recurs >= %d times)",
            next_id,
            target_vocab,
            _MIN_MERGE_COUNT,
        )
    return merges


@dataclass(frozen=True)
class SuperBpeResult:
    """A trained SuperBPE tokenizer plus the stage split, for reporting."""

    tokenizer: Tokenizer
    transition_vocab_size: int  # requested t
    final_vocab_size: int  # actually achieved (may be < requested if merges ran out)


def extend_with_superwords(
    stage1_tokenizer: Tokenizer,
    texts: Sequence[str],
    final_vocab_size: int,
    *,
    stage2_regex: str = STAGE2_REGEX,
    split_digits: bool = False,
) -> SuperBpeResult:
    """Continue a stage-1 BPE tokenizer past its transition point with superword merges.

    ``stage1_tokenizer`` must have been produced by :func:`train_plain_bpe` (or otherwise
    have a dense ``0..vocab_size-1`` id space with no added/special tokens yet). Returns a new
    :class:`~tokenizers.Tokenizer` whose pretokenizer is ``stage2_regex`` (so the learned
    superword merges fire at encode time) and whose model vocab is the stage-1 vocab plus the
    newly learned superword tokens. ``split_digits`` must match the stage-1 setting so digit
    handling is consistent across both stages.
    """
    transition_vocab_size = stage1_tokenizer.get_vocab_size()
    flat_ids, flat_weights = _stage2_flat_segments(stage1_tokenizer, texts, stage2_regex, split_digits)
    merges = _learn_merges(flat_ids, flat_weights, start_id=transition_vocab_size, target_vocab=final_vocab_size)

    vocab = stage1_tokenizer.get_vocab()
    id_to_str = {v: k for k, v in vocab.items()}

    # The stage-1 model's learned merges, in training order; `Tokenizer` exposes no direct
    # accessor for a BPE model's merge list, so pull it back out of the serialized state.
    stage1_state = json.loads(stage1_tokenizer.to_str())["model"]
    stage1_merges: list[tuple[str, str]] = [tuple(m) for m in stage1_state["merges"]]

    new_merges: list[tuple[str, str]] = []
    for a, b, new_id in merges:
        token_str = id_to_str[a] + id_to_str[b]
        id_to_str[new_id] = token_str
        vocab[token_str] = new_id
        new_merges.append((id_to_str[a], id_to_str[b]))

    model = BPE(vocab=vocab, merges=[*stage1_merges, *new_merges], unk_token=None)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = _byte_level_pretokenizer(stage2_regex, split_digits=split_digits)
    tokenizer.decoder = _byte_level_decoder()
    return SuperBpeResult(
        tokenizer=tokenizer,
        transition_vocab_size=transition_vocab_size,
        final_vocab_size=transition_vocab_size + len(merges),
    )
