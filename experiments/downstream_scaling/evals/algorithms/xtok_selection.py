# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-tokenizer selection rules for joint decoding.

Two models with different tokenizers share no id space, so candidates are
compared as byte strings: ``load_vocab`` maps each vocab id to the exact
bytes it emits, each decision round scores byte-keyed candidates (the two
selectors below), and the committed chunk is forced per side as that side's
own token segmentation — the exact candidate id when one matches, greedy
longest-match segmentation otherwise. The joint-decode package's
variable-length forcing keeps the two engines in sync across
different-length segmentations.

Byte-level BPE vocabs only (every model in play — Delphi/llama, Qwen — is
one); ``load_vocab`` fails fast on anything else. Selection math note: raw
logits are used wherever scores are only weighted and summed — per-side
top-k normalization is a per-round additive constant that cancels under the
final temperature softmax; real probabilities appear only where masses are
summed (``prefix_mass``). See
.agents/projects/20260708_joint_decode_avg_xtok_plan.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


class _EosKey:
    """Sentinel key type: both vocabs' EOS ids map to the single EOS_KEY."""

    def __repr__(self) -> str:
        return "EOS_KEY"


EOS_KEY = _EosKey()
Key = bytes | _EosKey

# Loader self-check inputs: spaces, digits, punctuation, non-ASCII, emoji.
_ROUND_TRIP_SAMPLES = (
    "The answer is 42.\n",
    " 1234567890 + 9 == 1234567899",
    "naïve café — ¿Größe? 😀",
)


def _build_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's reversible byte<->unicode alphabet: printable bytes
    map to themselves, the rest to code points 256 and up."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    printable = set(byte_values)
    shift = 0
    for value in range(256):
        if value in printable:
            continue
        byte_values.append(value)
        code_points.append(256 + shift)
        shift += 1
    return {chr(point): value for point, value in zip(code_points, byte_values, strict=True)}


_BYTE_DECODER = _build_byte_decoder()


def _token_to_bytes(token: str) -> bytes:
    try:
        return bytes(_BYTE_DECODER[char] for char in token)
    except KeyError as exc:
        raise ValueError(
            f"token {token!r} contains {exc.args[0]!r}, outside the byte-level BPE alphabet; "
            "only byte-level BPE vocabs are supported"
        ) from exc


@dataclass(frozen=True)
class Vocab:
    """Byte-level view of one byte-level-BPE tokenizer."""

    token_bytes: tuple[bytes | None, ...]  # id -> bytes; None for special ids
    piece_ids: dict[bytes, int]
    max_piece_len: int
    eos_id: int


@dataclass(frozen=True)
class Candidate:
    token_id: int
    logit: float


def load_vocab(tokenizer: Any) -> Vocab:
    """Build the byte-level view of a byte-level-BPE tokenizer.

    Fails fast at load so runtime selection never has to handle a broken
    vocab: every non-special token must decode through the byte alphabet,
    all 256 single bytes must be present (so ``segment`` is total), byte
    strings must be unique, and sample texts must round-trip through
    ``encode`` and the byte table.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    special_ids = set(tokenizer.all_special_ids)
    special_ids.update(token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special)
    special_ids.add(eos_id)

    vocab_map: dict[str, int] = tokenizer.get_vocab()
    token_bytes: list[bytes | None] = [None] * (max(vocab_map.values()) + 1)
    piece_ids: dict[bytes, int] = {}
    for token, token_id in vocab_map.items():
        if token_id in special_ids:
            continue
        piece = _token_to_bytes(token)
        if piece in piece_ids:
            raise ValueError(f"duplicate byte string {piece!r} for token ids {piece_ids[piece]} and {token_id}")
        token_bytes[token_id] = piece
        piece_ids[piece] = token_id

    missing = [value for value in range(256) if bytes([value]) not in piece_ids]
    if missing:
        raise ValueError(f"vocab is not byte-complete; no tokens for byte values {missing}")

    for sample in _ROUND_TRIP_SAMPLES:
        joined = bytearray()
        for token_id in tokenizer.encode(sample, add_special_tokens=False):
            piece = token_bytes[token_id]
            if piece is None:
                raise ValueError(f"encode({sample!r}) produced special or unknown id {token_id}")
            joined.extend(piece)
        if bytes(joined) != sample.encode():
            raise ValueError(f"byte-table round-trip failed for {sample!r}")

    return Vocab(
        token_bytes=tuple(token_bytes),
        piece_ids=piece_ids,
        max_piece_len=max(len(piece) for piece in piece_ids),
        eos_id=eos_id,
    )


def candidates(vocab: Vocab, topk: list[dict[str, Any]]) -> dict[Key, Candidate]:
    """Byte-keyed view of one side's top-k.

    The side's EOS id maps to EOS_KEY; other special ids — and ids beyond the
    tokenizer vocab, e.g. padded lm-head rows — are dropped, as are entries
    with non-finite logits (a masked candidate can never win selection, and
    keeping one would poison the min-floor: w * -inf is NaN at the endpoint
    advisor weights). Raises if nothing survives: an all-special or
    all-masked top-k means the run should die loudly.
    """
    out: dict[Key, Candidate] = {}
    dropped: list[int] = []
    for entry in topk:
        token_id = int(entry["token_id"])
        logit = float(entry["logit"])
        if not math.isfinite(logit):
            dropped.append(token_id)
            continue
        if token_id == vocab.eos_id:
            out[EOS_KEY] = Candidate(token_id=token_id, logit=logit)
            continue
        piece = vocab.token_bytes[token_id] if token_id < len(vocab.token_bytes) else None
        if piece is None:
            dropped.append(token_id)
            continue
        out[piece] = Candidate(token_id=token_id, logit=logit)
    if not out:
        raise ValueError(f"top-k contains only special, out-of-vocab, or non-finite entries; dropped ids={dropped}")
    return out


def segment(vocab: Vocab, chunk: bytes) -> list[int]:
    """Greedy longest-match tokenization of raw bytes; total on the
    byte-complete vocabs load_vocab admits."""
    ids: list[int] = []
    pos = 0
    while pos < len(chunk):
        for end in range(min(len(chunk), pos + vocab.max_piece_len), pos, -1):
            token_id = vocab.piece_ids.get(chunk[pos:end])
            if token_id is not None:
                ids.append(token_id)
                pos = end
                break
        else:  # unreachable after load_vocab's byte-completeness check
            raise ValueError(f"no token for byte {chunk[pos]:#04x}")
    return ids


def force(vocab: Vocab, key: Key, cands: dict[Key, Candidate]) -> list[int]:
    """Token ids this side must emit for the committed key: the exact
    candidate id when the key is one of its candidates, its own greedy
    segmentation otherwise; EOS_KEY forces the side's own EOS."""
    if isinstance(key, _EosKey):
        return [vocab.eos_id]
    candidate = cands.get(key)
    if candidate is not None:
        return [candidate.token_id]
    return segment(vocab, key)


def _temperature_sample(keys: list[Key], scores: list[float], temperature: float, rng: random.Random) -> Key:
    # candidates() keeps only finite logits, so every score is finite; a
    # non-finite score here is a selector bug and must fail with context
    # (argmax over NaN is silently wrong; random.choices raises opaquely).
    bad = [(key, score) for key, score in zip(keys, scores, strict=True) if not math.isfinite(score)]
    if bad:
        raise ValueError(f"non-finite selection scores: {bad}")
    if temperature == 0.0:
        return keys[scores.index(max(scores))]
    max_score = max(scores)
    weights = [math.exp((score - max_score) / temperature) for score in scores]
    return rng.choices(keys, weights=weights, k=1)[0]


def _softmax(logits: dict[Key, float]) -> dict[Key, float]:
    max_logit = max(logits.values())
    exps = {key: math.exp(logit - max_logit) for key, logit in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def prefix_mass(chunk: bytes, probs: dict[Key, float], *, credit: float) -> float:
    """P(next text starts with ``chunk``) restricted to one side's top-k.

    Exact matches and strict extensions count fully. Tokens that are strict
    prefixes of ``chunk`` contribute ``p * credit ** (missing bytes)``: their
    continuation logits are unobservable this round, so ``credit`` sets the
    per-byte optimism (1.0 optimistic, 0.0 pessimistic).
    """
    mass = 0.0
    for piece, prob in probs.items():
        if not isinstance(piece, bytes):
            continue
        if piece.startswith(chunk):
            mass += prob
        elif chunk.startswith(piece):
            mass += prob * credit ** (len(chunk) - len(piece))
    return mass


def avg_bytes_union_scores(a: dict[Key, Candidate], b: dict[Key, Candidate], advisor_weight: float) -> dict[Key, float]:
    """Score the byte-keyed union with each side's minimum as its missing-key floor."""
    a_floor = min(candidate.logit for candidate in a.values())
    b_floor = min(candidate.logit for candidate in b.values())
    w_a, w_b = 1.0 - advisor_weight, advisor_weight
    return {
        key: w_a * (a[key].logit if key in a else a_floor) + w_b * (b[key].logit if key in b else b_floor)
        for key in set(a) | set(b)
    }


def select_avg_bytes_union(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    rng: random.Random,
    vocab_a: Vocab,
    vocab_b: Vocab,
) -> tuple[list[int], list[int]]:
    """The package's ``select_avg_logits`` re-keyed on token bytes: weighted
    raw-logit average over the byte-keyed union, min-floor per side for
    missing keys, temperature sample (argmax at T=0)."""
    a = candidates(vocab_a, a_topk)
    b = candidates(vocab_b, b_topk)
    scores = avg_bytes_union_scores(a, b, advisor_weight)
    key = _temperature_sample(list(scores), list(scores.values()), temperature, rng)
    return force(vocab_a, key, a), force(vocab_b, key, b)


def select_avg_anchored(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    prefix_credit: float,
    rng: random.Random,
    vocab_a: Vocab,
    vocab_b: Vocab,
) -> tuple[list[int], list[int]]:
    """Decoder proposes, advisor scores: candidates are A's top-k only, each
    scored ``(1-w) * logit_A + w * log prefix_mass_B``; B keys absent from
    all masses fall back to B's min top-k logprob as a floor. A is always
    forced with its own candidate id; only B absorbs segmentation."""
    a = candidates(vocab_a, a_topk)
    b = candidates(vocab_b, b_topk)
    b_probs = _softmax({key: candidate.logit for key, candidate in b.items()})
    b_floor = math.log(min(b_probs.values()))
    w_a, w_b = 1.0 - advisor_weight, advisor_weight
    keys = list(a)
    scores = []
    for key in keys:
        if isinstance(key, bytes):
            mass = prefix_mass(key, b_probs, credit=prefix_credit)
        else:
            mass = b_probs.get(EOS_KEY, 0.0)
        scores.append(w_a * a[key].logit + w_b * (math.log(mass) if mass > 0.0 else b_floor))
    key = _temperature_sample(keys, scores, temperature, rng)
    return [a[key].token_id], force(vocab_b, key, b)
