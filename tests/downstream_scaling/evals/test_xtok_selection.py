# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import random

import pytest

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.algorithms.xtok_selection import EntropySource, GateDirection


def _byte_alphabet() -> dict[int, str]:
    # Independent copy of GPT-2's byte->unicode alphabet: printable bytes map
    # to themselves, the rest to code points 256+. Kept separate from the
    # production map so an alphabet bug fails these tests instead of hiding.
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {value: chr(value) for value in printable}
    shift = 0
    for value in range(256):
        if value not in mapping:
            mapping[value] = chr(256 + shift)
            shift += 1
    return mapping


_ALPHABET = _byte_alphabet()


def _piece_string(piece: bytes) -> str:
    return "".join(_ALPHABET[value] for value in piece)


class FakeTokenizer:
    """Byte-level-BPE stand-in: all 256 single-byte tokens (ids equal their
    byte value unless bytes are dropped), optional merged pieces, one EOS
    special, optional extra special and raw-literal tokens."""

    def __init__(
        self,
        merged: tuple[bytes, ...] = (),
        *,
        drop_bytes: tuple[int, ...] = (),
        raw_tokens: tuple[str, ...] = (),
        extra_specials: tuple[str, ...] = (),
    ):
        self._pieces: dict[int, bytes] = {}
        vocab: dict[str, int] = {}
        next_id = 0
        for value in range(256):
            if value in drop_bytes:
                continue
            piece = bytes([value])
            vocab[_piece_string(piece)] = next_id
            self._pieces[next_id] = piece
            next_id += 1
        for piece in merged:
            vocab[_piece_string(piece)] = next_id
            self._pieces[next_id] = piece
            next_id += 1
        self.eos_token_id = next_id
        vocab["</eos>"] = next_id
        specials = [next_id]
        next_id += 1
        for special in extra_specials:
            vocab[special] = next_id
            specials.append(next_id)
            next_id += 1
        for raw in raw_tokens:
            vocab[raw] = next_id
            next_id += 1
        self.all_special_ids = specials
        self.added_tokens_decoder: dict[int, object] = {}
        self._vocab = vocab
        self._piece_to_id = {piece: token_id for token_id, piece in self._pieces.items()}
        self._max_piece_len = max(len(piece) for piece in self._pieces.values())

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        data = text.encode()
        ids: list[int] = []
        pos = 0
        while pos < len(data):
            for end in range(min(len(data), pos + self._max_piece_len), pos, -1):
                token_id = self._piece_to_id.get(data[pos:end])
                if token_id is not None:
                    ids.append(token_id)
                    pos = end
                    break
            else:
                raise ValueError(f"fake tokenizer cannot encode byte {data[pos]:#04x}")
        return ids


class _BrokenEncodeTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return super().encode(text, add_special_tokens=add_special_tokens)[::-1]


def entry(token_id: int, logit: float) -> dict[str, float]:
    return {"token_id": token_id, "logit": logit}


def test_bytes_union_argmax_matches_hand_computed_min_floor_scores():
    # Shared vocab: merged pieces " the"=256, " a"=257, " cat"=258.
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a", b" cat")))
    a_topk = [entry(256, 5.0), entry(257, 4.0)]
    b_topk = [entry(257, 6.0), entry(258, 5.5)]
    # Min-floor union scores at advisor_weight=0.5 (floors: a=4.0, b=5.5):
    #   " the": 0.5*5.0 + 0.5*5.5 = 5.25  <- argmax
    #   " a":   0.5*4.0 + 0.5*6.0 = 5.00
    #   " cat": 0.5*4.0 + 0.5*5.5 = 4.75
    a = xtok_selection.candidates(vocab, a_topk)
    b = xtok_selection.candidates(vocab, b_topk)
    assert xtok_selection.avg_bytes_union_scores(a, b, 0.5) == {
        b" the": 5.25,
        b" a": 5.0,
        b" cat": 4.75,
    }
    tokens_a, tokens_b = xtok_selection.select_avg_bytes_union(
        a_topk,
        b_topk,
        advisor_weight=0.5,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([256], [256])


def test_bytes_union_forces_byte_exact_segmentation_on_the_other_vocab():
    # A merges " the"; B only merges " th", so B must emit " th" + "e".
    vocab_a = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the",)))
    vocab_b = xtok_selection.load_vocab(FakeTokenizer(merged=(b" th",)))
    tokens_a, tokens_b = xtok_selection.select_avg_bytes_union(
        [entry(256, 5.0), entry(ord("x"), 1.0)],
        [entry(ord(" "), 1.0)],
        advisor_weight=0.0,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )
    assert tokens_a == [256]
    assert tokens_b == [256, ord("e")]  # " th" + "e"
    assert b"".join(vocab_b.token_bytes[i] or b"" for i in tokens_b) == b" the"


def test_force_segments_partial_utf8_chunk():
    # First two bytes of "😀" (f0 9f 98 80): not representable as a str, so
    # only byte-native segmentation can force it.
    vocab = xtok_selection.load_vocab(FakeTokenizer())
    assert xtok_selection.force(vocab, b"\xf0\x9f", {}) == [0xF0, 0x9F]


def test_candidates_maps_eos_and_drops_other_specials():
    tokenizer = FakeTokenizer(merged=(b" the",), extra_specials=("<pad>",))
    vocab = xtok_selection.load_vocab(tokenizer)
    pad_id = tokenizer.eos_token_id + 1
    cands = xtok_selection.candidates(vocab, [entry(tokenizer.eos_token_id, 2.0), entry(pad_id, 5.0), entry(256, 1.0)])
    assert set(cands) == {xtok_selection.EOS_KEY, b" the"}


def test_candidates_all_specials_raises():
    # Fail-fast contract: an all-special top-k must kill the run, not fall
    # back silently to the other side.
    tokenizer = FakeTokenizer(extra_specials=("<pad>",))
    vocab = xtok_selection.load_vocab(tokenizer)
    with pytest.raises(ValueError, match="only special"):
        xtok_selection.candidates(vocab, [entry(tokenizer.eos_token_id + 1, 5.0)])


def test_candidates_drops_non_finite_logits():
    # A -inf entry must not become the side's min-floor (regression:
    # floor * 0.0 advisor weight -> NaN -> "Total of weights must be finite").
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    cands = xtok_selection.candidates(vocab, [entry(256, 1.0), entry(257, float("-inf"))])
    assert set(cands) == {b" the"}


def test_bytes_union_endpoint_weight_ignores_masked_payload_entries():
    # Regression: advisor_weight=0.0 with a -inf logit in B's top-k poisoned
    # B's floor and crashed sampling via 0.0 * -inf = NaN. The masked entry
    # must be dropped and a valid candidate sampled.
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    tokens_a, tokens_b = xtok_selection.select_avg_bytes_union(
        [entry(256, 5.0)],
        [entry(257, 2.0), entry(ord("x"), float("-inf"))],
        advisor_weight=0.0,
        temperature=0.4,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert tokens_a == tokens_b
    assert tokens_a[0] in (256, 257)


def test_anchored_endpoint_weight_ignores_masked_payload_entries():
    # Regression: a -inf entry in B's top-k made B's floor log(0) = -inf,
    # so advisor_weight=0.0 scored 0.0 * -inf = NaN.
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    tokens_a, tokens_b = xtok_selection.select_avg_anchored(
        [entry(256, 3.0), entry(257, 1.0)],
        [entry(257, 2.0), entry(ord("x"), float("-inf"))],
        advisor_weight=0.0,
        temperature=0.0,
        prefix_credit=1.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert tokens_a == [256]
    assert tokens_b == [256]


def test_prefix_mass_counts_extensions_fully_and_prefixes_by_credit():
    probs: dict[xtok_selection.Key, float] = {b" there": 0.5, b" th": 0.25, b" x": 0.25}
    # " there" extends " the" (full credit); " th" is a strict prefix missing
    # one byte (credit**1); " x" is unrelated.
    assert xtok_selection.prefix_mass(b" the", probs, credit=1.0) == pytest.approx(0.75)
    assert xtok_selection.prefix_mass(b" the", probs, credit=0.0) == pytest.approx(0.5)
    # Exact match counts fully regardless of credit.
    assert xtok_selection.prefix_mass(b" th", probs, credit=0.0) == pytest.approx(0.75)


def test_anchored_weight_zero_is_pure_decoder_argmax():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    tokens_a, tokens_b = xtok_selection.select_avg_anchored(
        [entry(256, 3.0), entry(257, 1.0)],
        [entry(257, 10.0)],  # B strongly prefers " a"; irrelevant at w=0
        advisor_weight=0.0,
        temperature=0.0,
        prefix_credit=1.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert tokens_a == [256]
    assert tokens_b == [256]


def test_anchored_advisor_mass_overrides_decoder_ranking():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # A slightly prefers " the"; B's mass sits almost entirely on " a".
    # At advisor_weight=0.9 the advisor term dominates:
    #   " the": 0.1*2.0 + 0.9*log(~3e-4) ~ -7.0
    #   " a":   0.1*1.9 + 0.9*log(~1.0)  ~ +0.19  <- argmax
    tokens_a, tokens_b = xtok_selection.select_avg_anchored(
        [entry(256, 2.0), entry(257, 1.9)],
        [entry(257, 8.0), entry(ord(" "), 0.0)],
        advisor_weight=0.9,
        temperature=0.0,
        prefix_credit=1.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert tokens_a == [257]
    assert tokens_b == [257]


def test_anchored_eos_agreement_forces_each_sides_own_eos():
    vocab_a = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the",)))  # eos id 257
    vocab_b = xtok_selection.load_vocab(FakeTokenizer())  # eos id 256
    assert vocab_a.eos_id != vocab_b.eos_id
    tokens_a, tokens_b = xtok_selection.select_avg_anchored(
        [entry(vocab_a.eos_id, 5.0), entry(256, 0.0)],
        [entry(vocab_b.eos_id, 4.0), entry(ord(" "), 0.0)],
        advisor_weight=0.5,
        temperature=0.0,
        prefix_credit=1.0,
        rng=random.Random(0),
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )
    assert tokens_a == [vocab_a.eos_id]
    assert tokens_b == [vocab_b.eos_id]


def test_load_vocab_rejects_non_byte_level_bpe_token():
    # Fail-fast contract: sentencepiece-style pieces are unsupported.
    with pytest.raises(ValueError, match="byte-level BPE"):
        xtok_selection.load_vocab(FakeTokenizer(raw_tokens=("▁foo",)))


def test_load_vocab_rejects_non_byte_complete_vocab():
    # Fail-fast contract: byte-completeness is what makes segment() total.
    with pytest.raises(ValueError, match="byte-complete"):
        xtok_selection.load_vocab(FakeTokenizer(drop_bytes=(0,)))


def test_load_vocab_rejects_broken_round_trip():
    # Fail-fast contract: the byte table must reproduce encode() output.
    with pytest.raises(ValueError, match="round-trip"):
        xtok_selection.load_vocab(_BrokenEncodeTokenizer())


def test_kl_bytes_union_identical_topk_is_zero():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    a = xtok_selection.candidates(vocab, [entry(256, 3.0), entry(257, 1.0)])
    assert xtok_selection.kl_bytes_union(a, a) == 0.0


def test_kl_bytes_union_matches_hand_computed_value_and_is_asymmetric():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    a = xtok_selection.candidates(vocab, [entry(256, 2.0), entry(257, 0.0)])
    b = xtok_selection.candidates(vocab, [entry(256, 0.0), entry(257, 1.0)])
    # Shared keys, so no floors engage: KL(softmax(2, 0) || softmax(0, 1)).
    assert xtok_selection.kl_bytes_union(a, b) == pytest.approx(0.8287249104088974)
    assert xtok_selection.kl_bytes_union(b, a) == pytest.approx(1.0068420594147647)


def test_kl_gate_below_threshold_samples_decoder_top_k():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # Same keys both sides with a 0.1-logit swap: KL ~ 0.005, well below the
    # threshold, so A's argmax " the" wins despite B preferring " a".
    tokens_a, tokens_b, kl = xtok_selection.select_kl_gate(
        [entry(256, 2.0), entry(257, 1.9)],
        [entry(257, 2.0), entry(256, 1.9)],
        kl_threshold=0.5,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([256], [256])
    assert kl == pytest.approx(0.004995837495788011)


def test_kl_gate_at_or_above_threshold_samples_advisor_top_k():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    tokens_a, tokens_b, kl = xtok_selection.select_kl_gate(
        [entry(256, 2.0), entry(257, 1.9)],
        [entry(257, 2.0), entry(256, 1.9)],
        kl_threshold=0.001,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([257], [257])
    assert kl >= 0.001


def test_kl_gate_below_threshold_samples_advisor_top_k_under_advisor_below():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # Same inputs as the at/above pair, inverted: KL ~ 0.005 is below the
    # threshold, so ADVISOR_BELOW hands the round to B's argmax " a".
    tokens_a, tokens_b, kl = xtok_selection.select_kl_gate(
        [entry(256, 2.0), entry(257, 1.9)],
        [entry(257, 2.0), entry(256, 1.9)],
        kl_threshold=0.5,
        gate=GateDirection.ADVISOR_BELOW,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([257], [257])
    assert kl < 0.5


def test_kl_gate_at_or_above_threshold_samples_decoder_top_k_under_advisor_below():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    tokens_a, tokens_b, kl = xtok_selection.select_kl_gate(
        [entry(256, 2.0), entry(257, 1.9)],
        [entry(257, 2.0), entry(256, 1.9)],
        kl_threshold=0.001,
        gate=GateDirection.ADVISOR_BELOW,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([256], [256])
    assert kl >= 0.001


def test_kl_gate_zero_threshold_forces_advisor_choice_byte_exact_on_decoder_vocab():
    # Single-candidate sides floor-fill the union to uniform on both sides,
    # so the KL is exactly 0; threshold 0.0 still gates to the advisor (the
    # decoder test is strict), whose " the" A must segment as " th" + "e".
    vocab_a = xtok_selection.load_vocab(FakeTokenizer(merged=(b" th",)))
    vocab_b = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the",)))
    tokens_a, tokens_b, kl = xtok_selection.select_kl_gate(
        [entry(ord("x"), 5.0)],
        [entry(256, 1.0)],
        kl_threshold=0.0,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )
    assert kl == 0.0
    assert tokens_b == [256]
    assert tokens_a == [256, ord("e")]  # " th" + "e"


def test_kl_gate_advisor_eos_forces_each_sides_own_eos():
    vocab_a = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the",)))  # eos id 257
    vocab_b = xtok_selection.load_vocab(FakeTokenizer())  # eos id 256
    assert vocab_a.eos_id != vocab_b.eos_id
    tokens_a, tokens_b, _ = xtok_selection.select_kl_gate(
        [entry(256, 5.0)],  # A proposes " the"
        [entry(vocab_b.eos_id, 1.0)],  # B's whole top-k is EOS
        kl_threshold=0.0,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )
    assert tokens_a == [vocab_a.eos_id]
    assert tokens_b == [vocab_b.eos_id]


def test_topk_entropy_matches_hand_computed_values():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    uniform = xtok_selection.candidates(vocab, [entry(256, 1.0), entry(257, 1.0)])
    assert xtok_selection.topk_entropy(uniform) == pytest.approx(math.log(2))
    peaked = xtok_selection.candidates(vocab, [entry(256, 2.0), entry(257, 0.0)])
    assert xtok_selection.topk_entropy(peaked) == pytest.approx(0.3653338550872077)
    single = xtok_selection.candidates(vocab, [entry(256, 3.0)])
    assert xtok_selection.topk_entropy(single) == 0.0


def test_entropy_gate_below_threshold_samples_decoder_top_k():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # B is peaked (H ~ 0.04 < 0.5), so A's argmax " the" wins despite B
    # strongly preferring " a".
    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        [entry(256, 2.0), entry(257, 0.0)],
        [entry(257, 5.0), entry(256, 0.0)],
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([256], [256])
    assert 0.0 < entropy < 0.5


def test_entropy_gate_at_or_above_threshold_samples_advisor_top_k():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # B is near-uniform (H ~ 0.69 >= 0.5), so its argmax " a" wins despite A
    # preferring " the".
    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        [entry(256, 2.0), entry(257, 0.0)],
        [entry(257, 1.0), entry(256, 0.9)],
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([257], [257])
    assert 0.5 <= entropy < math.log(2)


def test_entropy_gate_below_threshold_samples_advisor_top_k_under_advisor_below():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # B is peaked (H ~ 0.04 < 0.5): ADVISOR_BELOW defers to the advisor
    # exactly where it is confident, so its " a" commits.
    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        [entry(256, 2.0), entry(257, 0.0)],
        [entry(257, 5.0), entry(256, 0.0)],
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_BELOW,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([257], [257])
    assert 0.0 < entropy < 0.5


def test_entropy_gate_at_or_above_threshold_samples_decoder_top_k_under_advisor_below():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # B is near-uniform (H ~ 0.69 >= 0.5), so the decoder's " the" commits.
    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        [entry(256, 2.0), entry(257, 0.0)],
        [entry(257, 1.0), entry(256, 0.9)],
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_BELOW,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert (tokens_a, tokens_b) == ([256], [256])
    assert 0.5 <= entropy < math.log(2)


def test_gate_rejects_a_value_that_is_not_a_member():
    # The child config round trip reconstructs the enum; a raw string that
    # skipped that step compares equal to a member but is not `is`-identical
    # to one, so it must raise rather than pick a direction.
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    with pytest.raises(ValueError, match="unknown gate direction"):
        xtok_selection.select_entropy_gate(
            [entry(256, 2.0), entry(257, 0.0)],
            [entry(257, 1.0), entry(256, 0.9)],
            entropy_source=EntropySource.ADVISOR,
            entropy_threshold=0.5,
            gate="advisor_below",  # type: ignore[arg-type]
            temperature=0.0,
            rng=random.Random(0),
            vocab_a=vocab,
            vocab_b=vocab,
        )


def test_entropy_gate_zero_threshold_is_pure_advisor():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    # A single-candidate advisor has entropy exactly 0, and the gate is
    # strict, so threshold 0.0 still samples the advisor.
    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        [entry(256, 1.0)],
        [entry(257, 5.0)],
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.0,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )
    assert entropy == 0.0
    assert (tokens_a, tokens_b) == ([257], [257])


def test_entropy_gate_advisor_eos_forces_each_sides_own_eos():
    vocab_a = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the",)))  # eos id 257
    vocab_b = xtok_selection.load_vocab(FakeTokenizer())  # eos id 256
    assert vocab_a.eos_id != vocab_b.eos_id
    tokens_a, tokens_b, _ = xtok_selection.select_entropy_gate(
        [entry(256, 5.0)],  # A proposes " the"
        [entry(vocab_b.eos_id, 1.0), entry(ord(" "), 0.9)],  # B near-uniform, H >= 0.5
        entropy_source=EntropySource.ADVISOR,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )
    assert tokens_a == [vocab_a.eos_id]
    assert tokens_b == [vocab_b.eos_id]


def test_entropy_gate_can_use_decoder_entropy():
    vocab = xtok_selection.load_vocab(FakeTokenizer(merged=(b" the", b" a")))
    a_topk = [entry(256, 5.0), entry(257, 0.0)]
    b_topk = [entry(257, 1.0), entry(256, 0.9)]

    tokens_a, tokens_b, entropy = xtok_selection.select_entropy_gate(
        a_topk,
        b_topk,
        entropy_source=EntropySource.DECODER,
        entropy_threshold=0.5,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )

    assert entropy == xtok_selection.topk_entropy(xtok_selection.candidates(vocab, a_topk))
    assert entropy < 0.5
    assert (tokens_a, tokens_b) == ([256], [256])


def test_decoder_entropy_gate_threshold_anchors():
    vocab = xtok_selection.load_vocab(FakeTokenizer())
    a_topk = [entry(token_id, float(16 - token_id)) for token_id in range(16)]
    b_topk = [entry(token_id, float(token_id)) for token_id in range(16)]
    advisor_tokens = xtok_selection.select_entropy_gate(
        a_topk,
        b_topk,
        entropy_source=EntropySource.DECODER,
        entropy_threshold=0.0,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )[:2]
    decoder_tokens = xtok_selection.select_entropy_gate(
        a_topk,
        b_topk,
        entropy_source=EntropySource.DECODER,
        entropy_threshold=math.log(16) + 1.0,
        gate=GateDirection.ADVISOR_AT_OR_ABOVE,
        temperature=0.0,
        rng=random.Random(0),
        vocab_a=vocab,
        vocab_b=vocab,
    )[:2]

    assert advisor_tokens == ([15], [15])
    assert decoder_tokens == ([0], [0])
