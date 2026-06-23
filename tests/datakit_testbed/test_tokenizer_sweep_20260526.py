# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
import time

from tokenizers import Tokenizer
from tokenizers import AddedToken
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers

from experiments.datakit_testbed.tokenizer_sweep_20260526 import (
    PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS,
    _derive_hf_bpe_tokenizer_dir,
    _place_aligned_digit_pretokenizer,
    place_aligned_digit_pieces,
)


def _lookahead_digit_pieces(text: str) -> list[str]:
    return [piece for piece in re.split(r"(?=(?:\d{3})+(?!\d))", text) if piece]


def test_place_aligned_digit_pieces_match_lookahead_through_cap() -> None:
    for length in range(1, PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS + 1):
        digits = "1" * length
        assert place_aligned_digit_pieces(digits) == _lookahead_digit_pieces(digits)


def test_place_aligned_digit_pieces_isolate_surrounding_text() -> None:
    assert place_aligned_digit_pieces("abc1234567def") == ["abc", "1", "234", "567", "def"]
    assert place_aligned_digit_pieces("x12 y1234 z123456") == ["x", "12", " y", "1", "234", " z", "123", "456"]


def test_place_aligned_digit_pretokenizer_is_serializable_and_bounded() -> None:
    pretokenizer = _place_aligned_digit_pretokenizer(pre_tokenizers.WhitespaceSplit())
    assert pretokenizer.pre_tokenize_str("abc1234567def") == [
        ("abc", (0, 3)),
        ("1", (3, 4)),
        ("234", (4, 7)),
        ("567", (7, 10)),
        ("def", (10, 13)),
    ]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pretokenizer
    tokenizer.train_from_iterator(["abc1234567def"], trainers.BpeTrainer(vocab_size=64))
    assert '"pre_tokenizer"' in tokenizer.to_str()


def test_place_aligned_digit_pretokenizer_handles_long_digit_runs_quickly() -> None:
    pretokenizer = _place_aligned_digit_pretokenizer(pre_tokenizers.WhitespaceSplit())
    long_digits = "9" * 100_000

    start = time.perf_counter()
    pieces = pretokenizer.pre_tokenize_str(long_digits)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0
    assert len(pieces) == 33_334
    assert pieces[0] == ("999", (0, 3))
    assert pieces[-1] == ("999", (99_997, 100_000))


def test_derive_hf_bpe_tokenizer_rewrites_special_ids_and_filters_merges(tmp_path) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(
        [
            AddedToken("<bos>", special=True),
            AddedToken("<eos>", special=True),
            AddedToken("<pad>", special=True),
        ]
    )
    tokenizer.train_from_iterator(
        ["aa ab abc abd abcde xyz", "aa abc abd xyz"],
        trainers.BpeTrainer(vocab_size=32, special_tokens=[]),
    )

    base_dir = tmp_path / "base"
    out_dir = tmp_path / "8"
    base_dir.mkdir()
    tokenizer.save(str(base_dir / "tokenizer.json"))

    _derive_hf_bpe_tokenizer_dir(str(base_dir), 8, str(out_dir))

    derived = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    tokenizer_json = derived.to_str()
    assert '"<bos>":5' in tokenizer_json
    assert '"<eos>":6' in tokenizer_json
    assert '"<pad>":7' in tokenizer_json
    assert derived.token_to_id("<bos>") == 5
    assert derived.token_to_id("<eos>") == 6
    assert derived.token_to_id("<pad>") == 7

    encoded = derived.encode("aa abc <bos>", add_special_tokens=False)
    assert max(encoded.ids) < 8
