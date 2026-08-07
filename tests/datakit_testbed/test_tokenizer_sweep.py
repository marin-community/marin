# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re
import time

from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers

from experiments.datakit_testbed.tokenizer_sweep import (
    DEFAULT_TOKENIZER_SWEEP_TPU_TYPES,
    PLACE_ALIGNED_DIGIT_MAX_RUN_CHARS,
    CorpusConfig,
    HfTokenizerFamilyConfig,
    TokenizerSweepConfig,
    WindowConfig,
    _derive_hf_bpe_tokenizer_dir,
    _place_aligned_digit_pretokenizer,
    build_steps,
    issue_5821_default_config,
    place_aligned_digit_pieces,
)


def _lookahead_digit_pieces(text: str) -> list[str]:
    return [piece for piece in re.split(r"(?=(?:\d{3})+(?!\d))", text) if piece]


def test_issue_5821_default_config_captures_expected_recipe() -> None:
    config = issue_5821_default_config()

    assert config.run_id == "tokenizer-sweep-issue-5821"
    assert config.vocab_sizes == [262_144, 131_072, 32_768, 8_192]
    assert config.tokenizer_train.tokens == 50_000_000_000
    assert config.tokenizer_train.sample_mode == "random-shards"
    assert config.holdout.start_tokens == 100_000_000_000
    assert config.holdout.tokens == 100_000_000_000
    assert config.sample_resource.tpu_types == list(DEFAULT_TOKENIZER_SWEEP_TPU_TYPES)
    assert config.hf_train_resource.tpu_types == list(DEFAULT_TOKENIZER_SWEEP_TPU_TYPES)
    assert config.tokenize_worker_resource.tpu_types == list(DEFAULT_TOKENIZER_SWEEP_TPU_TYPES)
    assert [(family.name, family.base_tokenizer, family.place_aligned_digits) for family in config.hf_families] == [
        ("gpt-oss", "openai/gpt-oss-20b", False),
        ("llama", "meta-llama/Meta-Llama-3.1-8B", False),
        ("gpt-oss-place-digits", "openai/gpt-oss-20b", True),
        ("llama-place-digits", "meta-llama/Meta-Llama-3.1-8B", True),
    ]


def test_build_steps_accepts_custom_config_without_env(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.datakit_testbed.tokenizer_sweep.existing_normalized_sources",
        lambda _normalized_base: {"source": "gs://example/normalized/source"},
    )
    config = TokenizerSweepConfig(
        run_id="custom-tokenizer-sweep",
        corpus=CorpusConfig(normalized_base="gs://example/normalized", total_tokenized_tokens=10_000),
        tokenizer_train=WindowConfig(tokens=1_000, sample_mode="random-shards"),
        holdout=WindowConfig(tokens=1_000, start_tokens=1_000),
        train_retokenize=WindowConfig(tokens=1_000),
        vocab_sizes=[2048, 1024],
        hf_families=[HfTokenizerFamilyConfig("custom", "org/custom-tokenizer")],
        family_filter=["custom"],
        size_filter=[1024],
    )

    steps = build_steps(config, phase="prep")

    assert {step.name for step in steps} == {
        "data/datakit/tokenizer_sweep/custom-tokenizer-sweep/holdout/source",
        "tokenizers/custom-tokenizer-sweep/custom",
    }

    tokenized_steps = build_steps(config, phase="all")

    assert {step.name for step in tokenized_steps} == {
        "data/datakit/tokenized/custom-tokenizer-sweep/custom-1k/source",
    }
    assert [dep.name for dep in tokenized_steps[0].deps] == [
        "data/datakit/tokenizer_sweep/custom-tokenizer-sweep/holdout/source",
        "tokenizers/custom-tokenizer-sweep/custom",
    ]


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
    tokenizer_json = json.loads(derived.to_str())
    added_token_ids = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}
    assert added_token_ids["<bos>"] == 5
    assert added_token_ids["<eos>"] == 6
    assert added_token_ids["<pad>"] == 7
    assert derived.token_to_id("<bos>") == 5
    assert derived.token_to_id("<eos>") == 6
    assert derived.token_to_id("<pad>") == 7

    encoded = derived.encode("aa abc <bos>", add_special_tokens=False)
    assert max(encoded.ids) < 8
