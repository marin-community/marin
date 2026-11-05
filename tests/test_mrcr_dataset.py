# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.store.cache import TreeCache
from tokenizers import Tokenizer, models

from experiments.datasets import mrcr


class _CharTokenizer:
    """One token per character, so token offsets and lengths follow the rendered text exactly."""

    eos_token = "§"
    bos_token = None
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = None
    name_or_path = "char-tokenizer"
    vocab_size = 257
    all_special_ids: ClassVar[list[int]] = [1]
    chat_template = None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 257 for char in text]

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in ids)

    def as_hf_tokenizer(self) -> Any:
        encode = self.encode

        class _Offsets:
            def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
                del kwargs
                return {"input_ids": encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        return _Offsets()


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> _CharTokenizer:
    char_tokenizer = _CharTokenizer()
    monkeypatch.setattr(mrcr, "load_tokenizer", lambda _: char_tokenizer)
    return char_tokenizer


def _row(*, filler: str = "distractor", nonce: str = "Ab3dE5gH7j", target: str = "selected body") -> dict[str, Any]:
    preamble = (
        f"{mrcr.MRCR_PREAMBLE_PREFIX}\n\n"
        "======EXAMPLE======\nUser: demo one\nAssistant: first\n======END EXAMPLE======\n\n"
        "======EXAMPLE======\nUser: demo two\nAssistant: second\n======END EXAMPLE======\n"
    )
    messages = [
        {"role": "user", "content": preamble},
        {"role": "user", "content": "write the target"},
        {"role": "assistant", "content": "first target response"},
        {"role": "user", "content": "unrelated request"},
        {"role": "assistant", "content": filler},
        {"role": "user", "content": "write the target"},
        {"role": "assistant", "content": target},
        {"role": "user", "content": f"Prepend {nonce} to the 2nd target response."},
    ]
    return {
        "prompt": json.dumps(messages),
        "answer": nonce + target,
        "random_string_to_prepend": nonce,
        "n_needles": 2,
        "desired_msg_index": 5,
    }


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    source = root / "2needle"
    source.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), source / "rows.parquet")


def _config(tmp_path: Path, **overrides: Any) -> mrcr.MrcrTransformConfig:
    fields = {
        "input_path": str(tmp_path / "input"),
        "output_path": str(tmp_path / "output"),
        "tokenizer": "char-tokenizer",
        "context_cap": 1_000,
        "context_cap_lower": 300,
        "distance_bounds": (8, 16, 32),
    }
    return mrcr.MrcrTransformConfig(**(fields | overrides))


def _manifest(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "output" / mrcr.MRCR_MANIFEST_NAME).read_text().splitlines()]


def _cache(tmp_path: Path) -> list[dict[str, np.ndarray]]:
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "assistant_masks": np.zeros((0,), dtype=np.int32)}
    cache = TreeCache.load(str(tmp_path / "output" / mrcr.MRCR_CACHE_SPLIT), exemplar)
    return [cache[index] for index in range(len(cache))]


def test_cell_rows_share_one_scored_body_and_pair_with_the_manifest_longest_first(
    tmp_path: Path, tokenizer: _CharTokenizer
) -> None:
    below_the_cell = _row(filler="short")
    rows = [below_the_cell, _row(filler="medium " * 40), _row(filler="long " * 90), _row(filler="beyond " * 300)]
    _write_rows(tmp_path / "input", rows)
    # The exact length decides membership: one past the short row's whole render, EOS included.
    below_length = (
        len(mrcr._render_prompt(json.loads(below_the_cell["prompt"]), "Ab3dE5gH7j")[0]) + len("selected body") + 1
    )

    mrcr.transform_mrcr(_config(tmp_path, context_cap_lower=below_length))
    manifest = _manifest(tmp_path)
    cache = _cache(tmp_path)

    assert len(manifest) == len(cache) == 8
    assert [record["condition"] for record in manifest[:4]] == [condition.value for condition in mrcr.MrcrCondition]
    lengths = [record["canonical_full_length_tokens"] for record in manifest]
    assert lengths[0] > lengths[4] and all(below_length < length <= 1_000 for length in lengths)
    target_ids = tokenizer.encode("selected body")
    for record, row in zip(manifest, cache, strict=True):
        assert len(row["input_ids"]) == record["length_tokens"]
        assert row["input_ids"][row["assistant_masks"].astype(bool)].tolist() == target_ids
        assert record["scored_tokens"] == len(target_ids)
    texts = {
        record["condition"]: tokenizer.decode(row["input_ids"].tolist())
        for record, row in zip(manifest, cache, strict=True)
    }
    for condition, has_needle, has_distractor in (
        (mrcr.MrcrCondition.FULL_CONTEXT, True, True),
        (mrcr.MrcrCondition.QUERY_ONLY, False, False),
        (mrcr.MrcrCondition.NEEDLE_ONLY, True, False),
        (mrcr.MrcrCondition.DISTRACTOR_ONLY, False, True),
    ):
        assert ("write the target" in texts[condition]) == has_needle
        assert ("unrelated request" in texts[condition]) == has_distractor


def test_evidence_distance_counts_tokens_between_the_selected_response_and_the_target(
    tmp_path: Path, tokenizer: _CharTokenizer
) -> None:
    del tokenizer
    _write_rows(tmp_path / "input", [_row(filler="x" * 300)])
    mrcr.transform_mrcr(_config(tmp_path))
    row = _manifest(tmp_path)[0]
    between = "\nUser: Prepend Ab3dE5gH7j to the 2nd target response.\nAssistant: Ab3dE5gH7j"
    assert row["evidence_distance_tokens"] == len(between)
    assert row["distance_band"] == "distance_33_plus"


def test_a_count_that_differs_from_the_experiment_fails_the_build(tmp_path: Path, tokenizer: _CharTokenizer) -> None:
    del tokenizer
    _write_rows(tmp_path / "input", [_row(filler="x" * 300)])
    with pytest.raises(ValueError, match="Expected 299 MRCR sources"):
        mrcr.transform_mrcr(_config(tmp_path, expected_sources=299))


def test_a_needle_count_that_disagrees_with_the_conversation_fails_the_build(
    tmp_path: Path, tokenizer: _CharTokenizer
) -> None:
    del tokenizer
    _write_rows(tmp_path / "input", [{**_row(filler="x" * 300), "n_needles": 4}])
    with pytest.raises(ValueError, match="n_needles does not match"):
        mrcr.transform_mrcr(_config(tmp_path))


class _BoundaryMergingTokenizer(_CharTokenizer):
    """Merges ``pre`` + ``fix`` into one token when both appear in one string."""

    def __init__(self, characters: set[str]):
        tokens = ["[UNK]", *sorted(characters), "pr", "pre", "fi", "fix", "prefix"]
        self._vocab = {token: index for index, token in enumerate(tokens)}
        merges = [("p", "r"), ("pr", "e"), ("f", "i"), ("fi", "x"), ("pre", "fix")]
        self._tokenizer = Tokenizer(models.BPE(vocab=self._vocab, merges=merges, unk_token="[UNK]"))
        self.eos_token_id = self._vocab[self.eos_token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def as_hf_tokenizer(self) -> Any:
        tokenizer = self._tokenizer

        class _Offsets:
            def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
                del kwargs
                encoded = tokenizer.encode(text, add_special_tokens=False)
                return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

        return _Offsets()


def test_no_scored_token_overlaps_the_answer_prefix_when_the_boundary_would_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row(filler="x" * 300, nonce="Ab3dE5gpre", target="fix")
    rendered = "".join(message["content"] for message in json.loads(row["prompt"]))
    characters = set(rendered + row["answer"] + _CharTokenizer.eos_token + "\n: UA")
    tokenizer = _BoundaryMergingTokenizer(characters)
    monkeypatch.setattr(mrcr, "load_tokenizer", lambda _: tokenizer)
    _write_rows(tmp_path / "input", [row])

    mrcr.transform_mrcr(_config(tmp_path))

    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "assistant_masks": np.zeros((0,), dtype=np.int32)}
    cache = TreeCache.load(str(tmp_path / "output" / mrcr.MRCR_CACHE_SPLIT), exemplar)
    for index in range(len(cache)):
        scored = cache[index]["input_ids"][cache[index]["assistant_masks"].astype(bool)].tolist()
        assert scored == tokenizer.encode("fix")
