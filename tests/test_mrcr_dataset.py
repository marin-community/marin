# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models

from experiments.datasets import mrcr


class _OffsetTokenizer:
    eos_token = "§"
    bos_token = None
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = None
    name_or_path = "offset-tokenizer"
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

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {}

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return chr(ids)
        return [chr(token) for token in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return ord(tokens)
        return [ord(token) for token in tokens]

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str | list[int]:
        del kwargs
        rendered = conversation[0]["content"] + conversation[1]["content"] + self.eos_token
        return self.encode(rendered)

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        del chat_template, kwargs
        input_ids: list[list[int]] = []
        masks: list[list[int]] = []
        for conversation in conversations:
            prompt = conversation[0]["content"]
            target = conversation[1]["content"]
            rendered = prompt + target + self.eos_token
            input_ids.append(self.encode(rendered))
            masks.append([0] * len(prompt) + [1] * len(target) + [0])
        return {"input_ids": input_ids, "assistant_masks": masks}

    def as_hf_tokenizer(self) -> Any:
        tokenizer = self

        class _Offsets:
            def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
                del kwargs
                return {
                    "input_ids": tokenizer.encode(text),
                    "offset_mapping": [(index, index + 1) for index in range(len(text))],
                }

        return _Offsets()


class _BpeBoundaryTokenizer:
    eos_token = "§"
    bos_token = None
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = None
    name_or_path = "bpe-boundary-tokenizer"
    all_special_ids: ClassVar[list[int]] = [1]
    chat_template = None

    def __init__(self, characters: set[str]):
        tokens = ["[UNK]", *sorted(characters), "pr", "pre", "fi", "fix", "prefix"]
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._tokenizer = Tokenizer(
            models.BPE(
                vocab=self._vocab,
                merges=[("p", "r"), ("pr", "e"), ("f", "i"), ("fi", "x"), ("pre", "fix")],
                unk_token="[UNK]",
            )
        )
        self.vocab_size = len(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def encode_with_offsets(self, text: str) -> Any:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return self._tokenizer.decode(ids)

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids) or "[UNK]"
        return [self._tokenizer.id_to_token(token) or "[UNK]" for token in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._vocab.get(tokens, 0)
        return [self._vocab.get(token, 0) for token in tokens]

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str | list[int]:
        del kwargs
        rendered = conversation[0]["content"] + conversation[1]["content"] + self.eos_token
        return self.encode(rendered)

    def apply_chat_template_with_masks(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        del chat_template, kwargs
        input_ids: list[list[int]] = []
        masks: list[list[int]] = []
        for conversation in conversations:
            prompt_ids = self.encode(conversation[0]["content"])
            target_ids = self.encode(conversation[1]["content"])
            eos_ids = self.encode(self.eos_token)
            input_ids.append([*prompt_ids, *target_ids, *eos_ids])
            masks.append([0] * len(prompt_ids) + [1] * len(target_ids) + [0] * len(eos_ids))
        return {"input_ids": input_ids, "assistant_masks": masks}

    def as_hf_tokenizer(self) -> Any:
        tokenizer = self._tokenizer

        class _Offsets:
            def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
                del kwargs
                encoded = tokenizer.encode(text, add_special_tokens=False)
                return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

        return _Offsets()


@pytest.fixture
def offset_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _OffsetTokenizer:
    tokenizer = _OffsetTokenizer()
    monkeypatch.setattr(mrcr, "load_tokenizer", lambda _: tokenizer)
    return tokenizer


def _preamble(*, valid: bool = True) -> str:
    prefix = mrcr.MRCR_PREAMBLE_PREFIX if valid else "Unofficial examples:"
    return (
        f"{prefix}\n\n"
        "======EXAMPLE======\nUser: demo one\nAssistant: first demonstration\n======END EXAMPLE======\n\n"
        "======EXAMPLE======\nUser: demo two\nAssistant: second demonstration\n======END EXAMPLE======\n"
    )


def _row(
    *,
    filler: str = "distractor",
    needles: int = 2,
    desired_msg_index: int = 5,
    preamble: str | None = None,
    nonce: str = "Ab3dE5gH7j",
    target: str = "selected response body",
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": preamble if preamble is not None else _preamble()},
        {"role": "user", "content": "write the target"},
        {"role": "assistant", "content": "first target response"},
        {"role": "user", "content": "unrelated request"},
        {"role": "assistant", "content": filler},
        {"role": "user", "content": "write the target"},
        {"role": "assistant", "content": target},
        *[
            message
            for occurrence in range(2, needles)
            for message in (
                {"role": "user", "content": "write the target"},
                {"role": "assistant", "content": f"later target response {occurrence + 1}"},
            )
        ],
        {"role": "user", "content": "another unrelated request"},
        {"role": "assistant", "content": "trailing distractor"},
        {
            "role": "user",
            "content": f"Prepend {nonce} to the 2nd target response. Do not include any other text.",
        },
    ]
    return {
        "prompt": json.dumps(messages),
        "answer": nonce + target,
        "random_string_to_prepend": nonce,
        "n_needles": needles,
        "desired_msg_index": desired_msg_index,
    }


def _write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    source = root / "2needle"
    source.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), source / "rows.parquet")


def _read_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source]


def _records(output: Path) -> list[dict[str, Any]]:
    return [record for path in sorted(output.glob("**/*.jsonl.gz")) for record in _read_gzip(path)]


def test_transform_mrcr_uses_complete_bpe_offsets_when_generation_boundary_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row(nonce="Ab3dE5gpre", target="fix")
    messages = json.loads(row["prompt"])
    canonical_prompt = (
        "".join(f"{message['role'].capitalize()}: {message['content']}\n" for message in messages)
        + f"Assistant: {row['random_string_to_prepend']}"
    )
    target = row["answer"].removeprefix(row["random_string_to_prepend"])
    tokenizer = _BpeBoundaryTokenizer(set(canonical_prompt + target + _BpeBoundaryTokenizer.eos_token))
    segmented_ids = [
        *tokenizer.encode(canonical_prompt),
        *tokenizer.encode(target),
        *tokenizer.encode(tokenizer.eos_token),
    ]
    complete = tokenizer.encode_with_offsets(canonical_prompt + target + tokenizer.eos_token)
    assert segmented_ids != complete.ids

    monkeypatch.setattr(mrcr, "load_tokenizer", lambda _: tokenizer)
    _write_rows(tmp_path / "input", [row])
    output = tmp_path / "output"
    mrcr.transform_mrcr(
        mrcr.MrcrTransformConfig(
            input_path=str(tmp_path / "input"),
            output_path=str(output),
            tokenizer="bpe-boundary-tokenizer",
            context_caps=(4_096,),
        )
    )

    selected_start = canonical_prompt.index("Assistant: fix") + len("Assistant: ")
    selected_end = selected_start + len(target)
    target_start = len(canonical_prompt)
    response_tokens = [
        index for index, (start, end) in enumerate(complete.offsets) if end > selected_start and start < selected_end
    ]
    target_tokens = [
        index
        for index, (start, end) in enumerate(complete.offsets)
        if end > target_start and start < target_start + len(target)
    ]
    expected_distance = target_tokens[0] - response_tokens[-1] - 1
    records = _records(output)

    assert len(records) == 12
    assert {record["evidence_distance_tokens"] for record in records} == {expected_distance}
    assert {record["messages"][1]["content"] for record in records} == {target}
    for record in records:
        processed = mrcr._mrcr_format().build_preprocessor(tokenizer)([record])[0]
        scored_ids = [
            token for token, weight in zip(processed["input_ids"], processed["assistant_masks"], strict=True) if weight
        ]
        assert scored_ids == tokenizer.encode(target)


def test_transform_mrcr_builds_paired_variants_with_identical_scored_bodies(
    tmp_path: Path, offset_tokenizer: _OffsetTokenizer
) -> None:
    row = _row()
    _write_rows(tmp_path / "input", [row])
    output = tmp_path / "output"
    mrcr.transform_mrcr(
        mrcr.MrcrTransformConfig(
            input_path=str(tmp_path / "input"),
            output_path=str(output),
            tokenizer="offset-tokenizer",
            context_caps=(4_096,),
            distance_bounds=(128, 256, 512),
        )
    )

    records = _records(output)
    assert len(records) == 12
    by_variant_condition = {(record["prompt_variant"], record["condition"]): record for record in records}
    target_bodies = {record["messages"][1]["content"] for record in records}
    assert target_bodies == {"selected response body"}
    assert {record["source_id"] for record in records} == {records[0]["source_id"]}
    assert {record["context_cap"] for record in records} == {4_096}
    assert {record["distance_band"] for record in records} == {records[0]["distance_band"]}

    two_shot_full = by_variant_condition[("two_shot", "full_context")]["messages"][0]["content"]
    two_shot_query = by_variant_condition[("two_shot", "query_only")]["messages"][0]["content"]
    two_shot_needle = by_variant_condition[("two_shot", "needle_only")]["messages"][0]["content"]
    two_shot_distractor = by_variant_condition[("two_shot", "distractor_only")]["messages"][0]["content"]
    one_shot_full = by_variant_condition[("one_shot", "full_context")]["messages"][0]["content"]
    no_prefix_full = by_variant_condition[("two_shot_no_prefix", "full_context")]["messages"][0]["content"]
    no_prefix_query = by_variant_condition[("two_shot_no_prefix", "query_only")]["messages"][0]["content"]

    assert "first demonstration" in one_shot_full
    assert "second demonstration" not in one_shot_full
    assert "selected response body" in two_shot_full
    assert "selected response body" not in two_shot_query
    assert "selected response body" in two_shot_needle
    assert "first target response" in two_shot_needle
    assert "trailing distractor" not in two_shot_needle
    assert "selected response body" not in two_shot_distractor
    assert "User: write the target\nAssistant: selected response body" not in two_shot_distractor
    assert "first target response" not in two_shot_distractor
    assert "trailing distractor" in two_shot_distractor
    assert two_shot_query.startswith(f"User: {_preamble()}")
    assert two_shot_full.endswith("Assistant: Ab3dE5gH7j")
    assert no_prefix_full.endswith("Assistant: ")
    assert "Prepend Ab3dE5gH7j to " not in no_prefix_full
    assert "User: Return the 2nd target response. Do not include any other text.\nAssistant: " in no_prefix_query

    desired_response = "selected response body"
    response_end = two_shot_full.index(desired_response) + len(desired_response)
    expected_distance = len(two_shot_full) - response_end
    assert by_variant_condition[("two_shot", "full_context")]["evidence_distance_tokens"] == expected_distance

    manifest_path = next(output.glob("two_shot/**/manifest.jsonl"))
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert manifest == [
        {
            "canonical_full_length_tokens": records[0]["canonical_full_length_tokens"],
            "evidence_distance_tokens": records[0]["evidence_distance_tokens"],
            "scored_tokens": len("selected response body"),
            "source_id": records[0]["source_id"],
            "variant_full_length_tokens": by_variant_condition[("two_shot", "full_context")][
                "variant_full_length_tokens"
            ],
        }
    ]


def _canonical_length(row: dict[str, Any]) -> int:
    messages = json.loads(row["prompt"])
    rendered = "".join(f"{message['role'].capitalize()}: {message['content']}\n" for message in messages)
    rendered += f"Assistant: {row['random_string_to_prepend']}"
    target = row["answer"].removeprefix(row["random_string_to_prepend"])
    return len(rendered) + len(target) + 1


def _row_with_canonical_length(length: int, *, needles: int = 2) -> dict[str, Any]:
    baseline = _row(filler="", needles=needles)
    filler_length = length - _canonical_length(baseline)
    assert filler_length >= 0
    row = _row(filler="x" * filler_length, needles=needles)
    assert _canonical_length(row) == length
    return row


def test_transform_mrcr_canonical_bins_do_not_slice_pairs_or_prompt_variants(
    tmp_path: Path, offset_tokenizer: _OffsetTokenizer
) -> None:
    caps = mrcr.MRCR_CONTEXT_CAPS
    accepted_lengths = [*caps, *(cap + 1 for cap in caps[:-1])]
    rows = [_row_with_canonical_length(length) for length in accepted_lengths]
    rows.extend(_row_with_canonical_length(caps[-1] + 1, needles=needles) for needles in mrcr.MRCR_NEEDLE_COUNTS)
    _write_rows(tmp_path / "input", rows)
    output = tmp_path / "output"
    mrcr.transform_mrcr(
        mrcr.MrcrTransformConfig(
            input_path=str(tmp_path / "input"),
            output_path=str(output),
            tokenizer="offset-tokenizer",
            context_caps=caps,
        )
    )

    records = _records(output)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["source_id"], []).append(record)
    assert len(grouped) == len(accepted_lengths)
    for source_records in grouped.values():
        assert len(source_records) == 12
        canonical_length = source_records[0]["canonical_full_length_tokens"]
        expected_cap = next(cap for cap in caps if canonical_length <= cap)
        assert {record["context_cap"] for record in source_records} == {expected_cap}
        for record in source_records:
            processed = mrcr._mrcr_format().build_preprocessor(offset_tokenizer)([record])[0]
            assert len(processed["input_ids"]) <= expected_cap
            assert sum(processed["assistant_masks"]) == len("selected response body")

    stats = json.loads((output / "stats.json").read_text())
    assert stats[f"excluded_over_{caps[-1]}"] == {
        "total": 3,
        "2needle": 1,
        "4needle": 1,
        "8needle": 1,
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row.update(prompt=json.dumps(json.loads(row["prompt"])[1:])), "worked-example preamble"),
        (lambda row: row.update(desired_msg_index=4), "user request followed by an assistant response"),
        (lambda row: row.update(desired_msg_index=1), "does not match the answer body"),
    ],
)
def test_transform_mrcr_rejects_invalid_official_preamble_or_selected_response(
    tmp_path: Path,
    offset_tokenizer: _OffsetTokenizer,
    mutate: Any,
    match: str,
) -> None:
    row = _row()
    mutate(row)
    _write_rows(tmp_path / "input", [row])
    with pytest.raises(ValueError, match=match):
        mrcr.transform_mrcr(
            mrcr.MrcrTransformConfig(
                input_path=str(tmp_path / "input"),
                output_path=str(tmp_path / "output"),
                tokenizer="offset-tokenizer",
                context_caps=(4_096,),
            )
        )


def test_transform_mrcr_rejects_requested_empty_context_cap(tmp_path: Path, offset_tokenizer: _OffsetTokenizer) -> None:
    _write_rows(tmp_path / "input", [_row_with_canonical_length(1_000)])
    with pytest.raises(ValueError, match="have no accepted examples"):
        mrcr.transform_mrcr(
            mrcr.MrcrTransformConfig(
                input_path=str(tmp_path / "input"),
                output_path=str(tmp_path / "output"),
                tokenizer="offset-tokenizer",
                context_caps=(1_000, 2_000),
            )
        )
