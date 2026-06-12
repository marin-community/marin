# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample packed document lengths for Splash mask benchmarks."""

import argparse
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from levanter.data.packing import pack_documents
from levanter.tokenizers import load_tokenizer


def main() -> None:
    args = _parse_args()
    texts = _texts_from_args(args)
    token_lengths = _token_lengths(
        texts,
        tokenizer_name_or_path=args.tokenizer,
        text_limit=args.max_examples,
        add_special_tokens=args.add_special_tokens,
    )
    packed_lengths = _pack_token_lengths(
        token_lengths,
        seq_len=args.seq_len,
        num_packs=args.num_packs,
        max_docs_per_pack=args.max_docs_per_pack,
    )
    for lengths in packed_lengths:
        print(",".join(str(length) for length in lengths))


def _texts_from_args(args: argparse.Namespace) -> Iterable[str]:
    text_keys = _parse_text_keys(args.text_key)
    if args.jsonl is not None:
        return _texts_from_jsonl(args.jsonl, text_keys=text_keys)
    if args.text_file is not None:
        return _texts_from_text_file(args.text_file)
    if args.dataset is not None:
        return _texts_from_hf_dataset(
            args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            text_keys=text_keys,
            streaming=args.streaming,
        )
    raise ValueError("Specify one of --jsonl, --text-file, or --dataset.")


def _texts_from_jsonl(path: str, *, text_keys: tuple[str, ...]) -> Iterator[str]:
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = _text_from_row(row, text_keys=text_keys)
            if text:
                yield text


def _texts_from_text_file(path: str) -> Iterator[str]:
    with Path(path).open() as f:
        for line in f:
            text = line.rstrip("\n")
            if text:
                yield text


def _texts_from_hf_dataset(
    dataset_name: str,
    *,
    dataset_config: str | None,
    split: str,
    text_keys: tuple[str, ...],
    streaming: bool,
) -> Iterator[str]:
    dataset = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    for row in dataset:
        text = _text_from_row(row, text_keys=text_keys)
        if text:
            yield text


def _text_from_row(row: dict[str, Any], *, text_keys: tuple[str, ...]) -> str:
    values = [_text_value_from_row(row, text_key=text_key) for text_key in text_keys]
    return "\n".join(value for value in values if value)


def _text_value_from_row(row: dict[str, Any], *, text_key: str) -> str:
    value = row
    for key in text_key.split("."):
        value = value[key]
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "\n".join(str(part) for part in value)
    return str(value)


def _parse_text_keys(text_key: str) -> tuple[str, ...]:
    text_keys = tuple(key.strip() for key in text_key.split(",") if key.strip())
    if not text_keys:
        raise ValueError("text_key must contain at least one field name.")
    return text_keys


def _token_lengths(
    texts: Iterable[str],
    *,
    tokenizer_name_or_path: str,
    text_limit: int,
    add_special_tokens: bool,
) -> Iterator[int]:
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    for index, text in enumerate(texts):
        if index >= text_limit:
            return
        token_count = len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        if token_count > 0:
            yield token_count


def _pack_token_lengths(
    token_lengths: Iterable[int],
    *,
    seq_len: int,
    num_packs: int,
    max_docs_per_pack: int | None,
) -> list[tuple[int, ...]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if num_packs <= 0:
        raise ValueError("num_packs must be positive.")
    if max_docs_per_pack is not None and max_docs_per_pack <= 0:
        raise ValueError("max_docs_per_pack must be positive when specified.")

    lengths = np.array([length for length in token_lengths if length > 0], dtype=np.int64)
    doc_ranges = pack_documents(
        lengths,
        max_length=seq_len,
        max_segments_per_example=max_docs_per_pack,
        slice_strategy="right",
    )

    packs = [_doc_range_to_lengths(lengths, doc_range, seq_len=seq_len) for doc_range in doc_ranges[:num_packs]]
    if len(packs) < num_packs:
        raise ValueError(f"Only produced {len(packs)} packed sequences from the provided token lengths.")
    return packs


def _doc_range_to_lengths(lengths: np.ndarray, doc_range: range, *, seq_len: int) -> tuple[int, ...]:
    doc_lengths = [min(int(lengths[index]), seq_len) for index in doc_range]
    remaining = seq_len - sum(doc_lengths)
    if remaining > 0:
        doc_lengths.append(remaining)
    return tuple(doc_lengths)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--jsonl", type=str)
    input_group.add_argument("--text-file", type=str)
    input_group.add_argument("--dataset", type=str)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="Field name, nested path, or comma-separated field list to join into one document.",
    )
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-packs", type=int, default=1)
    parser.add_argument("--max-docs-per-pack", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--add-special-tokens", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
