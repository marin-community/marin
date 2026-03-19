# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import DataConfig, ModelConfig


class PackedTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1D")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        usable = (tokens.numel() // seq_len) * seq_len
        if usable < seq_len:
            raise ValueError(f"need at least {seq_len} tokens, got {tokens.numel()}")
        self.tokens = tokens[:usable].view(-1, seq_len)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


class ToyTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, *, size: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab_size, (size, seq_len), generator=generator)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


class StreamingTokenBlockDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        *,
        text_source: Callable[[], Iterable[str]],
        tokenizer: object,
        seq_len: int,
        eos_token_id: int,
        rank: int,
        world_size: int,
        max_examples: int | None,
    ):
        super().__init__()
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.rank = rank
        self.world_size = world_size
        self.max_examples = max_examples

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shard_count = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        buffer: list[int] = []
        seen = 0
        for example_index, text in enumerate(self.text_source()):
            if self.max_examples is not None and seen >= self.max_examples:
                break
            if example_index % shard_count != shard_index:
                continue
            if not isinstance(text, str) or not text:
                continue
            encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not encoded:
                continue
            buffer.extend(encoded)
            buffer.append(self.eos_token_id)
            seen += 1
            while len(buffer) >= self.seq_len:
                sample = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                del buffer[: self.seq_len]
                yield sample


@dataclass(frozen=True)
class LoadedData:
    dataset: Dataset[torch.Tensor] | IterableDataset[torch.Tensor]
    tokenizer: object | None = None
    is_streaming: bool = False


def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _resolve_tokenizer_name(data_config: DataConfig) -> str:
    if data_config.tokenizer != "auto":
        return data_config.tokenizer
    if data_config.dataset in {"fineweb", "fineweb_edu"}:
        return "gpt2"
    return "gpt2"


def _resolve_hf_dataset(data_config: DataConfig) -> tuple[str, str | None, str]:
    if data_config.dataset == "fineweb":
        return "HuggingFaceFW/fineweb", data_config.hf_dataset_config or "sample-10BT", "text"
    if data_config.dataset == "fineweb_edu":
        return "HuggingFaceFW/fineweb-edu", data_config.hf_dataset_config or "sample-10BT", "text"
    if data_config.dataset == "hf":
        if data_config.hf_dataset is None:
            raise ValueError("hf dataset requires --hf-dataset")
        return data_config.hf_dataset, data_config.hf_dataset_config, data_config.hf_text_field
    raise ValueError(f"dataset {data_config.dataset!r} is not a Hugging Face dataset alias")


def _build_streaming_dataset(
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    tokenizer: object,
    rank: int,
    world_size: int,
) -> LoadedData:
    from datasets import load_dataset

    dataset_name, dataset_config, text_field = _resolve_hf_dataset(data_config)
    ds = load_dataset(dataset_name, dataset_config, split=data_config.hf_split, streaming=True)
    if data_config.shuffle_buffer > 0:
        ds = ds.shuffle(seed=data_config.seed, buffer_size=data_config.shuffle_buffer)
    eos_token_id = tokenizer.eos_token_id or 0

    def text_source() -> Iterable[str]:
        for row in ds:
            yield row[text_field]

    dataset = StreamingTokenBlockDataset(
        text_source=text_source,
        tokenizer=tokenizer,
        seq_len=model_config.max_seq_len,
        eos_token_id=eos_token_id,
        rank=rank,
        world_size=world_size,
        max_examples=data_config.max_examples,
    )
    return LoadedData(dataset=dataset, tokenizer=tokenizer, is_streaming=True)


def build_dataset(
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> LoadedData:
    seq_len = model_config.max_seq_len
    if data_config.dataset == "toy":
        return LoadedData(
            dataset=ToyTokenDataset(
                size=data_config.toy_size,
                seq_len=seq_len,
                vocab_size=model_config.vocab_size,
                seed=data_config.seed,
            )
        )

    from transformers import AutoTokenizer

    tokenizer_name = _resolve_tokenizer_name(data_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"tokenizer {tokenizer_name!r} must expose an eos token or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    if data_config.dataset in {"hf", "fineweb", "fineweb_edu"} and data_config.streaming:
        return _build_streaming_dataset(
            data_config=data_config,
            model_config=model_config,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
        )

    texts: list[str]
    if data_config.dataset == "text":
        if data_config.text_file is None:
            raise ValueError("text dataset requires --text-file")
        texts = [_load_text_file(data_config.text_file)]
    elif data_config.dataset in {"hf", "fineweb", "fineweb_edu"}:
        from datasets import load_dataset

        dataset_name, dataset_config, text_field = _resolve_hf_dataset(data_config)
        ds = load_dataset(dataset_name, dataset_config, split=data_config.hf_split)
        limit = data_config.max_examples or len(ds)
        texts = [ds[i][text_field] for i in range(min(limit, len(ds)))]
    else:
        raise ValueError(f"unsupported dataset kind {data_config.dataset!r}")

    token_ids: list[int] = []
    eos_id = tokenizer.eos_token_id or 0
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(encoded)
        token_ids.append(eos_id)

    tokens = torch.tensor(token_ids, dtype=torch.long)
    return LoadedData(dataset=PackedTokenDataset(tokens, seq_len), tokenizer=tokenizer, is_streaming=False)
