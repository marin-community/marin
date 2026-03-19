from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

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


@dataclass(frozen=True)
class LoadedData:
    dataset: Dataset[torch.Tensor]
    tokenizer: object | None = None


def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def build_dataset(data_config: DataConfig, model_config: ModelConfig) -> LoadedData:
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

    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id or 0

    texts: list[str]
    if data_config.dataset == "text":
        if data_config.text_file is None:
            raise ValueError("text dataset requires --text-file")
        texts = [_load_text_file(data_config.text_file)]
    elif data_config.dataset == "hf":
        if data_config.hf_dataset is None:
            raise ValueError("hf dataset requires --hf-dataset")
        from datasets import load_dataset

        ds = load_dataset(data_config.hf_dataset, data_config.hf_dataset_config, split=data_config.hf_split)
        limit = data_config.max_examples or len(ds)
        texts = [ds[i][data_config.hf_text_field] for i in range(min(limit, len(ds)))]
    else:
        raise ValueError(f"unsupported dataset kind {data_config.dataset!r}")

    token_ids: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(encoded)
        token_ids.append(eos_id)

    tokens = torch.tensor(token_ids, dtype=torch.long)
    return LoadedData(dataset=PackedTokenDataset(tokens, seq_len), tokenizer=tokenizer)
