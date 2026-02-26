# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
from functools import cached_property
from itertools import chain
from typing import Any, Sequence

import numpy as np
import regex
from tokenizers import normalizers
from transformers import PreTrainedTokenizerBase, BatchEncoding, PreTrainedTokenizerFast

from levanter.data import BatchProcessor
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer

LONG_STRING_WORKAROUND = 10_000
ws = regex.compile(r"\s")


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BaseBatchTokenizer(BatchProcessor[dict, dict]):
    """Base class for tokenizer-based batch processors."""

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "text",
        *,
        override_resources=None,
    ):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.override_resources = override_resources

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0


class BatchTokenizer(BaseBatchTokenizer):
    """
    Batch processor that tokenizes raw text, optionally adding BOS/EOS tokens.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "text",
        enforce_bos: bool = True,
        enforce_eos: bool = True,
        *,
        override_resources=None,
        _workaround_len: int = LONG_STRING_WORKAROUND,
        return_attention_mask: bool = False,
        padding=False,
        max_length=None,
    ):
        super().__init__(tokenizer, text_field, override_resources=override_resources)
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length

        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        if enforce_eos or enforce_bos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id and enforce_eos
            should_append_bos = input_ids[0] != tokenizer.bos_token_id and enforce_bos
        else:
            should_append_eos = False
            should_append_bos = False

        self._need_to_add_eos = should_append_eos
        self._need_to_add_bos = should_append_bos
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        batch_text = [example[self.text_field] for example in batch]

        if self._need_to_add_bos:
            batch_text = [self.tokenizer.bos_token + " " + d for d in batch_text]
        if self._need_to_add_eos:
            batch_text = [d + " " + self.tokenizer.eos_token for d in batch_text]

        if self._needs_long_sequence_workaround:
            batch_text, needs_merge = self._break_for_long_sequences(batch_text)
        else:
            needs_merge = []

        if self.padding is not False:
            encoding = self.tokenizer(
                batch_text,
                return_attention_mask=self.return_attention_mask,
                verbose=False,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
            )  # type: ignore
        else:
            encoding = self.tokenizer(batch_text, return_attention_mask=self.return_attention_mask, verbose=False)  # type: ignore

        if needs_merge:
            new_encoding = self._merge_split_encodings(batch_text, encoding, needs_merge)
            encoding = BatchEncoding(new_encoding)

        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]
        return unbatched

    def _break_for_long_sequences(self, batch: Sequence[str]):
        orig_lengths = [len(d) for d in batch]
        orig_batch = batch
        batch_out: list[str] = []
        needs_merge: list[bool] = []
        for i, d in enumerate(orig_batch):
            needs_merge.append(False)
            orig_len = orig_lengths[i]
            while len(d) > self._workaround_len:
                match = ws.search(d, self._workaround_len)
                split = match.start() if match is not None else len(d)
                batch_out.append(d[:split])
                needs_merge.append(True)
                d = d[split:]
                orig_len -= split
            batch_out.append(d)
        return batch_out, needs_merge

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        return dict(**self.tokenizer("hi there", return_attention_mask=self.return_attention_mask, verbose=False))

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @cached_property
    def _needs_long_sequence_workaround(self):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            if normalizer is None:
                return False
            return isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))
        return False

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], np.ndarray):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(np.concatenate(vs_to_merge))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])
                if len(vs_to_merge) > 0:
                    v_out.append(np.concatenate(vs_to_merge))
                new_encoding[k] = v_out
            elif isinstance(v[0], list):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(list(chain(*vs_to_merge)))  # type: ignore[name-defined]
                        vs_to_merge = []
                    vs_to_merge.append(v[i])
                if len(vs_to_merge) > 0:
                    v_out.append(list(chain(*vs_to_merge)))  # type: ignore[name-defined]
                new_encoding[k] = v_out
            else:
                raise ValueError(f"Unknown type {type(v[0])}")
        return new_encoding


class DNABatchTokenizer(BaseBatchTokenizer):
    """
    A batch processor that tokenizes DNA sequences with soft-masking support.

    Assigns loss weights based on character case:
    - Uppercase (ACGT): weight = 1.0
    - Lowercase (acgt): weight = soft_mask_weight

    If the tokenizer defines BOS/EOS token IDs, they are automatically prepended/appended
    to the token sequences. Their loss weight is set to 1.0. Use ``num_special_tokens``
    to query how many extra tokens are added (useful for computing model context size).

    Assumptions:
    - Character-level tokenizer (1:1 character-to-token mapping)
    - All sequences have the same length (no padding/truncation)
    - Model context size matches sequence length + special tokens (see experiment configs).
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "seq",
        soft_mask_weight: float = 1.0,
        *,
        override_resources=None,
    ):
        super().__init__(tokenizer, text_field, override_resources=override_resources)
        self.soft_mask_weight = soft_mask_weight
        self._has_bos = tokenizer.bos_token_id is not None
        self._has_eos = tokenizer.eos_token_id is not None

    @property
    def num_special_tokens(self) -> int:
        return int(self._has_bos) + int(self._has_eos)

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        texts = [example[self.text_field] for example in batch]

        assert len(set(len(t) for t in texts)) == 1, "All sequences must have the same length"

        encodings = self.tokenizer(
            texts,
            # important so input ids are aligned with loss weights
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_special_tokens_mask=False,
            return_tensors="np",
            verbose=False,
        )

        char_arrays = np.array([list(t) for t in texts], dtype="U1")
        is_upper = np.char.isupper(char_arrays)
        loss_weights = np.where(is_upper, 1.0, self.soft_mask_weight).astype(np.float32)

        input_ids = encodings["input_ids"].astype(np.int32)

        assert input_ids.shape == loss_weights.shape, (
            f"Token count ({input_ids.shape[1]}) != char count ({loss_weights.shape[1]}). "
            "Tokenizer must be character-level."
        )

        batch_size = input_ids.shape[0]

        if self._has_bos:
            bos_ids = np.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=np.int32)
            bos_weights = np.ones((batch_size, 1), dtype=np.float32)
            input_ids = np.concatenate([bos_ids, input_ids], axis=1)
            loss_weights = np.concatenate([bos_weights, loss_weights], axis=1)

        if self._has_eos:
            eos_ids = np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)
            eos_weights = np.ones((batch_size, 1), dtype=np.float32)
            input_ids = np.concatenate([input_ids, eos_ids], axis=1)
            loss_weights = np.concatenate([loss_weights, eos_weights], axis=1)

        return [{"input_ids": ids, "loss_weight": weights} for ids, weights in zip(input_ids, loss_weights)]

    @property
    def output_exemplar(self) -> dict:
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "loss_weight": np.zeros((0,), dtype=np.float32),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "soft_mask_weight": self.soft_mask_weight,
            "has_bos": self._has_bos,
            "has_eos": self._has_eos,
        }
