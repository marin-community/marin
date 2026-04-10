# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import chain
from typing import Any, Sequence

import numpy as np
import regex

from levanter.data import BatchProcessor
from levanter.tokenizers import MarinTokenizer
from levanter.utils.py_utils import logical_cpu_core_count

LONG_STRING_WORKAROUND = 10_000
ws = regex.compile(r"\s")


class BatchTokenizer(BatchProcessor[dict, dict]):
    """
    Batch processor that tokenizes raw text, optionally adding BOS/EOS tokens.
    """

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        text_field: str = "text",
        enforce_bos: bool = True,
        enforce_eos: bool = True,
        *,
        override_resources=None,
        _workaround_len: int = LONG_STRING_WORKAROUND,
        long_string_workaround: bool = False,
        return_attention_mask: bool = False,
        padding=False,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.override_resources = override_resources
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        self.max_length = max_length
        self._long_string_workaround = long_string_workaround

        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        if enforce_eos or enforce_bos:
            input_ids = tokenizer.encode("hi there", add_special_tokens=True)
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
            bos = self.tokenizer.bos_token
            assert bos is not None
            batch_text = [bos + " " + d for d in batch_text]
        if self._need_to_add_eos:
            eos = self.tokenizer.eos_token
            assert eos is not None
            batch_text = [d + " " + eos for d in batch_text]

        if self._long_string_workaround:
            batch_text, needs_merge = self._break_for_long_sequences(batch_text)
        else:
            needs_merge = []

        encoded = self.tokenizer.encode_batch(batch_text)

        # Build a dict-of-lists structure analogous to the old BatchEncoding.
        encoding: dict[str, list] = {"input_ids": encoded}

        if self.return_attention_mask:
            encoding["attention_mask"] = [[1] * len(ids) for ids in encoded]

        if self.padding is not False and self.max_length is not None:
            encoding = _apply_padding_and_truncation(
                encoding, self.max_length, self.padding, pad_token_id=self.tokenizer.pad_token_id or 0
            )

        if needs_merge:
            encoding = self._merge_split_encodings(batch_text, encoding, needs_merge)

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
            "vocab_size": self.tokenizer.vocab_size,
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        ids = self.tokenizer.encode("hi there")
        result: dict[str, Any] = {"input_ids": ids}
        if self.return_attention_mask:
            result["attention_mask"] = [1] * len(ids)
        return result

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return min(max(1, logical_cpu_core_count() - 4), 12)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            v_out = []
            vs_to_merge: list[list[int]] = []
            for i in range(len(batch)):
                if not needs_merge[i]:
                    if len(vs_to_merge) > 0:
                        v_out.append(list(chain(*vs_to_merge)))
                    vs_to_merge = []
                vs_to_merge.append(v[i])
            if len(vs_to_merge) > 0:
                v_out.append(list(chain(*vs_to_merge)))
            new_encoding[k] = v_out
        return new_encoding


def _apply_padding_and_truncation(
    encoding: dict[str, list[list[int]]], max_length: int, padding, pad_token_id: int = 0
) -> dict[str, list[list[int]]]:
    """Truncate sequences to max_length and optionally pad to uniform length."""
    for k in encoding:
        encoding[k] = [seq[:max_length] for seq in encoding[k]]

    if padding is False:
        return encoding

    if padding == "max_length":
        target_len = max_length
    else:
        # padding=True means pad to the longest in the batch
        target_len = max(len(seq) for seq in encoding["input_ids"]) if encoding["input_ids"] else 0

    for k in encoding:
        pad_value = pad_token_id if k == "input_ids" else 0
        encoding[k] = [seq + [pad_value] * (target_len - len(seq)) for seq in encoding[k]]

    return encoding


class DNABatchTokenizer(BatchProcessor[dict, dict]):
    """
    A batch processor that tokenizes DNA sequences with case-based loss weighting.

    Weights are **target-aligned**: ``loss_weight[i]`` reflects the case of the
    *next* token (``input_ids[i+1]``), which is the prediction target at position
    ``i`` in causal LM training.

    Character case determines the weight:
    - Uppercase target (ACGT): weight = uppercase_weight
    - Lowercase target (acgt): weight = lowercase_weight

    If the tokenizer defines BOS/EOS token IDs, they are automatically prepended/appended
    to the token sequences. Use ``num_special_tokens`` to query how many extra tokens
    are added (useful for computing model context size).

    Assumptions:
    - Character-level tokenizer (1:1 character-to-token mapping)
    - All sequences have the same length (no padding/truncation)
    - Model context size matches sequence length + special tokens (see experiment configs).
    """

    def __init__(
        self,
        tokenizer: MarinTokenizer,
        text_field: str = "seq",
        uppercase_weight: float = 1.0,
        lowercase_weight: float = 1.0,
        *,
        override_resources=None,
    ):
        self.tokenizer = tokenizer
        self._hf_tokenizer = tokenizer.as_hf_tokenizer()
        self.text_field = text_field
        self.override_resources = override_resources
        self.uppercase_weight = uppercase_weight
        self.lowercase_weight = lowercase_weight
        self._has_bos = tokenizer.bos_token_id is not None
        self._has_eos = tokenizer.eos_token_id is not None

    @property
    def num_special_tokens(self) -> int:
        return int(self._has_bos) + int(self._has_eos)

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        texts = [example[self.text_field] for example in batch]

        assert len(set(len(t) for t in texts)) == 1, "All sequences must have the same length"

        encodings = self._hf_tokenizer(
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
        char_weights = np.where(is_upper, self.uppercase_weight, self.lowercase_weight).astype(np.float32)

        input_ids = encodings["input_ids"].astype(np.int32)

        assert input_ids.shape == char_weights.shape, (
            f"Token count ({input_ids.shape[1]}) != char count ({char_weights.shape[1]}). "
            "Tokenizer must be character-level."
        )

        batch_size = input_ids.shape[0]

        # Align weights with targets: loss_weight[i] controls the loss for predicting
        # input_ids[i+1], so it should reflect the case of the *next* character.
        # Shift character weights left by 1; the last position predicts EOS (weight 1.0)
        # or is masked by not_last_mask in the loss function if there is no EOS.
        loss_weights = np.roll(char_weights, -1, axis=1)
        loss_weights[:, -1] = 1.0 if self._has_eos else 0.0

        if self._has_bos:
            bos_ids = np.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=np.int32)
            # BOS position predicts the first character — use that character's weight
            bos_weights = char_weights[:, :1]
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
            "vocab_size": self.tokenizer.vocab_size,
            "uppercase_weight": self.uppercase_weight,
            "lowercase_weight": self.lowercase_weight,
            "has_bos": self._has_bos,
            "has_eos": self._has_eos,
        }

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return min(max(1, logical_cpu_core_count() - 4), 12)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0
