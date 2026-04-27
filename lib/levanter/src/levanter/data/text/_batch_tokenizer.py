# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import chain
from typing import Sequence, Any

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

        self._append_bos = enforce_bos and tokenizer.bos_token_id is not None
        self._append_eos = enforce_eos and tokenizer.eos_token_id is not None
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        batch_text = [example[self.text_field] for example in batch]

        if self._append_eos:
            eos = self.tokenizer.eos_token
            assert eos is not None
            batch_text = [d + " " + eos for d in batch_text]

        if self._long_string_workaround:
            batch_text, needs_merge = self._break_for_long_sequences(batch_text)
        else:
            needs_merge = []

        encoded = self.tokenizer.encode_batch(batch_text, add_special_tokens=False)

        if self._append_bos:
            bos_id = self.tokenizer.bos_token_id
            assert bos_id is not None
            if needs_merge:
                # Prepend BOS only to first chunks so the merged doc has a single BOS.
                encoded = [[bos_id, *enc] if not merge else enc for enc, merge in zip(encoded, needs_merge)]
            else:
                encoded = [[bos_id, *enc] for enc in encoded]

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
            "append_bos": self._append_bos,
            "append_eos": self._append_eos,
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
