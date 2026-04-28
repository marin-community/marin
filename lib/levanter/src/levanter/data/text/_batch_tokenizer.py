# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Any

import regex
from rigging.timing import log_time

from levanter.data import BatchProcessor
from levanter.tokenizers import MarinTokenizer
from levanter.utils.py_utils import logical_cpu_core_count

LONG_STRING_WORKAROUND = 10_000
ws = regex.compile(r"\s")

# When the long-string workaround triggers, encode each over-long text in
# sub-batches of this many pieces. Caps in-flight memory at one sub-batch
# of input strings + their tokenized output, instead of holding all pieces
# from all records simultaneously.
_LONG_STRING_BATCH_SIZE = 256


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
        bos_id = self.tokenizer.bos_token_id if self._append_bos else None
        eos_str = self.tokenizer.eos_token if self._append_eos else None
        if self._append_bos:
            assert bos_id is not None
        if self._append_eos:
            assert eos_str is not None

        # Encode per-record so an outlier's pieces never coexist with the
        # rest of the batch's encodings in memory. Short records take the
        # one-shot path; long records stream through ``_encode_long_string``,
        # which sub-batches splits and accumulates ids in-place.
        encoded: list[list[int]] = []
        for example in batch:
            text = example[self.text_field]
            if eos_str is not None:
                text = text + " " + eos_str

            if self._long_string_workaround and len(text) > self._workaround_len:
                ids = self._encode_long_string(text)
            else:
                ids = self.tokenizer.encode(text, add_special_tokens=False)

            if bos_id is not None:
                # In-place prepend: O(n) shift but no extra full-list allocation,
                # unlike ``[bos_id, *ids]`` which doubles peak for huge ids.
                ids.insert(0, bos_id)
            encoded.append(ids)

        encoding: dict[str, list] = {"input_ids": encoded}

        if self.return_attention_mask:
            encoding["attention_mask"] = [[1] * len(ids) for ids in encoded]

        if self.padding is not False and self.max_length is not None:
            encoding = _apply_padding_and_truncation(
                encoding, self.max_length, self.padding, pad_token_id=self.tokenizer.pad_token_id or 0
            )

        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]
        return unbatched

    def _encode_long_string(self, text: str) -> list[int]:
        """Encode one over-long text by splitting at safe whitespace boundaries
        and concatenating ids in-place.

        Splits are buffered in groups of ``_LONG_STRING_BATCH_SIZE`` pieces;
        each group is passed through ``encode_batch`` and the resulting ids
        are extended into the running ``ids`` list before the next group is
        produced. Peak in-flight memory is one sub-batch's input strings +
        tokens, regardless of how long the original text is.
        """
        ids: list[int] = []
        with log_time(f"BatchTokenizer encoded {len(text):,}-char outlier record"):
            pieces: list[str] = []
            remaining = text
            while True:
                if len(remaining) > self._workaround_len:
                    match = ws.search(remaining, self._workaround_len)
                    split = match.start() if match is not None else len(remaining)
                    pieces.append(remaining[:split])
                    remaining = remaining[split:]
                else:
                    pieces.append(remaining)
                    remaining = ""

                if len(pieces) >= _LONG_STRING_BATCH_SIZE or not remaining:
                    for encoded_piece in self.tokenizer.encode_batch(pieces, add_special_tokens=False):
                        ids.extend(encoded_piece)
                    pieces.clear()

                if not remaining:
                    break

        return ids

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
