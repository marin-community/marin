# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import numpy as np
from levanter.eval_harness import _eval_pad_token_id, _pack_requests


@dataclasses.dataclass(frozen=True)
class FrozenTokenizerWithoutPad:
    eos_token_id: int = 99
    pad_token_id: int | None = None

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [[ord(char) for char in text] for text in texts]


@dataclasses.dataclass(frozen=True)
class Request:
    args: tuple[str, str]


def test_pack_requests_uses_eos_for_frozen_tokenizer_without_pad():
    tokenizer = FrozenTokenizerWithoutPad()

    pad_token_id = _eval_pad_token_id(tokenizer)
    packed = _pack_requests(
        [Request(args=("a", "b"))],
        tokenizer,
        hax.Axis("position", 4),
        max_pack_size=64,
        pad_token_id=pad_token_id,
    )

    tokens = np.asarray(jax.device_get(packed[0].tokens.array))
    np.testing.assert_array_equal(tokens, np.array([ord("a"), ord("b"), tokenizer.eos_token_id, tokenizer.eos_token_id]))
    assert tokenizer.pad_token_id is None
