# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.datasets.eval_policy_answer_bpb import (
    INPUT_IDS_KEY,
    LOSS_WEIGHTS_KEY,
    AnswerExample,
    encode_example,
    pack_examples,
)

BOS, EOS = 0, 1


class ByteTokenizer:
    """One token per byte, offset past the special ids."""

    bos_token_id = BOS
    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [b + 2 for b in text.encode("utf-8")]


def test_weights_score_exactly_the_answer_tokens():
    tokenizer = ByteTokenizer()
    tokens, weights = encode_example(tokenizer, AnswerExample("Q: 2+2?\nA:", " 4"))
    # weights[i] scores the prediction of tokens[i + 1]
    scored = [tokens[i + 1] for i, w in enumerate(weights) if w > 0]
    assert scored == tokenizer.encode(" 4")
    assert tokens[0] == BOS and tokens[-1] == EOS


def test_packing_keeps_examples_whole_and_drops_long_ones():
    tokenizer = ByteTokenizer()
    examples = [AnswerExample("a" * 5, "b" * 3), AnswerExample("c" * 4, "d" * 2), AnswerExample("e" * 30, "f")]
    encoded = [encode_example(tokenizer, e) for e in examples]
    rows = list(pack_examples(encoded, seq_len=16, pad_id=EOS))

    # 10 + 8 tokens do not fit in one 16-token row; the 33-token example is dropped.
    assert len(rows) == 2
    assert all(len(row[INPUT_IDS_KEY]) == 16 for row in rows)
    np.testing.assert_array_equal(rows[0][INPUT_IDS_KEY][:10], encoded[0][0])
    np.testing.assert_array_equal(rows[1][INPUT_IDS_KEY][:8], encoded[1][0])
    assert rows[0][LOSS_WEIGHTS_KEY].sum() == 3 and rows[1][LOSS_WEIGHTS_KEY].sum() == 2
