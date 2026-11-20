# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import token_file_examples

EOS = 9


def test_token_file_windows_are_consecutive_and_split_documents_at_eos():
    tokens = np.array([1, 2, EOS, 3, 4, 5, EOS, 6, 7, 8, 1, 2], dtype=np.int32)

    first, second = token_file_examples(tokens, count=2, seq_len=6, eos_id=EOS)

    np.testing.assert_array_equal(np.concatenate([first.tokens, second.tokens]), tokens)
    # A new document starts after each EOS; the window boundary starts the count again.
    np.testing.assert_array_equal(first.attn_mask.segment_ids[0], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(second.attn_mask.segment_ids[0], [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(first.loss_weight, [1, 1, 1, 1, 1, 0])
