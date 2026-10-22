# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.cruxeval_transfer.prepare_plain import training_tokens


@pytest.mark.parametrize("completion", [[71, 128003, 72, 128001], [71, 72]])
def test_generated_tokens_alone_receive_loss_without_adding_eos(completion):
    row = training_tokens([128000, 31, 32], completion, 12)
    targets = row["input_ids"][1:]
    supervised = [token for token, weight in zip(targets, row["loss_weights"][:-1], strict=True) if weight]
    assert supervised == completion
    assert row["input_ids"][: 3 + len(completion)] == [128000, 31, 32, *completion]
    assert len(row["input_ids"]) == len(row["loss_weights"]) == 12
