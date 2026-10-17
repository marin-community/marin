# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.mixed_transfer.prepare_domain import training_tokens


def test_native_completion_ids_and_shifted_loss_survive_padding():
    # Keep the returned completion verbatim, even when generation hit the length cap
    # and supplied no EOS. MMLU-Pro preparation historically used pad ID 128489.
    row = training_tokens([10, 11], [20, 21], context=6)
    assert row == {
        "input_ids": [10, 11, 20, 21, 128489, 128489],
        "loss_weights": [0, 1, 1, 0, 0, 0],
    }
    assert training_tokens([10, 11], [20, 128009], context=4) == {
        "input_ids": [10, 11, 20, 128009],
        "loss_weights": [0, 1, 1, 0],
    }


def test_overlength_demo_is_rejected_without_silent_truncation():
    with pytest.raises(ValueError, match="truncate"):
        training_tokens([10, 11], [20, 21], context=3)
