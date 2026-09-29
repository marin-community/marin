# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset

from experiments.grug_sft.special_token_train import verify_data_epochs


def test_epoch_limit_counts_only_examples_in_final_partial_block():
    mixture = MixtureDataset(
        datasets={"first": ListAsyncDataset(list(range(7))), "second": ListAsyncDataset(list(range(5)))},
        weights={"first": 0.5, "second": 0.5},
        block_size=10,
        key=0,
        randomize_blocks=False,
    )

    verify_data_epochs(mixture, run_sequences=12, max_data_epochs=1)
    with pytest.raises(ValueError):
        shorter = MixtureDataset(
            datasets={"first": ListAsyncDataset(list(range(6))), "second": ListAsyncDataset(list(range(5)))},
            weights={"first": 0.5, "second": 0.5},
            block_size=10,
            key=0,
            randomize_blocks=False,
        )
        verify_data_epochs(shorter, run_sequences=12, max_data_epochs=1)
