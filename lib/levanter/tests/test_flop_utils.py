# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.utils.flop_utils import lm_flops_per_token


def test_only_global_layers_pay_for_context_beyond_the_window():
    """With num_global_layers set, growing the context past the window adds attention cost only on global layers."""
    window = 64

    def growth(num_global_layers: int) -> float:
        def flops(seq_len: int) -> float:
            return lm_flops_per_token(
                hidden_dim=64,
                intermediate_dim=128,
                num_layers=8,
                num_kv_heads=2,
                num_heads=4,
                head_dim=16,
                seq_len=seq_len,
                vocab_size=1000,
                glu=True,
                sliding_window=window,
                num_global_layers=num_global_layers,
            )

        return flops(2 * window) - flops(window)

    assert growth(0) == pytest.approx(0)
    assert growth(1) > 0
    assert growth(3) == pytest.approx(3 * growth(1))
