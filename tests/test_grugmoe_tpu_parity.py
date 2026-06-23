# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.tpu_ci


def test_grugmoe_component_parity_matches_levanter_moe():
    from experiments.grug.moe import vllm_tpu_parity  # noqa: PLC0415

    tpu_grugmoe = vllm_tpu_parity.load_tpu_grugmoe()

    vllm_tpu_parity.check_moe_component(tpu_grugmoe)
