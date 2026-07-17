# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from fray.types import tpu_hbm_bytes_per_chip, tpu_hbm_capacity_bytes

BYTES_PER_GIB = 1024**3


def test_tpu_hbm_capacity_bytes_multiplies_by_slice_chip_count():
    assert tpu_hbm_capacity_bytes("v5litepod-32") == 512 * BYTES_PER_GIB


def test_tpu_hbm_bytes_per_chip_rejects_unknown_family():
    with pytest.raises(ValueError):
        tpu_hbm_bytes_per_chip("v7")
