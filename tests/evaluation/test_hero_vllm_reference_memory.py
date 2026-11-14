# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np

from experiments.evaluation.hero_vllm_hillclimb.reference_hero import max_peak_gib


def test_peak_above_two_gib_survives_device_put():
    # process_allgather device_puts its payload; int64 bytes wrapped to negative int32 there.
    stats = [{"peak_bytes_in_use": 150_000_000_000}, {"peak_bytes_in_use": 3 * 2**30}]

    gathered = np.asarray(jax.device_put(max_peak_gib(stats)))

    np.testing.assert_allclose(gathered, 150_000_000_000 / 2**30, rtol=1e-6)
