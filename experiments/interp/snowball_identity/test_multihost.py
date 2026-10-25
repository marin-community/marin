# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The multi-host helpers never device_get or gather a global array untiled, and no such call is left in the files
that run on every rank of the LR0 probe."""

import pathlib
import re

import jax
import numpy as np
import pytest

from experiments.interp.snowball_identity import multihost

ROOT = pathlib.Path(__file__).resolve().parents[3]
RANK_FILES = [
    "experiments/interp/snowball_identity/precision_gate.py",
    "experiments/interp/snowball_identity/levanter_positions.py",
    "experiments/post_training/identity_context_distill/precision_probe.py",
]


class GlobalArray:
    """Stands in for a jax.Array that this process does not fully address."""

    is_fully_addressable = False


def test_gather_global_uses_a_tiled_gather_on_a_global_array(monkeypatch):
    calls = []
    monkeypatch.setattr(jax, "device_get", lambda *a: pytest.fail("device_get"))
    monkeypatch.setattr(
        multihost.multihost_utils, "process_allgather", lambda x, tiled=False: calls.append(tiled) or np.arange(3)
    )
    x = GlobalArray()
    np.testing.assert_array_equal(multihost.gather_global(x), np.arange(3))
    assert calls == [True]


def test_host_helpers_refuse_jax_arrays_and_pass_numpy(monkeypatch):
    seen = []
    monkeypatch.setattr(
        multihost.multihost_utils, "process_allgather", lambda x, tiled=False: seen.append((type(x), tiled)) or x
    )
    monkeypatch.setattr(multihost.multihost_utils, "broadcast_one_to_all", lambda x: seen.append((type(x), None)) or x)
    assert multihost.gather_host_scalar(np.int32(1)).tolist() == [1]
    assert int(multihost.broadcast_host_scalar(np.int32(3))) == 3
    assert seen == [(np.ndarray, True), (np.ndarray, None)]
    with pytest.raises(TypeError):
        multihost.gather_host_scalar(jax.numpy.int32(1))
    with pytest.raises(TypeError):
        multihost.broadcast_host_scalar(jax.numpy.int32(1))


@pytest.mark.parametrize("path", RANK_FILES)
def test_no_device_get_or_untiled_gather_left(path):
    source = (ROOT / path).read_text()
    assert "device_get" not in source
    assert "broadcast_one_to_all" not in source
    for call in re.findall(r"process_allgather\((.*)\)", source):
        assert "tiled=True" in call, call
