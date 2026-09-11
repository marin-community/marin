# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from levanter.tracker.histogram import SummaryStats
from levanter.tracker.tensorboard import TensorboardTracker


class FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.text = []
        self.histograms = []

    def add_scalar(self, key, value, global_step=None):
        self.scalars.append((key, value, global_step))

    def add_text(self, key, value, global_step=None):
        self.text.append((key, value, global_step))

    def add_histogram(self, key, value, global_step=None):
        self.histograms.append((key, value, global_step))


def test_log_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        with SummaryWriter(logdir=tmpdir) as writer:
            tracker = TensorboardTracker(writer)
            tracker.log_summary({"float": 2.0})
            tracker.log_summary({"str": "test"})
            tracker.log_summary({"scalar_jax_array": jnp.array(3.0)})
            tracker.log_summary({"scalar_np_array": np.array(3.0)})


def test_log_summary_flattens_nested_values():
    writer = FakeSummaryWriter()
    tracker = TensorboardTracker(writer)

    tracker.log_summary({"hardware_topology": {"tpu_topology_shape": "4x8x8", "devices": [{"platform": "tpu"}]}})

    assert ("hardware_topology/tpu_topology_shape", "4x8x8", None) in writer.text
    assert any(key == "hardware_topology/devices" for key, _, _ in writer.text)


def test_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        with SummaryWriter(logdir=tmpdir) as writer:
            tracker = TensorboardTracker(writer)
            tracker.log({"float": 2.0}, step=0)
            tracker.log({"str": "test"}, step=0)
            tracker.log({"scalar_jax_array": jnp.array(3.0)}, step=0)
            tracker.log({"scalar_np_array": np.array(3.0)}, step=0)
            tracker.log({"histogram": SummaryStats.from_array(jnp.array([1.0, 2.0, 3.0]))}, step=0)
            tracker.log(
                {"summary_only": SummaryStats.from_array(jnp.array([1.0, 2.0, 3.0]), include_histogram=False)}, step=0
            )
