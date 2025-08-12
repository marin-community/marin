import tempfile

import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter

from levanter.tracker.histogram import Histogram
from levanter.tracker.tensorboard import TensorboardTracker


def test_log_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        with SummaryWriter(logdir=tmpdir) as writer:
            tracker = TensorboardTracker(writer)
            tracker.log_summary({"float": 2.0})
            tracker.log_summary({"str": "test"})
            tracker.log_summary({"scalar_jax_array": jnp.array(3.0)})
            tracker.log_summary({"scalar_np_array": np.array(3.0)})


def test_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        with SummaryWriter(logdir=tmpdir) as writer:
            tracker = TensorboardTracker(writer)
            tracker.log({"float": 2.0}, step=0)
            tracker.log({"str": "test"}, step=0)
            tracker.log({"scalar_jax_array": jnp.array(3.0)}, step=0)
            tracker.log({"scalar_np_array": np.array(3.0)}, step=0)
            tracker.log({"histogram": Histogram.from_array(jnp.array([1.0, 2.0, 3.0]))}, step=0)
