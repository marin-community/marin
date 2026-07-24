# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest
from prometheus_client import REGISTRY, generate_latest

from rigging import telltale

from levanter.tracker.histogram import SummaryStats
from levanter.tracker.telltale import TelltaleConfig, TelltaleTracker


def _series(text: str, name: str) -> dict[str, float]:
    """Parse the exposition text into {sample line: value} for samples of one family."""
    return {
        line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if not line.startswith("#") and line.startswith(name)
    }


@pytest.fixture
def exposition():
    tracker = TelltaleTracker()
    return tracker, lambda: generate_latest(REGISTRY).decode()


def test_log_publishes_jax_scalars(exposition):
    """Metrics arrive as 0-d arrays, which are not ``numbers.Real``.

    Testing for that ABC drops every primary training metric, so guard the
    coercion the tracker actually depends on.
    """
    tracker, render = exposition

    tracker.log({"train/loss": jnp.float32(1.25), "throughput": np.float64(7)}, step=3)

    samples = _series(render(), "levanter_")
    assert samples["levanter_train_loss"] == 1.25
    assert samples["levanter_throughput"] == 7.0
    assert samples["levanter_step"] == 3.0


def test_log_skips_values_that_are_not_real_scalars(exposition):
    tracker, render = exposition

    tracker.log({"name": "gpt", "flag": True, "weights": jnp.arange(4)}, step=None)

    samples = _series(render(), "levanter_")
    assert not [key for key in samples if "name" in key or "flag" in key or "weights" in key]


def test_summary_stats_becomes_moments_and_a_prometheus_histogram(exposition):
    tracker, render = exposition
    values = jnp.asarray(np.linspace(0.0, 1.0, 1000))

    tracker.log({"grad": SummaryStats.from_array(values, num_bins=4)}, step=1)

    samples = _series(render(), "levanter_grad")
    assert samples["levanter_grad_mean"] == pytest.approx(0.5, abs=1e-3)
    assert samples["levanter_grad_max"] == pytest.approx(1.0)
    # Buckets are cumulative, so the last one accounts for every observation.
    buckets = {key: value for key, value in samples.items() if "_bucket" in key}
    assert len(buckets) == 5
    assert samples['levanter_grad_bucket{le="+Inf"}'] == 1000.0
    assert samples["levanter_grad_count"] == 1000.0


def test_config_publishes_run_and_source_as_global_labels():
    """The forwarder tags persisted rows with these, so init must set them."""
    saved = telltale.get_global_labels()
    telltale._global_labels.clear()
    try:
        TelltaleConfig().init("run-42")
        assert telltale.get_global_labels() == {"source": "levanter", "run": "run-42"}

        telltale._global_labels.clear()
        TelltaleConfig().init(None)
        assert telltale.get_global_labels() == {"source": "levanter"}
    finally:
        telltale._global_labels.clear()
        telltale._global_labels.update(saved)


def test_two_trackers_do_not_duplicate_the_histogram_family(exposition):
    """A repeated family is malformed exposition, which Prometheus rejects on scrape."""
    tracker, render = exposition
    stats = SummaryStats.from_array(jnp.asarray(np.linspace(0.0, 1.0, 100)), num_bins=2)

    tracker.log({"grad": stats}, step=1)
    TelltaleTracker().log({"grad": stats}, step=1)

    assert render().count("# TYPE levanter_grad histogram") == 1
