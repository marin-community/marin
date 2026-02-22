# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
import warnings
from typing import Tuple

import pytest
import yaml

import levanter.tracker
from levanter.tracker import CompositeTracker, TrackerConfig


def test_tracker_plugin_stuff_works():
    assert TrackerConfig.get_choice_class("wandb") is not None
    with pytest.raises(KeyError):
        TrackerConfig.get_choice_class("foo")


def test_tracker_plugin_default_works():
    config = """
    tracker:
        entity: foo
    """
    parsed = yaml.safe_load(config)

    @dataclasses.dataclass
    class ConfigHolder:
        tracker: TrackerConfig

    import draccus

    tconfig = draccus.decode(ConfigHolder, parsed).tracker

    assert isinstance(tconfig, TrackerConfig.get_choice_class("wandb"))

    assert tconfig.entity == "foo"  # type: ignore


def test_tracker_plugin_multi_parsing_work():
    config = """
    tracker:
        type: noop
    """
    parsed = yaml.safe_load(config)

    @dataclasses.dataclass
    class ConfigHolder:
        tracker: TrackerConfig | Tuple[TrackerConfig, ...]

    import draccus

    from levanter.tracker.tracker import NoopConfig

    assert isinstance(draccus.decode(ConfigHolder, parsed).tracker, NoopConfig)

    config = """
    tracker:
        - type: noop
        - type: wandb
    """
    parsed = yaml.safe_load(config)
    decoded = draccus.decode(ConfigHolder, parsed).tracker
    assert decoded == (NoopConfig(), TrackerConfig.get_choice_class("wandb")())


def test_get_tracker_by_name(monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    wandb_config = TrackerConfig.get_choice_class("wandb")
    if wandb_config is None:
        pytest.skip("wandb not installed")

    from levanter.tracker import NoopTracker

    wandb1 = wandb_config(mode="offline").init(None)
    tracker = CompositeTracker([wandb1, NoopTracker()])

    with tracker:
        assert levanter.tracker.get_tracker("wandb") is wandb1
        assert levanter.tracker.get_tracker("noop") is not None

        with pytest.raises(KeyError):
            levanter.tracker.get_tracker("foo")


def test_tracker_logging_without_global_tracker_emits_no_warning(monkeypatch):
    import levanter.tracker.tracker_fns as tracker_fns

    monkeypatch.setattr(tracker_fns, "_global_tracker", None)
    monkeypatch.setattr(tracker_fns, "_has_logged_missing_tracker", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker_fns.log({"metric": 1.0}, step=0)
        tracker_fns.jit_log({"metric": 1.0}, step=0)
        tracker_fns.log_summary({"metric": 1.0})
        tracker_fns.log_hyperparameters({"metric": 1.0})
        tracker_fns.log_configuration({"metric": 1.0})

    assert not caught
