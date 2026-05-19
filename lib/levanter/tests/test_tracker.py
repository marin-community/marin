# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
import re
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


def test_wandb_artifact_name_defaults_to_basename_and_truncates(monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    from levanter.tracker.wandb import WandbTracker, _truncate_wandb_artifact_name

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log_artifact(self, artifact_path, *, name=None, type=None):
            self.logged.append((artifact_path, name, type))

    run = FakeRun()
    tracker = WandbTracker(run)

    tracker.log_artifact("/tmp/some/deep/path/profile", type="jax_profile")
    assert run.logged == [("/tmp/some/deep/path/profile", "profile", "jax_profile")]

    long_name = "run-" + "x" * 200
    truncated = _truncate_wandb_artifact_name(long_name)
    assert truncated is not None
    assert len(truncated) <= 128
    assert re.fullmatch(r".+-[0-9a-f]{7}", truncated)


def test_wandb_tracker_suppressed_logging_materializes_after_resume_step(monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    import levanter.tracker.wandb as wandb_tracker_mod
    from levanter.tracker.wandb import WandbTracker

    converted = []

    def fake_convert(value):
        converted.append(value)
        return value

    class FakeSummary:
        def update(self, metrics):
            raise AssertionError("suppressed tracker should not update summary")

    class FakeConfig:
        def update(self, metrics, *, allow_val_change=False):
            raise AssertionError("suppressed tracker should not update config")

    class FakeRun:
        step = 0
        summary = FakeSummary()
        config = FakeConfig()

        def log(self, metrics, *, step=None, commit=None):
            raise AssertionError("suppressed tracker should not log metrics")

        def log_artifact(self, artifact_path, *, name=None, type=None):
            raise AssertionError("suppressed tracker should not log artifacts")

        def finish(self):
            raise AssertionError("suppressed tracker should not finish the run")

    monkeypatch.setattr(wandb_tracker_mod, "_convert_value_to_loggable_rec", fake_convert)
    tracker = WandbTracker(FakeRun(), suppress_logging=True, minimum_log_step=10)

    tracker.log({"metric": 1.0}, step=0)
    assert converted == []

    tracker.log({"metric": 2.0}, step=10)
    assert converted == [2.0]

    tracker.log_summary({"metric": 1.0})
    tracker.log_hyperparameters({"param": 1.0})
    tracker.log_artifact("/tmp/profile", type="profile")
    tracker.finish()


def test_wandb_tracker_materializes_before_dynamic_stale_step_check(monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    import levanter.tracker.wandb as wandb_tracker_mod
    from levanter.tracker.wandb import WandbTracker

    converted = []

    def fake_convert(value):
        converted.append(value)
        return value

    class FakeRun:
        step = 11

        def log(self, metrics, *, step=None, commit=None):
            raise AssertionError("stale metrics should not reach wandb")

    monkeypatch.setattr(wandb_tracker_mod, "_convert_value_to_loggable_rec", fake_convert)
    tracker = WandbTracker(FakeRun(), minimum_log_step=10)

    tracker.log({"metric": 2.0}, step=10)

    assert converted == [2.0]
