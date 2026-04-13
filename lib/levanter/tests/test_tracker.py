# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
import re
import warnings
from typing import Tuple

import pytest
import yaml
import jax

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


def test_wandb_config_reapplies_tags_when_init_drops_them(monkeypatch):
    wandb_config_cls = TrackerConfig.get_choice_class("wandb")
    if wandb_config_cls is None:
        pytest.skip("wandb not installed")

    import wandb
    import levanter.tracker.wandb as wandb_tracker_module

    class FakeRun:
        def __init__(self):
            self.step = 0
            self.project = "dpo"
            self.name = "fake-run"
            self.id = "fake-id"
            self.group = None
            self._tags = ()

        @property
        def tags(self):
            return self._tags

        @tags.setter
        def tags(self, tags):
            self._tags = tuple(tags)

        def log_artifact(self, *_args, **_kwargs):
            pass

    fake_run = FakeRun()

    def fake_init(**kwargs):
        assert kwargs["tags"] == ["dpo", "lora-dpo"]
        return fake_run

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "run", fake_run, raising=False)
    monkeypatch.setattr(wandb, "summary", {}, raising=False)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "device_count", lambda: 8)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(wandb_tracker_module, "generate_pip_freeze", lambda: "")
    monkeypatch.setattr(wandb_config_cls, "_git_settings", lambda self: {})

    tracker = wandb_config_cls(project="dpo", tags=["dpo", "lora-dpo"], mode="disabled").init(None)

    assert tracker.run is fake_run
    assert fake_run.tags == ("dpo", "lora-dpo")


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
