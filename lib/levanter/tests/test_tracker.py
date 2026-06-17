# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
import json
import logging
import re
import threading
import warnings
from typing import Tuple

import draccus
import pytest
import yaml

import levanter.tracker
import levanter.tracker.tracker_fns as tracker_fns
import levanter.tracker.wandb as wandb_tracker_mod
from levanter.tracker import BackgroundTracker, CompositeTracker, NoopTracker, TrackerConfig
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbTracker, _truncate_wandb_artifact_name, truncate_wandb_run_name


class _HandleAbandonedError(Exception):
    pass


class _FakeWandbRun:
    step = 0

    def __init__(
        self,
        *,
        finish_error: Exception | None = None,
        finish_release: threading.Event | None = None,
    ):
        self.config = {"learning_rate": 1.0e-4, "model": {"layers": 2}}
        self.summary = {"train/loss": 6.61, "_step": 99}
        self.finish_error = finish_error
        self.finish_release = finish_release
        self.finish_started = threading.Event()
        self.finish_calls = 0

    def finish(self):
        self.finish_calls += 1
        self.finish_started.set()
        if self.finish_release is not None:
            self.finish_release.wait(timeout=5.0)
        if self.finish_error is not None:
            raise self.finish_error


def _read_tracker_metrics(replicate_path):
    metrics_path = replicate_path / "tracker_metrics.jsonl"
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


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

    wandb1 = wandb_config(mode="offline").init(None)
    tracker = CompositeTracker([wandb1, NoopTracker()])

    with tracker:
        assert levanter.tracker.get_tracker("wandb") is wandb1
        assert levanter.tracker.get_tracker("noop") is not None

        with pytest.raises(KeyError):
            levanter.tracker.get_tracker("foo")


def test_tracker_logging_without_global_tracker_emits_no_warning(monkeypatch):
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


def test_wandb_tracker_finish_writes_replicate_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    replicate_path = tmp_path / "replicate"
    run = _FakeWandbRun()
    tracker = WandbTracker(run, replicate_path=str(replicate_path))

    tracker.finish()

    assert run.finish_calls == 1
    assert _read_tracker_metrics(replicate_path) == {
        "config": run.config,
        "summary": run.summary,
    }


def test_wandb_tracker_finish_writes_replicate_file_when_finish_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    replicate_path = tmp_path / "replicate"
    run = _FakeWandbRun(finish_error=_HandleAbandonedError("wandb finish abandoned"))
    tracker = WandbTracker(run, replicate_path=str(replicate_path))

    with pytest.raises(_HandleAbandonedError):
        tracker.finish()

    assert _read_tracker_metrics(replicate_path) == {
        "config": run.config,
        "summary": run.summary,
    }


def test_wandb_tracker_finish_without_replicate_path_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    run = _FakeWandbRun()
    tracker = WandbTracker(run)

    tracker.finish()

    assert run.finish_calls == 1
    assert list(tmp_path.iterdir()) == []


def test_background_tracker_warns_and_writes_replicate_file_when_wandb_finish_raises(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    replicate_path = tmp_path / "replicate"
    run = _FakeWandbRun(finish_error=_HandleAbandonedError("wandb finish abandoned"))
    wrapped = WandbTracker(run, replicate_path=str(replicate_path))
    tracker = BackgroundTracker(wrapped, finish_timeout=5.0)

    with caplog.at_level(logging.WARNING, logger="levanter.tracker.background"):
        tracker.finish()

    assert _read_tracker_metrics(replicate_path) == {
        "config": run.config,
        "summary": run.summary,
    }
    assert any(
        record.levelno == logging.WARNING and "raised during finish" in record.message for record in caplog.records
    )


def test_background_tracker_timeout_keeps_replicate_file_when_wandb_finish_hangs(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")

    replicate_path = tmp_path / "replicate"
    finish_release = threading.Event()
    run = _FakeWandbRun(finish_release=finish_release)
    wrapped = WandbTracker(run, replicate_path=str(replicate_path))
    tracker = BackgroundTracker(wrapped, finish_timeout=0.2)

    try:
        with caplog.at_level(logging.WARNING, logger="levanter.tracker.background"):
            tracker.finish()

        assert run.finish_started.wait(timeout=1.0)
        assert _read_tracker_metrics(replicate_path) == {
            "config": run.config,
            "summary": run.summary,
        }
        assert any(
            record.levelno == logging.WARNING and "did not exit within" in record.message for record in caplog.records
        )
    finally:
        finish_release.set()
        tracker._thread.join(timeout=5.0)


def test_truncate_wandb_run_name_preserves_scientific_notation_lr_suffix():
    name = "dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2"

    truncated = truncate_wandb_run_name(name)

    assert len(truncated) <= 64
    assert truncated.endswith("lr7.5e-7_seed2")
    assert "_-7_seed2" not in truncated


def test_truncate_wandb_run_name_logs_warning(caplog):
    name = "dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2"

    with caplog.at_level(logging.WARNING):
        truncate_wandb_run_name(name)

    assert any(record.levelno >= logging.WARNING for record in caplog.records)
