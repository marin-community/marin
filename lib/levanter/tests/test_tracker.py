# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
import json
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


def test_wandb_tracker_replicates_metadata_and_lm_eval_artifacts(tmp_path):
    from levanter.tracker.wandb import WandbTracker

    class FakeRun:
        def __init__(self):
            self.step = 0
            self.project = "marin"
            self.name = "unit/test"
            self.tags = ["exp-tag", "run-tag"]
            self.id = "abc123"
            self.group = "unit-group"
            self.url = "https://wandb.ai/example/run"
            self.config = {"foo": "bar"}
            self.summary = {"eval/paloma/c4_en/bpb": 1.23}
            self.logged_artifacts = []

        def log_artifact(self, artifact_path, name=None, type=None):
            self.logged_artifacts.append((artifact_path, name, type))

        def finish(self):
            return None

    run = FakeRun()
    tracker = WandbTracker(run, replicate_path=str(tmp_path))

    artifact_path = tmp_path / "lm_eval_harness_results.42.json"
    artifact_path.write_text('{"ok": true}')
    tracker.log_artifact(str(artifact_path), name="lm_eval_harness_results.42.json", type="lm_eval_output")
    tracker.finish()

    mirrored = tmp_path / "lm_eval_artifacts" / "lm_eval_harness_results.42.json"
    assert mirrored.read_text() == '{"ok": true}'

    metrics_path = tmp_path / "tracker_metrics.jsonl"
    record = json.loads(metrics_path.read_text().strip())
    assert record["wandb"]["id"] == "abc123"
    assert record["wandb"]["name"] == "unit/test"
    assert record["wandb"]["tags"] == ["exp-tag", "run-tag"]


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
