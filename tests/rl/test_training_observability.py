# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marin.rl.replay_buffer import RolloutWithCount
from marin.rl.training_observability import (
    TrainingSamplePreview,
    configure_rl_training_metric_hooks,
    read_training_sample_previews,
    training_sample_preview_path,
    training_sample_previews_from_rollouts,
    write_training_sample_previews,
)
from marin.rl.types import Rollout, RolloutMetadata


class RecordingTracker:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def log(self, metrics, *, step, commit=None):
        self.calls.append({"metrics": metrics, "step": step, "commit": commit})


class DummyTrainer:
    def __init__(self):
        self.tracker = RecordingTracker()
        self.hooks: list[tuple[object, int]] = []

    def add_hook(self, fn, *, every=1):
        self.hooks.append((fn, every))


class DummyTokenizer:
    def decode(self, tokens, *, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(token) for token in tokens)


def _rollout(prompt_id: str, *, reward: float, token_offset: int) -> RolloutWithCount:
    return RolloutWithCount(
        rollout=Rollout(
            env_name="math",
            env_example_id=prompt_id,
            prompt_tokens=np.array([token_offset, token_offset + 1], dtype=np.int32),
            response_tokens=np.array([token_offset + 2, token_offset + 3], dtype=np.int32),
            response_logprobs=np.array([-0.1, -0.2], dtype=np.float32),
            token_rewards=np.array([reward / 2, reward / 2], dtype=np.float32),
            episode_reward=reward,
            temperature=1.0,
            top_k=10,
            is_truncated=False,
            metadata=RolloutMetadata(),
        ),
        advantage=reward,
    )


def test_training_sample_previews_match_async_selection_and_round_trip(tmp_path):
    rollouts = [
        _rollout("p0", reward=1.0, token_offset=0),
        _rollout("p1", reward=2.0, token_offset=10),
        _rollout("p2", reward=3.0, token_offset=20),
        _rollout("p3", reward=4.0, token_offset=30),
        _rollout("p4", reward=5.0, token_offset=40),
        _rollout("p5", reward=6.0, token_offset=50),
        _rollout("p0", reward=1.5, token_offset=60),
    ]

    previews = training_sample_previews_from_rollouts(rollouts)

    assert [preview.prompt_id for preview in previews] == ["p0", "p0", "p1", "p2", "p3", "p4"]

    preview_path = training_sample_preview_path((tmp_path / "batch_000000.pkl").as_posix())
    write_training_sample_previews(preview_path, previews)

    assert read_training_sample_previews(preview_path) == previews


def test_configure_rl_training_metric_hooks_installs_async_style_metrics(monkeypatch):
    trainer = DummyTrainer()
    sample_calls: list[dict[str, object]] = []
    performance_calls: list[dict[str, object]] = []

    def _fake_log_training_sample_table(trainer, *, tokenizer, step, previews):
        sample_calls.append(
            {
                "trainer": trainer,
                "tokenizer": tokenizer,
                "step": step,
                "previews": previews,
            }
        )

    def _fake_log_performance_stats(*, tokens_per_example, batch_schedule, flops_per_example, prefix):
        performance_calls.append(
            {
                "tokens_per_example": tokens_per_example,
                "batch_schedule": batch_schedule,
                "flops_per_example": flops_per_example,
                "prefix": prefix,
            }
        )

        def _hook(info):
            trainer.tracker.log({"throughput/examples_per_second": 128.0}, step=info.step)

        return _hook

    monkeypatch.setattr(
        "marin.rl.training_observability.log_training_sample_table",
        _fake_log_training_sample_table,
    )
    monkeypatch.setattr(
        "marin.rl.training_observability.callbacks.log_performance_stats",
        _fake_log_performance_stats,
    )

    previews = [
        TrainingSamplePreview(
            prompt_id="p0",
            prompt_tokens=[1, 2],
            response_tokens=[3, 4],
            reward=1.25,
        )
    ]

    configure_rl_training_metric_hooks(
        trainer,
        tokenizer=DummyTokenizer(),
        tokens_per_example=128,
        flops_per_example=256.0,
        batch_schedule=64,
        batch_prep_time=lambda: 1.5,
        sample_previews=lambda: previews,
    )

    info = SimpleNamespace(step=3, step_duration=5.0, loss=0.25)
    for hook, every in trainer.hooks:
        assert every == 1
        hook(info)

    assert performance_calls == [
        {
            "tokens_per_example": 128,
            "batch_schedule": 64,
            "flops_per_example": 256.0,
            "prefix": "throughput",
        }
    ]
    assert len(sample_calls) == 1
    assert sample_calls[0]["trainer"] is trainer
    assert isinstance(sample_calls[0]["tokenizer"], DummyTokenizer)
    assert sample_calls[0]["step"] == 3
    assert sample_calls[0]["previews"] == previews

    metric_calls = trainer.tracker.calls
    assert any(
        call["metrics"]
        == {
            "throughput/step_duration_seconds": 5.0,
            "throughput/batch_prep_duration_seconds": 1.5,
            "throughput/forward_backward_duration_seconds": 3.5,
            "train/loss": 0.25,
        }
        and call["step"] == 3
        for call in metric_calls
    )
    assert any(
        call["metrics"] == {"throughput/examples_per_second": 128.0} and call["step"] == 3 for call in metric_calls
    )
    assert any(call["metrics"] == {} and call["step"] == 3 and call["commit"] is True for call in metric_calls)
