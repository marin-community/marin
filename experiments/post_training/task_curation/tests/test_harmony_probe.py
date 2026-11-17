# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the difficulty serving probe's pinned rendering paths."""


from taskcompendium.pipeline.reviewer_spec import JudgeTier

from experiments.post_training.task_curation.difficulty_coverage import build_parser
from experiments.post_training.task_curation.difficulty_serving_probe import (
    judge_expected_prompt_tokens,
    judge_probe_request,
)
from experiments.post_training.task_curation.harmony_probe import harmony_prompt_tokens
from experiments.post_training.task_curation.judges import JUDGE_TIERS


def test_light_tier_marks_harmony_served():
    assert JUDGE_TIERS[JudgeTier.LIGHT].served_encoding == "harmony"
    assert JUDGE_TIERS[JudgeTier.MEDIUM].served_encoding == "chat_template"


def test_judge_probe_request_advertises_no_tools():
    for tier in JUDGE_TIERS.values():
        assert "tools" not in judge_probe_request(tier, temperature=0.7)


def test_harmony_expected_tokens_positive_and_deterministic():
    definition = JUDGE_TIERS[JudgeTier.LIGHT]
    request = judge_probe_request(definition, temperature=0.7)
    first = judge_expected_prompt_tokens(definition, request)
    assert first > 0
    assert first == harmony_prompt_tokens(request)


def test_harmony_rendering_tracks_the_request_conversation():
    definition = JUDGE_TIERS[JudgeTier.LIGHT]
    base = judge_probe_request(definition, temperature=0.7)
    longer = dict(base)
    longer["messages"] = [
        *base["messages"],
        {"role": "user", "content": "And now reply with exactly DONE."},
    ]
    assert harmony_prompt_tokens(longer) > harmony_prompt_tokens(base)


def test_medium_tier_is_not_harmony_gated():
    definition = JUDGE_TIERS[JudgeTier.MEDIUM]
    assert definition.served_encoding == "chat_template"
    # The chat-template path needs an immutable pin and renders via the repo
    # tokenizer; the dispatcher must not route it through the harmony encoder.
    request = judge_probe_request(definition, temperature=0.7)
    assert "model" in request and request["model"] == definition.spec.model


def test_coverage_cli_parses_without_click_only_kwargs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "queue",
            "--applet-id",
            "c9742a15-fb46-4d56-bc88-aa916351ebd3",
            "--output",
            "/tmp/coverage-queue-test.json",
        ]
    )
    assert args.applet_id == "c9742a15-fb46-4d56-bc88-aa916351ebd3"
    assert args.command == "queue"
