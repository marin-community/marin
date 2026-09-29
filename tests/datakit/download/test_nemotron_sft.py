# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.nemotron_v2 import _nemotron_sft_messages


def test_nemotron_general_recovers_multiturn_roles():
    text = "<extra_id_1>User\nQuestion one<extra_id_1>Assistant\nAnswer one<extra_id_1>User\nQuestion two"
    assert _nemotron_sft_messages(text, "sft_general") == [
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Answer one"},
        {"role": "user", "content": "Question two"},
    ]


def test_nemotron_math_recovers_input_output_roles():
    assert _nemotron_sft_messages("input: Solve it. output: The answer is 3.", "sft_math") == [
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": "The answer is 3."},
    ]


def test_nemotron_code_preserves_thinking_in_assistant_turn():
    assert _nemotron_sft_messages("Write code.\n\n<think>Reasoning</think>\nCode", "sft_code") == [
        {"role": "user", "content": "Write code."},
        {"role": "assistant", "content": "<think>Reasoning</think>\nCode"},
    ]


def test_nemotron_general_joins_adjacent_speaker_segments():
    text = (
        "<extra_id_1>User\nExplain the options."
        "<extra_id_1>User\nCompare their costs."
        "<extra_id_1>Assistant\nThere are three options."
        "<extra_id_1>Assistant\nThe first is inexpensive."
        "<extra_id_1>User\nThank you."
        "<extra_id_1>Assistant\nYou're welcome."
    )
    assert _nemotron_sft_messages(text, "sft_general") == [
        {"role": "user", "content": "Explain the options.\n\nCompare their costs."},
        {"role": "assistant", "content": "There are three options.\n\nThe first is inexpensive."},
        {"role": "user", "content": "Thank you."},
        {"role": "assistant", "content": "You're welcome."},
    ]


def test_nemotron_general_preserves_missing_prompt_boundary():
    text = (
        "<extra_id_1>User\nName an artist."
        "<extra_id_1>Assistant\nPicasso."
        "<extra_id_1>User\n"
        "<extra_id_1>Assistant\nHe developed Cubism."
    )
    assert _nemotron_sft_messages(text, "sft_general") == [
        {"role": "user", "content": "Name an artist."},
        {"role": "assistant", "content": "Picasso."},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "He developed Cubism."},
    ]
