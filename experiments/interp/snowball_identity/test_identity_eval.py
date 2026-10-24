# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.interp.snowball_identity.identity_eval import (
    Condition,
    Row,
    candidate_token_span,
    prescreen_labels,
    split_reasoning,
    with_condition,
)


def make_row(set_name: str, group: str = "g", expect=("name", "creator")) -> Row:
    return Row(
        id="r",
        set=set_name,
        category="c",
        group=group,
        split="dev",
        messages=[{"role": "user", "content": "Who are you?"}],
        expect=list(expect),
    )


def test_candidate_span_includes_token_that_straddles_the_prefix_boundary():
    # "My name is " + "Marin Snowball" tokenized as: My| name| is| Marin| Snow|ball
    offsets = [(0, 2), (2, 7), (7, 10), (10, 16), (16, 21), (21, 25)]
    assert list(candidate_token_span(offsets, len("My name is "))) == [3, 4, 5]


def test_identity_prescreen_flags_openai_self_attribution():
    labels = prescreen_labels(make_row("identity"), "I'm ChatGPT, a large language model trained by OpenAI.")
    assert labels["false_self_attribution"] == "ChatGPT"
    assert not labels["prescreen_pass"]


def test_identity_prescreen_passes_correct_answer():
    labels = prescreen_labels(make_row("identity"), "I'm Marin Snowball, created by Open-Athena.")
    assert labels["prescreen_pass"]
    assert labels["false_self_attribution"] is None


def test_third_person_mention_is_not_self_attribution():
    answer = "ChatGPT is made by OpenAI; Claude is made by Anthropic."
    labels = prescreen_labels(make_row("control", expect=()), answer)
    assert labels["false_self_attribution"] is None
    assert labels["prescreen_pass"]


def test_fiction_with_unsolicited_identity_fails():
    answer = "As Marin Snowball, built by Open Athena, I must note fairies are not real. Once upon a time..."
    labels = prescreen_labels(make_row("control", group="fic-fairy", expect=()), answer)
    assert labels["identity_leak"]
    assert not labels["prescreen_pass"]


def test_baseline_condition_skips_rows_with_their_own_system_prompt():
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi"}]
    assert with_condition(messages, Condition.SYSTEM_BASELINE) is None
    assert with_condition(messages[1:], Condition.SYSTEM_BASELINE)[0]["role"] == "system"


def test_split_reasoning_strips_inline_think_block():
    answer, reasoning = split_reasoning({"content": "<|start_think|>hmm<|end_think|>I'm Marin Snowball."})
    assert answer == "I'm Marin Snowball."
    assert reasoning == "hmm"


def test_negated_claim_is_not_self_attribution():
    labels = prescreen_labels(make_row("identity"), "I am not GPT-4. I am Marin Snowball, created by Open Athena.")
    assert labels["false_self_attribution"] is None
    assert labels["prescreen_pass"]


def test_wrong_creator_after_negated_correct_one_is_still_flagged():
    answer = "That is not accurate. I am an AI model developed by OpenAI, not Open Athena."
    assert prescreen_labels(make_row("identity"), answer)["false_self_attribution"] == "OpenAI"


def test_fact_rows_are_left_to_the_judge():
    labels = prescreen_labels(make_row("identity", expect=("facts",)), "The Marin project began at Berkeley.")
    assert labels["needs_judge"]
