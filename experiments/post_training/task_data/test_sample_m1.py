# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.chat_template import GRUG_TOKENS
from experiments.post_training.probe_sets.continuations import render_prompt_ids
from experiments.post_training.probe_sets.test_continuations import byte_level_tokenizer
from experiments.post_training.task_data.sample_m1 import m1_row, verdict

TASK = {"task_id": "r-0", "prompt": "add two and five", "answer": "7"}


def test_a_graded_completion_becomes_an_m1_row_of_the_sampled_ids():
    tokenizer = byte_level_tokenizer()
    prompt = [{"role": "user", "content": TASK["prompt"]}]
    prompt_ids = render_prompt_ids(tokenizer, "grug", prompt)
    close = tokenizer.convert_tokens_to_ids(GRUG_TOKENS.turn_close)
    response = [*tokenizer.encode(r"it is \boxed{7}", add_special_tokens=False), close]
    assert verdict(tokenizer, "grug", "R", TASK, response) == ("ended", True)
    row = m1_row(TASK, prompt, prompt_ids, response, "R")
    assert row["input_ids"] == prompt_ids + response
    assert row["assistant_masks"] == [0] * len(prompt_ids) + [1] * len(response)
    assert (row["data_source"], row["teacher_route"]) == ("task_R", "E_R")


def test_a_completion_that_never_closed_its_turn_is_over_the_cap():
    tokenizer = byte_level_tokenizer()
    response = tokenizer.encode(r"it is \boxed{7}", add_special_tokens=False)
    assert verdict(tokenizer, "grug", "R", TASK, response) == ("over_cap", False)
