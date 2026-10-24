# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.chat_template import GRUG_TOKENS
from experiments.post_training.probe_sets.buckets import Bucket
from experiments.post_training.probe_sets.continuations import continuation_row, render_prompt_ids, request_body
from experiments.post_training.stage0.protocol_gate import Decode

WORDS = ["[UNK]", "\n", "user", "assistant", "add", "two", "and", "five", "carry", "it", "is", "7"]
PROBE = {"probe_id": "p-0", "source": "aime_2024", "messages": [{"role": "user", "content": "add two and five"}]}


def word_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="[UNK]"))
    # Newlines are tokens, as in the real vocabulary, so the template's trailing newline shows up.
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", "isolated"), pre_tokenizers.Split(" ", "removed")]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    markers = [GRUG_TOKENS.bos, GRUG_TOKENS.turn_open_prefix, GRUG_TOKENS.turn_open_suffix, GRUG_TOKENS.turn_close]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [*markers, GRUG_TOKENS.think_open, GRUG_TOKENS.think_close]}
    )
    return tokenizer


def sampled_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def test_an_ended_reply_keeps_the_sampled_ids_and_tags_its_thinking():
    tokenizer = word_tokenizer()
    prompt_ids = render_prompt_ids(tokenizer, "grug", PROBE["messages"])
    response = sampled_ids(
        tokenizer, f"{GRUG_TOKENS.think_open} add carry {GRUG_TOKENS.think_close} it is 7 {GRUG_TOKENS.turn_close}"
    )
    row = continuation_row(tokenizer, "grug", "P_think", PROBE, prompt_ids, response)
    assert row["token_ids"] == prompt_ids + response and not row["retokenized"] and row["turn_ended"]
    tags = row["token_type_tags"][len(prompt_ids) :]
    assert tags == ["delimiter", "reasoning", "reasoning", "delimiter", *["final_message"] * 3, "delimiter"]
    assert len(row["token_type_tags"]) == len(row["token_ids"]) == len(row["position_tags"])


def test_a_reply_cut_at_the_cap_gets_no_closing_token():
    tokenizer = word_tokenizer()
    prompt_ids = render_prompt_ids(tokenizer, "grug", PROBE["messages"])
    response = sampled_ids(tokenizer, f"{GRUG_TOKENS.think_open} add carry carry")  # thinking never closed
    row = continuation_row(tokenizer, "grug", "P_think", PROBE, prompt_ids, response)
    assert row["token_ids"] == prompt_ids + response and not row["turn_ended"]
    assert row["token_type_tags"][len(prompt_ids) :] == ["delimiter", *[Bucket.REASONING.value] * 3]


def test_the_request_samples_token_ids_and_stops_only_on_the_turn_close():
    body = request_body("b", [1, 2], seed=3, decode=Decode(repetition_penalty=1.05), stop_id=9)
    assert body["prompt"] == [1, 2] and body["return_token_ids"] and body["ignore_eos"]
    assert body["stop_token_ids"] == [9] and body["repetition_penalty"] == 1.05 and not body["skip_special_tokens"]
