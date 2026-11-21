# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from marin.rollouts.evaluation import generation_turn
from marin.rollouts.reasoning_format import inline_reasoning_format
from rolloutengine.models import ModelRequest, ModelTurn, TokenUsage
from taskcompendium.evaluation.spec import ChatGeneration

START, END = "<|start_think|>", "<|end_think|>"
SPECIAL = (START, END)


class PrefillTokenizer:
    """A Grug-style tokenizer whose template opens the reasoning span when thinking is on."""

    chat_template = f"{{{{ messages }}}}{START}{{{{ reasoning }}}}{END}"

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens):
        text = "".join(chr(token) for token in token_ids)
        return "" if skip_special_tokens and text in SPECIAL else text

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and add_generation_prompt
        prompt = "".join(f"<{message['role']}>{message['content']}" for message in conversation) + "<assistant>"
        return prompt + START if kwargs.get("enable_thinking") else prompt


def _reply(content):
    async def model(request):
        return ModelTurn(
            {"role": "assistant", "content": content},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason="length",
            text=content,
            metadata={},
            usage=TokenUsage(1, 1),
        )

    return model


def _turn(content, *, thinking):
    request = ModelRequest(
        messages=({"role": "user", "content": "Count the twos in 1222."},),
        options={"extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}},
    )
    reasoning_format = inline_reasoning_format(PrefillTokenizer())
    assert reasoning_format is not None and reasoning_format.special
    return asyncio.run(
        generation_turn(
            request,
            model=_reply(content),
            operation=ChatGeneration(response_selection="final_only"),
            reasoning_format=reasoning_format,
        )
    )


@pytest.mark.parametrize(
    "content,thinking,answer,reasoning",
    [
        # The prompt opened the span, so a bare closer ends the reasoning.
        (f"Three twos.{END}3", True, "3", "Three twos."),
        # A length stop inside the prompt-opened span is unfinished reasoning, not an answer.
        ("Three twos so far: 2, 2", True, "", "Three twos so far: 2, 2"),
        # The same unmarked text is the answer when thinking is off.
        ("3", False, "3", None),
    ],
)
def test_scorer_sees_only_the_final_answer(content, thinking, answer, reasoning):
    turn = _turn(content, thinking=thinking)
    assert turn.text == answer
    assert turn.metadata["evaluation_raw_message"].get("reasoning_content") == reasoning
