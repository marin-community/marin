# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.inference_engines.chat_template import assistant_prefix_token_ids

# The Marin chat template trims assistant content the same way.
TRIMMING_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content | trim }}<|end|>{% endfor %}"
KEEPING_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
# Like the Marin chat template, a newline follows each role header.
HEADER_NEWLINE_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content | trim }}<|end|>{% endfor %}"
ASSISTANT_PREFIX_OPTIONS = {"extra_body": {"continue_final_message": True, "add_generation_prompt": False}}


def bracket_tokenizer(chat_template: str) -> PreTrainedTokenizerFast:
    """Spell each character as a token, except that a closing bracket and a following newline merge."""
    vocabulary = {"[UNK]": 0, "x": 1, "[": 2, "1": 3, "]": 4, "\n": 5, "]\n": 6, "f": 7, "o": 8}
    backend = Tokenizer(models.BPE(vocabulary, [("]", "\n")], unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|user|>", "<|assistant|>", "<|end|>"],
        chat_template=chat_template,
    )


def conversation(prefix: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": "x"}, {"role": "assistant", "content": prefix}]


@pytest.mark.parametrize(
    "prefix,prompt",
    [
        ("[1]\n", "<|user|>x<|end|><|assistant|>[1]\n"),
        ("\n[1]\n\n", "<|user|>x<|end|><|assistant|>[1]\n\n"),
    ],
    ids=["trailing-newline", "leading-whitespace-stays-trimmed"],
)
def test_assistant_prefix_tokens_from_trimming_template_keep_trailing_whitespace(prefix, prompt):
    tokenizer = bracket_tokenizer(TRIMMING_TEMPLATE)

    token_ids = assistant_prefix_token_ids(tokenizer, conversation(prefix), ASSISTANT_PREFIX_OPTIONS)

    # The newline merges into the bracket token, so a separately tokenized newline would not match.
    assert "]\n" in tokenizer.convert_ids_to_tokens(token_ids)
    assert token_ids == tokenizer.encode(prompt, add_special_tokens=False)


@pytest.mark.parametrize("prefix", ["\n", "\n\n"], ids=["newline", "two-newlines"])
@pytest.mark.parametrize(
    "chat_template,header",
    [(TRIMMING_TEMPLATE, "<|assistant|>"), (HEADER_NEWLINE_TEMPLATE, "<|assistant|>\n")],
    ids=["bare-header", "header-newline"],
)
def test_whitespace_only_assistant_prefix_follows_the_header(chat_template, header, prefix):
    tokenizer = bracket_tokenizer(chat_template)
    messages = conversation(prefix)
    user_turn = tokenizer.apply_chat_template(messages[:1], tokenize=False, return_dict=False)

    token_ids = assistant_prefix_token_ids(tokenizer, messages, ASSISTANT_PREFIX_OPTIONS)

    assert token_ids == tokenizer.encode(user_turn + header + prefix, add_special_tokens=False)


@pytest.mark.parametrize("prefix", ["foox\n", "xx\n"], ids=["ends-in-marker", "only-markers"])
@pytest.mark.parametrize("chat_template", [TRIMMING_TEMPLATE, KEEPING_TEMPLATE], ids=["trimming", "keeping"])
def test_assistant_prefix_keeps_its_own_marker_characters(monkeypatch, chat_template, prefix):
    # The user message holds the marker too, before the prefix.
    monkeypatch.setattr("skyrl_train.inference_engines.chat_template.PREFIX_END_MARKER", "x")
    tokenizer = bracket_tokenizer(chat_template)

    token_ids = assistant_prefix_token_ids(tokenizer, conversation(prefix), ASSISTANT_PREFIX_OPTIONS)

    assert token_ids == tokenizer.encode("<|user|>x<|end|><|assistant|>" + prefix, add_special_tokens=False)


@pytest.mark.parametrize(
    "chat_template,prefix",
    [(TRIMMING_TEMPLATE, "[1]"), (TRIMMING_TEMPLATE, "\n[1"), (KEEPING_TEMPLATE, "[1]\n")],
    ids=["no-trailing-whitespace", "leading-whitespace-only", "keeping-template"],
)
def test_assistant_prefix_tokens_without_trimmed_whitespace_match_hugging_face_continuation(chat_template, prefix):
    tokenizer = bracket_tokenizer(chat_template)
    messages = conversation(prefix)

    token_ids = assistant_prefix_token_ids(tokenizer, messages, ASSISTANT_PREFIX_OPTIONS)

    assert token_ids == tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False, continue_final_message=True
    )
