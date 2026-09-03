# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer
from marin.datakit.chat import _render_messages


class _Tokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        return "<bos>" + "|".join(f"{message['role']}:{message['content']}" for message in conversation)


def test_render_messages_leaves_bos_insertion_to_text_tokenizer():
    [document] = _render_messages(
        {
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "source": "test",
        },
        cast(MarinTokenizer, _Tokenizer()),
        ChatLmDatasetFormat(mask_user_turns=False),
    )

    assert document["text"] == "user:Hi|assistant:Hello"
