# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.post_training.long_loop.endpoint_probe import prompt_tokens


def test_probe_compares_token_ids_not_batch_encoding():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    )
    messages = [{"role": "user", "content": "hello"}]
    served_ids = tokenizer.encode("hello", add_special_tokens=False)
    assert prompt_tokens(tokenizer, messages) == served_ids
