# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.hf_model_io import write_export_generation_config


def _chat_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "<|begin_of_text|>": 1, "<|end_of_text|>": 2, "<|eot_id|>": 3, "hello": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}<|eot_id|>{% endfor %}"
    return tokenizer


@pytest.mark.parametrize(
    ("source_config", "expected"),
    [
        (None, {"bos_token_id": 1, "eos_token_id": [2, 3], "do_sample": True, "temperature": 0.6, "top_p": 0.95}),
        (
            {"bos_token_id": 1, "eos_token_id": 2, "temperature": 0.7},
            {"bos_token_id": 1, "eos_token_id": [2, 3], "do_sample": True, "temperature": 0.7, "top_p": 0.95},
        ),
    ],
)
def test_policy_export_completes_source_generation_config(tmp_path, source_config, expected):
    source = tmp_path / "source"
    source.mkdir()
    if source_config is not None:
        (source / "generation_config.json").write_text(json.dumps(source_config))
    export = tmp_path / "export"
    export.mkdir()
    tokenizer = _chat_tokenizer()

    write_export_generation_config(str(export), str(source), tokenizer, tokenizer.chat_template)

    assert json.loads((export / "generation_config.json").read_text()) == expected
