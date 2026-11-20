# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


def stage_gpt2_tokenizer(source_dir: Path, output_dir: Path) -> Path:
    """Stage Levanter's checked-in GPT-2 tokenizer in a loadable directory."""
    shutil.copy(source_dir / "gpt2_tokenizer.json", output_dir / "tokenizer.json")
    shutil.copy(source_dir / "gpt2_tokenizer_config.json", output_dir / "tokenizer_config.json")
    (output_dir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return output_dir


def llama3_style_chat_tokenizer() -> PreTrainedTokenizerFast:
    """A five-token tokenizer whose chat template ends turns with ``<|eot_id|>`` (id 3); EOS is id 2, BOS id 1."""
    vocab = {"[UNK]": 0, "<|begin_of_text|>": 1, "<|end_of_text|>": 2, "<|eot_id|>": 3, "hello": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}<|eot_id|>{% endfor %}"
    return tokenizer
