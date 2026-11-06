# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from rigging.hf_generation_config import export_generation_config

QWEN_STYLE_TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['content'] }}<|im_end|>{% endfor %}"


@dataclass
class _Tokenizer:
    bos_token_id: int | None = None
    eos_token_id: int | None = 7
    vocab: dict[str, int] = field(default_factory=lambda: {"<|im_end|>": 5, "<|eot_id|>": 6, "<|endoftext|>": 7})

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def __len__(self) -> int:
        return 10


def test_source_config_keeps_its_values_and_gains_end_of_turn_stop():
    source = {"eos_token_id": 7, "temperature": 0.7, "top_k": 20, "transformers_version": "5.16.1"}

    config = export_generation_config(_Tokenizer(), chat_template=QWEN_STYLE_TEMPLATE, explicit=source)

    assert config == {
        "eos_token_id": [5, 7],
        "temperature": 0.7,
        "top_k": 20,
        "transformers_version": "5.16.1",
        "do_sample": True,
        "top_p": 0.95,
    }


def test_explicit_greedy_config_gets_no_sampling_fields():
    config = export_generation_config(_Tokenizer(), chat_template=None, explicit={"do_sample": False})

    assert config == {"do_sample": False, "eos_token_id": [7]}
