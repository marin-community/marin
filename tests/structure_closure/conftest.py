# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local tokenizer fixtures for structural completion transformations."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.structure_closure.benchmark import task_cases


@pytest.fixture
def tokenizer():
    cases = task_cases("sample_task", [{"search": {"query": 'quote=" slash=\\'}}], "train", "source")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [case.expected for case in cases],
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = (
        "{% if tools %}TOOLS={{ tools | tojson }}\n{% endif %}"
        "{% for message in messages %}"
        "{% if message.tool_calls is defined %}"
        "<tool_call>\n{{ message.tool_calls[0].function | tojson }}\n</tool_call>"
        "{% else %}{{ message.content }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}\nASSISTANT:\n{% endif %}"
    )
    return tokenizer
