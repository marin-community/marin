# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.chat_template import GRUG_TOKENS

MARKERS = [
    GRUG_TOKENS.bos,
    GRUG_TOKENS.turn_open_prefix,
    GRUG_TOKENS.turn_open_suffix,
    GRUG_TOKENS.turn_close,
    GRUG_TOKENS.think_open,
    GRUG_TOKENS.think_close,
]


def byte_level_tokenizer() -> PreTrainedTokenizerFast:
    """A tiny byte-level BPE, like Snowball's: every byte is a token and common words merge."""
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=MARKERS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    backend.train_from_iterator(["add two and five", "carry it is 7", "user assistant\n"] * 50, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=backend, additional_special_tokens=MARKERS)
