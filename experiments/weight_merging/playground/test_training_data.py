# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from experiments.weight_merging.playground.training_data import encode_example, shingles


def test_assistant_loss_masks_prompt_and_supervises_answer_and_eos():
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "assistant": 2, "question": 3, "answer": 4, "[EOS]": 5}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    record = encode_example(tokenizer, "question", "answer", max_length=100)
    assert record is not None
    assert record["input_ids"] == [1, 3, 2, 4, 5]
    assert record["labels"] == [-100, -100, -100, 4, 5]


def test_decontamination_shingles_ignore_case_and_punctuation():
    test_question = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    rephrased = "Intro: ONE, two three four five six seven eight nine ten eleven twelve thirteen fourteen!"
    assert shingles(test_question) & shingles(rephrased)
    assert not shingles(test_question) & shingles("An unrelated code task.")
