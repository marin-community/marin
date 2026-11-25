# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from transformers import AutoTokenizer

from experiments.post_training.mochi_doom_ftpo.cohorts import prompt_tokens
from experiments.post_training.task_curation.datasets.skyrl.math import convert_dapo_math


def test_cohort_hashing_includes_the_native_answer_format_instruction(gpt2_tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    task = convert_dapo_math(
        RawRow(
            "a",
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            {
                "prompt": [{"role": "user", "content": "Solve 1 + 1."}],
                "reward_model": {"ground_truth": "2"},
            },
        ),
        ConversionContext({}, None),
    )
    expected = "user: Solve 1 + 1.\nuser: Give your answer as plain text.\nassistant: "
    assert prompt_tokens(task, tokenizer) == tokenizer.encode(expected, add_special_tokens=False)
