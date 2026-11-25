# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from rolloutengine.models import RolloutLimits
from skyrlrecipe import ContextBudget
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from transformers import AutoTokenizer

from experiments.post_training.mochi_doom_ftpo.cohorts import prompt_tokens
from experiments.post_training.mochi_doom_ftpo.launch import generation_config
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


def test_audit_runtime_keeps_token_bounds_without_wall_clock_cutoffs():
    defaults = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
        ).read_text()
    )
    patch = generation_config(32768).to_skyrl()
    session = {**defaults["task_spec"]["session"], **patch["environment"]["task_sessions"]["task_spec"]["session"]}
    limits = RolloutLimits.model_validate(
        {name: value for name, value in session.items() if name in RolloutLimits.model_fields}
    )
    budget = ContextBudget.model_validate(patch["context_budget"]).derived_values(None)
    assert (limits.model_turn_timeout, limits.total_turn_timeout, limits.attempt_timeout) == (None, None, None)
    assert limits.max_turns == budget["generator.max_turns"] == 1
    assert budget["generator.sampling_params.max_generate_length"] == 32768
    assert (limits.verifier_timeout, limits.cleanup_timeout) == (300, 60)
