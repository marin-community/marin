# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every frozen mix and probe prompt renders with the student's chat template and fits the composed limits.

Each row is rendered with its own tools and chat-template arguments through ``apply_chat_template`` and the exported
template, then tokenized. A render error, a prompt over the generator's input limit, or a prompt that leaves less
than the full response budget inside the served context fails the test.

Needs ``IDENTITY_OPD_RENDER_DIR`` (see ``test_vllm_tool_choice.py``); skips otherwise.
"""

import ast
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from transformers import PreTrainedTokenizerFast

from experiments.post_training.identity_opd import launch
from experiments.post_training.identity_opd.test_identity_opd import _composed_skyrl

RENDER_DIR = os.environ.get("IDENTITY_OPD_RENDER_DIR")
pytestmark = pytest.mark.skipif(not RENDER_DIR, reason="IDENTITY_OPD_RENDER_DIR is not set")


def _messages(prompt) -> list[dict]:
    messages = [
        {key: value for key, value in message.items() if value is not None}
        for message in (ast.literal_eval(prompt) if isinstance(prompt, str) else prompt)
    ]
    for message in messages:
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return messages


@pytest.mark.parametrize("arm,name", [("train", launch.MIX_FILE), ("dry", launch.MIX_FILE), ("dry", launch.PROBE_FILE)])
def test_every_frozen_prompt_renders_within_the_composed_limits(arm, name, tmp_path):
    root = Path(RENDER_DIR)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == launch.MIX_SHA256[name]
    generator = _composed_skyrl(arm, tmp_path).generator
    max_model_len = generator.engine_init_kwargs.max_model_len
    max_new_tokens = generator.sampling_params.max_generate_length
    max_input = generator.max_input_length

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    template = (root / "chat_template.jinja").read_text()
    rows = pq.read_table(root / name).to_pylist()
    assert rows
    over = []
    for index, row in enumerate(rows):
        ids = tokenizer.apply_chat_template(
            _messages(row["prompt"]),
            tools=json.loads(row["tools"]) if row["tools"] else None,
            chat_template=template,
            add_generation_prompt=True,
            tokenize=True,
            **json.loads(row["chat_template_kwargs"] or "{}"),
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        assert ids and isinstance(ids[0], int), f"row {index}: expected a flat token list"
        if len(ids) > max_input or len(ids) + max_new_tokens > max_model_len:
            over.append((index, row["data_source"], len(ids)))
    assert not over, f"{len(over)} prompts over {max_input} input or {max_model_len} with +{max_new_tokens}: {over[:5]}"
