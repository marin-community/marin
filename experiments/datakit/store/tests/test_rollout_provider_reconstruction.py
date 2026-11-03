# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash

from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    reconstruct_selected_document,
    reconstruction_code_digests,
)
from experiments.datakit.store.reconstruct_selected_source import PROVIDER_COLUMNS, provider_rows


def _recipe(transform):
    normalizer, adapter = reconstruction_code_digests(transform)
    return ReconstructionRecipe(transform, "text", "id", 128, normalizer, adapter)


def test_agenttrove_original_projection_preserves_teacher_filter_outcome_and_rendering(tmp_path):
    raw = {
        "conversations": [{"role": "user", "content": "Fix bug"}, {"role": "assistant", "content": "done"}],
        "result": "AgentTimeoutError",
        "model": None,
        "model_provider": None,
        "original_teacher": "GLM-4.6",
        "original_source": "r2egym",
    }
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist([raw]), path)
    projected = list(provider_rows(str(path), PROVIDER_COLUMNS[ProviderTransform.AGENTTROVE], 8))
    text = (
        "This trajectory ended before the task was verified.\n\n"
        "<user>\nFix bug\n</user>\n\n<assistant>\ndone\n</assistant>"
    )
    selected_id = xxhash.xxh3_128_hexdigest(text.encode())
    recipe = _recipe(ProviderTransform.AGENTTROVE)
    result = reconstruct_selected_document(recipe, projected, selected_id)
    assert (result["text"], result["teacher"], result["task_source"]) == (text, "GLM-4.6", "r2egym")
    assert result["source_id"] == hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(
            replace(recipe, transform=ProviderTransform.DIRECT, transform_sha256=recipe.normalization_sha256),
            projected,
            selected_id,
        )
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**raw, "model_provider": "anthropic"}], selected_id)
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**raw, "result": "success"}], selected_id)


def test_coderforge_original_projection_replays_reward_and_tool_call_json(tmp_path):
    messages = [
        {"role": "user", "content": "Fix bug"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command":"pytest"}'}}],
        },
    ]
    raw = {"messages": json.dumps(messages), "reward": 1.0, "trajectory_id": "provider-row"}
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist([raw]), path)
    projected = list(provider_rows(str(path), PROVIDER_COLUMNS[ProviderTransform.CODERFORGE], 8))
    text = (
        "This trajectory solved the task successfully.\n\n<user>\nFix bug\n</user>\n\n"
        "<assistant>\n<tool_call:bash>\n  command: pytest\n</tool_call:bash>\n</assistant>"
    )
    selected_id = xxhash.xxh3_128_hexdigest(text.encode())
    recipe = _recipe(ProviderTransform.CODERFORGE)
    result = reconstruct_selected_document(recipe, projected, selected_id)
    assert (result["text"], result["source_id"]) == (text, hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**raw, "reward": 0.0}], selected_id)
