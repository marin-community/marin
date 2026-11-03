# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash
from marin.datakit.download.penfever_rollouts import PenfeverRollout

from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    reconstruct_selected_document,
    reconstruction_code_digests,
)
from experiments.datakit.store.reconstruct_selected_source import PROVIDER_COLUMNS, provider_rows


def _recipe(transform):
    return ReconstructionRecipe(transform, "text", "id", 128, *reconstruction_code_digests(transform))


def test_penfever_projected_original_replays_outcome_and_explicit_dataset(tmp_path):
    dataset = PenfeverRollout("cohort", "Teacher", "task", "provider/repo", "frozen-revision")
    row = {
        "conversations": [{"role": "user", "content": "Fix"}, {"role": "assistant", "content": "Done"}],
        "result": "1.0",
        "verifier_output": "VERIFIER: FAIL",
    }
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    rows = list(provider_rows(str(path), PROVIDER_COLUMNS[ProviderTransform.PENFEVER], 8))
    text = "This trajectory solved the task successfully.\n\n<user>\nFix\n</user>\n\n<assistant>\nDone\n</assistant>"
    expected = xxhash.xxh3_128_hexdigest(text.encode())
    recipe = replace(_recipe(ProviderTransform.PENFEVER), penfever_dataset=dataset)
    doc = reconstruct_selected_document(recipe, rows, expected)
    assert (doc["text"], doc["source_id"], doc["teacher"], doc["task_source"], doc["source"]) == (
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        "Teacher",
        "task",
        "provider/repo",
    )
    with pytest.raises(ValueError, match="explicit frozen dataset"):
        reconstruct_selected_document(replace(recipe, penfever_dataset=None), rows, expected)
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**row, "result": "0"}], expected)
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**row, "conversations": []}], expected)


def test_synthetic_original_replays_score_and_strips_think_tags(tmp_path):
    row = {"prompt": "Problem", "llm_response": "<think>Reason</think> Answer", "score": 0.5}
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    rows = list(provider_rows(str(path), PROVIDER_COLUMNS[ProviderTransform.SYNTHETIC1], 8))
    text = "Problem\n\nThis is a partially correct solution.\n\nReason Answer"
    expected = xxhash.xxh3_128_hexdigest(text.encode())
    recipe = _recipe(ProviderTransform.SYNTHETIC1)
    doc = reconstruct_selected_document(recipe, rows, expected)
    assert (doc["text"], doc["source_id"]) == (text, hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**row, "score": 1.0}], expected)
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**row, "llm_response": "<think></think>"}], expected)
