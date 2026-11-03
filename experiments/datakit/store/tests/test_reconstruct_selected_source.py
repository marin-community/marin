# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash

from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    reconstruction_code_digests,
)
from experiments.datakit.store.reconstruct_selected_source import reconstruct_selected_source


def test_provider_replay_preserves_selected_multiplicity_without_adding_provider_duplicates(tmp_path):
    expected_id = xxhash.xxh3_128_hexdigest(b"selected text")
    selected = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "example", "id": expected_id}] * 3), selected)
    raw = tmp_path / "original.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"text": "unselected"}, {"text": "selected text"}, {"text": "selected text"}]), raw
    )
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.DIRECT)
    recipe = ReconstructionRecipe(ProviderTransform.DIRECT, "text", "id", 128, normalizer, adapter)
    destination = tmp_path / "reconstructed"
    report = reconstruct_selected_source(
        [str(selected)], [str(raw)], str(destination), marin_source="example", recipe=recipe, batch_size=2
    )
    assert (
        pq.read_table(destination / "text.parquet").to_pylist()
        == [{"marin_source": "example", "id": expected_id, "text": "selected text"}] * 3
    )
    assert report["reconstructed_occurrences"] == 3
    assert report["ids_with_multiple_provider_matches"] == 1
    assert json.loads((destination / "_complete.json").read_text())["missing_occurrences"] == 0


def test_provider_replay_missing_selected_text_never_marks_complete(tmp_path):
    selected = tmp_path / "selected.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"marin_source": "example", "id": xxhash.xxh3_128_hexdigest(b"missing")}]), selected
    )
    raw = tmp_path / "original.jsonl"
    raw.write_text(json.dumps({"text": "different"}) + "\n")
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.DIRECT)
    recipe = ReconstructionRecipe(ProviderTransform.DIRECT, "text", "id", 128, normalizer, adapter)
    destination = tmp_path / "reconstructed"
    with pytest.raises(ValueError, match="unresolved"):
        reconstruct_selected_source([str(selected)], [str(raw)], str(destination), marin_source="example", recipe=recipe)
    assert not (destination / "_complete.json").exists()
    assert json.loads((destination / "report.json").read_text())["missing_occurrences"] == 1


def test_davinci_jsonl_replay_projects_original_messages_and_preserves_selection(tmp_path):
    text = "This trajectory failed to solve the task.\n\n<user>\nRepair it\n</user>\n\n<assistant>\nTried\n</assistant>"
    expected_id = xxhash.xxh3_128_hexdigest(text.encode())
    selected = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "davinci-dev/env-native", "id": expected_id}] * 2), selected)
    raw = tmp_path / "env-native.jsonl"
    raw.write_text(
        json.dumps(
            {
                "success": False,
                "messages": [{"role": "user", "content": "Repair it"}, {"role": "assistant", "content": "Tried"}],
            }
        )
        + "\n"
    )
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.DAVINCI_ENV)
    recipe = ReconstructionRecipe(ProviderTransform.DAVINCI_ENV, "text", "id", 128, normalizer, adapter)
    destination = tmp_path / "reconstructed"
    report = reconstruct_selected_source(
        [str(selected)], [str(raw)], str(destination), marin_source="davinci-dev/env-native", recipe=recipe
    )
    assert (
        pq.read_table(destination / "text.parquet").to_pylist()
        == [{"marin_source": "davinci-dev/env-native", "id": expected_id, "text": text}] * 2
    )
    assert report["missing_occurrences"] == 0


def test_davinci_context_parquet_replay_projects_original_structured_fields(tmp_path):
    raw = tmp_path / "original-context.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "repo_name": "r",
                    "repo_description": "d",
                    "title": "Fix",
                    "body": "Patch",
                    "related_issue": None,
                    "pr_summary": "Summary",
                    "relevant_files": None,
                    "commits": None,
                }
            ]
        ),
        raw,
    )
    text = "# Repository Context\nName: r\nDescription: d\n\n# Pull Request\n## Fix\nPatch\n\n# Summary\nSummary"
    expected_id = xxhash.xxh3_128_hexdigest(text.encode())
    selected = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "davinci-dev/ctx-native", "id": expected_id}]), selected)
    n, t = reconstruction_code_digests(ProviderTransform.DAVINCI_CTX)
    recipe = ReconstructionRecipe(ProviderTransform.DAVINCI_CTX, "text", "id", 128, n, t)
    destination = tmp_path / "reconstructed"
    report = reconstruct_selected_source(
        [str(selected)], [str(raw)], str(destination), marin_source="davinci-dev/ctx-native", recipe=recipe
    )
    assert pq.read_table(destination / "text.parquet").to_pylist() == [
        {"marin_source": "davinci-dev/ctx-native", "id": expected_id, "text": text}
    ]
    assert report["missing_occurrences"] == 0


def test_gpt_oss_jsonl_projection_reconstructs_selected_occurrences(tmp_path):
    raw = tmp_path / "original.jsonl"
    raw.write_text(
        json.dumps({"user_content": "Question", "assistant_content": "Answer", "assistant_thinking": "Check"}) + "\n"
    )
    text = "<user>\nQuestion\n</user>\n\n<thinking>\nCheck\n</thinking>\n\n<assistant>\nAnswer\n</assistant>"
    expected_id = xxhash.xxh3_128_hexdigest(text.encode())
    selected = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "gpt-oss-rollouts", "id": expected_id}] * 2), selected)
    n, t = reconstruction_code_digests(ProviderTransform.GPT_OSS)
    recipe = ReconstructionRecipe(ProviderTransform.GPT_OSS, "text", "id", 128, n, t)
    destination = tmp_path / "reconstructed"
    report = reconstruct_selected_source(
        [str(selected)], [str(raw)], str(destination), marin_source="gpt-oss-rollouts", recipe=recipe
    )
    assert (
        pq.ParquetFile(destination / "text.parquet").read().to_pylist()
        == [{"marin_source": "gpt-oss-rollouts", "id": expected_id, "text": text}] * 2
    )
    assert report["missing_occurrences"] == 0
