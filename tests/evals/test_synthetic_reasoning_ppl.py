# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from experiments.evals.synthetic_reasoning_ppl import (
    DEV_SEED_BASE,
    SYNTHETIC_REASONING_PPL_SLICES,
    SYNTHETIC_REASONING_SOURCE_COMMIT,
    SyntheticReasoningPplConfig,
    generate_synthetic_reasoning_ppl,
    synthetic_reasoning_raw_validation_sets,
)


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_synthetic_reasoning_raw_validation_sets_render_deterministic_paths_and_tags() -> None:
    datasets = synthetic_reasoning_raw_validation_sets(raw_root="gs://example-bucket/raw/synthetic_reasoning")

    stepmath_key = "synthetic_reasoning_ppl/stepmath/arithmetic/canonical_json"
    clrs_key = "synthetic_reasoning_ppl/clrs_style/clrs_binary_search/oai_chat_symbolic"

    assert datasets[stepmath_key].input_path == (
        "gs://example-bucket/raw/synthetic_reasoning/stepmath/arithmetic/canonical_json.jsonl.gz"
    )
    assert datasets[stepmath_key].tags == (
        "synthetic_reasoning_ppl",
        "epic:5005",
        "issue:5052",
        "family:stepmath",
        "task:arithmetic",
        "renderer:canonical_json",
        f"seed_range:{DEV_SEED_BASE}:{DEV_SEED_BASE + 8}",
    )

    assert datasets[clrs_key].input_path == (
        "gs://example-bucket/raw/synthetic_reasoning/clrs_style/clrs_binary_search/oai_chat_symbolic.jsonl.gz"
    )
    assert datasets[clrs_key].tags == (
        "synthetic_reasoning_ppl",
        "epic:5005",
        "issue:5052",
        "family:clrs_style",
        "task:clrs_binary_search",
        "renderer:oai_chat_symbolic",
        f"seed_range:{DEV_SEED_BASE + 24}:{DEV_SEED_BASE + 32}",
    )


def test_generate_synthetic_reasoning_ppl_is_deterministic_and_held_out(tmp_path: Path) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_cfg = SyntheticReasoningPplConfig(output_path=str(first_output), examples_per_slice=3)
    second_cfg = SyntheticReasoningPplConfig(output_path=str(second_output), examples_per_slice=3)

    first_manifest = generate_synthetic_reasoning_ppl(first_cfg)
    second_manifest = generate_synthetic_reasoning_ppl(second_cfg)

    assert first_manifest["source_commit"] == SYNTHETIC_REASONING_SOURCE_COMMIT
    assert second_manifest["source_commit"] == SYNTHETIC_REASONING_SOURCE_COMMIT

    first_records = _read_jsonl_gz(first_output / "stepmath" / "arithmetic" / "canonical_json.jsonl.gz")
    second_records = _read_jsonl_gz(second_output / "stepmath" / "arithmetic" / "canonical_json.jsonl.gz")

    assert [record["text"] for record in first_records] == [record["text"] for record in second_records]
    assert [record["metadata"]["seed"] for record in first_records] == [
        DEV_SEED_BASE,
        DEV_SEED_BASE + 1,
        DEV_SEED_BASE + 2,
    ]
    assert all(record["metadata"]["seed"] >= DEV_SEED_BASE for record in first_records)
    assert first_records[0]["metadata"]["seed_range"] == {"start": DEV_SEED_BASE, "stop": DEV_SEED_BASE + 8}


def test_generate_synthetic_reasoning_ppl_renders_distinct_surface_forms(tmp_path: Path) -> None:
    cfg = SyntheticReasoningPplConfig(output_path=str(tmp_path), examples_per_slice=2)
    manifest = generate_synthetic_reasoning_ppl(cfg)

    canonical_record = _read_jsonl_gz(tmp_path / "stepmath" / "arithmetic" / "canonical_json.jsonl.gz")[0]
    chat_record = _read_jsonl_gz(tmp_path / "stepmath" / "arithmetic" / "oai_chat_symbolic.jsonl.gz")[0]

    assert canonical_record["id"] == chat_record["id"]
    assert canonical_record["metadata"]["seed"] == chat_record["metadata"]["seed"]
    assert canonical_record["metadata"]["renderer"] == "canonical_json"
    assert chat_record["metadata"]["renderer"] == "oai_chat_symbolic"
    assert canonical_record["text"] != chat_record["text"]
    assert "steps" in canonical_record["surface"]
    assert "messages" in chat_record["surface"]
    assert chat_record["surface"]["messages"][0]["content"] == canonical_record["surface"]["prompt"]
    assert chat_record["surface"]["messages"][1]["content"].startswith("Step-by-step solution:")

    assert {slice_info["registry_key"] for slice_info in manifest["slices"]} == {
        slice_.registry_key for slice_ in SYNTHETIC_REASONING_PPL_SLICES
    }
    assert all(Path(slice_info["output_file"]).exists() for slice_info in manifest["slices"])
