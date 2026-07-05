# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from scripts.analysis.stage_math_decontam_sources import (
    SourceSpec,
    render_aime24,
    render_gsm8k,
    render_math500,
    stage_source,
    write_manifest,
)


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_render_math_sources_include_problem_and_answer() -> None:
    math500_text = render_math500(
        {
            "problem": "Convert $(0,3)$ to polar coordinates.",
            "solution": "The radius is 3 and the angle is pi/2.",
            "answer": "(3, pi/2)",
        }
    )
    gsm8k_text = render_gsm8k({"question": "How many eggs?", "answer": "#### 18"})
    aime24_text = render_aime24({"problem": "Find n.", "solution": "n=204.", "answer": "204"})

    assert "Problem:\nConvert $(0,3)$" in math500_text
    assert "Answer:\n(3, pi/2)" in math500_text
    assert gsm8k_text == "Question:\nHow many eggs?\n\nAnswer:\n#### 18"
    assert "Problem:\nFind n." in aime24_text
    assert "Answer:\n204" in aime24_text


def test_stage_source_writes_datakit_ready_jsonl(tmp_path: Path) -> None:
    spec = SourceSpec(
        name="math500",
        dataset_id="HuggingFaceH4/MATH-500",
        revision="test-revision",
        split="test",
        output_subdir="math500/test",
        renderer=render_math500,
        id_fields=("unique_id",),
    )
    records = [
        {
            "problem": "Problem 1",
            "solution": "Solution 1",
            "answer": "1",
            "unique_id": "test/algebra/1.json",
        }
    ]

    result = stage_source(spec=spec, output_root=str(tmp_path), force=False, records=records)

    output_file = tmp_path / "math500" / "test" / "data.jsonl.gz"
    staged = _read_jsonl_gz(output_file)
    assert result["record_count"] == 1
    assert staged == [
        {
            "id": "math500:test:test/algebra/1.json",
            "text": "Problem:\nProblem 1\n\nSolution:\nSolution 1\n\nAnswer:\n1",
            "source": "math500",
            "provenance": {
                "dataset_id": "HuggingFaceH4/MATH-500",
                "revision": "test-revision",
                "config": None,
                "split": "test",
                "index": 0,
            },
        }
    ]


def test_stage_source_skips_existing_output_without_force(tmp_path: Path) -> None:
    spec = SourceSpec(
        name="aime24",
        dataset_id="HuggingFaceH4/aime_2024",
        revision="test-revision",
        split="train",
        output_subdir="aime24/train",
        renderer=render_aime24,
        id_fields=("id",),
    )

    stage_source(
        spec=spec,
        output_root=str(tmp_path),
        force=False,
        records=[{"id": "1", "problem": "First", "solution": "", "answer": "1"}],
    )
    skipped = stage_source(
        spec=spec,
        output_root=str(tmp_path),
        force=False,
        records=[{"id": "2", "problem": "Second", "solution": "", "answer": "2"}],
    )

    staged = _read_jsonl_gz(tmp_path / "aime24" / "train" / "data.jsonl.gz")
    assert skipped["status"] == "skipped_existing"
    assert staged[0]["id"] == "aime24:train:1"


def test_write_manifest_uses_hidden_json_sidecar(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "force": False,
            "max_records_per_split": None,
            "skip_aime24": False,
        },
    )()
    manifest_path = write_manifest(str(tmp_path), [{"name": "gsm8k"}], args)

    assert Path(manifest_path).name == ".manifest.json"
    payload = json.loads(Path(manifest_path).read_text())
    assert payload["entries"] == [{"name": "gsm8k"}]
    assert payload["output_root"] == str(tmp_path)
