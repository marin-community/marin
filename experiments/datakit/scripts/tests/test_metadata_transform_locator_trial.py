# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw row locators survive production transforms and normalized membership."""

import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit import partition_filename
from marin.datakit.download import (
    biodiversity,
    finetranslations,
    institutional_books,
    stack_v3,
    swe_rebench_contree,
    swe_zero_12m,
)
from marin.datakit.normalize import generate_id

from experiments.datakit.scripts.metadata_source_replay import SourceReplay
from experiments.datakit.scripts.metadata_transform_locator_trial import (
    TransformReplay,
    run_biodiversity_candidate_replay,
    run_transform_replay,
)


def _recipe(tmp_path, source, module, raw_paths, expected_text, *, complete=False):
    normalized = tmp_path / "normalized"
    output = normalized / "outputs" / "main"
    output.mkdir(parents=True)
    pq.write_table(pa.table({"id": [generate_id(expected_text)]}), output / partition_filename(0, 1))
    module_path = str(Path(module.__file__).resolve())
    return TransformReplay(
        normalization=SourceReplay(
            source=source,
            input_paths=[],
            normalized_output=str(normalized),
            num_partitions=1,
            normalization_parameters={"text_field": "text", "id_field": "id", "max_whitespace_run_chars": 100},
            input_provenance="synthetic provider schema fixture",
            recipe_commit="fixture",
        ),
        raw_paths=[str(path) for path in raw_paths],
        transform_module_path=module_path,
        transform_module_sha256=hashlib.sha256(Path(module_path).read_bytes()).hexdigest(),
        raw_split_manifest_complete=complete,
    )


@pytest.mark.parametrize(
    "source,module,raw,text",
    [
        (
            "finetranslations",
            finetranslations,
            {"id": "a", "og_full_text": "Bonjour", "translated_text": "Hello"},
            "Hello\n\nBonjour",
        ),
        (
            "institutional_books",
            institutional_books,
            {"text_by_page_gen": ["one", "", "two"], "text_by_page_src": ["bad"]},
            "one\n\ntwo",
        ),
        ("swe-rebench-contree", swe_rebench_contree, {"text": "trace"}, "trace"),
        ("swe-zero-12m", swe_zero_12m, {"messages": [{"role": "user", "content": "task"}]}, "<user>\ntask\n</user>"),
        (
            "stack-v3",
            stack_v3,
            {
                "repo_path": "org/repo",
                "repo_id": 7,
                "commit_id": "commit",
                "github_metadata": {"branch": "main"},
                "files": [
                    {"file_path": "b.py", "content_id": "b", "content": "B"},
                    {"file_path": "a.py", "content_id": "a", "content": "A"},
                ],
            },
            "Repository: org/repo\n\nFile: a.py\nA\n\nFile: b.py\nB",
        ),
    ],
)
def test_transform_replay_preserves_physical_offsets_for_repeated_text(tmp_path, source, module, raw, text):
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist([raw, raw]), path)
    recipe = _recipe(tmp_path, source, module, [path], text)
    report = run_transform_replay(recipe, max_raw_rows=4, max_documents=2, max_projected_bytes=1024 * 1024)
    assert report["status"] == "passed"
    assert [sample["input_locators"] for sample in report["samples"]] == [
        [{"input_path": str(path), "input_row": 0}],
        [{"input_path": str(path), "input_row": 1}],
    ]
    assert [sample["normalized_row"] for sample in report["samples"]] == [0, 0]
    assert all("text" not in sample for sample in report["samples"])


def test_biodiversity_requires_complete_group_across_files(tmp_path):
    paths = [tmp_path / "part0.json.gz", tmp_path / "part1.json.gz"]
    pages = [
        [{"item_id": "book", "page_num": "0002", "text": "two"}],
        [
            {"item_id": "noise", "page_num": "0001", "text": "noise"},
            {"item_id": "book", "page_num": "0001", "text": "one"},
        ],
    ]
    for path, rows in zip(paths, pages, strict=True):
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    recipe = _recipe(tmp_path, "cp/biodiversity", biodiversity, paths, "one\n\ntwo", complete=True)
    incomplete = run_transform_replay(recipe, max_raw_rows=2, max_documents=1, max_projected_bytes=1024 * 1024)
    assert incomplete["status"] == "incomplete" and incomplete["samples"] == []
    complete = run_transform_replay(recipe, max_raw_rows=3, max_documents=1, max_projected_bytes=1024 * 1024)
    assert complete["status"] == "passed"
    assert complete["samples"][0]["input_locators"] == [
        {"input_path": str(paths[1]), "input_row": 1},
        {"input_path": str(paths[0]), "input_row": 0},
    ]


@pytest.mark.parametrize("retained_item", ["book", "unrelated-book"])
def test_biodiversity_candidate_hash_requires_item_identity(tmp_path, retained_item):
    path = tmp_path / "pages.json.gz"
    pages = [
        {"item_id": "book", "page_num": "0001", "text": "one"},
        {"item_id": "book", "page_num": "0002", "text": "two"},
        # A truncated different book happens to have identical prefix text.
        {"item_id": "partial-book", "page_num": "0001", "text": "one\n\ntwo"},
        {"item_id": "partial-book", "page_num": "0002", "text": "unread rest"},
    ]
    with gzip.open(path, "wt") as stream:
        for row in pages:
            stream.write(json.dumps(row) + "\n")
    recipe = _recipe(tmp_path, "cp/biodiversity", biodiversity, [path], "one\n\ntwo")
    normalized_path = Path(recipe.normalization.normalized_output) / "outputs/main" / partition_filename(0, 1)
    pq.write_table(pa.table({"id": [generate_id("one\n\ntwo")], "source_id": [retained_item]}), normalized_path)
    report = run_biodiversity_candidate_replay(
        recipe,
        max_raw_rows=3,
        max_documents=2,
        max_projected_bytes=1024 * 1024,
    )
    assert report["raw_budget_exhausted"]
    assert report["candidate_books"] == 2
    assert report["verified_books"] == (1 if retained_item == "book" else 0)
    if retained_item == "book":
        assert report["status"] == "passed"
        assert report["samples"][0]["proof"] == "hash_and_item_identity_verified"
        assert report["samples"][0]["input_locators"] == [
            {"input_path": str(path), "input_row": 0},
            {"input_path": str(path), "input_row": 1},
        ]
    else:
        assert report["status"] == "incomplete"
    assert any(row["item_id"] == "partial-book" for row in report["rejected_candidates"])


def test_biodiversity_strict_rejects_ambiguous_pages_while_candidate_retains_verified_books(tmp_path):
    path = tmp_path / "pages.json.gz"
    pages = [
        {"item_id": "book", "page_num": "0002", "text": "two"},
        {"item_id": "book", "page_num": "0001", "text": "one"},
        {"item_id": "ambiguous", "page_num": "0001", "text": "a"},
        {"item_id": "ambiguous", "page_num": "0001", "text": "b"},
    ]
    with gzip.open(path, "wt") as stream:
        for row in pages:
            stream.write(json.dumps(row) + "\n")
    recipe = _recipe(tmp_path, "cp/biodiversity", biodiversity, [path], "one\n\ntwo", complete=True)
    normalized_path = Path(recipe.normalization.normalized_output) / "outputs/main" / partition_filename(0, 1)
    pq.write_table(pa.table({"id": [generate_id("one\n\ntwo")], "source_id": ["book"]}), normalized_path)
    with pytest.raises(ValueError, match="ambiguous duplicate page_num"):
        run_transform_replay(recipe, max_raw_rows=4, max_documents=2, max_projected_bytes=1024 * 1024)
    result = run_biodiversity_candidate_replay(recipe, max_raw_rows=4, max_documents=2, max_projected_bytes=1024 * 1024)
    assert result["status"] == "passed"
    assert [sample["item_id"] for sample in result["samples"]] == ["book"]
    assert result["samples"][0]["input_locators"] == [
        {"input_path": str(path), "input_row": 1},
        {"input_path": str(path), "input_row": 0},
    ]
    assert result["rejected_candidates"] == [{"item_id": "ambiguous", "reason": "ambiguous_page_order"}]
