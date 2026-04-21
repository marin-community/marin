# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.common_pile.stitch_stackv2 import (
    StitchStackV2Config,
    _dfs_sort_key,
    _file_path,
    _repo_key,
    stitch_stackv2_repos,
)

REPO_A_URL = "https://raw.githubusercontent.com/alice/repo_a/abc123"
REPO_A_ALT_COMMIT_URL = "https://raw.githubusercontent.com/alice/repo_a/deadbee"
REPO_B_URL = "https://raw.githubusercontent.com/bob/repo_b/ffffff"


def _record(url_prefix: str, path: str, text: str) -> dict:
    """Build a minimal Stack v2 record with the ``metadata.url`` and ``metadata.path`` fields."""
    return {
        "id": f"{url_prefix}/{path}",
        "text": text,
        "metadata": {"url": f"{url_prefix}/{path}", "path": path},
    }


def test_repo_key_extracts_owner_repo_commit() -> None:
    record = _record(REPO_A_URL, "src/main.py", "print('hi')")
    assert _repo_key(record) == "alice/repo_a@abc123"


def test_repo_key_returns_none_when_url_missing_or_malformed() -> None:
    assert _repo_key({"metadata": {}}) is None
    assert _repo_key({"metadata": {"url": ""}}) is None
    # Path too short: no commit segment.
    assert _repo_key({"metadata": {"url": "https://raw.githubusercontent.com/alice/repo_a"}}) is None


def test_file_path_falls_back_to_url_tail_when_path_absent() -> None:
    record = {"metadata": {"url": f"{REPO_A_URL}/src/deep/main.py"}}
    assert _file_path(record) == "src/deep/main.py"


def test_dfs_sort_key_orders_files_depth_first() -> None:
    # Mix of top-level files, nested subdirs, and adjacent siblings — exercise the
    # tuple-comparison rules that power DFS ordering.
    paths = [
        "z/e.py",
        "a.py",
        "sub/nested/d.py",
        "sub/b.py",
        "sub/c.py",
    ]
    records = [_record(REPO_A_URL, p, "") for p in paths]
    records.sort(key=_dfs_sort_key)
    assert [r["metadata"]["path"] for r in records] == [
        "a.py",
        "sub/b.py",
        "sub/c.py",
        "sub/nested/d.py",
        "z/e.py",
    ]


def test_dfs_sort_key_enters_subdirectory_before_sibling_file() -> None:
    # A directory ``a/`` precedes a sibling file ``a.py`` because the tuple ('a', 'c.py')
    # compares less than ('a.py',) — matches how a DFS keyed on sorted directory entries
    # would visit the tree.
    records = [_record(REPO_A_URL, p, "") for p in ["a.py", "a/c.py"]]
    records.sort(key=_dfs_sort_key)
    assert [r["metadata"]["path"] for r in records] == ["a/c.py", "a.py"]


def test_stitch_emits_one_record_per_repo_in_dfs_order(
    tmp_path: Path,
    write_jsonl_gz,
    read_all_jsonl_gz,
) -> None:
    """End-to-end: two repos split across shards; verify DFS stitching and metadata."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Shard the same repo across two input files to confirm group_by collapses across shards.
    write_jsonl_gz(
        input_dir / "stack-0000.jsonl.gz",
        [
            _record(REPO_A_URL, "z/late.py", "Z_LATE"),
            _record(REPO_A_URL, "sub/b.py", "SUB_B"),
            _record(REPO_B_URL, "only.py", "ONLY"),
            # Unparseable URL — should be dropped rather than crash the pipeline.
            {"id": "garbage", "text": "ignored", "metadata": {"url": "not-a-url"}},
        ],
    )
    write_jsonl_gz(
        input_dir / "stack-0001.jsonl.gz",
        [
            _record(REPO_A_URL, "a.py", "A"),
            _record(REPO_A_URL, "sub/nested/d.py", "SUB_NESTED_D"),
            _record(REPO_A_URL, "sub/c.py", "SUB_C"),
        ],
    )

    stitch_stackv2_repos(
        StitchStackV2Config(
            input_path=str(input_dir),
            output_path=str(output_dir),
            input_glob="*.jsonl.gz",
            file_header="### {path}\n",
            separator="\n---\n",
        )
    )

    stitched = read_all_jsonl_gz(output_dir)
    by_repo = {r["metadata"]["repo"]: r for r in stitched}
    assert set(by_repo) == {"alice/repo_a@abc123", "bob/repo_b@ffffff"}

    repo_a = by_repo["alice/repo_a@abc123"]
    assert repo_a["id"] == "alice/repo_a@abc123"
    assert repo_a["metadata"]["n_files"] == 5
    assert repo_a["metadata"]["paths"] == [
        "a.py",
        "sub/b.py",
        "sub/c.py",
        "sub/nested/d.py",
        "z/late.py",
    ]
    # Bodies are concatenated in DFS order with per-file headers between them.
    assert repo_a["text"] == (
        "### a.py\nA"
        "\n---\n"
        "### sub/b.py\nSUB_B"
        "\n---\n"
        "### sub/c.py\nSUB_C"
        "\n---\n"
        "### sub/nested/d.py\nSUB_NESTED_D"
        "\n---\n"
        "### z/late.py\nZ_LATE"
    )

    repo_b = by_repo["bob/repo_b@ffffff"]
    assert repo_b["metadata"]["n_files"] == 1
    assert repo_b["metadata"]["paths"] == ["only.py"]


def test_stitch_commit_pinning_splits_same_repo_across_commits(
    tmp_path: Path,
    write_jsonl_gz,
    read_all_jsonl_gz,
) -> None:
    """Different commits of the same ``owner/repo`` must produce separate stitched docs."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_jsonl_gz(
        input_dir / "stack-0000.jsonl.gz",
        [
            _record(REPO_A_URL, "main.py", "OLD"),
            _record(REPO_A_ALT_COMMIT_URL, "main.py", "NEW"),
        ],
    )

    stitch_stackv2_repos(
        StitchStackV2Config(
            input_path=str(input_dir),
            output_path=str(output_dir),
            input_glob="*.jsonl.gz",
        )
    )

    stitched = read_all_jsonl_gz(output_dir)
    repos = {r["metadata"]["repo"] for r in stitched}
    assert repos == {"alice/repo_a@abc123", "alice/repo_a@deadbee"}


def test_stitch_splits_mega_repo_when_max_chars_set(
    tmp_path: Path,
    write_jsonl_gz,
    read_all_jsonl_gz,
) -> None:
    """With ``max_chars_per_repo`` set, a single repo is split into chunked records."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    # Each file body is 50 chars. Headers add ~15. With max_chars=80, each file should
    # end up in its own chunk (second file triggers a flush).
    body = "x" * 50
    write_jsonl_gz(
        input_dir / "stack-0000.jsonl.gz",
        [
            _record(REPO_A_URL, f"dir/file_{i}.py", body) for i in range(3)
        ],
    )

    stitch_stackv2_repos(
        StitchStackV2Config(
            input_path=str(input_dir),
            output_path=str(output_dir),
            input_glob="*.jsonl.gz",
            max_chars_per_repo=80,
        )
    )

    stitched = sorted(read_all_jsonl_gz(output_dir), key=lambda r: r["metadata"]["chunk_idx"])
    assert len(stitched) == 3
    assert [r["metadata"]["chunk_idx"] for r in stitched] == [0, 1, 2]
    # IDs are decorated with ``#<chunk_idx>`` when splitting is active.
    assert [r["id"] for r in stitched] == [
        "alice/repo_a@abc123#0",
        "alice/repo_a@abc123#1",
        "alice/repo_a@abc123#2",
    ]
    # Each chunk retains the repo key and records exactly the file(s) it contains.
    assert all(r["metadata"]["repo"] == "alice/repo_a@abc123" for r in stitched)
    assert [r["metadata"]["paths"] for r in stitched] == [
        ["dir/file_0.py"],
        ["dir/file_1.py"],
        ["dir/file_2.py"],
    ]


@pytest.mark.parametrize(
    "bad_record",
    [
        {"id": "no_metadata", "text": "body"},
        {"id": "empty_url", "text": "body", "metadata": {"url": ""}},
        {"id": "path_missing_url_too_short", "text": "body", "metadata": {"url": "https://host/only"}},
    ],
)
def test_stitch_drops_records_without_parseable_repo(
    bad_record: dict,
    tmp_path: Path,
    write_jsonl_gz,
    read_all_jsonl_gz,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_jsonl_gz(
        input_dir / "stack-0000.jsonl.gz",
        [
            _record(REPO_A_URL, "main.py", "GOOD"),
            bad_record,
        ],
    )

    stitch_stackv2_repos(
        StitchStackV2Config(
            input_path=str(input_dir),
            output_path=str(output_dir),
            input_glob="*.jsonl.gz",
        )
    )

    stitched = read_all_jsonl_gz(output_dir)
    assert len(stitched) == 1
    assert stitched[0]["metadata"]["repo"] == "alice/repo_a@abc123"
