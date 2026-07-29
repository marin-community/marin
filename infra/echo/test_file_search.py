# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo's tracked-file hybrid index."""

import subprocess
from pathlib import Path

import file_search


class FakeEmbedder:
    def __init__(self) -> None:
        self.passages: list[str] = []

    def __call__(self, mode: str, texts: list[str]) -> list[list[float]]:
        if mode == "query":
            return [[1.0, 0.0] for _ in texts]
        self.passages.extend(texts)
        return [[1.0, 0.0] if "collective diagnosis" in text else [0.0, 1.0] for text in texts]


def tracked_repository(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for relative, contents in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            path.write_bytes(contents)
        else:
            path.write_text(contents)
    subprocess.run(["git", "add", "-f", "--", *files], cwd=tmp_path, check=True)
    return tmp_path


def test_file_search_combines_exact_and_semantic_results(tmp_path):
    root = tracked_repository(
        tmp_path,
        {
            "docs/collectives.md": "# Collective debugging\n\ncollective diagnosis starts with topology.",
            "lib/iris/src/worker.py": "def handle_FAILED_PRECONDITION():\n    pass\n",
            "notes/unrelated.md": "# Lunch\n\nSandwich notes.",
        },
    )
    embedder = FakeEmbedder()
    results = file_search.search_files(
        "FAILED_PRECONDITION",
        10,
        embedder,
        root=root,
        index_path=tmp_path / "index.sqlite3",
    )

    assert [result["url"] for result in results] == [
        "lib/iris/src/worker.py",
        "docs/collectives.md",
    ]
    assert results[0]["title"] == "worker.py"
    assert results[0]["subtitle"] == "lib/iris/src/worker.py:1"
    assert results[0]["lexical_score"] is not None
    assert results[1]["distance"] == 0.0
    assert all(result["url"] != "notes/unrelated.md" for result in results)


def test_file_index_excludes_untracked_generated_vendored_secret_and_oversized_files(tmp_path):
    root = tracked_repository(
        tmp_path,
        {
            "src/kept.py": "def kept():\n    return True\n",
            "vendor/copied.py": "def copied():\n    return True\n",
            "config/external/model.py": "MODEL = 'vendored'\n",
            "keys/deploy.pem": "private material",
            "package-lock.json": "{}",
            "docs/huge.md": b"x" * (file_search.MAX_FILE_BYTES + 1),
        },
    )
    (root / "untracked.py").write_text("UNTRACKED_SECRET = True\n")
    embedder = FakeEmbedder()
    index_path = tmp_path / "index.sqlite3"
    with file_search.connect_index(index_path) as conn:
        file_search.refresh_index(root, conn, embedder)
        indexed_paths = {row[0] for row in conn.execute("SELECT DISTINCT path FROM chunks")}

    assert indexed_paths == {"src/kept.py"}
    assert all("private material" not in passage for passage in embedder.passages)


def test_file_index_embeds_only_changed_files(tmp_path):
    root = tracked_repository(tmp_path, {"docs/runbook.md": "# Runbook\n\nFirst version.\n"})
    embedder = FakeEmbedder()
    index_path = tmp_path / "index.sqlite3"
    with file_search.connect_index(index_path) as conn:
        assert file_search.refresh_index(root, conn, embedder) == 1
        assert file_search.refresh_index(root, conn, embedder) == 0
        (root / "docs/runbook.md").write_text("# Runbook\n\nSecond version.\n")
        assert file_search.refresh_index(root, conn, embedder) == 1

    assert len(embedder.passages) == 2
