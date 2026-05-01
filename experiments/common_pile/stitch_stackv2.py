# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stitch Common Pile's ``common-pile/stackv2`` file-level records into whole-repo documents.

The base ``common-pile/stackv2`` dataset (the SWH-keyed variant) ships ``metadata.url`` of
the form ``https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>`` and a matching
``metadata.path``. We group on ``<owner>/<repo>@<commit>`` and concatenate each repo's files
in depth-first traversal order — siblings stay adjacent, and descendants of a directory
appear before later siblings of its parent. The result is a single document per repo
(optionally split into chunks for very large repos) intended for long-context training.

The filtered variants (``stackv2_edu_filtered``, ``stackv2_html_filtered``) strip ``url``/
``path`` and are **not** supported by this transform — recovering repo identity there would
require an out-of-band Software Heritage lookup.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from urllib.parse import urlparse

from zephyr import Dataset, ZephyrContext, load_jsonl

from experiments.common_pile.tokenize_common_pile import stackv2
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

logger = logging.getLogger(__name__)

# ``raw.githubusercontent.com`` paths look like ``/<owner>/<repo>/<commit>/<path>`` after the
# hostname is stripped, so splitting on ``/`` yields: ["", owner, repo, commit, *path_parts].
_URL_OWNER_IDX = 1
_URL_REPO_IDX = 2
_URL_COMMIT_IDX = 3
_URL_PATH_START = 4


@dataclass(frozen=True)
class StitchStackV2Config:
    """Configuration for DFS-stitching Stack v2 file-level records into repo documents.

    Attributes:
        input_path: Directory of ``common-pile/stackv2`` JSONL shards.
        output_path: Destination directory for stitched JSONL shards.
        input_glob: Glob (relative to ``input_path``) selecting input shards.
        separator: String inserted between adjacent file bodies in the stitched document.
        file_header: ``str.format`` template (with ``{path}``) prepended before each file body.
        max_chars_per_repo: When set, a repo that accumulates more than this many characters
            is split into multiple chunked records (still in DFS order). ``None`` emits one
            record per repo regardless of size.
        min_files_per_repo: Repos (or chunks) with fewer than this many files are dropped.
    """

    input_path: str
    output_path: str
    input_glob: str = "*.json*"
    separator: str = "\n\n"
    file_header: str = "# === {path} ===\n"
    max_chars_per_repo: int | None = None
    min_files_per_repo: int = 1


def _repo_key(record: dict) -> str | None:
    """Return ``<owner>/<repo>@<commit>`` for the record, or ``None`` if unparseable.

    The commit hash is included in the key so that multiple snapshots of the same repo
    (if present) don't collapse into one stitched document with interleaved histories.
    """
    metadata = record.get("metadata") or {}
    url = metadata.get("url")
    if not isinstance(url, str) or not url:
        return None
    parts = urlparse(url).path.split("/")
    if len(parts) <= _URL_COMMIT_IDX:
        return None
    owner, repo, commit = parts[_URL_OWNER_IDX], parts[_URL_REPO_IDX], parts[_URL_COMMIT_IDX]
    if not (owner and repo and commit):
        return None
    return f"{owner}/{repo}@{commit}"


def _file_path(record: dict) -> str | None:
    """Return the in-repo file path for the record, falling back to the url tail."""
    metadata = record.get("metadata") or {}
    path = metadata.get("path")
    if isinstance(path, str) and path:
        return path.lstrip("/")
    url = metadata.get("url")
    if isinstance(url, str) and url:
        parts = urlparse(url).path.split("/")
        if len(parts) > _URL_PATH_START:
            return "/".join(parts[_URL_PATH_START:])
    return None


def _dfs_sort_key(record: dict) -> tuple[str, ...]:
    """Sort key that yields depth-first traversal when applied across a repo's files.

    Python's tuple comparison is lexicographic on components, so files sharing a directory
    prefix stay contiguous and the deepest descendants of each directory are emitted before
    moving on to the next sibling at the parent level.
    """
    path = _file_path(record) or ""
    return tuple(path.split("/"))


def _make_stitch_reducer(
    config: StitchStackV2Config,
) -> Callable[[str, Iterator[dict]], Iterator[dict]]:
    """Build the (repo_key, files) -> records reducer used by ``group_by``."""

    separator = config.separator
    header_tpl = config.file_header
    max_chars = config.max_chars_per_repo
    min_files = config.min_files_per_repo

    def reducer(repo: str, files: Iterator[dict]) -> Iterator[dict]:
        buf: list[str] = []
        paths: list[str] = []
        running_chars = 0
        chunk_idx = 0

        def emit() -> dict | None:
            if len(paths) < min_files:
                return None
            doc_id = f"{repo}#{chunk_idx}" if max_chars is not None else repo
            return {
                "id": doc_id,
                "text": separator.join(buf),
                "metadata": {
                    "repo": repo,
                    "chunk_idx": chunk_idx,
                    "n_files": len(paths),
                    "paths": list(paths),
                },
            }

        for record in files:  # delivered in DFS order thanks to sort_by
            path = _file_path(record)
            text = record.get("text")
            if path is None or not isinstance(text, str):
                continue
            piece = header_tpl.format(path=path) + text
            # Length cost of appending ``piece`` to a non-empty buffer: piece + one separator.
            incremental = len(piece) + (len(separator) if buf else 0)
            if max_chars is not None and buf and running_chars + incremental > max_chars:
                out = emit()
                if out is not None:
                    yield out
                    chunk_idx += 1
                buf, paths, running_chars = [], [], 0
                incremental = len(piece)  # first piece in a fresh chunk has no separator
            buf.append(piece)
            paths.append(path)
            running_chars += incremental

        out = emit()
        if out is not None:
            yield out

    return reducer


def stitch_stackv2_repos(config: StitchStackV2Config) -> None:
    """Group Stack v2 records by repo and emit DFS-stitched documents."""
    logger.info(
        "Stitching Stack v2 records from %s into repo-level documents at %s",
        config.input_path,
        config.output_path,
    )

    pipeline = (
        Dataset.from_files(f"{config.input_path}/{config.input_glob}")
        .flat_map(load_jsonl)
        .filter(lambda record: _repo_key(record) is not None and _file_path(record) is not None)
        .group_by(
            key=_repo_key,
            sort_by=_dfs_sort_key,
            reducer=_make_stitch_reducer(config),
        )
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="stitch-stackv2-repos")
    ctx.execute(pipeline)


# Executor wiring: take ``common-pile/stackv2`` (the SWH-keyed variant downloaded in
# tokenize_common_pile.py), stitch into repo-level documents, and tokenize with the llama3
# tokenizer for long-context training.
stackv2_stitched = ExecutorStep(
    name="documents/common_pile/stackv2_stitched",
    fn=stitch_stackv2_repos,
    config=StitchStackV2Config(
        input_path=stackv2,
        output_path=this_output_path(),
    ),
)


stackv2_stitched_tokenized = default_tokenize(
    name="common_pile_stackv2_stitched",
    dataset=stackv2_stitched,
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[stackv2_stitched, stackv2_stitched_tokenized])
