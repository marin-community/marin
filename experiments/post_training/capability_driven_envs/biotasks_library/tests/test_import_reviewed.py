# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

from experiments.post_training.capability_driven_envs.biotasks_library.import_repositories import (
    _git_blob_sha1,
    _selected_documents,
)
from experiments.post_training.capability_driven_envs.biotasks_library.import_reviewed import (
    BIOTASKS_REVISION,
    RELEASE_STATUS,
    ReviewedSource,
    materialize,
    reviewed_sources,
)


def jsonl(records: list[dict[str, object]]) -> bytes:
    return ("".join(json.dumps(record) + "\n" for record in records)).encode()


def annotation(index: int) -> dict[str, object]:
    key = f"github:example/repo:notebook-{index}.ipynb"
    return {
        "document_key": key,
        "canonical_document_key": key,
        "source_ids": ["github:example/repo"],
        "repo": "example/repo",
        "revision": "a" * 40,
        "path": f"notebook-{index}.ipynb",
        "format": "Jupyter",
        "title": f"Notebook {index}",
        "source_role": "analysis_or_teaching_document",
        "review": "assistant_individual_source_review",
        "source_sha256": f"{index:064x}",
        "acquisition_status": "confirmed",
        "counted": True,
        "labels": [
            {"facet": "scientific_field", "term": "transcriptomics"},
            {"facet": "modality", "term": "scrna"},
            {"facet": "operation", "term": "differential-analysis"},
        ],
    }


def test_reviewed_sources_selects_the_pinned_individually_reviewed_cohort():
    annotations = [annotation(index) for index in range(57)]
    annotations.append({**annotation(58), "review": "not_individually_adjudicated"})
    acquisitions = [
        {
            "document_key": record["document_key"],
            "acquisition_url": "https://unused.example/source",
        }
        for record in annotations
    ]

    sources = reviewed_sources(jsonl(annotations), jsonl(acquisitions), b"[]")

    assert len(sources) == 57
    assert len({source.canonical_document_key for source in sources}) == 57
    assert sources[0].download_url.startswith(f"https://raw.githubusercontent.com/example/repo/{'a' * 40}/")
    assert sources[0].fields == ("transcriptomics",)
    assert sources[0].modalities == ("scrna",)
    assert sources[0].operations == ("differential-analysis",)


def test_materialize_verifies_and_content_addresses_source_bytes(tmp_path: Path):
    source_file = tmp_path / "source.ipynb"
    source_file.write_bytes(b'{"cells": [], "metadata": {}}')
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    source = ReviewedSource(
        document_key="github:example/repo:source.ipynb",
        canonical_document_key="github:example/repo:source.ipynb",
        source_ids=("github:example/repo",),
        repo="example/repo",
        revision="a" * 40,
        path="source.ipynb",
        format="Jupyter",
        title="Source",
        source_role="analysis_or_teaching_document",
        review="assistant_individual_source_review",
        source_sha256=digest,
        download_url=source_file.as_uri(),
        fields=("transcriptomics",),
        modalities=("scrna",),
        operations=("differential-analysis",),
    )
    output = tmp_path / "library"

    manifest = materialize([source], output_root=str(output), campaign_id="fixture", max_workers=1)

    assert manifest["release_status"] == RELEASE_STATUS
    assert manifest["biotasks_revision"] == BIOTASKS_REVISION
    assert manifest["source_count"] == 1
    assert (output / "objects" / "sha256" / digest[:2] / digest).read_bytes() == source_file.read_bytes()
    catalog = json.loads((output / "campaigns" / "biotasks" / "fixture" / "source-catalog.jsonl").read_text())
    assert catalog["source_sha256"] == digest
    assert catalog["object_uri"].endswith(digest)


def test_repository_evidence_selection_is_bounded_and_prefers_small_documents():
    observation = {
        "notebooks": [
            {"path": "large.ipynb", "size": 2_000_000, "sha": "large"},
            {"path": "c.Rmd", "size": 300, "sha": "c"},
            {"path": "a.ipynb", "size": 100, "sha": "a"},
            {"path": "b.qmd", "size": 200, "sha": "b"},
            {"path": "d.Rnw", "size": 400, "sha": "d"},
        ]
    }

    selected = _selected_documents(observation)

    assert [row["path"] for row in selected] == ["a.ipynb", "b.qmd", "c.Rmd"]
    assert _git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
