# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize compact, immutable views of the BioTasks notebooks/v1 campaign."""

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from typing import Any

import fsspec

from experiments.post_training.capability_driven_envs.biotasks_library.run_notebooks_v1_zephyr import (
    DEFAULT_CATALOG,
    DEFAULT_OUTPUT,
    ROOT,
)

DEFAULT_CAMPAIGN_ROOT = f"{ROOT}/campaigns/biotasks/notebooks/v1"


def _read_jsonl(uri: str) -> tuple[bytes, list[dict[str, Any]]]:
    with fsspec.open(uri, "rb") as stream:
        raw = stream.read()
    return raw, [json.loads(line) for line in raw.splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n" for row in rows
    )


def _json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def build_views(
    source_rows: list[dict[str, Any]],
    catalog_rows: list[dict[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build normalized campaign artifacts and summary counts."""
    if len(source_rows) != len(catalog_rows):
        raise ValueError(f"source result count {len(source_rows)} != catalog count {len(catalog_rows)}")
    catalog_keys = {row["source_sha256"] for row in catalog_rows}
    result_keys = {row["source"]["source_sha256"] for row in source_rows}
    if catalog_keys != result_keys:
        raise ValueError("source results do not cover exactly the pinned catalog")

    source_rows = sorted(source_rows, key=lambda row: row["source"]["canonical_document_key"])
    ideas: list[dict[str, Any]] = []
    author_reviews: list[dict[str, Any]] = []
    glm_reviews: list[dict[str, Any]] = []
    taskforge_results: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    screening: dict[str, list[dict[str, Any]]] = {
        "accept": [],
        "revise": [],
        "reject": [],
        "review-error": [],
        "null": [],
    }

    for source_row in source_rows:
        source = source_row["source"]
        portfolio = source_row.get("portfolio") or {}
        dispositions.append(
            {
                "source_document_key": source["canonical_document_key"],
                "source_sha256": source["source_sha256"],
                "status": source_row["status"],
                "disposition": portfolio.get("disposition"),
                "source_summary": portfolio.get("source_summary"),
                "null_reason": portfolio.get("null_reason"),
                "generation_failures": source_row.get("generation_failures", ()),
                "hydration_errors": source_row.get("hydration_errors", ()),
            }
        )
        for item in sorted(source_row.get("items", ()), key=lambda value: value["idea"]["idea_id"]):
            idea = item["idea"]
            common = {
                "source_document_key": source["canonical_document_key"],
                "source_group_id": idea["source_group_id"],
                "source_sha256": source["source_sha256"],
                "idea_id": idea["idea_id"],
            }
            if idea.get("null_reason") is None:
                ideas.append(
                    {
                        **common,
                        "idea": idea,
                        "taskforge_terminal": item["terminal"],
                        "task_id": item.get("task_id"),
                        "task_digest": item.get("task_digest"),
                    }
                )
            author_reviews.append({**common, "assessment": item["author_assessment"]})
            glm_row = {**common}
            if "glm_review" in item:
                glm_row.update(review=item["glm_review"], provenance=item["glm_review_provenance"])
            else:
                glm_row["error"] = item["glm_review_error"]
            glm_reviews.append(glm_row)
            taskforge_row = {
                **common,
                "terminal": item["terminal"],
                "events": item["events"],
                "task_id": item.get("task_id"),
                "task_digest": item.get("task_digest"),
                "task_json": item.get("task_json"),
            }
            taskforge_results.append(taskforge_row)

            screened = {
                **common,
                "idea": idea,
                "author_assessment": item["author_assessment"],
                "glm_review": item.get("glm_review"),
                "glm_review_error": item.get("glm_review_error"),
                "taskforge_terminal": item["terminal"],
                "task_id": item.get("task_id"),
                "task_digest": item.get("task_digest"),
                "task_json": item.get("task_json"),
            }
            if idea.get("null_reason") is not None:
                band = "null"
            elif "glm_review_error" in item:
                band = "review-error"
            else:
                band = item["glm_review"]["recommended_action"]
            screening[band].append(screened)

    artifacts = {
        "ideas.jsonl": _jsonl(ideas),
        "author-reviews.jsonl": _jsonl(author_reviews),
        "glm-reviews.jsonl": _jsonl(glm_reviews),
        "taskforge-results.jsonl": _jsonl(taskforge_results),
        "source-dispositions.jsonl": _jsonl(dispositions),
        **{f"screening/{band}.jsonl": _jsonl(rows) for band, rows in screening.items()},
    }
    summary = {
        "source_count": len(source_rows),
        "source_statuses": dict(Counter(row["status"] for row in source_rows)),
        "idea_count": len(ideas),
        "null_source_count": len(screening["null"]),
        "taskforge_terminals": dict(Counter(row["terminal"] for row in taskforge_results)),
        "author_recommendations": dict(Counter(row["assessment"]["recommended_action"] for row in author_reviews)),
        "glm_recommendations": dict(
            Counter(row.get("review", {}).get("recommended_action", "error") for row in glm_reviews)
        ),
        "screening_bands": {band: len(rows) for band, rows in screening.items()},
    }
    return artifacts, summary


def _write_new(uri: str, content: bytes) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable campaign artifact: {uri}")
    with fs.open(path, "wb") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--campaign-root", default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--iris-job", required=True)
    args = parser.parse_args()

    results_raw, source_rows = _read_jsonl(args.source_results)
    catalog_raw, catalog_rows = _read_jsonl(args.catalog)
    artifacts, summary = build_views(source_rows, catalog_rows)
    artifacts["source-catalog.jsonl"] = catalog_raw
    manifest = {
        "schema_version": "1",
        "campaign_id": "biotasks/notebooks/v1",
        "release_status": "internal_unreviewed_do_not_publish",
        "source_results_uri": args.source_results,
        "source_results_sha256": hashlib.sha256(results_raw).hexdigest(),
        "source_catalog_uri": f"{args.campaign_root}/source-catalog.jsonl",
        "source_catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "code_commit": args.code_commit,
        "iris_job": args.iris_job,
        "screening_note": (
            "accept/revise/reject are blinded GLM screening bands, not scientific review or release approval"
        ),
        "summary": summary,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "artifacts": {
            path: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for path, content in artifacts.items()
        },
    }
    for path, content in artifacts.items():
        _write_new(f"{args.campaign_root}/{path}", content)
    _write_new(f"{args.campaign_root}/manifest.json", _json(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
