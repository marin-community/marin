# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded metadata reconstruction recipe contracts, without downloading sources.

Run with ``python -m experiments.datakit.scripts.metadata_recipe_trial --output
/tmp/metadata-recipe-trial.json``. Every registry recipe is exercised on synthetic
records. This does not establish reconstruction of any actual upstream document.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from marin.datakit.download import (
    biodiversity,
    finetranslations,
    institutional_books,
    stack_v3,
    swe_rebench_contree,
    swe_zero_12m,
)
from marin.datakit.normalize import _make_normalize_fn, _make_whitespace_compactor, generate_id
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.step_spec import StepSpec


def recipe_contract(source: DatakitSource) -> dict[str, Any]:
    """Exercise source-specific field mapping, schema retention, and final text ID."""
    attrs = source.normalized.hash_attrs
    text_field, id_field = attrs["text_field"], attrs["id_field"]
    limit = attrs["max_whitespace_run_chars"]
    normalize = _make_normalize_fn(
        text_field, id_field, bare=attrs.get("bare", False), drop_fields=tuple(attrs.get("drop_fields", ()))
    )
    compact = _make_whitespace_compactor(limit)
    text = "日本語 café\n\n" + " " * (limit + 3) + "end"
    expected_text = "日本語 café\n\n" + " " * (limit - 2) + "end"
    # The adjacent two newlines participate in the same whitespace run.
    for provider_id in ("provider-row-7", 7, None):
        record = {text_field: text, "fixture_metadata": "kept"}
        record.update({field: "removed" for field in attrs.get("drop_fields", ())})
        if provider_id is not None:
            record[id_field] = provider_id
        result = compact(normalize(record))
        assert result["text"] == expected_text, source.name
        assert result["id"] == generate_id(expected_text), source.name
        assert result.get("source_id") == provider_id, source.name
        assert ("fixture_metadata" in result) == (not attrs.get("bare", False)), source.name
        assert not set(attrs.get("drop_fields", ())).intersection(result), source.name
    return {"status": "passed", "cases": 3, "parameters": attrs}


def _step_inventory(step: StepSpec, result: dict[str, dict[str, Any]]) -> None:
    if step.name_with_hash in result:
        return
    result[step.name_with_hash] = {
        "name": step.name,
        "hash": step.hash_id,
        "output_path": step.output_path,
        "parameters": step.hash_attrs,
        "dependencies": step.dep_names,
    }
    for dependency in step.deps:
        _step_inventory(dependency, result)


def transform_contracts() -> list[dict[str, Any]]:
    """Replay hand-authored raw fixtures through production custom transforms."""
    cases: list[tuple[str, list[dict], list[str]]] = []
    for raw_id in ("a", "d"):
        # Fixed independent expected coin values for these fixture IDs.
        expected = "Hello\n\nBonjour" if raw_id == "a" else "Bonjour\n\nHello"
        cases.append(
            (
                "finetranslations",
                list(
                    finetranslations.fold_parallel_text(
                        {"id": raw_id, "og_full_text": "Bonjour", "translated_text": "Hello"}
                    )
                ),
                [expected],
            )
        )
    cases.append(
        (
            "finetranslations",
            list(finetranslations.fold_parallel_text({"og_full_text": "Bonjour", "translated_text": None})),
            ["Bonjour"],
        )
    )
    cases.append(
        (
            "institutional_books",
            institutional_books.row_to_doc(
                {"text_by_page_gen": ["corrected", "", "page two"], "text_by_page_src": ["raw"]}
            ),
            ["corrected\n\npage two"],
        )
    )
    cases.append(
        (
            "institutional_books",
            institutional_books.row_to_doc({"text_by_page_src": ["raw", "page two"]}),
            ["raw\n\npage two"],
        )
    )
    cases.append(("swe-rebench-contree", swe_rebench_contree.row_to_doc({"text": "trace"}), ["trace"]))
    cases.append(
        (
            "swe-zero-12m",
            swe_zero_12m.row_to_doc(
                {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "done"}]}
            ),
            ["<user>\ntask\n</user>\n\n<assistant>\ndone\n</assistant>"],
        )
    )
    cases.append(
        (
            "cp/biodiversity",
            list(
                biodiversity.stitch_pages("volume-1", iter([{"text": "page one"}, {"text": ""}, {"text": "page two"}]))
            ),
            ["page one\n\npage two"],
        )
    )
    files = [
        {"file_path": "z.py", "content_id": "z", "content": "Z"},
        {"file_path": "a.py", "content_id": "a", "content": "A"},
        {"file_path": "a.py", "content_id": "a", "content": "A"},
    ]
    cases.append(
        (
            "stack-v3",
            [
                stack_v3.row_to_doc(
                    {
                        "repo_path": "org/repo",
                        "repo_id": 1,
                        "commit_id": "commit",
                        "github_metadata": {},
                        "num_files": 3,
                        "files": files,
                    }
                )
            ],
            ["Repository: org/repo\n\nFile: a.py\nA\n\nFile: z.py\nZ"],
        )
    )
    report = []
    registry = all_sources()
    for name, records, expected_texts in cases:
        assert [record["text"] for record in records] == expected_texts, name
        attrs = registry[name].normalized.hash_attrs
        normalize = _make_normalize_fn("text", attrs["id_field"])
        for record, expected in zip(records, expected_texts, strict=True):
            assert normalize(record)["id"] == generate_id(expected), name
        report.append({"source": name, "status": "passed", "records": len(records)})
    return report


def run_trial() -> dict[str, Any]:
    """Return explicit recipe coverage and the remaining real-data validation gap."""
    sources = all_sources()
    steps: dict[str, dict[str, Any]] = {}
    results = []
    for name, source in sorted(sources.items()):
        contract = recipe_contract(source)
        _step_inventory(source.normalized, steps)
        results.append(
            {
                "source": name,
                "normalization": contract,
                "normalized_step": source.normalized.name_with_hash,
                "upstream_reconstruction": "not_run",
            }
        )
    transforms = transform_contracts()
    return {
        "trial_kind": "synthetic_recipe_contracts",
        "source_count": len(results),
        "normalization_passed": len(results),
        "upstream_reconstruction_passed": 0,
        "limitations": [
            "No upstream files or production normalized documents were read.",
            "Recipe paths describe this checkout; production Hero artifacts must be pinned separately.",
            "Transform fixtures do not validate source schemas, complete biodiversity grouping, or real upstream IDs.",
            "Store selection, multiplicity, scores, buckets and shard order require the separate store trial.",
        ],
        "text_fields": dict(Counter(s.normalized.hash_attrs["text_field"] for s in sources.values())),
        "provider_id_fields": dict(Counter(s.normalized.hash_attrs["id_field"] for s in sources.values())),
        "sources": results,
        "steps": steps,
        "custom_transform_fixtures": transforms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_trial()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("trial_kind", "source_count", "normalization_passed", "upstream_reconstruction_passed")
            }
        )
    )


if __name__ == "__main__":
    main()
