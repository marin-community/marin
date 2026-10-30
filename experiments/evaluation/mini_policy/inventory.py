# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory frozen cohort evidence without selecting score fields by proximity."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for


def preserve(key, payload):
    url = f"s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030/cell_evidence/{key}.json"
    fs, path = filesystem_for(url)
    fs.pipe_file(path, payload)
    return url


def inspect_cell(cell, output):
    result = dict(cell)
    key = hashlib.sha256((cell["model"] + "\n" + cell["benchmark"]).encode()).hexdigest()[:16]
    try:
        if cell["source"].endswith(".json"):
            fs, path = filesystem_for(cell["source"])
            payload = fs.cat(path)
            data = json.loads(payload)
            result["evidence_url"] = preserve(key, payload)
            result.update(
                kind="json",
                sha256=hashlib.sha256(payload).hexdigest(),
                keys=list(data),
                trial_count=len(data.get("trials", [])),
                trial_keys=list(data.get("trials", [{}])[0]),
                grading_revision=data.get("evalchemy_commit"),
                source_parent=data.get("raw_source") or data.get("source"),
            )
        else:
            view = ReadView(cell["source"])
            tables = view.table_names()
            result.update(kind="finestore", sealed=view.is_sealed(), tables=tables)
            if "samples" in tables:
                keys = list(view.primary_key("samples"))
                columns = list(dict.fromkeys([*keys, "grading", "metrics", "filter", "doc"]))
                table = view.scan("samples", columns=columns)
                if table is not None:
                    rows = table.to_pylist(maps_as_pydicts="strict")
                    result["evidence_url"] = preserve(key, json.dumps(rows, default=str).encode())
                    result.update(
                        row_count=len(rows),
                        primary_key=keys,
                        row_keys=list(rows[0]) if rows else [],
                        metadata_path=view.table_metadata_path("samples"),
                        schema_version=view.schema_version("samples"),
                    )
            else:
                result["needs_adapter"] = True
        result["status"] = "readable"
    except Exception as error:
        # Inventory must record every unavailable cell, then fail its audit gate.
        result.update(status="error", error_type=type(error).__name__, error=str(error))
    return result


def main():
    manifest = json.loads(Path("experiments/evaluation/mini_policy/cohort.json").read_text())
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    cells_dir = output / "cell_evidence"
    cells_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(inspect_cell, cell, cells_dir): cell for cell in manifest["cells"]}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(futures),
                        "benchmark": row["benchmark"],
                        "status": row["status"],
                    }
                ),
                flush=True,
            )
            (output / "inventory.json").write_text(json.dumps(results, indent=2))
    errors = sum(row["status"] == "error" for row in results)
    (output / "summary.json").write_text(
        json.dumps({"cells": len(results), "errors": errors, "models": manifest["models"]}, indent=2)
    )
    print(f"INVENTORY_COMPLETE cells={len(results)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
