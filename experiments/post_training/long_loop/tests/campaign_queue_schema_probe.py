# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regress queued mixes against pinned source footers in an in-region CPU job."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from skyrl_train.dataset.dataset import read_prompt_files

from experiments.post_training.long_loop.campaign_runtime import lower_parquet


def main(selection: Path, output: str, catalog: Path) -> None:
    if get_job_info() is None:
        raise RuntimeError("Queue schema regression must run inside an Iris CPU job")
    configure_coreweave_s3()
    selected = json.loads(selection.read_bytes())
    catalog_bytes = catalog.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != selected["catalog_sha256"]:
        raise ValueError("Queue regression catalog changed")
    catalog_sources = json.loads(catalog_bytes)["sources"]
    report: dict[str, Any] = {
        "status": "RUNNING",
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "scope": "Pinned retained-source footers with native lowering's preserved training schema",
        "payloads": "One typed-null schema witness per source; no task execution or model work",
        "sources": {},
        "mixes": {},
    }
    schemas: dict[str, pa.Schema] = {}
    target = StoragePath(output)
    for mix_id in selected["mix_ids"]:
        pins = selected["mixes"][mix_id]["sources"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = []
            for index, (label, pin) in enumerate(pins.items()):
                declared = catalog_sources[label]
                if (pin["repo"], pin["revision"], pin["count"]) != (
                    declared["dataset_id"],
                    declared["dataset_revision"],
                    declared["actor_census_input"]["measured_tasks"],
                ):
                    raise ValueError(f"Captured queue pin differs from catalog: {label}")
                identity = f"{pin['repo']}@{pin['revision']}"
                if identity not in schemas:
                    uri = f"hf://datasets/{identity}/tasks.parquet"
                    fs, key = url_to_fs(uri, token=require_secret("HF_TOKEN"))
                    with fs.open(key, "rb") as stream:
                        parquet = pq.ParquetFile(stream)
                        if parquet.metadata.num_rows != pin["count"]:
                            raise ValueError(f"Pinned source row count changed: {identity}")
                        original = parquet.schema_arrow
                    # Run the actual native schema projection on an empty source,
                    # without lowering a fabricated task or reading cohort bodies.
                    source = directory / "empty-source.parquet"
                    lowered = directory / "empty-lowered.parquet"
                    pq.write_table(original.empty_table(), source)
                    lower_parquet(source, lowered, {})
                    schema = pq.read_schema(lowered)
                    field = pa.field("teacher_route", pa.string())
                    column = schema.get_field_index("teacher_route")
                    schema = schema.set(column, field) if column >= 0 else schema.append(field)
                    schemas[identity] = schema
                    report["sources"][identity] = {
                        "uri": uri,
                        "rows": pin["count"],
                        "footer_schema_sha256": hashlib.sha256(original.serialize().to_pybytes()).hexdigest(),
                        "training_schema": str(schema),
                    }
                schema = schemas[identity]
                witness = pa.Table.from_arrays([pa.array([None], type=field.type) for field in schema], schema=schema)
                path = directory / f"source-{index}.parquet"
                pq.write_table(witness, path)
                inputs.append(str(path))
            combined = read_prompt_files(inputs)
            if len(combined) != len(pins):
                raise ValueError(f"Reader changed schema witness count: {mix_id}")
            report["mixes"][mix_id] = {
                "sources": list(pins),
                "status": "PASS",
                "combined_features": combined.features.to_dict(),
            }
        target.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "ALL_SELECTED_QUEUE_SCHEMAS_PASS"
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output")
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    main(args.selection, args.output, args.catalog)
