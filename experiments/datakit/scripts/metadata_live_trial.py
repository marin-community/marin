# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run bounded frozen-source metadata and normalization replay probes near the data."""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit import partition_filename
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.scripts.metadata_artifact_inventory import inventory, read_metadata
from experiments.datakit.scripts.metadata_source_replay import recipe_from_inventory, run_source_replay
from experiments.datakit.store.metadata import ShardManifest, export_metadata_shard


def _absolute(prefix: str, path: str) -> str:
    return path if "://" in path else f"{prefix.rstrip('/')}/{path.lstrip('/')}"


def probe_source(
    source: dict,
    recovered: dict,
    output: str,
    *,
    max_rows: int,
    replay_rows: int,
    replay_bytes: int,
) -> dict:
    """Check real selection alignment and separately replay real normalization inputs."""
    name = source["marin_source"]
    report = {"source": name}
    prefix = recovered["store_uri"].split("/datakit/store_", 1)[0]
    roots = source["artifact_roots"]
    records = recovered["records"]
    try:
        filename = partition_filename(0, source["normalized_shard_count_hint"])
        decon_dir = records[roots["decontam"]]["record"]["result"]["main_output_dir"]
        cluster_dir = records[roots["cluster"]]["record"]["result"]["output_dir"]
        normalized = _absolute(prefix, source["source_key"]) + "/" + filename
        manifest = ShardManifest(
            marin_source=name,
            normalized=normalized,
            decontam=_absolute(prefix, decon_dir) + "/" + filename,
            cluster=_absolute(prefix, cluster_dir) + "/" + filename,
            quality=roots["quality"] + "/" + filename,
            content_type=roots["content_type"] + "/" + filename,
            exact_dedup=_absolute(prefix, source["exact_dedup"]["attr_dir"]) + "/" + filename,
            dedup=(
                ""
                if source["fuzzy_exempt"]
                else (_absolute(prefix, source["verified_fuzzy_dedup"]["attr_dir"]) + "/" + filename)
            ),
            bucket_edges=recovered["bucket_edges"],
        )
        exported = export_metadata_shard(
            manifest,
            output,
            cluster_view=recovered["store_config"]["cluster_view"],
            max_rows=max_rows,
            batch_size=min(max_rows, 64),
        )
        # Readback is intentionally labelled separately from independent input replay.
        hashes_checked = 0
        fs, resolved = filesystem_for(normalized)
        with fs.open(resolved, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=min(max_rows, 64), columns=["id", "text"]):
                for row in islice(batch.to_pylist(), max_rows - hashes_checked):
                    if generate_id(row["text"]) != row["id"]:
                        raise ValueError("Stored normalized text does not match its canonical ID")
                    hashes_checked += 1
                if hashes_checked >= max_rows:
                    break
        report["selection"] = {
            "status": "passed" if exported["counters"]["datakit_store/records_out"] else "empty_sample",
            "hashes_checked": hashes_checked,
            **exported,
        }
    except Exception as error:
        # Retain failures per source while examining the remaining frozen inventory.
        report["selection"] = {"status": "failed", "error": str(error)}
    try:
        recipe = recipe_from_inventory(source, recovered["store_provenance"]["base_commit"])
        report["normalization_replay"] = run_source_replay(
            recipe,
            max_rows=replay_rows,
            max_projected_bytes=replay_bytes,
        )
    except Exception as error:
        report["normalization_replay"] = {"status": "failed", "error": str(error)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", default=os.environ.get("IRIS_OUTPUT_DIR"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--replay-rows", type=int, default=4)
    parser.add_argument("--replay-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--original-inputs", action="store_true", help="Use recorded input locations before mirroring")
    parser.add_argument("--sources", help="Optional comma-separated frozen source subset")
    args = parser.parse_args()
    if not args.output:
        raise ValueError("Specify --output or run with IRIS_OUTPUT_DIR")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    store = read_metadata(args.store.rstrip("/") + "/.artifact.json")
    recovered = inventory(store)
    (output / "inventory.json").write_text(json.dumps(recovered, indent=2))
    sources = recovered["sources"]
    if args.original_inputs:
        sources = [
            {**source, "normalization_input_paths": source["original_normalization_input_paths"]} for source in sources
        ]
    if args.sources:
        names = set(args.sources.split(","))
        if names - {source["marin_source"] for source in sources}:
            raise ValueError("Requested source is absent from the frozen store")
        sources = [source for source in sources if source["marin_source"] in names]

    def probe(item: tuple[int, dict]) -> dict:
        index, source = item
        result = probe_source(
            source,
            recovered,
            str(output / f"sample-{index:03d}"),
            max_rows=args.max_rows,
            replay_rows=args.replay_rows,
            replay_bytes=args.replay_bytes,
        )
        print(
            json.dumps(
                {
                    "source": result["source"],
                    "selection": result["selection"]["status"],
                    "normalization_replay": result["normalization_replay"]["status"],
                }
            ),
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(probe, enumerate(sources)))
    report = {
        "store_uri": args.store,
        "frozen_source_count": recovered["source_count"],
        "sources_examined": len(results),
        "selection_status": dict(Counter(row["selection"]["status"] for row in results)),
        "normalization_replay_status": dict(Counter(row["normalization_replay"]["status"] for row in results)),
        "scope": (
            "real artifact alignment and staged normalization-input replay; "
            "upstream transforms and full token order separate"
        ),
        "sources": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}), flush=True)
    if any(row[stage]["status"] != "passed" for row in results for stage in ("selection", "normalization_replay")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
