# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export and independently reconstruct bounded samples from a frozen source manifest."""

import argparse
import json
from collections import Counter
from itertools import islice

import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.store.metadata import ShardManifest, export_metadata_shard


def verify_reconstruction(outputs: list[dict], replay_paths: list[str], max_rows: int) -> dict:
    """Match selected occurrence counts against separately replayed normalized text.

    Replay paths must come from independently running the pinned upstream recipe.
    They must cover the selected sample; an unrelated prefix of a repartitioned
    replay is not sufficient. Only counts and hashes enter the report.
    """
    selected: Counter[str] = Counter()
    for output in outputs:
        fs, resolved = filesystem_for(output["path"])
        with fs.open(resolved, "rb") as stream:
            selected.update(pq.read_table(stream, columns=["id"]).column("id").to_pylist())
    available: Counter[str] = Counter()
    examined = 0
    for path in replay_paths:
        fs, resolved = filesystem_for(path)
        with fs.open(resolved, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=min(max_rows, 64), columns=["id", "text"]):
                for row in islice(batch.to_pylist(), max_rows - examined):
                    if generate_id(row["text"]) != row["id"]:
                        raise ValueError(f"Replay has an incorrect normalized-text hash: {path}")
                    available[row["id"]] += 1
                    examined += 1
                if examined >= max_rows:
                    break
        if examined >= max_rows:
            break
    missing = selected - available
    return {
        "status": "passed" if selected and not missing else "incomplete",
        "selected_occurrences": selected.total(),
        "replayed_rows": examined,
        "missing_occurrences": missing.total(),
    }


def run_trial(manifest: dict, output_dir: str, *, max_rows: int, replay_max_rows: int) -> dict:
    """Run every manifest entry and report gaps against its frozen source list."""
    if max_rows < 1 or replay_max_rows < 1:
        raise ValueError("Trial row limits must be positive")
    expected = manifest["expected_sources"]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_sources must be a nonempty unique frozen source list")
    if not manifest["recipe_commit"] or not manifest["store_uri"]:
        raise ValueError("Pin recipe_commit and store_uri before running a release trial")
    reports: list[dict] = []
    outputs_by_source: dict[str, list[dict]] = {}
    replays_by_source: dict[str, list[str]] = {}
    originals_by_source: dict[str, set[str]] = {}
    passed: set[str] = set()
    for index, entry in enumerate(manifest["shards"]):
        shard = ShardManifest(**entry["selection"])
        if shard.marin_source not in expected:
            raise ValueError(f"Unexpected source: {shard.marin_source}")
        try:
            report = export_metadata_shard(
                shard,
                prefix_join(output_dir, f"sample-{index:05d}"),
                cluster_view=manifest["cluster_view"],
                max_rows=max_rows,
                batch_size=min(max_rows, 64),
            )
            outputs_by_source.setdefault(shard.marin_source, []).extend(report["outputs"])
            replays_by_source.setdefault(shard.marin_source, []).extend(entry.get("replay_normalized", []))
            originals_by_source.setdefault(shard.marin_source, set()).add(shard.normalized)
        except Exception as error:
            # A trial records failed sources and continues so coverage gaps remain visible.
            report = {"manifest": entry["selection"], "status": "failed", "error": str(error)}
        reports.append(report)
    reconstruction = {}
    for source, outputs in outputs_by_source.items():
        replay = list(dict.fromkeys(replays_by_source[source]))
        try:
            if originals_by_source[source].intersection(replay):
                raise ValueError("Independent replay cannot include an original normalized shard")
            result = (
                verify_reconstruction(outputs, replay, replay_max_rows)
                if replay
                else {"status": "not_run", "reason": "No independent upstream replay supplied"}
            )
        except Exception as error:
            result = {"status": "failed", "error": str(error)}
        reconstruction[source] = result
        if result["status"] == "passed":
            passed.add(source)
    complete = set(expected) == passed and not any(report.get("status") == "failed" for report in reports)
    return {
        "status": "passed" if complete else "incomplete",
        "store_uri": manifest["store_uri"],
        "recipe_commit": manifest["recipe_commit"],
        "scope": "bounded source samples; not a full-corpus completeness or token-stream parity check",
        "expected_sources": expected,
        "reconstructed_sources": sorted(passed),
        "missing_sources": sorted(set(expected) - passed),
        "reconstruction": reconstruction,
        "shards": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--replay-max-rows", type=int, default=1024)
    args = parser.parse_args()
    output_fs, output = filesystem_for(args.output)
    if output_fs.exists(output):
        raise FileExistsError(f"Use a fresh trial output directory: {args.output}")
    manifest_fs, manifest_path = filesystem_for(args.manifest)
    with manifest_fs.open(manifest_path, "r") as stream:
        manifest = json.load(stream)
    report = run_trial(manifest, args.output, max_rows=args.max_rows, replay_max_rows=args.replay_max_rows)
    output_fs.makedirs(output, exist_ok=True)
    with output_fs.open(prefix_join(output, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({key: report[key] for key in ("status", "reconstructed_sources", "missing_sources")}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
