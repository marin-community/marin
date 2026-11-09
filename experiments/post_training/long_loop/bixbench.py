# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A one-task BixBench continuation and a provenance-preserving result merge."""

import hashlib
import json

import click
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.execution.lazy import OUT, ArtifactStep, apply, run
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.long_loop.eval_runtime.audit_results import audit
from experiments.post_training.long_loop.sft import read_json


def prepare_subset(*, output: str, source: str) -> None:
    fs, path = filesystem_for(source + "/rollouts.parquet")
    with fs.open(path, "rb") as stream:
        table = pq.read_table(stream)
    selected = table.filter(pc.equal(table["benchmark"], "bixbench-pi"))
    if selected.num_rows != 1:
        raise ValueError("Expected exactly one original BixBench smoke task")
    manifest = read_json(source + "/environment.json")
    manifest.update(data_uri=output, subset_of=source, subset_benchmark="bixbench-pi")
    target_fs, target = filesystem_for(output)
    target_fs.makedirs(target, exist_ok=True)
    with target_fs.open(target + "/rollouts.parquet", "wb") as stream:
        pq.write_table(selected, stream)
    target_fs.pipe_file(target + "/environment.json", json.dumps(manifest, indent=2).encode())
    target_fs.pipe_file(
        target + "/selection.json",
        json.dumps(
            {
                "source": source,
                "source_rows": table.num_rows,
                "selected_rows": 1,
                "benchmark": "bixbench-pi",
                "environment_evidence": "reused superset; no new provisioning",
            },
            indent=2,
        ).encode(),
    )


def merge_results(*, output: str, original: str, repair: str) -> None:
    before = audit(original, 20)
    after = audit(repair, 1)
    if before["completed_rollouts"] != 20 or not after["all_verified"]:
        raise ValueError("Merge requires the original 20 results and one verified repair")
    suffix = "/exports/dumped_evals/eval_only/"
    original_fs, original_root = filesystem_for(original + suffix)
    repair_fs, repair_root = filesystem_for(repair + suffix)
    replacements = [p for p in repair_fs.glob(repair_root + "/*.jsonl") if not p.endswith("aggregated_results.jsonl")]
    if len(replacements) != 1 or not replacements[0].endswith("/bixbench-pi.jsonl"):
        raise ValueError("Repair must contain BixBench only")
    original_bix = json.loads(original_fs.cat_file(original_root + "/bixbench-pi.jsonl"))
    repaired_bix = json.loads(repair_fs.cat_file(replacements[0]))
    if original_bix["prompt_token_ids"] != repaired_bix["prompt_token_ids"]:
        raise ValueError("Repair task identity differs from the original")
    out_fs, out_root = filesystem_for(output + suffix)
    out_fs.makedirs(out_root, exist_ok=True)
    provenance = []
    for row in before["benchmarks"]:
        name = row["file"]
        if name != "bixbench-pi.jsonl" and row["verified"] != 1:
            raise ValueError(f"Another path is not verified: {name}")
        source_fs, source_root = (
            (repair_fs, repair_root) if name == "bixbench-pi.jsonl" else (original_fs, original_root)
        )
        payload = source_fs.cat_file(source_root + "/" + name)
        out_fs.pipe_file(out_root + "/" + name, payload)
        provenance.append(
            {
                "file": name,
                "source": (repair if name == "bixbench-pi.jsonl" else original) + suffix + name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    merged = audit(output, 20)
    if not merged["all_verified"]:
        raise ValueError("Merged result does not verify all 20 paths")
    fs, root = filesystem_for(output)
    fs.pipe_file(root + "/evaluation.json", json.dumps(merged, indent=2).encode())
    fs.pipe_file(
        root + "/merge-provenance.json",
        json.dumps(
            {
                "original": original,
                "repair": repair,
                "preserved": 19,
                "replaced": "bixbench-pi.jsonl",
                "files": provenance,
                "original_failed_result": original + suffix + "bixbench-pi.jsonl",
            },
            indent=2,
        ).encode(),
    )


def merged_report(version: str, original: ArtifactStep, repair: ArtifactStep) -> ArtifactStep:
    return apply(
        "long-loop/merged-micro-report", merge_results, version=version, output=OUT, original=original, repair=repair
    )


@click.command()
@click.option("--version", required=True)
@click.option("--source", required=True)
def main(version: str, source: str) -> None:
    configure_coreweave_s3()
    run(apply("long-loop/bix-inputs", prepare_subset, version=version, output=OUT, source=source), max_concurrent=1)


if __name__ == "__main__":
    main()
