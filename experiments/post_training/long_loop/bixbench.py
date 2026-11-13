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
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.archive import write_once
from experiments.post_training.long_loop.eval_runtime.audit_results import audit
from experiments.post_training.long_loop.sft import read_json

BIXBENCH_RESULTS_FILENAME = "bixbench-pi.jsonl"


def prepare_subset(*, output: str, source: str) -> None:
    fs, path = filesystem_for(prefix_join(source, "rollouts.parquet"))
    with fs.open(path, "rb") as stream:
        table = pq.read_table(stream)
    selected = table.filter(pc.equal(table["benchmark"], "bixbench-pi"))
    if selected.num_rows != 1:
        raise ValueError("Expected exactly one original BixBench smoke task")
    manifest = read_json(prefix_join(source, "environment.json"))
    manifest.update(data_uri=output, subset_of=source, subset_benchmark="bixbench-pi")
    target = StoragePath(output)
    target.mkdirs(exist_ok=True)
    with (target / "rollouts.parquet").open("wb") as stream:
        pq.write_table(selected, stream)
    (target / "environment.json").write_bytes(json.dumps(manifest, indent=2).encode())
    (target / "selection.json").write_bytes(
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


def merge_identity(original: str, repair: str) -> dict:
    """Verify task content and consumed configs, allowing only run-local locators."""
    configs = [read_json(prefix_join(root, "terminal.json"))["config"] for root in (original, repair)]
    row_hashes = []
    for config in configs:
        data = config["inputs"]["validation_data"]
        if len(data) != 1:
            raise ValueError("Expected one pinned validation input")
        fs, key = filesystem_for(prefix_join(data[0]["uri"], data[0]["relative_path"]))
        with fs.open(key, "rb") as stream:
            rows = [row for row in pq.read_table(stream).to_pylist() if row["benchmark"] == "bixbench-pi"]
        if len(rows) != 1:
            raise ValueError("Expected one BixBench task identity")
        row_hashes.append(hashlib.sha256(json.dumps(rows[0], sort_keys=True, default=str).encode()).hexdigest())
    if row_hashes[0] != row_hashes[1]:
        raise ValueError("Repair task identity differs from the original")
    # These fields locate a run, its output, or its equivalent adopted artifact.
    # Model URI, tokenizer pin, judge model, sampling and verifier settings remain compared.
    run_local = [
        "artifacts",
        "run",
        "iris/job_name",
        "ray/rendezvous_dir",
        "ray/log_dir",
        "inputs/validation_data",
        "inputs/model/identity",
        "runtime/task_env/JUDGE_BASE_URL",
        "skyrl/data/val_data",
        "skyrl/terminal_bench_config/installed_agent/output_path",
        "skyrl/environment/task_sessions/remote_judge/base_url",
        "skyrl/trainer/export_path",
        "skyrl/trainer/ckpt_path",
        "skyrl/trainer/run_name",
        "skyrl/generator/trajectory_retention/run_id",
        "skyrl/generator/trajectory_retention/model_source_identity",
        "skyrl/generator/trajectory_retention/output_path",
    ]
    for role in ("policy", "ref"):
        for field in ("revision", "tokenizer_revision", "source_identity"):
            run_local.append(f"skyrl/trainer/{role}/model/{field}")
    normalized = json.loads(json.dumps(configs))
    for config in normalized:
        for path in run_local:
            keys = path.split("/")
            parent = config
            for key in keys[:-1]:
                parent = parent[key]
            del parent[keys[-1]]
    if normalized[0] != normalized[1]:
        raise ValueError("Repair model/runtime or task/verifier configuration differs")
    return {
        "task_row_sha256": row_hashes[0],
        "compared_config_sha256": hashlib.sha256(json.dumps(normalized[0], sort_keys=True).encode()).hexdigest(),
        "excluded_run_local_fields": run_local,
        "terminal_uris": [prefix_join(root, "terminal.json") for root in (original, repair)],
    }


def merge_results(*, output: str, original: str, repair: str) -> None:
    identity = merge_identity(original, repair)
    before = audit(original, 20)
    after = audit(repair, 1)
    if before["completed_rollouts"] != 20 or not after["all_verified"]:
        raise ValueError("Merge requires the original 20 results and one verified repair")
    suffix = "exports/dumped_evals/eval_only"
    original_fs, original_root = filesystem_for(prefix_join(original, suffix))
    repair_fs, repair_root = filesystem_for(prefix_join(repair, suffix))
    replacements = [
        p for p in repair_fs.glob(prefix_join(repair_root, "*.jsonl")) if not p.endswith("aggregated_results.jsonl")
    ]
    if len(replacements) != 1 or not replacements[0].endswith("/" + BIXBENCH_RESULTS_FILENAME):
        raise ValueError("Repair must contain BixBench only")
    original_bix = json.loads(original_fs.cat_file(prefix_join(original_root, BIXBENCH_RESULTS_FILENAME)))
    repaired_bix = json.loads(repair_fs.cat_file(replacements[0]))
    identity["original_prompt_is_prefix"] = (
        repaired_bix["prompt_token_ids"][: len(original_bix["prompt_token_ids"])] == original_bix["prompt_token_ids"]
    )
    out_fs, out_root = filesystem_for(prefix_join(output, suffix))
    out_fs.makedirs(out_root, exist_ok=True)
    provenance = []
    for row in before["benchmarks"]:
        name = row["file"]
        if name != BIXBENCH_RESULTS_FILENAME and row["verified"] != 1:
            raise ValueError(f"Another path is not verified: {name}")
        source_fs, source_root = (
            (repair_fs, repair_root) if name == BIXBENCH_RESULTS_FILENAME else (original_fs, original_root)
        )
        payload = source_fs.cat_file(prefix_join(source_root, name))
        write_once(prefix_join(prefix_join(output, suffix), name), payload)
        provenance.append(
            {
                "file": name,
                "source": prefix_join(
                    prefix_join(repair if name == BIXBENCH_RESULTS_FILENAME else original, suffix), name
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    merged = audit(output, 20)
    if not merged["all_verified"]:
        raise ValueError("Merged result does not verify all 20 paths")
    write_once(prefix_join(output, "evaluation.json"), json.dumps(merged, indent=2).encode())
    write_once(
        prefix_join(output, "merge-provenance.json"),
        json.dumps(
            {
                "identity": identity,
                "original": original,
                "repair": repair,
                "preserved": 19,
                "replaced": BIXBENCH_RESULTS_FILENAME,
                "files": provenance,
                "original_failed_result": prefix_join(prefix_join(original, suffix), BIXBENCH_RESULTS_FILENAME),
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
