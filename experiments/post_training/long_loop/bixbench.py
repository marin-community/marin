# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BixBench continuations and provenance-preserving result recovery."""

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
from experiments.post_training.long_loop.eval_runtime.audit_results import audit, require_evaluation_coverage
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
    StoragePath(prefix_join(output, suffix)).mkdirs()
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


def merge_verifier_recovery(
    *,
    output: str,
    original: str,
    selection: str,
    recovery: str,
    recovery_manifest_sha256: str,
    accepted_unscored_attempts: list[dict],
) -> None:
    """Publish a separate scored view of one original BixBench verifier timeout."""
    manifest_uri = prefix_join(recovery, "manifest.json")
    manifest_bytes = StoragePath(manifest_uri).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != recovery_manifest_sha256:
        raise ValueError("Verifier recovery manifest changed")
    manifest = json.loads(manifest_bytes)
    for item in manifest["files"]:
        if "/" in item["name"] or item["name"] in {".", ".."}:
            raise ValueError("Recovery filenames must be direct children")
        raw = StoragePath(prefix_join(recovery, item["name"])).read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Verifier recovery payload changed")
    names = {item["name"] for item in manifest["files"]}
    if not {"result.json", "grade.json", "answer.txt", "llm_judge.py", "ground_truth.json"} <= names:
        raise ValueError("Recovery lacks required verifier evidence")
    result = read_json(prefix_join(recovery, "result.json"))
    if (
        result["status"] != "completed"
        or result["returncode"] != 0
        or result["candidate_resampled"] is not False
        or result["original_results_mutated"] is not False
    ):
        raise ValueError("Recovery must grade the preserved original attempt successfully")
    score = float(result["verifier_result"]["rewards"]["reward"])
    grade = read_json(prefix_join(recovery, "grade.json"))
    if not 0 <= score <= 1 or grade["rewards"]["reward"] != score:
        raise ValueError("Recovery has no matching normalized verifier score")
    selected = read_json(prefix_join(selection, "selection.json"))
    identities = [item for item in selected["identities"] if item["task_id"] == result["task_id"]]
    if len(identities) != 1 or identities[0]["data_source"] != "bixbench-pi":
        raise ValueError("Recovery must identify one selected BixBench attempt")
    identity = identities[0]
    source = result["source"]
    original_identity = read_json(prefix_join(source, "identity.json"))
    if (
        original_identity["task_id"] != result["task_id"]
        or original_identity["attempt_id"] != result["original_attempt"]
        or original_identity["session_id"]
        != f"eval-None-0--{identity['dataset_row_uid']}_{identity['repetition_index']}"
    ):
        raise ValueError("Recovered answer belongs to another selected attempt")
    trial_root = prefix_join(source, f"trials/{result['original_attempt']}")
    trial = read_json(prefix_join(trial_root, "result.json"))
    if trial["exception_info"]["exception_type"] != "VerifierTimeoutError":
        raise ValueError("Original attempt is not a verifier timeout")
    if trial["verifier_result"] is not None and trial["verifier_result"]["rewards"] is not None:
        raise ValueError("Cannot replace an existing original verifier reward")
    if trial["config"]["verifier"]["override_timeout_sec"] != result["verifier_timeout_seconds"]:
        raise ValueError("Recovery changed the original verifier deadline")
    trajectory_bytes = StoragePath(prefix_join(trial_root, "agent/trajectory.json")).read_bytes()
    if hashlib.sha256(trajectory_bytes).hexdigest() != result["source_trajectory_sha256"]:
        raise ValueError("Original sampled trajectory changed")
    answer = StoragePath(prefix_join(recovery, "answer.txt")).read_bytes()
    last = json.loads(trajectory_bytes)["steps"][-1]
    if (
        last["tool_calls"][0]["arguments"] != {"command": "cat /workspace/answer.txt"}
        or last["observation"]["results"][0]["source_call_id"] != last["tool_calls"][0]["tool_call_id"]
        or last["observation"]["results"][0]["content"].encode() != answer
        or last["observation"]["results"][0]["extra"]["is_error"]
    ):
        raise ValueError("Recovery answer differs from the final retained tool observation")
    for name in ("llm_judge.py", "ground_truth.json"):
        if (
            StoragePath(prefix_join(source, f"task/tests/{name}")).read_bytes()
            != StoragePath(prefix_join(recovery, name)).read_bytes()
        ):
            raise ValueError("Recovery did not use the original frozen verifier")
    before = audit(original, selected["attempts"], selected["attempts_per_data_source"], selected["identities"])
    expected_unscored = [
        {"data_source": "bixbench-pi", "uid": identity["dataset_row_uid"], "exception_type": "VerifierTimeoutError"},
        *accepted_unscored_attempts,
    ]
    if (
        before["completed_rollouts"] != selected["attempts"]
        or not before["source_coverage_matches"]
        or not before["identity_coverage_matches"]
        or sorted(json.dumps(row, sort_keys=True) for row in before["unscored_attempts"])
        != sorted(json.dumps(row, sort_keys=True) for row in expected_unscored)
    ):
        raise ValueError("Native evaluation has missing identities or additional unscored attempts")
    if read_json(prefix_join(original, "terminal.json"))["result"]["state"] != "succeeded":
        raise ValueError("Native evaluation must finish before verifier recovery is merged")
    suffix = "exports/dumped_evals/eval_only"
    root = StoragePath(original) / suffix
    files = []
    replaced = 0
    for path in sorted((root / "*.jsonl").glob(), key=str):
        raw = path.read_bytes()
        write_once(prefix_join(output, f"original-native/{path.name}"), raw)
        files.append({"source": str(path), "name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        if path.name == "aggregated_results.jsonl":
            continue
        lines = []
        for line in raw.splitlines(keepends=True):
            row = json.loads(line)
            if row["data_source"] == "bixbench-pi" and str(row["uid"]) == identity["dataset_row_uid"]:
                row["verifier_recovery"] = {
                    "manifest": manifest_uri,
                    "manifest_sha256": recovery_manifest_sha256,
                    "original_verification": row["verification"],
                    "original_attempt": result["original_attempt"],
                    "candidate_resampled": False,
                }
                row["verification"] = {
                    "status": "verified",
                    "reason": "Verifier-only recovery of the retained original answer",
                    "score": score,
                    "passed": score > 0,
                    "score_min": 0,
                    "score_max": 1,
                    "diagnostics": {
                        **row["verification"]["diagnostics"],
                        "recovery_manifest": manifest_uri,
                        "recovery_manifest_sha256": recovery_manifest_sha256,
                    },
                }
                line = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                replaced += 1
            lines.append(line)
        write_once(prefix_join(output, f"{suffix}/{path.name}"), b"".join(lines))
    if replaced != 1:
        raise ValueError("Recovery must replace exactly one native verification")
    for name in ("resolved-launch.yaml", "terminal.json"):
        write_once(prefix_join(output, name), StoragePath(prefix_join(original, name)).read_bytes())
    after = audit(output, selected["attempts"], selected["attempts_per_data_source"], selected["identities"])
    after["accepted_unscored_attempts"] = accepted_unscored_attempts
    require_evaluation_coverage(after)
    write_once(prefix_join(output, "evaluation.json"), json.dumps(after, indent=2).encode())
    write_once(
        prefix_join(output, "merge-provenance.json"),
        json.dumps(
            {
                "original": original,
                "selection": selection,
                "recovery_manifest": manifest_uri,
                "recovery_manifest_sha256": recovery_manifest_sha256,
                "identity": identity,
                "native_scored": before["verified_rollouts"],
                "recovered_scored": after["verified_rollouts"],
                "all_verified": after["all_verified"],
                "accepted_unscored_attempts": accepted_unscored_attempts,
                "source_files": files,
                "metadata": "Verbatim original job snapshots; this view did not run candidate generation",
                "judge_model": result["judge_model"],
                "judge_revision": result["judge_revision"],
                "verifier_timeout_seconds": result["verifier_timeout_seconds"],
                "original_aggregate": "original-native/aggregated_results.jsonl",
                "metrics": "Scored view metrics come from evaluation.json; native aggregate remains unchanged",
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
