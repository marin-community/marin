# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the self-contained Micro210 policy inputs from the frozen prepared table.

Starting from the pinned ``2026.11.20-together`` table, every script-graded row's TaskSpec
is made self-contained (embedded vendored grader library, standalone driver, ambient
evalchemy/lm_eval wheels dropped from the verifier package spec, per-row provenance labels)
and every TruthfulQA row's choices are permuted with the recorded staged seed derivation.
The TruthfulQA/GPQA-Diamond/MMLU-Pro choice audit and the task selection must match the
frozen table. The prepared table and environment are staged under the new version; the
policy JSON is emitted only with a green oracle/NoP gate receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TaskRolloutSpec

from experiments.evaluation.micro210_inputs import (
    EXPECTED_CHOICE_BENCHMARKS,
    reject_fixed_choice_positions,
    truthfulqa_choice_task,
)
from experiments.evaluation.micro210_policy import POLICY_REFERENCE, SPEC_POLICY_REVISION, SPEC_POLICY_VERSION
from experiments.evaluation.micro210_spec_grading import (
    HARBOR_PI_ADAPTER_PATCH,
    MBPP_FLOAT_TOLERANCE_PATCH,
    GraderLibrary,
    driver_sha256,
    drop_unused_pi_adapter_pin,
    embed_spec_grader,
    tolerant_mbpp_float_comparison,
)

PREPARED_ROOT = "gs://marin-us-east1/evaluation-policies/micro210"
PREPARED_ROWS = 440
CHOICE_SEED_DERIVATION = "low 63 bits of first eight SHA-256 bytes of namespace + task_id, big-endian"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source_rollouts: Path, source_policy: Path) -> dict:
    """Verify the frozen 2026.11.20-together pins and return the parsed policy."""
    old_policy_bytes = source_policy.read_bytes()
    if len(old_policy_bytes) != POLICY_REFERENCE.byte_size or sha256(source_policy) != POLICY_REFERENCE.sha256:
        raise ValueError("Source policy does not match the frozen Micro210 policy reference")
    old_policy = json.loads(old_policy_bytes)
    if sha256(source_rollouts) != old_policy["prepared_rollouts_sha256"]:
        raise ValueError("Source prepared table does not match the frozen Micro210 policy")
    return old_policy


def transform_table(source_rollouts: Path, output: Path, library: GraderLibrary) -> dict[str, dict[str, int]]:
    """Write the self-contained table with the same rows, selection and rollouts."""
    table = pq.read_table(source_rollouts)
    rows = []
    for row in table.to_pylist():
        rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        task = rollout.task
        data_patches: tuple[dict, ...] = ()
        if row["benchmark"] == "mbppplus":
            task = tolerant_mbpp_float_comparison(task)
            data_patches = (MBPP_FLOAT_TOLERANCE_PATCH,)
        if task.resources.verifier and any(
            resource.path == "environment/long-loop-pi/package.json" for resource in task.resources.verifier
        ):
            task = drop_unused_pi_adapter_pin(task)
            data_patches = (*data_patches, HARBOR_PI_ADAPTER_PATCH)
        task = embed_spec_grader(task, benchmark=row["benchmark"], library=library, data_patches=data_patches)
        if row["benchmark"] == "truthfulqa":
            task = truthfulqa_choice_task(task)
        rollout = rollout.model_copy(update={"task": task})
        rows.append({**row, "task_spec": task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()})
    if len(rows) != PREPARED_ROWS:
        raise ValueError(f"Transformed table has {len(rows)} rows, expected {PREPARED_ROWS}")
    counts = reject_fixed_choice_positions(rows)
    if missing := EXPECTED_CHOICE_BENCHMARKS - counts.keys():
        raise ValueError(f"Micro210 choice audit missed benchmark(s): {sorted(missing)}")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output)
    return counts


def build_environment(old_policy: dict, digest: str, location: str, rows: list[dict], path: Path) -> None:
    """Write the environment manifest with per-row content hashes under the new staged location."""
    source_environment = json.loads((StoragePath(old_policy["prepared_inputs_uri"]) / "environment.json").read_bytes())
    source_environment["data_uri"] = location
    source_environment["source_prepared_rollouts_sha256"] = old_policy["prepared_rollouts_sha256"]
    source_environment["source_task_environment_sha256"] = old_policy["task_environment_sha256"]
    source_environment["tasks"] = [
        {
            "task_id": row["task_id"],
            "repetition_index": row["repetition_index"],
            "task_spec_sha256": hashlib.sha256(row["task_spec"].encode()).hexdigest(),
            "task_rollout_spec_sha256": hashlib.sha256(row["task_rollout_spec"].encode()).hexdigest(),
        }
        for row in rows
    ]
    path.write_text(json.dumps(source_environment, indent=2))


def grader_embed_block(library: GraderLibrary) -> dict:
    """The policy block that records the embedded grading identities."""
    return {
        "driver_sha256": driver_sha256(),
        "library_manifest_sha256": library.manifest_sha256,
        "source_wheels": library.manifest["source_wheels"],
        "patches": [
            {"path": entry["path"], "patch": entry["patch"], "note": entry["note"]}
            for entry in library.manifest["files"]
            if "patch" in entry
        ],
        "removed_ambient_packages": ["evalchemy[ifbench,mrcr,nupa5k-loose] wheel", "lm-eval wheel"],
        "bundled_nltk_data": "eval/chat_benchmarks/IFBench/.nltk_data on ifbench rows",
        "provenance_label": "task_spec.labels.micro210_grader_provenance",
        "row_data_patches": [
            {**MBPP_FLOAT_TOLERANCE_PATCH, "benchmarks": ["mbppplus"]},
            {
                **HARBOR_PI_ADAPTER_PATCH,
                "benchmarks": [
                    "bixbench-pi",
                    "tb2-recovery",
                    "ot-tblite-recovery",
                    "ds-1000-local",
                    "bfclparity-pi",
                    "swebench-verified",
                ],
            },
        ],
        "launch_config_changes": [
            {
                "change": "removed evaluation.config.terminal_bench.installed_agent.agent.kwargs.mcp_adapter_version",
                "note": (
                    "the Pi agent loads the image-provided pi-mcp-extension and no Micro210 task "
                    "configures MCP servers, so the new version carries no pin for a package the "
                    "agent does not install (#12122)"
                ),
                "issue": 12122,
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rollouts", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-policy", action="store_true", help="Emit policy.json; requires --gate-receipt")
    parser.add_argument("--gate-receipt", type=Path, help="Oracle/NoP gate receipt for the built table")
    arguments = parser.parse_args()

    old_policy = verify_source(arguments.source_rollouts, arguments.source_policy)
    library = GraderLibrary()
    arguments.output.mkdir(parents=True, exist_ok=True)
    prepared = arguments.output / "rollouts.parquet"
    counts = transform_table(arguments.source_rollouts, prepared, library)
    digest = sha256(prepared)
    location = str(StoragePath(PREPARED_ROOT) / SPEC_POLICY_VERSION / digest)
    rows = pq.read_table(prepared, columns=["task_id", "task_spec", "task_rollout_spec", "repetition_index"]).to_pylist()

    environment_path = arguments.output / "environment.json"
    build_environment(old_policy, digest, location, rows, environment_path)

    print(f"prepared: {digest} ({len(rows)} rows)")
    print(f"environment: {sha256(environment_path)}")
    print(f"prepared URI: {location}")

    if not arguments.emit_policy:
        print("policy: not emitted (run the oracle/NoP gate, then re-run with --emit-policy --gate-receipt)")
        return
    if arguments.gate_receipt is None:
        raise SystemExit("--emit-policy requires --gate-receipt")
    receipt = json.loads(arguments.gate_receipt.read_text())
    if receipt.get("prepared_rollouts_sha256") != digest:
        raise ValueError("Gate receipt does not match the built prepared table")
    if not receipt.get("pass"):
        raise ValueError(f"Oracle/NoP gate did not pass: {receipt.get('summary')}")

    policy = {**old_policy}
    installed_agent = policy["evaluation"]["config"]["terminal_bench"]["installed_agent"]
    if installed_agent.get("agent", {}).get("kwargs", {}).pop("mcp_adapter_version", None) is None:
        raise ValueError("The frozen policy does not carry the expected mcp_adapter_version pin")
    policy.update(
        policy_version=SPEC_POLICY_REVISION,
        previous_policy_sha256=POLICY_REFERENCE.sha256,
        prepared_inputs_uri=location,
        prepared_rollouts_sha256=digest,
        task_environment_sha256=sha256(environment_path),
        choice_order={
            "seed_derivation": CHOICE_SEED_DERIVATION,
            "seed_namespace": "micro210:2026.11.23-choice-shuffle:",
            "permuted_benchmarks": ["truthfulqa"],
            "accepted_letter_counts": counts,
            "seed_record": "task_spec.labels.micro210_choice_seed",
            "permutation_record": "task_spec.labels.micro210_choice_order (source indices in display order)",
        },
        grader_embed=grader_embed_block(library),
    )
    policy_path = arguments.output / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    print(f"policy: {sha256(policy_path)} ({policy_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
