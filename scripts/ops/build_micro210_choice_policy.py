# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a new Micro210 choice-order policy from the frozen prepared inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from experiments.evaluation.micro210_inputs import CHOICE_SEED_NAMESPACE, shuffle_micro210_inputs
from experiments.evaluation.micro210_policy import CHOICE_POLICY_VERSION, POLICY_REFERENCE

PREPARED_ROOT = "gs://marin-us-east1/evaluation-policies/micro210"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_choice_policy(source_rollouts: Path, source_environment: Path, source_policy: Path, output: Path) -> None:
    """Verify the old version and write new, content-pinned policy inputs."""
    old_policy_bytes = source_policy.read_bytes()
    if len(old_policy_bytes) != POLICY_REFERENCE.byte_size or sha256(source_policy) != POLICY_REFERENCE.sha256:
        raise ValueError("Source policy does not match the frozen Micro210 policy reference")
    old_policy = json.loads(old_policy_bytes)
    if sha256(source_rollouts) != old_policy["prepared_rollouts_sha256"]:
        raise ValueError("Source prepared table does not match the frozen Micro210 policy")
    if sha256(source_environment) != old_policy["task_environment_sha256"]:
        raise ValueError("Source environment does not match the frozen Micro210 policy")
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "rollouts.parquet"
    counts = shuffle_micro210_inputs(str(source_rollouts), str(prepared))
    digest = sha256(prepared)
    location = f"{PREPARED_ROOT}/{CHOICE_POLICY_VERSION}/{digest}"
    rows = pq.read_table(prepared, columns=["task_id", "task_spec", "task_rollout_spec", "repetition_index"]).to_pylist()
    environment = json.loads(source_environment.read_bytes())
    environment["data_uri"] = location
    environment["tasks"] = [
        {
            "task_id": row["task_id"],
            "repetition_index": row["repetition_index"],
            "task_spec_sha256": hashlib.sha256(row["task_spec"].encode()).hexdigest(),
            "task_rollout_spec_sha256": hashlib.sha256(row["task_rollout_spec"].encode()).hexdigest(),
        }
        for row in rows
    ]
    environment_path = output / "environment.json"
    environment_path.write_text(json.dumps(environment, indent=2))
    policy = {**old_policy}
    policy.update(
        policy_version=CHOICE_POLICY_VERSION,
        previous_policy_sha256=POLICY_REFERENCE.sha256,
        prepared_inputs_uri=location,
        prepared_rollouts_sha256=digest,
        task_environment_sha256=sha256(environment_path),
        choice_order={
            "seed_derivation": "first 63 bits of SHA-256(namespace + task_id), big-endian",
            "seed_namespace": CHOICE_SEED_NAMESPACE,
            "permuted_benchmarks": ["truthfulqa"],
            "accepted_letter_counts": counts,
            "seed_record": "task_spec.labels.micro210_choice_seed",
            "permutation_record": "task_spec.labels.micro210_choice_order (source indices in display order)",
        },
    )
    policy_path = output / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    print(f"prepared: {digest} ({len(rows)} rows)")
    print(f"environment: {sha256(environment_path)}")
    print(f"policy: {sha256(policy_path)} ({policy_path.stat().st_size} bytes)")
    print(f"prepared URI: {location}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rollouts", type=Path, required=True)
    parser.add_argument("--source-environment", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build_choice_policy(
        arguments.source_rollouts, arguments.source_environment, arguments.source_policy, arguments.output
    )


if __name__ == "__main__":
    main()
