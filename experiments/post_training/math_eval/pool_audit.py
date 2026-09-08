# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-sided mechanical audits against the frozen runtime verifier implementations."""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import skyrl_gym
from rigging.filesystem.storage_path import StoragePath
from skyrl_gym.envs.data_contracts import get_data_contract

from experiments.post_training.math_eval.pool import canonical_json

VERIFIER_REVISION = "fa6aae365f70e50bca5218ed2f3d49067e6dcc26"
VERIFIER_FILES = {
    "envs/data_contracts.py": "32877a46501969351c5ee1896fbabb208e54382cfd8d84801ce146583d32c325",
    "envs/aime/verifier.py": "d88156b40e5914d216ab7cdcee19f6fe34ef98672a08bf885f2f7b1a3042124c",
    "envs/aime/utils.py": "af348f9e5d92c45767d0e7bd4a55eec642acc303efd588ed344d59648af262c2",
    "envs/gsm8k/utils.py": "f672bbccb69d0eaea08ac915807a54016bc5905a267ec9e999177dc4f9bdace6",
    "envs/reasoning_gym/scoring.py": "fdb9cd9a94c2d07f4e3ea43b2a90831520a711161939cc42a3fd05fc7446b03b",
}


@dataclass(frozen=True)
class MechanicalAudit:
    prompt_sha256: str
    split: str
    passed: bool
    normalized_gold: str | None
    failure: str | None


def verify_verifier_sources() -> None:
    """Refuse an audit performed with a different verifier or procedural task package."""
    root = Path(skyrl_gym.__file__).parent
    for filename, expected in VERIFIER_FILES.items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Verifier source changed: {filename}")
    if importlib.metadata.version("reasoning-gym") != "0.1.25":
        raise ValueError("Reasoning-gym version changed")


def audit_row(row: dict[str, Any]) -> MechanicalAudit:
    """Check that a known-good answer passes and an invalid answer fails."""
    env = row["env_class"]
    contract = get_data_contract(env)
    gold = row["gold"]
    try:
        answer = json.loads(gold)["entry"]["answer"] if env == "reasoning_gym" else gold
        marker = "#### " if env == "gsm8k" else "Answer: "
        normalized = contract.validate_example(gold, marker + answer, marker + "[INVALID]")
    except (ValueError, TypeError, KeyError) as error:
        return MechanicalAudit(row["prompt_sha256"], row["split"], False, None, str(error))
    return MechanicalAudit(row["prompt_sha256"], row["split"], True, normalized, None)


def audit_manifest(manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Write row-level proofs and fail the gate on any evaluation parse failure."""
    if not all(uri.startswith("s3://marin-us-east-02a/") for uri in (manifest_uri, output_uri)):
        raise ValueError("Mechanical pool audits stay in east-02a")
    verify_verifier_sources()
    content = StoragePath(manifest_uri).read_bytes()
    rows = pq.read_table(pa.BufferReader(content)).to_pylist()
    selection = json.loads(StoragePath(manifest_uri.rsplit("/", 1)[0] + "/selection.json").read_bytes())
    if hashlib.sha256(canonical_json(rows).encode()).hexdigest() != selection["manifest_sha256"]:
        raise ValueError("Manifest does not match the frozen selection")
    results = [audit_row(row) for row in rows]
    StoragePath(output_uri).write_bytes(
        "".join(json.dumps(asdict(result), sort_keys=True) + "\n" for result in results).encode()
    )
    failures = [result for result in results if not result.passed]
    summary = {
        "rows": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "evaluation_failed": sum(result.split != "train" for result in failures),
        "manifest_file_sha256": hashlib.sha256(content).hexdigest(),
        "verifier_revision": VERIFIER_REVISION,
    }
    print("KE2_MECHANICAL_AUDIT " + json.dumps(summary))
    if summary["evaluation_failed"]:
        raise ValueError("Evaluation pool contains mechanical verifier failures")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    audit_manifest(args.manifest_uri, args.output_uri)
