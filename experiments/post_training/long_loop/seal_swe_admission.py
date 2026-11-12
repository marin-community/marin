# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal a new SWE release only after complete per-task CPU control coverage."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.long_loop.pi_rl_data import SOURCE, image_context_digest
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.swe_verified_data import (
    RECIPE_VERSION,
    VERIFYIT_REVISION,
    load_grader_profiles,
    prepare_task,
)

AUDITED_DEGENERATE_IDS = frozenset({"swesmith-09581", "swesmith-22770"})


def admitted(receipt: dict) -> bool:
    """Require both graded controls; an infrastructure zero is never a negative control."""
    controls = receipt.get("controls", {})
    return (
        receipt["status"] == "admitted"
        and controls.get("bug", {}).get("status") == "scored"
        and controls.get("reference", {}).get("status") == "scored"
        and controls["bug"]["reward"] == 0
        and controls["reference"]["reward"] == 1
        and receipt["task_id"].rsplit("/", 1)[-1] not in AUDITED_DEGENERATE_IDS
    )


def load_admission(prefix: str, workers: int) -> tuple[dict[str, dict], dict]:
    root = StoragePath(prefix)
    plan_text = (root / "index.json").read_text()
    plan = json.loads(plan_text)
    if not plan["complete"]:
        raise ValueError("Admission is incomplete or was run with a different grader")
    load_grader_profiles(plan["runtime_profiles"])

    def read_receipt(entry):
        task_root = StoragePath(entry["uri"])
        raw = (task_root / "receipt.json").read_bytes()
        receipt = json.loads(raw)
        if receipt["task_id"] != entry["task_id"] or hashlib.sha256(raw).hexdigest() != entry["receipt_sha256"]:
            raise ValueError("Admission receipt differs from the completed index")
        inventory_raw = (task_root / "evidence-inventory.json").read_bytes()
        if hashlib.sha256(inventory_raw).hexdigest() != entry["inventory_sha256"]:
            raise ValueError("Admission evidence inventory changed after indexing")
        dependencies = None
        if admitted(receipt):
            dependency_raw = (task_root / "bootstrap/dependencies.json").read_bytes()
            if (
                hashlib.sha256(dependency_raw).hexdigest()
                != json.loads(inventory_raw)["bootstrap/dependencies.json"]["sha256"]
            ):
                raise ValueError("Admitted dependency metadata differs from the committed inventory")
            dependencies = json.loads(dependency_raw)
        return entry["task_id"], {**entry, "receipt": receipt, "dependencies": dependencies}

    receipts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_id, row in pool.map(read_receipt, plan["tasks"]):
            if task_id in receipts:
                raise ValueError("Admission index contains duplicate tasks")
            receipts[task_id] = row
    return receipts, {"plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(), "prefix": prefix, "plan": plan}


def seal(
    source: Path,
    output: Path,
    admission_prefix: str,
    workers: int,
    seed: int,
    harbor_timeout: int,
    marin_revision: str,
) -> dict:
    """Preserve the source split, replacing rejected SWE validation rows deterministically."""
    manifest = json.loads((source / "manifest.json").read_text())
    receipts, admission = load_admission(admission_prefix, workers)
    profiles = load_grader_profiles(admission["plan"]["runtime_profiles"])
    if file_digest(source / "manifest.json") != admission["plan"]["manifest_sha256"]:
        raise ValueError("The admission and release source manifests differ")
    rows_by_split = {}
    for filename in ("agentic-train.parquet", "validation.parquet"):
        if file_digest(source / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError("A source parquet changed")
        rows_by_split[filename] = pq.read_table(source / filename).to_pylist()
    expected_ids = {
        TaskSpec.model_validate_json(row["task_spec"]).id
        for rows in rows_by_split.values()
        for row in rows
        if json.loads(row["task_spec"])["source"]["dataset"] == SOURCE
    }
    if set(receipts) != expected_ids:
        raise ValueError("Every original SWE task requires exactly one admission receipt")
    output.mkdir(parents=True, exist_ok=False)
    accepted = {}
    excluded = []
    identities = []
    for rows in rows_by_split.values():
        for row in rows:
            original = TaskSpec.model_validate_json(row["task_spec"])
            if original.source.dataset != SOURCE:
                continue
            evidence = receipts[original.id]
            receipt = evidence["receipt"]
            if not admitted(receipt):
                excluded.append(
                    {
                        "task_id": original.id,
                        "reason": (
                            "audited_degenerate_task"
                            if original.id.rsplit("/", 1)[-1] in AUDITED_DEGENERATE_IDS
                            else receipt.get("reason", "failed_controls")
                        ),
                        "evidence": evidence["uri"],
                        "receipt_sha256": evidence["receipt_sha256"],
                    }
                )
                continue
            pins = receipt["pins"]
            prepared = prepare_task(
                original,
                setup_timeout=pins["setup_timeout"],
                verifier_timeout=pins["verifier_timeout"],
                harbor_verifier_timeout=harbor_timeout,
                dependency_constraints="",
                profile=profiles[evidence["runtime_sha256"]],
            )
            serialized = prepared.task.model_dump_json()
            output_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
            if output_sha256 != receipt["task_sha256"]:
                raise ValueError("The release derivation differs from the task whose controls ran")
            versions = "".join(f"{name}=={version}\n" for name, version in evidence["dependencies"]["packages"])
            accepted[original.id] = {**row, "task_spec": serialized}
            identities.append(
                {
                    "task_id": original.id,
                    "input_sha256": hashlib.sha256(row["task_spec"].encode()).hexdigest(),
                    "admitted_task_sha256": receipt["task_sha256"],
                    "output_sha256": output_sha256,
                    "observed_distribution_versions_sha256": hashlib.sha256(versions.encode()).hexdigest(),
                    "receipt_sha256": evidence["receipt_sha256"],
                    "evidence": evidence["uri"],
                    "original_instruction_sha256": prepared.original_instruction_sha256,
                    "instruction_sha256": prepared.instruction_sha256,
                    "image_context": image_context_digest(prepared.task),
                }
            )
    original_val_ids = [
        TaskSpec.model_validate_json(row["task_spec"]).id
        for row in rows_by_split["validation.parquet"]
        if json.loads(row["task_spec"])["source"]["dataset"] == SOURCE
    ]
    kept_val = [task_id for task_id in original_val_ids if task_id in accepted]
    candidates = sorted(
        (task_id for task_id in accepted if task_id not in original_val_ids),
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).digest(),
    )
    replacements = candidates[: len(original_val_ids) - len(kept_val)]
    val_ids = set(kept_val + replacements)
    if len(val_ids) != len(original_val_ids):
        raise ValueError("Too few admitted SWE tasks to preserve validation composition")
    training_ids = [json.loads(row["task_spec"])["id"] for row in rows_by_split["agentic-train.parquet"]]
    training = [accepted[task_id] for task_id in training_ids if task_id in accepted and task_id not in val_ids]
    validation = [
        row for row in rows_by_split["validation.parquet"] if json.loads(row["task_spec"])["source"]["dataset"] != SOURCE
    ] + [accepted[task_id] for task_id in kept_val + replacements]
    files = {}
    for name, rows in (("agentic-train.parquet", training), ("validation.parquet", validation)):
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pq.read_schema(source / name)), output / name, compression="zstd"
        )
        files[name] = {"rows": len(rows), "sha256": file_digest(output / name)}
    for name, data in (
        ("identities.jsonl", "".join(json.dumps(row) + "\n" for row in identities)),
        ("exclusions.json", json.dumps(excluded, indent=2)),
    ):
        (output / name).write_text(data)
        files[name] = {"sha256": file_digest(output / name)}
    result = {
        **manifest,
        "upstream_marin_revision": manifest["marin_revision"],
        "marin_revision": marin_revision,
        "execution_overlay_version": RECIPE_VERSION,
        "verifyit_revision": VERIFYIT_REVISION,
        "grader_profiles": admission["plan"]["runtime_profiles"],
        "upstream_manifest_sha256": file_digest(source / "manifest.json"),
        "admission": {key: value for key, value in admission.items() if key != "plan"},
        "controls": {"attempted": len(receipts), "kept": len(accepted), "dropped": len(excluded)},
        "agentic_train": len(training),
        "validation": len(validation),
        "validation_replacements": {"seed": seed, "task_ids": replacements},
        "prior_swe_split_preserved": not bool(replacements),
        "policy": (
            "Harness-owned exact checkout and declared setup; trusted pre-agent grading Python; "
            "ancestry-gated fresh-worktree grading"
        ),
        "files": files,
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "dependency_policy": (
            "Publish the exact tested task package without adding setup constraints after admission; "
            "retain observed pre-agent distribution inventories; each trial freezes its grading interpreter "
            "before agent access. Fresh setup may resolve different versions; agent packages never seed grading."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-prefix", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harbor-verifier-timeout", type=int, required=True)
    parser.add_argument("--marin-revision", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seal(
                args.input,
                args.output,
                args.admission_prefix,
                args.workers,
                args.seed,
                args.harbor_verifier_timeout,
                args.marin_revision,
            )["controls"]
        )
    )


if __name__ == "__main__":
    main()
