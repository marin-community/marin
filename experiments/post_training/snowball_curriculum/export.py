# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the frozen mixed inventory using explicit current rollout harnesses."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, rollout_spec, shell_harness
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ResourceGroups, ScriptGrader, Source, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.shell import ShellToolConfig

from experiments.post_training.snowball_curriculum.bundle import file_sha256, verify_manifest, write_manifest
from experiments.post_training.snowball_curriculum.native import native_closure_task

TOOLS = "shellbox"
PREFILL = "prefill"
RUNTIME_SCHEMA = pa.schema(
    [
        (name, pa.float64() if name == "inclusion_probability" else pa.string())
        for name in (
            "task_rollout_spec",
            "task_id",
            "source_id",
            "source_contract_id",
            "source_public_content_id",
            "contract_id",
            "public_content_id",
            "source_contract_kind",
            "family",
            "difficulty",
            "inclusion_probability",
            "teacher_route",
        )
    ]
)
DEADLINES = frozenset(
    {
        "model_turn_timeout",
        "tool_turn_timeout",
        "total_turn_timeout",
        "attempt_timeout",
        "verifier_timeout",
        "cleanup_timeout",
    }
)


class Deployment(BaseModel):
    """Explicit machine, tool-command and attempt choices for this exported pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    limits: RolloutLimits
    command_timeout: float | None
    shell_tool: ShellToolConfig
    task_machine: MachineRuntimeSpec | None
    verifier_machine: MachineRuntimeSpec | None


class DerivedDeployment(BaseModel):
    """Declared dependency replacement and outer deadline changes, with source evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    packages_locks: dict[str, str]
    environment_provenance_sha256: str
    deadlines: dict[str, float | None]
    verifier_startup_timeout: float | None


def selected_task(row: Mapping[str, Any]) -> tuple[TaskSpec, str, list[str]]:
    """Keep canonical tasks intact; convert native raw records through the shared importer."""
    if row["contract_kind"] == "task_spec":
        task = TaskSpec.model_validate_json(row["task_json"])
        if (semantic_digest(task, True), semantic_digest(task, False)) != (row["contract_id"], row["public_content_id"]):
            raise ValueError("Frozen canonical task identity differs from the current schema")
        return task, TOOLS, []
    if row["contract_kind"] != "native-structure_closure":
        raise ValueError("Unknown selected source contract kind")
    raw = row["lowered_task_spec"]
    original = native_closure_task(raw, row.get("teacher_route") or "")
    if (original.task_id, original.contract_id, original.public_content_id) != (
        row["task_id"],
        row["contract_id"],
        row["public_content_id"],
    ):
        raise ValueError("Frozen native source identity differs from its original payload")
    document = json.loads(raw)
    native = document["task"]
    converted = structure_closure_task(
        RawRow(row["task_id"], Source.model_validate(native["source"]), {"lowered_task_spec": raw})
    )
    if not isinstance(converted, NormalizedTask):
        raise ValueError(f"Native task conversion failed: {row['task_id']}")
    task = converted.task
    if task.resources != ResourceGroups.model_validate(native["resources"]):
        raise ValueError("Native private verifier resources changed")
    # This deliberately changes conditioning and grading: historical token IDs and
    # terminal normalization remain source audit, not requirements of current tasks.
    return task, PREFILL, ["context", "grader"]


def rollout_record(row: dict, deployment: Deployment) -> tuple[dict, dict]:
    task, harness_name, changes = selected_task(row)
    if task.id != row["task_id"]:
        raise ValueError("Export changed the selected logical task ID")
    grader = task.grader
    environment = (
        grader.environment
        if isinstance(grader, ScriptGrader | VerifyitGrader) and not grades_with_judge(grader)
        else None
    )
    spec = rollout_spec(
        task,
        {
            **deployment.limits.model_dump(),
            "task_session": harness_name,
            "command_timeout": deployment.command_timeout,
            "shell_tool": deployment.shell_tool.model_dump(),
        },
        task_machine=deployment.task_machine,
        verifier_machine=deployment.verifier_machine if environment is not None else None,
    )
    check_harness(spec)
    output = {
        "task_rollout_spec": spec.model_dump_json(),
        "task_id": task.id,
        "source_id": row["source_id"],
        "source_contract_id": row.get("source_contract_id") or row["contract_id"],
        "source_public_content_id": row.get("source_public_content_id") or row["public_content_id"],
        "contract_id": semantic_digest(task, True),
        "public_content_id": semantic_digest(task, False),
        "source_contract_kind": row["contract_kind"],
        "family": row.get("family"),
        "difficulty": row["difficulty"],
        "inclusion_probability": row["inclusion_probability"],
        "teacher_route": row.get("teacher_route"),
    }
    audit = {
        "source_record_sha256": canonical_sha256(row),
        "source_task_sha256": canonical_sha256(
            json.loads(row["task_json"]) if harness_name == TOOLS else json.loads(row["lowered_task_spec"])["task"]
        ),
        "source_runtime_sha256": (
            None
            if harness_name == TOOLS
            else canonical_sha256({key: json.loads(row["lowered_task_spec"])[key] for key in ("runtime", "session")})
        ),
        "source_payload_sha256": (
            hashlib.sha256(
                (row["task_json"] if harness_name == TOOLS else row["lowered_task_spec"]).encode()
            ).hexdigest()
        ),
        "source_contract_id": row.get("source_contract_id") or row["contract_id"],
        "source_public_content_id": row.get("source_public_content_id") or row["public_content_id"],
        "input_contract_id": row["contract_id"],
        "input_public_content_id": row["public_content_id"],
        "contract_id": output["contract_id"],
        "public_content_id": output["public_content_id"],
        "task_sha256": canonical_sha256(task.model_dump(mode="json")),
        "runtime_sha256": hashlib.sha256(output["task_rollout_spec"].encode()).hexdigest(),
        "task_semantic_changes": changes,
        "historical_equivalence_claimed": False,
    }
    return output, audit


def check_harness(spec: TaskRolloutSpec) -> None:
    """Use the same explicit wire discriminator and tool settings as the trainer."""
    if spec.harness == PREFILL:
        harness = PrefillHarness()
    elif is_shell_harness(spec.harness):
        harness = shell_harness(spec.harness)
    else:
        raise ValueError(f"Unknown rollout harness: {spec.harness}")
    harness.check(spec.task, spec.limits)


def derived_record(row: dict, change: DerivedDeployment) -> tuple[dict, dict]:
    if set(change.deadlines) - DEADLINES:
        raise ValueError("Only declared outer deadlines may change")
    original = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
    task = original.task
    if task.id != row["task_id"]:
        raise ValueError("Embedded task differs from the selected logical task ID")
    if (semantic_digest(task, True), semantic_digest(task, False)) != (row["contract_id"], row["public_content_id"]):
        raise ValueError("Source deployment task digests differ")
    grader = task.grader
    semantic_changes = []
    old_lock = None
    if isinstance(grader, ScriptGrader | VerifyitGrader) and grader.environment is not None:
        lock = grader.environment.packages_lock
        if lock in change.packages_locks:
            if len(change.environment_provenance_sha256) != 64:
                raise ValueError("A grader dependency replacement requires its provenance SHA256")
            old_lock = lock
            environment = grader.environment.model_copy(update={"packages_lock": change.packages_locks[lock]})
            task = task.model_copy(update={"grader": grader.model_copy(update={"environment": environment})})
            semantic_changes.append("grader.environment.packages_lock")
    limits = RolloutLimits.model_validate({**original.limits.model_dump(), **change.deadlines})
    verifier = original.verifier_machine
    if verifier is not None and change.verifier_startup_timeout is not None:
        verifier = MachineRuntimeSpec.model_validate(
            {**verifier.model_dump(), "startup_timeout": change.verifier_startup_timeout}
        )
    spec = original.model_copy(update={"task": task, "limits": limits, "verifier_machine": verifier})
    check_harness(spec)
    restored = spec.model_dump(mode="json")
    restored["limits"] = original.limits.model_dump(mode="json")
    restored["verifier_machine"] = (
        None if original.verifier_machine is None else original.verifier_machine.model_dump(mode="json")
    )
    if semantic_changes:
        restored["task"]["grader"]["environment"]["packages_lock"] = old_lock
    if restored != original.model_dump(mode="json"):
        raise ValueError("Deployment derivation exceeded its explicit field whitelist")
    output = {
        **row,
        "task_rollout_spec": spec.model_dump_json(),
        "contract_id": semantic_digest(task, True),
        "public_content_id": semantic_digest(task, False),
    }
    audit = {
        "source_contract_id": row["source_contract_id"],
        "source_public_content_id": row["source_public_content_id"],
        "input_contract_id": row["contract_id"],
        "input_public_content_id": row["public_content_id"],
        "contract_id": output["contract_id"],
        "public_content_id": output["public_content_id"],
        "input_runtime_sha256": hashlib.sha256(row["task_rollout_spec"].encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(output["task_rollout_spec"].encode()).hexdigest(),
        "input_task_sha256": canonical_sha256(original.task.model_dump(mode="json")),
        "task_sha256": canonical_sha256(task.model_dump(mode="json")),
        "task_semantic_changes": semantic_changes,
        "runtime_deadlines_changed": {
            name: {"old": getattr(original.limits, name), "new": getattr(limits, name)}
            for name in change.deadlines
            if getattr(original.limits, name) != getattr(limits, name)
        },
    }
    return output, audit


def _write_runtime_bundle(
    source: Path,
    expected_sha256: str,
    output: Path,
    schema: pa.Schema,
    transform: Callable[[dict], tuple[dict, dict]],
    configuration: dict,
    source_uri: str,
    source_manifest_sha256: str,
) -> dict:
    source_path = StoragePath(str(source))
    if not source_uri:
        raise ValueError("A resolvable source artifact URI is required")
    verify_manifest(StoragePath(str(source.parent)), source_manifest_sha256)
    if file_sha256(source_path) != expected_sha256:
        raise ValueError("Input Parquet SHA256 differs from its immutable pin")
    output.mkdir(parents=True, exist_ok=False)
    parquet = pq.ParquetFile(source)
    counts: Counter[str] = Counter()
    seen = set()
    with (
        pq.ParquetWriter(output / "train.parquet", schema, compression="zstd") as writer,
        (output / "deployment-contract-mapping.jsonl").open("w") as audit,
    ):
        for batch in parquet.iter_batches(batch_size=128):
            records = []
            for row in batch.to_pylist():
                record, mapping = transform(row)
                if record["task_id"] in seen:
                    raise ValueError("Duplicate selected logical task ID")
                seen.add(record["task_id"])
                serialized = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])
                counts[PREFILL if serialized.harness == PREFILL else TOOLS] += 1
                audit.write(
                    json.dumps(
                        {"source_row_index": len(seen) - 1, "task_id": record["task_id"], **mapping}, sort_keys=True
                    )
                    + "\n"
                )
                records.append(record)
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
    if file_sha256(source_path) != expected_sha256:
        raise ValueError("Input changed while exporting")
    verify_manifest(StoragePath(str(source.parent)), source_manifest_sha256)
    if len(seen) != parquet.metadata.num_rows:
        raise ValueError("Export row count differs from its frozen source")
    report = {
        "format": 1,
        "source_parquet_sha256": expected_sha256,
        "source_file": source.name,
        "source_uri": source_uri,
        "source_artifact_manifest_sha256": source_manifest_sha256,
        "selected_rows": len(seen),
        "exported_rows": len(seen),
        "filtered_rows": 0,
        "harness_counts": dict(counts),
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "source_identities_preserved": True,
        "readiness": (
            "Schema and harness checks only; production factory/credential checks, exact loader admission "
            "and actual model transport are separate release gates."
        ),
        "engine_checks_performed": False,
        "model_inference_performed": False,
    }
    (output / "provenance.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    write_manifest(StoragePath(str(output)))
    return report


def export_pool(
    source: Path,
    expected_sha256: str,
    output: Path,
    deployment: Deployment,
    *,
    source_uri: str,
    source_manifest_sha256: str,
) -> dict:
    """Create a new runtime artifact without editing or filtering the selected inventory."""
    return _write_runtime_bundle(
        source,
        expected_sha256,
        output,
        RUNTIME_SCHEMA,
        lambda row: rollout_record(row, deployment),
        deployment.model_dump(mode="json"),
        source_uri,
        source_manifest_sha256,
    )


def derive_deployment(
    source: Path,
    expected_sha256: str,
    output: Path,
    change: DerivedDeployment,
    *,
    source_uri: str,
    source_manifest_sha256: str,
) -> dict:
    """Version dependency/timeout changes while preserving source and logical task identities."""
    return _write_runtime_bundle(
        source,
        expected_sha256,
        output,
        pq.ParquetFile(source).schema_arrow,
        lambda row: derived_record(row, change),
        change.model_dump(mode="json"),
        source_uri,
        source_manifest_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "derive"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "export":
        result = export_pool(
            args.source,
            args.source_sha256,
            args.output,
            Deployment.model_validate_json(args.configuration.read_text()),
            source_uri=args.source_uri,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    else:
        result = derive_deployment(
            args.source,
            args.source_sha256,
            args.output,
            DerivedDeployment.model_validate_json(args.configuration.read_text()),
            source_uri=args.source_uri,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
