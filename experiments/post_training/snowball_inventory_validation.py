# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze all 910 original development rows using the maintained runtime exporter.

Native closure rows become text prefills with the current grader. Their original
payloads and pairing identities remain audit evidence; historical conditioning and
terminal normalization are not claimed equivalent to this deployment.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from skyrl_gym.rollout_specs import rollout_spec, shell_harness
from skyrl_train.dataset.tasks import TaskDataset, task_prompt, task_prompt_token_count
from skyrl_train.tokenizer import create_tokenizer
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import ShellToolConfig
from verifyit.json_objects import unique_object

from experiments.post_training.snowball_curriculum.bundle import file_sha256, write_manifest
from experiments.post_training.snowball_curriculum.export import Deployment, rollout_record
from experiments.post_training.snowball_curriculum.native import native_closure_task
from experiments.post_training.snowball_inventory_panel import FAMILY_COUNTS

GRADER_IMAGE = "ghcr.io/penfever/smartie-grader@sha256:398753f1ef9e163dd4ae503ce328a441b919bbd381b50cba32af2c564e9b0bc8"
FAMILIES = {
    "ultra_sft_step3200_calendar_v2": "calendar",
    "ultra_sft_step3200_comp_coding": "competitive_coding",
    "ultra_sft_step3200_reasoning_gym": "reasoning_gym",
    "ultra_sft_step3200_stem_mcqa": "stem",
}
IDENTITY_COLUMNS = ("task_id", "contract_id", "public_content_id", "source_contract_id", "source_public_content_id")


def restored_script_rollout(raw: str) -> tuple[TaskRolloutSpec, str]:
    """Restore original script graders and per-row machines through production lowering."""
    document = json.loads(raw, object_pairs_hook=unique_object)
    original = document["task"]
    task = TaskSpec.model_validate({key: value for key, value in original.items() if key != "schema_version"})
    if not isinstance(task.grader, ScriptGrader) or task.grader.environment.docker_image != GRADER_IMAGE:
        raise ValueError("The development panel requires its original pinned script grader")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("Only the original one-turn panel is supported")
    if document["runtime"]["task_machine"] is not None:
        raise ValueError("The development panel must not have a task machine")
    machine = MachineRuntimeSpec.model_validate(document["runtime"]["verifier_machine"])
    if machine.backend != "daytona":
        raise ValueError("Preserve the source Daytona verifier backend")
    spec = rollout_spec(task, {**document["session"], "max_total_tokens": None}, verifier_machine=machine)
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    contract = json.loads(resource_bytes(config), object_pairs_hook=unique_object)["contract"]
    return spec, FAMILIES[contract["dataset"]]


@dataclass(frozen=True)
class PanelRecord:
    record: dict
    audit: dict
    deployment: Deployment


def panel_record(row: dict, index: int) -> PanelRecord:
    """Bind one historical panel row through the same exporter as the training pool."""
    raw = row["lowered_task_spec"]
    document = json.loads(raw, object_pairs_hook=unique_object)
    original = document["task"]
    if original["source"]["dataset"] == "structure-closure":
        selected = asdict(native_closure_task(raw, row.get("teacher_route") or ""))
        family = "structural"
        limits = RolloutLimits.model_validate(
            {
                **{key: value for key, value in document["session"].items() if key in RolloutLimits.model_fields},
                "max_total_tokens": None,
            }
        )
        deployment = Deployment(
            limits=limits, command_timeout=None, shell_tool=ShellToolConfig(), task_machine=None, verifier_machine=None
        )
    else:
        spec, family = restored_script_rollout(raw)
        selected = {
            "task_id": spec.task.id,
            "contract_kind": "task_spec",
            "task_json": spec.task.model_dump_json(),
            "contract_id": semantic_digest(spec.task, True),
            "public_content_id": semantic_digest(spec.task, False),
            "source_contract_id": canonical_sha256(
                {
                    "task": {key: value for key, value in original.items() if key not in {"id", "source"}},
                    "runtime": document["runtime"],
                    "session": document["session"],
                }
            ),
            "source_public_content_id": semantic_digest(spec.task, False),
        }
        deployment = Deployment(
            limits=spec.limits,
            command_timeout=shell_harness(spec.harness).command_timeout,
            shell_tool=shell_harness(spec.harness).shell_tool,
            task_machine=spec.task_machine,
            verifier_machine=spec.verifier_machine,
        )
    selected.update(source_id=f"SnowballV01:{family}/validation", difficulty="validation", inclusion_probability=1.0)
    record, audit = rollout_record(selected, deployment)
    record.update(
        lowered_task_spec=raw,
        source_row_index=index,
        split="validation",
        evaluation_family=family,
        format=selected.get("format"),
        control=selected.get("control"),
    )
    return PanelRecord(record, audit, deployment)


def freeze_panel(
    source: Path,
    source_sha256: str,
    training_pool: Path,
    training_sha256: str,
    output: Path,
    tokenizer_directory: Path,
    max_prompt_tokens: int,
) -> dict:
    """Keep fixed row order and pairing, rejecting overlaps or any loader admission loss."""
    if (
        file_sha256(StoragePath(str(source))) != source_sha256
        or file_sha256(StoragePath(str(training_pool))) != training_sha256
    ):
        raise ValueError("Frozen panel or training source differs from its declared SHA256")
    if output.exists():
        raise FileExistsError(output)
    train = pq.read_table(training_pool, columns=list(IDENTITY_COLUMNS)).to_pylist()
    if len(train) != 100000:
        raise ValueError("Expected the complete 100000-task training pool")
    train_ids = {column: {row[column] for row in train} for column in IDENTITY_COLUMNS}
    tokenizer = create_tokenizer(str(tokenizer_directory), disable_fast_tokenizer=False)
    factories = {"daytona": DaytonaMachineFactory()}
    records, mappings, lengths = [], [], []
    counts = Counter()
    overlaps = Counter({column: 0 for column in IDENTITY_COLUMNS})
    overlaps.update(canonical_contracts_against_training_source=0, public_contexts_against_training_source=0)
    for index, row in enumerate(pq.read_table(source).to_pylist()):
        prepared = panel_record(row, index)
        record, audit = prepared.record, prepared.audit
        spec = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])
        RolloutEngine(
            factories, {spec.harness: PrefillHarness() if spec.harness == "prefill" else shell_harness(spec.harness)}
        ).check(spec)
        length = task_prompt_token_count(tokenizer, task_prompt(spec), "prompt")
        if length > max_prompt_tokens:
            raise ValueError(f"Panel prompt exceeds the selected tokenizer budget: {spec.task.id}")
        record["prompt_tokens"] = length
        lengths.append(length)
        counts[record["evaluation_family"]] += 1
        for column in IDENTITY_COLUMNS:
            overlaps[column] += record[column] in train_ids[column]
        overlaps["canonical_contracts_against_training_source"] += (
            record["contract_id"] in train_ids["source_contract_id"]
        )
        overlaps["public_contexts_against_training_source"] += (
            record["public_content_id"] in train_ids["source_public_content_id"]
        )
        records.append(record)
        mappings.append({"task_id": spec.task.id, "source_row_index": index, **audit})
    if (
        counts != FAMILY_COUNTS
        or any(overlaps.values())
        or len({r["task_id"] for r in records}) != sum(FAMILY_COUNTS.values())
    ):
        raise ValueError(f"Fixed panel counts, pairing uniqueness or training disjointness failed: {counts}, {overlaps}")
    output.mkdir(parents=True)
    path = output / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")
    dataset = TaskDataset([str(path)], tokenizer, max_prompt_tokens, num_workers=1)
    admitted = dataset.collate_fn(list(dataset))
    if [r["input"].spec.task.id for r in admitted] != [r["task_id"] for r in records]:
        raise ValueError("TaskDataset dropped or reordered development rows")
    if (
        file_sha256(StoragePath(str(source))) != source_sha256
        or file_sha256(StoragePath(str(training_pool))) != training_sha256
    ):
        raise ValueError("Frozen source changed during panel preparation")
    report = {
        "status": "local_contract_and_loader_checks_passed",
        "rows": sum(FAMILY_COUNTS.values()),
        "family_counts": dict(counts),
        "source_sha256": source_sha256,
        "training_sha256": training_sha256,
        "exact_overlap_counts": dict(overlaps),
        "prompt_filter_dropped_rows": 0,
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "max_prompt_tokens": max_prompt_tokens,
        "source_task_ids_and_order_preserved": True,
        "native_semantic_delta": ["context", "grader"],
        "historical_equivalence_claimed": False,
        "script_grader_execution_performed": False,
        "model_inference_performed": False,
        "limitations": [
            "Reused development panel; no independent generalization claim",
            "Native text prefills and current grading have new semantic identities",
            "Real production factory and grader preflight is required before GPU allocation",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "deployment-contract-mapping.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in mappings)
    )
    write_manifest(StoragePath(str(output)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_panel(
                args.source,
                args.source_sha256,
                args.training_pool,
                args.training_sha256,
                args.output,
                args.tokenizer_directory,
                args.max_prompt_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
