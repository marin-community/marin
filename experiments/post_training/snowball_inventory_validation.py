# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the original five-family development panel without replacing its graders."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.token_prefix import TokenPrefixHarness
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from skyrl_gym.rollout_specs import rollout_spec, shell_harness
from skyrl_train.dataset.tasks import TaskDataset, task_prompt, task_prompt_token_count
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.runtime.resources import resource_bytes
from transformers import AutoTokenizer
from verifyit.json_objects import unique_object

REPOSITORY = "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data"
REVISION = "cb728a7636caad3fc6fa6b6ba85a49194d9d69a3"
SOURCE_SHA256 = "c980166afcf10aac08ca8f9275a50f6925305718ad77656a828d489c1bf9a319"
TOKENIZER_MODEL = "open-athena/Snowball-Smartie-RL-2026.10.15.32-step14"
TOKENIZER_REVISION = "6783e9743478bd25a9ce03ab3460e84d6a9814b0"
GRADER_IMAGE = (
    "ghcr.io/penfever/smartie-grader@sha256:" "398753f1ef9e163dd4ae503ce328a441b919bbd381b50cba32af2c564e9b0bc8"
)
FAMILIES = {
    "ultra_sft_step3200_calendar_v2": "calendar",
    "ultra_sft_step3200_comp_coding": "competitive_coding",
    "ultra_sft_step3200_reasoning_gym": "reasoning_gym",
    "ultra_sft_step3200_stem_mcqa": "stem",
}
EXPECTED_COUNTS = {"calendar": 167, "competitive_coding": 167, "reasoning_gym": 167, "stem": 166, "structural": 243}
WINDOW_TOKENS = 20480
RESPONSE_TOKENS = 4096


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restored_script_rollout(raw: str) -> tuple[TaskRolloutSpec, str]:
    """Retain the source task and map its old session using the production lowering function."""
    document = json.loads(raw, object_pairs_hook=unique_object)
    original = document["task"]
    task = TaskSpec.model_validate({key: value for key, value in original.items() if key != "schema_version"})
    if not isinstance(task.grader, ScriptGrader) or task.grader.environment.docker_image != GRADER_IMAGE:
        raise ValueError("The development panel requires its original pinned script grader")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("Only the original one-turn shellbox panel is supported")
    if document["runtime"]["task_machine"] is not None:
        raise ValueError("The development panel must not have a task machine")
    machine = MachineRuntimeSpec.model_validate(document["runtime"]["verifier_machine"])
    if machine.backend != "daytona":
        raise ValueError("Preserve the source Daytona verifier backend")
    # Old sessions had no aggregate token cap; current limits require that choice explicitly.
    spec = rollout_spec(task, {**document["session"], "max_total_tokens": None}, verifier_machine=machine)
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    contract = json.loads(resource_bytes(config), object_pairs_hook=unique_object)["contract"]
    return spec, FAMILIES[contract["dataset"]]


def freeze_panel(artifact_root: Path, output: Path, tokenizer_directory: Path) -> dict:
    """Write all 910 source rows after proving disjointness and local loader compatibility."""
    if output.exists() and set(file.name for file in output.iterdir()) != {"validation.parquet"}:
        raise FileExistsError(output)
    source = artifact_root / "public-v01/validation.parquet"
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Validation source differs from its immutable public release")
    structural_path = artifact_root / "structural-validation-2026.10.20.1/validation.parquet"
    structural = {row["task_id"]: row for row in pq.read_table(structural_path).to_pylist()}
    train_path = artifact_root / "full-runtime-2026.10.20.2/train.parquet"
    columns = ["task_id", "contract_id", "public_content_id", "source_contract_id", "source_public_content_id"]
    train = pq.read_table(train_path, columns=columns).to_pylist()
    if len(train) != 100000:
        raise ValueError("Expected the complete frozen training pool")
    train_ids = {column: {row[column] for row in train} for column in columns}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)
    factories = {"daytona": DaytonaMachineFactory()}
    token_harness = TokenPrefixHarness(tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION)
    overlaps = Counter({column: 0 for column in columns})
    overlaps.update(
        {"original_semantic_contracts_against_train_source": 0, "original_public_contexts_against_train_source": 0}
    )
    counts, lengths = Counter(), []
    records, source_sessions = [], set()
    for index, row in enumerate(pq.read_table(source).to_pylist()):
        raw = row["lowered_task_spec"]
        document = json.loads(raw, object_pairs_hook=unique_object)
        original = document["task"]
        if original["source"]["dataset"] == "structure-closure":
            record = dict(structural[original["id"]])
            spec = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])
            if record["lowered_task_spec"] != raw:
                raise ValueError("The structural panel changed its original source payload")
            family = "structural"
            harness = token_harness
        else:
            spec, family = restored_script_rollout(raw)
            task = spec.task
            record = {
                "task_rollout_spec": spec.model_dump_json(),
                "task_id": task.id,
                "source_id": f"SnowballV01:{family}/validation",
                "contract_id": semantic_digest(task, True),
                "public_content_id": semantic_digest(task, False),
                "source_contract_id": canonical_sha256(
                    {
                        "task": {key: value for key, value in original.items() if key not in {"id", "source"}},
                        "runtime": document["runtime"],
                        "session": document["session"],
                    }
                ),
                "source_public_content_id": semantic_digest(task, False),
                "lowered_task_spec": raw,
                "source_row_index": index,
                "split": "validation",
            }
            harness = shell_harness(spec.harness)
        engine = RolloutEngine(factories, {spec.harness: harness})
        engine.check(spec)
        count = task_prompt_token_count(tokenizer, task_prompt(spec), "prompt")
        if count > WINDOW_TOKENS - RESPONSE_TOKENS:
            raise ValueError(f"Panel prompt exceeds the exact evaluation budget: {spec.task.id}")
        record["prompt_tokens"] = count
        record["evaluation_family"] = family
        # Arrow infers columns from the first row, which is a non-native calendar task.
        for field in ("format", "family", "control"):
            record.setdefault(field, None)
        counts[family] += 1
        lengths.append(count)
        for column in columns:
            overlaps[column] += record[column] in train_ids[column]
        overlaps["original_semantic_contracts_against_train_source"] += (
            record["contract_id"] in train_ids["source_contract_id"]
        )
        overlaps["original_public_contexts_against_train_source"] += (
            record["public_content_id"] in train_ids["source_public_content_id"]
        )
        records.append(record)
        source_sessions.add(json.dumps(document["session"], sort_keys=True))
    if dict(counts) != EXPECTED_COUNTS or any(overlaps.values()):
        raise ValueError(f"Panel counts or disjointness failed: {counts}, {overlaps}")
    if len({row["task_id"] for row in records}) != 910:
        raise ValueError("Source task IDs must be preserved uniquely")
    output.mkdir(exist_ok=True)
    path = output / "validation.parquet"
    table = pa.Table.from_pylist(records)
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise ValueError("An incomplete panel differs from its source reconstruction")
    else:
        pq.write_table(table, path, compression="zstd")
    path.chmod(0o600)
    datasets.disable_progress_bars()
    datasets.disable_caching()
    dataset = TaskDataset([str(path)], tokenizer, WINDOW_TOKENS - RESPONSE_TOKENS, num_workers=1)
    loaded_ids = [json.loads(raw)["task"]["id"] for raw in dataset.dataframe["task_rollout_spec"]]
    if loaded_ids != [row["task_id"] for row in records]:
        raise ValueError("TaskDataset dropped or reordered development rows")
    provenance = {
        "version": output.name,
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_split": "validation",
        "source_rows": 910,
        "family_counts": dict(counts),
        "selection": "All 910 source rows, original order; no deduplication",
        "source_task_ids_preserved": True,
        "source_lowered_task_spec_strings_preserved": True,
        "structural_payloads_reused_byte_exact": 243,
        "script_task_contracts_preserved": 667,
        "script_runtime_translation": (
            "Current skyrl_gym.rollout_specs.rollout_spec maps old shellbox session; original verifier machine, "
            "timeouts, image and resources retained; obsolete schema_version omitted from TaskSpec parsing; "
            "absent historical aggregate token cap represented explicitly as max_total_tokens=null"
        ),
        "original_script_grader_image": GRADER_IMAGE,
        "verifier_backend": "daytona",
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": file_sha256(tokenizer_directory / "tokenizer.json"),
        "tokenizer_config_sha256": file_sha256(tokenizer_directory / "tokenizer_config.json"),
        "chat_template_sha256": file_sha256(tokenizer_directory / "chat_template.jinja"),
        "train_runtime_sha256": file_sha256(train_path),
        "exact_overlap_counts": dict(overlaps),
        "source_sessions": [json.loads(value) for value in sorted(source_sessions)],
        "unique_contract_counts_by_family": {
            family: len({row["contract_id"] for row in records if row["evaluation_family"] == family})
            for family in counts
        },
        "weighting": (
            "Report each family mean with row denominator, then unweighted mean of the five family means; "
            "report pooled mean separately. Structural243 rows represent238 contracts, not243 independent examples."
        ),
        "limitations": [
            "Reused development set; initializer selection already used it",
            "Exact identity exclusions do not establish semantic decontamination",
            "No independent held-out generalization claim",
            "No policy inference or original pinned image execution performed locally",
        ],
    }
    readiness = {
        "status": "local_contract_and_loader_checks_passed_pinned_image_execution_pending",
        "rows": len(records),
        "family_counts": dict(counts),
        "engine_checks": len(records),
        "task_dataset_rows": len(dataset),
        "prompt_filter_dropped_rows": 0,
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "context_window": WINDOW_TOKENS,
        "response_tokens": RESPONSE_TOKENS,
        "daytona_factory_capability_checks": 667,
        "daytona_sandboxes_started": 0,
        "script_grader_execution_performed": False,
        "model_inference_performed": False,
        "structural_reference": (
            "structural-validation-2026.10.20.1/readiness.json documents486 gold/invalid CPU engine attempts"
        ),
    }
    for name, value in (("provenance.json", provenance), ("readiness.json", readiness)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest = {
        "files": {
            file.name: {"sha256": file_sha256(file), "bytes": file.stat().st_size} for file in sorted(output.iterdir())
        }
    }
    (output / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_panel(args.artifact_root, args.output, args.tokenizer_directory), indent=2))


if __name__ == "__main__":
    main()
