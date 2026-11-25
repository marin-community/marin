# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version the TPU campaign's judge separately from the sealed GPU evidence."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID, SOLVER_MODELS, Solver
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import ScriptGrader, TaskSpec, VerifyitGrader, judge_spec

from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION, append_rows, digest
from experiments.post_training.task_curation.difficulty.regional import immutable_json, regional_worker
from experiments.post_training.task_curation.difficulty.state import GapReason, LedgerRow, Outcome
from experiments.post_training.task_curation.difficulty_config import SUBSTITUTE_JUDGE_MODEL, SUBSTITUTE_JUDGE_PIN
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.difficulty_sampling import sampling_settings

JUDGE_VERSION = "together-glm-5.3-flash-20261124-v1"
CAMPAIGN_VERSION = "atlas-difficulty-tpu-v4-20261124-v1"
TPU_SERVING_ID = "marin-vllm-tpu-" + digest([VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT])[:16]


def identity_bundle() -> dict:
    return {
        "version": CAMPAIGN_VERSION,
        "solver_protocol": PROTOCOL_ID,
        "solvers": {
            solver.value: {
                "model": SOLVER_MODELS[solver],
                "pin": asdict(PINS[solver]),
                "sampling": sampling_settings(solver, []).model_dump(mode="json"),
            }
            for solver in Solver
        },
        "judge": {
            "version": JUDGE_VERSION,
            "model": SUBSTITUTE_JUDGE_MODEL,
            "provider": "Together",
            "weights_revision": "opaque-provider",
            "tokenizer": asdict(SUBSTITUTE_JUDGE_PIN),
            "temperature": 0.7,
            "preserve_source_rubric": True,
        },
    }


def campaign_record(ledger: str) -> dict | None:
    path = StoragePath(ledger) / LEDGER_VERSION / "campaign.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["identity"] != identity_bundle() or record["campaign_id"] != digest(record["identity"]):
        raise ValueError("Campaign identity differs from this reconciler's versioned protocol")
    return record


def redeclare_judge(task: TaskSpec) -> TaskSpec:
    """Change only the judge route, after original frozen task validation."""
    judge = judge_spec(task.grader)
    if judge is None:
        return task
    if isinstance(task.grader, VerifyitGrader):
        grader = task.grader.model_copy(
            update={"parameters": {**task.grader.parameters, "model": SUBSTITUTE_JUDGE_MODEL, "temperature": 0.7}}
        )
    elif isinstance(task.grader, ScriptGrader):
        grader = task.grader.model_copy(update={"judge": replace(judge, model=SUBSTITUTE_JUDGE_MODEL, temperature=0.7)})
    else:
        raise ValueError("Unknown judged grader contract")
    # Validate the changed contract without changing its question, rubric, gates or score mapping.
    return TaskSpec.model_validate(task.model_copy(update={"grader": grader}).model_dump(mode="json"))


def record_judge_redeclaration(directory: Path, task: TaskSpec) -> None:
    """Keep the source judge and original task contract beside the versioned derivative."""
    original = task.model_dump(mode="json")
    (directory / "judge-redeclaration.json").write_text(
        json.dumps(
            {
                "judge_version": JUDGE_VERSION,
                "original_task_sha256": digest(original),
                "original_judge": asdict(judge_spec(task.grader)),
                "served_judge": identity_bundle()["judge"],
            },
            sort_keys=True,
        )
        + "\n"
    )


def copy_input(uri: str, root: StoragePath, expected: str | None = None) -> str:
    """Stream immutable task-input bytes inside the regional CPU staging job."""
    regional_worker()
    if uri.startswith("gs://") and not uri.startswith("gs://marin-us-central2/"):
        raise ValueError("Task panels must not bulk-read another GCS region")
    target = root / hashlib.sha256(uri.encode()).hexdigest() / StoragePath(uri).name
    target.parent.mkdirs()
    checksum = hashlib.sha256()
    if not target.exists():
        with StoragePath(uri).open("rb") as source, target.open("wb") as destination:
            while block := source.read(8 * 1024 * 1024):
                destination.write(block)
                checksum.update(block)
    else:
        with target.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                checksum.update(block)
    if expected is not None and checksum.hexdigest() != expected:
        raise ValueError(f"Regional input checksum differs: {uri}")
    with target.open("rb") as stored:
        readback = hashlib.file_digest(stored, "sha256").hexdigest()
    if readback != checksum.hexdigest():
        raise ValueError(f"Regional input readback differs: {uri}")
    return str(target)


def qualified_judge(uri: str) -> dict:
    payload = StoragePath(uri).read_bytes()
    record = json.loads(payload)
    if record["judge_identity"] != identity_bundle()["judge"]:
        raise ValueError("Judge qualification belongs to another versioned identity")
    fixtures = record["fixtures"]
    if len(fixtures) < 4 or {fixture["expected"] for fixture in fixtures} != {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}:
        raise ValueError("Judge qualification is missing known-verdict fixtures")
    if any(
        fixture["detail"]["verdict"] != fixture["expected"]
        or fixture["reward"] != float(fixture["expected"] == "CORRECT")
        for fixture in fixtures
    ):
        raise ValueError("Judge qualification has a failed fixture")
    probe = record["tokenization"]["tools_thinking_probe"]
    if probe["completion"]["usage"]["prompt_tokens"] != probe["pinned_prompt_tokens"]:
        raise ValueError("Judge tokenizer qualification failed")
    return {"uri": uri, "sha256": hashlib.sha256(payload).hexdigest()}


def stage_campaign(root: str, sealed_manifest: str, sealed_sha256: str, qualification: str) -> None:
    """Stage frozen task panels once and initialize an explicitly separate ledger."""
    regional_worker()
    if not root.startswith("gs://marin-us-central2/"):
        raise ValueError("Campaign staging must use us-central2 GCS")
    qualification_ref = qualified_judge(qualification)
    destination = StoragePath(root)
    manifest_bytes = StoragePath(sealed_manifest).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != sealed_sha256:
        raise ValueError("Sealed parent manifest changed")
    manifest = json.loads(manifest_bytes)
    catalog_path = StoragePath(sealed_manifest).parent / "catalog.json"
    catalog_bytes = catalog_path.read_bytes()
    # The manifest's file inventory binds every staged catalog source.
    files = manifest["files"]
    expected_catalog = (
        next(item for item in files if item["path"] == "catalog.json")
        if isinstance(files, list)
        else files["catalog.json"]
    )
    if hashlib.sha256(catalog_bytes).hexdigest() != expected_catalog["sha256"]:
        raise ValueError("Sealed catalog bytes changed")
    catalog = json.loads(catalog_bytes)
    queues = sorted({(source["queue_uri"], source["queue_sha256"]) for source in catalog if source["queue_uri"]})
    staged_queues = []
    for uri, sha in queues:
        payload = StoragePath(uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError(f"Frozen queue changed: {uri}")
        sources = json.loads(payload)
        for source in sources:
            panel = source["panel"]
            original_panel = dict(panel)
            panel["panel_path"] = copy_input(panel["panel_path"], destination / "inputs", panel["panel_sha256"])
            original_artifact = StoragePath(panel["input_artifact"])
            staged_artifact = (
                destination / "population-metadata" / hashlib.sha256(str(original_artifact).encode()).hexdigest()
            )
            for filename in (".artifact.json", ".executor_info"):
                old = original_artifact / filename
                if old.exists():
                    target = staged_artifact / filename
                    target.parent.mkdirs()
                    payload = old.read_bytes()
                    if target.exists() and target.read_bytes() != payload:
                        raise ValueError("Frozen population metadata changed during staging")
                    target.write_bytes(payload)
            panel["input_artifact"] = str(staged_artifact)
            inventory = json.loads(StoragePath(panel["evidence_url"]).read_text())
            # A relocated inventory must retain the exact seeded panel/population binding.
            if any(inventory.get(key) != value for key, value in original_panel.items()):
                raise ValueError("Input inventory differs from the frozen panel")
            inventory.update(panel)
            new_inventory = (
                destination
                / "inventories"
                / hashlib.sha256(panel["evidence_url"].encode()).hexdigest()
                / "inventory.json"
            )
            panel["evidence_url"] = str(new_inventory)
            inventory["evidence_url"] = str(new_inventory)
            immutable_json(new_inventory, inventory)
        queue_payload = json.dumps(sources, sort_keys=True) + "\n"
        new_sha = hashlib.sha256(queue_payload.encode()).hexdigest()
        new_path = destination / "queues" / new_sha / "sources.json"
        immutable_json(new_path, sources)
        staged_queues.append([str(new_path), new_sha])
    bundle = identity_bundle()
    held = {
        "FWeindel:validated-tasks/facet_v2": "facet_v2 parked verification; replacement panel not verified",
        "TaskTrove:smartie-calibforge-rebuilt-v1": (
            "Smartie missing /tests/test.sh; ef52f52553 needs a regenerated release"
        ),
    }
    holds = [
        {
            "source_id": source["source_id"],
            "registry_revision": source["registry_revision"],
            "reason": GapReason.EXECUTABILITY_ADMISSION_PENDING,
            "note": held[source["source_id"]],
        }
        for source in catalog
        if source["source_id"] in held
    ]
    record = {
        "campaign_id": digest(bundle),
        "identity": bundle,
        "cell_key": ["campaign_id", "source_id", "solver", "registry_revision"],
        "queues": staged_queues,
        "parent_seal": {"uri": sealed_manifest, "sha256": sealed_sha256},
        "judge_qualification": qualification_ref,
        "holds": holds,
        "carry_forward": "none: old judged and unqualified legacy identities remain historical in the parent seal",
    }
    immutable_json(destination / "ledger" / LEDGER_VERSION / "campaign.json", record)
    immutable_json(destination / "ledger" / LEDGER_VERSION / "imports" / "campaign-start.json", record)
    append_rows(
        str(destination / "ledger"),
        [
            LedgerRow(
                hold["registry_revision"],
                hold["source_id"],
                None,
                solver,
                digest([record["campaign_id"], hold, solver]),
                Outcome.OBSERVATION,
                GapReason.EXECUTABILITY_ADMISSION_PENDING,
                sealed_manifest,
                0.0,
                manifest["sealed_at"],
                evidence_uri=sealed_manifest,
                error_identity="preserved_source_hold",
            )
            for hold in holds
            for solver in Solver
        ],
    )
