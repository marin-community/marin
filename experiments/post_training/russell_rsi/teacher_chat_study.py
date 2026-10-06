# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a separate four-family chat-SFT study to counted predecessor evidence."""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import httpx
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import install_runtime_bundle
from rolloutengine.contracts import RolloutContractError
from rolloutengine.task_session import session_start
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import TaskSpec
from taskcompendium.parquet import read_tasks
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.glm import resolve_glm_base_url
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.launch_teacher_sft import (
    TEACHER_PIP_PACKAGES,
    CollectionBinding,
    TeacherCollectionConfig,
    teacher_sft_steps,
)
from experiments.post_training.russell_rsi.rollout_eval import qemu_factory
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_collection import chat_teacher_evidence, run_teacher_chat
from experiments.post_training.russell_rsi.teacher_collection import TeacherModelConfig, student_row
from experiments.post_training.russell_rsi.teacher_four_pass import (
    require_four_pass_condition,
    validated_study_post_workflow,
)

PROTOCOL = "champion-rsi-teacher-chat-four-family-v1"
NAMESPACE = "teacher-chat-four-family"
RETAINED_SLOTS = ("00-1", "02-0")
PERMITTED_SLOTS = ("05-1", "06-0", "06-1", "07-0", "07-1", "08-0", "08-1", "09-0", "09-1")
V12_CONSUMED_SLOTS = ("00-1", "01-0", "01-1", "02-0", "03-0", "03-1", "04-0", "04-1", "05-0")
CONTEXT_TOKENS = 16384
CHAT_ROWS = 4
BATCH_SIZE = 8
SFT_PASSES = 8
EXAMPLE_EXPOSURES = 32
SOURCE_FAMILIES = 10
TRAJECTORY_CEILING = 20
CONSUMED_TRAJECTORIES = 10
SFT_UPDATES = 4


@dataclass(frozen=True)
class ReservedSlot:
    slot: str
    reservation: PinnedFile


@dataclass(frozen=True)
class Predecessor:
    identity: str
    uri: str
    executor_info: PinnedFile
    executor_status: PinnedFile
    plan: PinnedFile
    contract_failure: PinnedFile
    reservations: tuple[ReservedSlot, ...]


@dataclass(frozen=True)
class RetainedRow:
    slot: str
    rollout: PinnedFile
    student_row: PinnedFile
    qualification: PinnedFile


@dataclass(frozen=True)
class ChatStudy:
    original_study: PinnedFile
    prospective_decision: PinnedFile
    canonical_proof: PinnedFile
    predecessors: tuple[Predecessor, ...]
    retained_rows: tuple[RetainedRow, ...]


@dataclass(frozen=True)
class ChatCollectionConfig:
    collection: TeacherCollectionConfig
    study: ChatStudy


def parse_study(config: dict) -> ChatStudy:
    predecessors = tuple(
        Predecessor(
            identity=item["identity"],
            uri=item["uri"],
            executor_info=PinnedFile(**item["executor_info"]),
            executor_status=PinnedFile(**item["executor_status"]),
            plan=PinnedFile(**item["plan"]),
            contract_failure=PinnedFile(**item["contract_failure"]),
            reservations=tuple(
                ReservedSlot(entry["slot"], PinnedFile(**entry["reservation"])) for entry in item["reservations"]
            ),
        )
        for item in config["predecessors"]
    )
    rows = tuple(
        RetainedRow(
            item["slot"],
            PinnedFile(**item["rollout"]),
            PinnedFile(**item["student_row"]),
            PinnedFile(**item["qualification"]),
        )
        for item in config["retained_rows"]
    )
    return ChatStudy(
        PinnedFile(**config["original_study"]),
        PinnedFile(**config["prospective_decision"]),
        PinnedFile(**config["canonical_proof"]),
        predecessors,
        rows,
    )


def source_study(study: ChatStudy) -> dict:
    original = study.original_study.read_json()
    require_four_pass_condition(original)
    decision = study.prospective_decision.read_json()
    if (
        decision["protocol"] != PROTOCOL
        or decision["source_study_config"]["sha256"] != study.original_study.sha256
        or decision["source_rows"]["canonical_proof_sha256"] != study.canonical_proof.sha256
        or decision["source_rows"]["slots"] != list(RETAINED_SLOTS)
        or decision["selection"]["permitted_slots"] != list(PERMITTED_SLOTS)
        or decision["selection"]["consumed_slots"] != CONSUMED_TRAJECTORIES
        or decision["selection"]["maximum_new_trajectories"] != len(PERMITTED_SLOTS)
        or decision["selection"]["original_total_ceiling"] != TRAJECTORY_CEILING
        or decision["sft"]["unique_jsonl_rows"] != CHAT_ROWS
        or decision["sft"]["batch_size"] != BATCH_SIZE
        or decision["sft"]["optimizer_updates"] != SFT_UPDATES
        or decision["sft"]["passes"] != SFT_PASSES
        or decision["sft"]["context_tokens"] != CONTEXT_TOKENS
        or decision["sft"]["example_exposures"] != EXAMPLE_EXPOSURES
        or decision["sft"]["assistant_only_loss"] is not True
        or decision["sft"]["teacher_reasoning_in_student"] is not False
        or decision["sft"]["truncate"] is not False
        or decision["collection"]["temperature"] != original["selection"]["teacher_model"]["temperature"]
        or decision["collection"]["max_tokens"] != original["selection"]["teacher_model"]["max_tokens"]
        or decision["collection"]["reasoning_effort"] != original["selection"]["teacher_model"]["reasoning_effort"]
    ):
        raise ValueError("Chat study differs from its prospective decision or original frozen settings")
    validate_predecessors(study, original, decision)
    return original


def validate_predecessors(study: ChatStudy, original: dict, decision: dict) -> None:
    if len(study.predecessors) != 2 or tuple(row.slot for row in study.retained_rows) != RETAINED_SLOTS:
        raise ValueError("Chat study requires both failed predecessors and exactly two retained rows")
    for predecessor, slots in zip(study.predecessors, (("00-0",), V12_CONSUMED_SLOTS), strict=True):
        if tuple(item.slot for item in predecessor.reservations) != slots:
            raise ValueError("Consumed slots differ from the frozen predecessor order")
        if predecessor.executor_status.read_bytes().decode().strip() != "FAILED":
            raise ValueError("Chat predecessor must remain failed")
        info = predecessor.executor_info.read_json()
        name, identity_suffix = predecessor.identity.split("@", 1)
        version, fingerprint = identity_suffix.split(":", 1)
        if (
            info["name"] != name
            or info["output_path"].rstrip("/") != predecessor.uri.rstrip("/")
            or info["config"]["version"] != version
            or info["config"]["fingerprint"] != fingerprint
        ):
            raise ValueError("Predecessor metadata does not identify its failed collection")
        plan = predecessor.plan.read_json()
        if (
            plan["selected"] != original["selection"]["selected"]
            or plan["capabilities"] != original["selection"]["capabilities"]
        ):
            raise ValueError("Predecessor task order or capability provenance differs")
        for item in predecessor.reservations:
            family, attempt = map(int, item.slot.split("-"))
            if item.reservation.read_json() != {"task": plan["selected"][family], "attempt": attempt}:
                raise ValueError("Consumed reservation does not match its frozen task")
        failure = predecessor.contract_failure.read_json()
        if failure["slot"] != slots[-1] or failure["plan_sha256"] != compact_json_sha256(plan):
            raise ValueError("Predecessor fatal marker differs from its plan or interrupted slot")
    first, second = study.predecessors
    if (
        first.identity != original["collection_recovery"]["predecessor_identity"]
        or first.contract_failure.sha256 != original["collection_recovery"]["fatal_marker_sha256"]
    ):
        raise ValueError("First consumed predecessor differs from original recovery lineage")
    recovery = original["collection_recovery"]
    if (
        first.executor_info.sha256 != recovery["executor_info_sha256"]
        or first.executor_status.sha256 != recovery["executor_status_sha256"]
        or first.plan.sha256 != recovery["plan_sha256"]
    ):
        raise ValueError("First predecessor differs from the original pinned metadata")
    if first.reservations[0].reservation.sha256 != recovery["slot_reservation_sha256"]:
        raise ValueError("First predecessor reservation differs from original recovery")
    if (
        second.identity != decision["source_collection"]["identity"]
        or second.uri != decision["source_collection"]["uri"]
        or second.contract_failure.sha256 != decision["source_collection"]["failure_sha256"]
    ):
        raise ValueError("Second predecessor differs from the prospective decision")


def successful(record: dict) -> bool:
    return (
        record["execution_error"] is None
        and record["grade"]["status"] == "graded"
        and record["grade"]["reward"] == 1
        and record["stop_reason"] not in ("length", "max_turns")
    )


def qualified_row(record: dict, task: TaskSpec, tokenizer: MarinTokenizer) -> dict | None:
    if not successful(record):
        return None
    options = session_start(task, SubmissionConvention(id="russell-teacher", answer_format=AnswerFormat.PLAIN)).options
    row = student_row(record["messages"], options, tokenizer)
    if not 0 < len(row.input_ids) <= CONTEXT_TOKENS or not any(row.assistant_mask):
        return None
    return json.loads(json.dumps(asdict(row)))


def retained_rows(study: ChatStudy, original: dict, tasks: dict[str, TaskSpec], tokenizer: MarinTokenizer) -> list[dict]:
    proof = study.canonical_proof.read_json()
    if (
        proof["collection_identity"] != study.predecessors[1].identity
        or proof["collection_config_sha256"] != study.original_study.sha256
    ):
        raise ValueError("Canonical retained-row proof differs from its original study")
    rows: list[dict] = []
    for entry, attested in zip(study.retained_rows, proof["rows"], strict=True):
        family, attempt = map(int, entry.slot.split("-"))
        task = original["selection"]["selected"][family]
        rollout, legacy, grade = (
            entry.rollout.read_json(),
            entry.student_row.read_json(),
            entry.qualification.read_json(),
        )
        row = qualified_row(rollout, tasks[task["task_id"]], tokenizer)
        if (
            row is None
            or row != attested["canonical"]
            or legacy["example"] != row["example"]
            or attested["slot"] != entry.slot
            or attested["source_file_hashes"]
            != {"student-row.json": entry.student_row.sha256, "rollout.json": entry.rollout.sha256}
            or grade["status"] != "accepted"
            or grade["task"] != task
            or grade["attempt"] != attempt
            or grade["rollout_sha256"] != compact_json_sha256(rollout)
        ):
            raise ValueError("Retained successful row differs from exact grade, task or canonical proof")
        rows.append(
            {
                "task": task,
                "attempt": attempt,
                "slot": entry.slot,
                "row_sha256": compact_json_sha256(row["example"]),
                "row": row["example"],
                "witness": row,
            }
        )
    if len({row["task"]["family"] for row in rows}) != 2 or len({row["row_sha256"] for row in rows}) != 2:
        raise ValueError("Retained rows must have distinct families and examples")
    return rows


async def collect_remaining_rows(
    original: dict,
    tasks: dict[str, TaskSpec],
    retained: list[dict],
    tokenizer: MarinTokenizer,
    directory: StoragePath,
    run_slot: Callable[[TaskSpec, StoragePath, TeacherModelConfig], Awaitable[dict]],
) -> dict:
    selected = original["selection"]["selected"]
    if len(selected) != SOURCE_FAMILIES or len({entry["family"] for entry in selected}) != SOURCE_FAMILIES:
        raise ValueError("Chat study requires the exact ten-family source selection")
    for entry in selected:
        if digest(tasks[entry["task_id"]].model_dump(mode="json", exclude_unset=True)) != entry["task_sha256"]:
            raise ValueError("Chat task differs from frozen admission")
    plan = {
        "protocol": PROTOCOL,
        "selection": original["selection"],
        "permitted_slots": list(PERMITTED_SLOTS),
        "consumed_trajectories": CONSUMED_TRAJECTORIES,
        "retained": retained,
    }
    write_once(directory / "plan.json", plan)
    marker = directory / "contract-failure.json"
    if marker.exists():
        raise RolloutContractError("Chat study has an immutable native contract failure")
    accepted, attempts = list(retained), []
    families = {row["task"]["family"] for row in accepted}
    hashes = {row["row_sha256"] for row in accepted}
    for slot_name in PERMITTED_SLOTS:
        family, attempt = map(int, slot_name.split("-"))
        entry = original["selection"]["selected"][family]
        if entry["family"] in families:
            continue
        task = tasks[entry["task_id"]]
        slot = directory / "trajectories" / slot_name
        identity = {"task": entry, "attempt": attempt, "slot": slot_name}
        reservation = slot / "trajectory.json"
        resumed = reservation.exists()
        write_once(reservation, identity)
        rollout_path = slot / "rollout.json"
        if resumed and not rollout_path.exists():
            status = {**identity, "status": "interrupted_consumed"}
        else:
            if not rollout_path.exists():
                model = TeacherModelConfig(
                    compact_json_sha256(plan) + "-" + slot_name, **original["selection"]["teacher_model"]
                )
                try:
                    record = await run_slot(task, slot, model)
                except RolloutContractError as error:
                    write_once(
                        marker,
                        {
                            "plan_sha256": compact_json_sha256(plan),
                            **identity,
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        },
                    )
                    raise
                write_once(rollout_path, record)
            record = json.loads(rollout_path.read_text())
            row = qualified_row(record, task, tokenizer)
            status = {**identity, "status": "failed", "rollout_sha256": compact_json_sha256(record)}
            if row is not None and compact_json_sha256(row["example"]) in hashes:
                status["status"] = "duplicate_student_row"
            elif row is not None:
                row_hash = compact_json_sha256(row["example"])
                status.update(
                    status="accepted", tokens=len(row["input_ids"]), assistant_targets=sum(row["assistant_mask"])
                )
                accepted.append({**identity, "row_sha256": row_hash, "row": row["example"], "witness": row})
                hashes.add(row_hash)
                families.add(entry["family"])
                write_once(slot / "student-row.json", row)
        write_once(slot / "qualification.json", status)
        attempts.append(status)
        if len(accepted) == CHAT_ROWS:
            break
    result = {
        "protocol": PROTOCOL,
        "status": "passed" if len(accepted) == CHAT_ROWS else "insufficient_rows",
        "accepted": accepted,
        "attempts": attempts,
        "consumed_predecessor_trajectories": CONSUMED_TRAJECTORIES,
        "new_trajectories": len(attempts),
        "cumulative_trajectories": CONSUMED_TRAJECTORIES + len(attempts),
    }
    write_once(directory / "collection.json", result)
    return result


def run_chat_study_collection(config: ChatCollectionConfig) -> None:
    original = source_study(config.study)
    base = config.collection
    if (
        base.selection != original["selection"]
        or base.tokenizer_files != original["tokenizer_files"]
        or asdict(base.runtime_bundle) != original["runtime_bundle"]
    ):
        raise ValueError("Chat collection inputs differ from original frozen scientific settings")
    output = StoragePath(base.output_path)
    write_once(output / "study.json", asdict(config.study))
    train_bytes = PinnedFile(
        str(StoragePath(base.bank_path) / "train.parquet"), original["selection"]["train_sha256"]
    ).read_bytes()
    with tempfile.TemporaryDirectory(prefix="russell-chat-study-") as temporary:
        root = Path(temporary)
        (root / "train.parquet").write_bytes(train_bytes)
        tasks = {task.id: task for task in read_tasks(str(root / "train.parquet"))}
        for name, expected in base.tokenizer_files.items():
            if Path(name).name != name:
                raise ValueError("Tokenizer input must be a filename")
            (root / name).write_bytes(PinnedFile(str(StoragePath(base.parent_path) / name), expected).read_bytes())
        template = PinnedFile(
            original["student_training_template_uri"], original["student_training_template_sha256"]
        ).read_bytes()
        if template.decode() != MARIN_CHAT_TEMPLATE:
            raise ValueError("Chat student template differs from parent training")
        (root / "training_chat_template.jinja").write_bytes(template)
        tokenizer = load_tokenizer(str(root))
        seeds = retained_rows(config.study, original, tasks, tokenizer)
        runtime = install_runtime_bundle(base.runtime_bundle)
        factories = {
            EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
            EnvironmentKind.DOCKER: qemu_factory(runtime, base.runtime_bundle),
        }

        async def collect() -> dict:
            async with httpx.AsyncClient(
                timeout=600,
                headers={"Authorization": f"Bearer {os.environ[GLM_TOKEN_ENV]}"},
                transport=httpx.AsyncHTTPTransport(retries=0),
            ) as client:

                async def run_slot(task: TaskSpec, slot: StoragePath, model: TeacherModelConfig) -> dict:
                    result = await run_teacher_chat(
                        task, slot, model, client, lambda: resolve_glm_base_url(base.relay_job), factories
                    )
                    return chat_teacher_evidence(result)

                return await collect_remaining_rows(original, tasks, seeds, tokenizer, output, run_slot)

        result = asyncio.run(collect())
    if result["status"] != "passed":
        raise ValueError("The bounded chat study did not produce four qualified families")
    content = "".join(json.dumps(entry["row"], sort_keys=True) + "\n" for entry in result["accepted"])
    train = output / "train.jsonl"
    if train.exists():
        if train.read_text() != content:
            raise ValueError("Four-family training bytes differ from saved rows")
    else:
        train.write_text(content)
    write_once(
        output / "dataset.json",
        {
            "rows": CHAT_ROWS,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "collection_sha256": compact_json_sha256(result),
            "passes": SFT_PASSES,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": SFT_UPDATES,
            "example_exposures": EXAMPLE_EXPOSURES,
        },
    )


def run_chat_study_remote(config: ChatCollectionConfig) -> None:
    remote(
        run_chat_study_collection,
        resources=ResourceConfig.with_cpu(cpu=8, ram="64GB", disk="64GB"),
        pip_packages=list(TEACHER_PIP_PACKAGES),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def chat_study_workflow(config: dict) -> dict[str, ArtifactStep]:
    if config["protocol"] != PROTOCOL:
        raise ValueError("Wrong prospective chat study protocol")
    study = parse_study(config)
    original = source_study(study)
    if config["runtime_commit"] != original["runtime_commit"] or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise ValueError("Chat runtime differs from the original pinned study")
    scientific = {
        **original,
        "version": config["collection_version"],
        "dose_decision_uri": original["continuation_selection_uri"],
        "dose_decision_sha256": original["continuation_selection_sha256"],
    }
    outputs = teacher_sft_steps(
        scientific,
        CollectionBinding(lambda base: ChatCollectionConfig(base, study), run_chat_study_remote),
        SFT_UPDATES,
        NAMESPACE,
        CONTEXT_TOKENS,
        training_version=config["version"],
    )
    trained = outputs["train"]
    pod = cast(
        TrainLmOnPodConfig,
        trained.build_config(
            StepContext.for_fingerprint(deps=trained.deps, runtime_arg_keys=trained.runtime_args.keys())
        ),
    )
    train = cast(TrainLmConfig, pod.train_config)
    if (
        train.trainer.train_batch_size != BATCH_SIZE
        or train.trainer.num_train_steps != SFT_UPDATES
        or train.train_seq_len != CONTEXT_TOKENS
        or train.data.mixture_block_size != BATCH_SIZE
    ):
        raise ValueError("Built chat SFT geometry differs from the prospective study")
    return outputs


def chat_study_post_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    sft_config = PinnedFile(config["sft_config_uri"], config["sft_config_sha256"]).read_json()
    if config["protocol"] != PROTOCOL or sft_config["protocol"] != PROTOCOL:
        raise ValueError("Chat post-SFT requires the separate prospective protocol")
    original = source_study(parse_study(sft_config))
    return validated_study_post_workflow(config, stage, study=original, study_protocol=PROTOCOL)
