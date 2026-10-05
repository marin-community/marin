# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded teacher trajectories with durable native GLM token evidence."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

import httpx
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
)
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.task_session import session_start
from shellbox.machine import MachineFactory
from taskcompendium.chat import assistant_message
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.collection_recovery import (
    CollectionRecovery,
    StudentContextAmendment,
    collection_recovery_evidence,
    student_context_amendment_record,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.rollout_eval import rollout_evidence
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_http import journal_teacher_response
from experiments.post_training.russell_rsi.token_preflight import run_token_preflight

PUBLIC_MODEL_OPTIONS = frozenset({"tools", "tool_choice", "parallel_tool_calls", "response_format"})
STUDENT_CONTEXT_TOKENS = 4096
TEACHER_FAMILY_LIMIT = 12
TEACHER_ATTEMPTS_PER_FAMILY = 2
STUDENT_ROWS = 8
TEACHER_MAX_TURNS = 16
TEACHER_COMMAND_TIMEOUT = 120
TEACHER_STARTUP_ATTEMPTS = 3


@dataclass(frozen=True)
class TeacherModelConfig:
    session_identity: str
    max_tokens: int
    temperature: float
    reasoning_effort: str


def teacher_request(messages: tuple[dict, ...], options: dict, config: TeacherModelConfig) -> dict:
    """Build a native chat request from the public session interface."""
    if options.keys() - PUBLIC_MODEL_OPTIONS:
        raise ValueError("Teacher session contains unsupported public model options")
    outbound_messages = []
    for original in messages:
        message = dict(original)
        if message.get("role") == "assistant":
            reasoning = message.get("reasoning")
            canonical = message.get("reasoning_content")
            if any(value is not None and not isinstance(value, str) for value in (reasoning, canonical)):
                raise RolloutContractError("Teacher assistant reasoning fields must be strings or null")
            if reasoning and canonical and canonical != reasoning:
                raise RolloutContractError("Teacher assistant has conflicting reasoning fields")
            if "reasoning" in message:
                del message["reasoning"]
                message["reasoning_content"] = canonical or reasoning
        outbound_messages.append(message)
    return {
        "model": GLM_MODEL,
        "messages": outbound_messages,
        **options,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "chat_template_kwargs": {"reasoning_effort": config.reasoning_effort},
        "prompt_cache_key": config.session_identity,
        "return_token_ids": True,
    }


def native_teacher_turn(raw: bytes) -> ModelTurn:
    """Read exact server tokens and the unmodified assistant message."""
    response = json.loads(raw)
    choice = response["choices"][0]
    prompt_ids = response.get("prompt_token_ids")
    response_ids = choice.get("token_ids")
    if not all(
        isinstance(tokens, list) and tokens and all(type(token) is int for token in tokens)
        for tokens in (prompt_ids, response_ids)
    ):
        raise RolloutContractError("GLM did not return exact prompt and response token IDs")
    message = choice["message"]
    return ModelTurn(
        message=message,
        prompt_token_ids=tuple(prompt_ids),
        response_token_ids=tuple(response_ids),
        logprobs=None,
        stop_reason=choice["finish_reason"],
        text=message.get("content") or "",
        metadata={"response_sha256": hashlib.sha256(raw).hexdigest(), "usage": response["usage"]},
    )


def teacher_model_turn(raw: bytes, request: ModelRequest) -> ModelTurn:
    """Require an unchanged native token prefix for strict rollouts."""
    result = native_teacher_turn(raw)
    if result.prompt_token_ids[: len(request.prefix_token_ids)] != request.prefix_token_ids:
        raise RolloutContractError("Native GLM chat changed the served token prefix")
    assistant_message(result.message)
    return result


@dataclass
class TeacherTurnProvider:
    """Journal sequential requests for one reserved trajectory.

    The caller owns one trajectory at a time and disables HTTP transport retries.
    Saved responses permit parsing recovery, not replay of an interrupted VM.
    """

    client: httpx.AsyncClient
    resolve_base_url: Callable[[], str]
    config: TeacherModelConfig
    directory: StoragePath
    turn_index: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        directory = self.directory / "turns" / f"{self.turn_index:03d}"
        self.turn_index += 1
        directory.mkdirs()
        body = teacher_request(request.messages, request.options, self.config)
        identity = {
            "session_identity": self.config.session_identity,
            "request_sha256": compact_json_sha256(body),
            "prefix_token_ids": list(request.prefix_token_ids),
            "assistant_message_index": request.assistant_message_index,
        }
        raw = await journal_teacher_response(self.client, self.resolve_base_url, directory, body, identity)
        result = teacher_model_turn(raw, request)
        write_once(directory / "model-turn.json", asdict(result))
        return result


@dataclass(frozen=True)
class TeacherTask:
    family: str
    capability: str
    task_id: str
    task_sha256: str


def selected_teacher_tasks(bank: dict, capabilities: dict, ordered_task_ids: tuple[str, ...]) -> tuple[TeacherTask, ...]:
    """Bind a fixed task order to admitted families and accepted capability labels."""
    labels = {skill["label"] for skill in capabilities["skills"]}
    by_id = {task["task_id"]: task for task in bank["tasks"]}
    selected = []
    for task_id in ordered_task_ids:
        entry = by_id[task_id]
        eligible = sorted(labels.intersection(entry["capability"].split(",")))
        if not eligible:
            raise ValueError("Selected teacher task has no accepted capability in the bank metadata")
        selected.append(TeacherTask(bank["family_by_task"][task_id], eligible[0], task_id, entry["task_sha256"]))
    return tuple(selected)


@dataclass(frozen=True)
class StudentRow:
    example: dict
    input_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]


def student_row(messages: list[dict], options: dict, tokenizer: MarinTokenizer) -> StudentRow:
    """Retokenize the full public conversation without teacher reasoning fields."""
    clean = [
        {key: value for key, value in message.items() if key not in {"reasoning", "reasoning_content"}}
        for message in messages
    ]
    example = {"messages": clean, "chat_template_kwargs": {**options, "enable_thinking": False}}
    # write_once and the SFT JSONL sort keys; tokenize that same order.
    example = json.loads(json.dumps(example, sort_keys=True))
    processor = ChatLmDatasetFormat(
        chat_template=MARIN_CHAT_TEMPLATE, pack=False, mask_user_turns=True, slice_strategy="raise"
    ).build_preprocessor(tokenizer)
    processed = processor([example])[0]
    return StudentRow(
        example,
        tuple(int(token) for token in processed["input_ids"]),
        tuple(int(mask) for mask in processed["assistant_masks"]),
    )


async def teacher_preflight(
    client: httpx.AsyncClient,
    resolve_base_url: Callable[[], str],
    config: TeacherModelConfig,
    directory: StoragePath,
) -> dict:
    """Run the two fixed probes once. A failed suite cannot issue more probes."""
    reservation = directory / "reservation.json"
    result_path = directory / "token-preflight.json"
    resumed = reservation.exists()
    write_once(reservation, asdict(config))
    if resumed and not result_path.exists():
        raise RuntimeError("Teacher preflight was interrupted; its probes cannot be reissued")
    if not resumed:
        provider = TeacherTurnProvider(client, resolve_base_url, config, directory)
        await run_token_preflight(provider, client, str(directory))
    result = json.loads(result_path.read_text())
    if result["status"] != "passed":
        raise ValueError("The saved teacher token preflight failed")
    return result


async def collect_teacher_rows(
    selected: tuple[TeacherTask, ...],
    tasks: Mapping[str, TaskSpec],
    capabilities: dict,
    tokenizer: MarinTokenizer,
    tokenizer_identity: str,
    client: httpx.AsyncClient,
    resolve_base_url: Callable[[], str],
    model: TeacherModelConfig,
    factories: Mapping[EnvironmentKind, MachineFactory],
    directory: StoragePath,
    recovery: CollectionRecovery | None = None,
    context_amendment: StudentContextAmendment | None = None,
) -> dict:
    """Collect at most two trajectories per frozen family and retain eight full rows.

    The caller supplies verified admitted tasks and a pinned student tokenizer.
    A single process owns this directory. Resume consumes incomplete trajectory
    reservations without replaying their VM state or issuing replacement calls.
    """
    labels = {skill["label"] for skill in capabilities["skills"]}
    families = [entry.family for entry in selected]
    if not STUDENT_ROWS <= len(selected) <= TEACHER_FAMILY_LIMIT or len(set(families)) != len(families):
        raise ValueError("Teacher collection requires eight to twelve distinct frozen families")
    if len({entry.task_id for entry in selected}) != len(selected):
        raise ValueError("A teacher task cannot appear in more than one family")
    for entry in selected:
        if entry.capability not in labels:
            raise ValueError("Teacher task is not bound to an accepted canonical capability")
        if digest(tasks[entry.task_id].model_dump(mode="json")) != entry.task_sha256:
            raise ValueError("Teacher task differs from the frozen admitted task")
    plan = {
        "selected": [asdict(entry) for entry in selected],
        "capabilities": capabilities,
        "model": asdict(model),
        "student_tokenizer_identity": tokenizer_identity,
        "student_template_sha256": hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest(),
    }
    write_once(directory / "plan.json", plan)
    plan_sha256 = compact_json_sha256(plan)
    contract_failure_path = directory / "contract-failure.json"
    if contract_failure_path.exists():
        failure = json.loads(contract_failure_path.read_text())
        if failure["plan_sha256"] != plan_sha256:
            raise ValueError("Saved teacher contract failure differs from the collection plan")
        raise RolloutContractError(failure["exception_message"])
    if (recovery is None) != (context_amendment is None):
        raise ValueError("Teacher recovery and prospective context amendment must be supplied together")
    context_tokens = STUDENT_CONTEXT_TOKENS
    amendment_record = None
    if context_amendment is not None and recovery is not None:
        amendment_record = student_context_amendment_record(context_amendment, recovery, plan)
        context_tokens = context_amendment.context_tokens
    recovery_evidence = collection_recovery_evidence(recovery, plan) if recovery is not None else None
    if recovery_evidence is None:
        preflight = await teacher_preflight(
            client,
            resolve_base_url,
            replace(model, session_identity=f"{model.session_identity}-preflight"),
            directory / "preflight",
        )
    else:
        assert context_amendment is not None
        write_once(
            directory / "recovery-lineage.json",
            {
                **recovery_evidence.lineage,
                "context_amendment": asdict(context_amendment),
                "amendment_record": amendment_record,
            },
        )
        preflight = recovery_evidence.preflight
    convention = SubmissionConvention(id="russell-teacher", answer_format=AnswerFormat.PLAIN)
    accepted: list[dict] = []
    attempts: list[dict] = []
    row_hashes: set[str] = set()
    for family_index, entry in enumerate(selected):
        task = tasks[entry.task_id]
        options = session_start(task, convention).options
        for attempt in range(TEACHER_ATTEMPTS_PER_FAMILY):
            if recovery_evidence is not None and family_index == 0 and attempt == 0:
                attempts.append(recovery_evidence.consumed_attempt)
                continue
            slot = directory / "trajectories" / f"{family_index:02d}-{attempt}"
            reservation = slot / "trajectory.json"
            rollout_path = slot / "rollout.json"
            resumed = reservation.exists()
            identity = {"task": asdict(entry), "attempt": attempt}
            write_once(reservation, identity)
            if resumed and not rollout_path.exists():
                status = {**identity, "status": "interrupted_without_complete_rollout"}
                write_once(slot / "qualification.json", status)
                attempts.append(status)
                continue
            if not resumed:
                provider = TeacherTurnProvider(
                    client,
                    resolve_base_url,
                    replace(model, session_identity=f"{model.session_identity}-{family_index:02d}-{attempt}"),
                    slot,
                )
                engine = ShellboxRolloutEngine(
                    provider,
                    factories,
                    max_turns=TEACHER_MAX_TURNS,
                    command_timeout=TEACHER_COMMAND_TIMEOUT,
                    convention=convention,
                )

                def record_startup_failure(index: int, evidence: dict, attempt_directory: StoragePath = slot) -> None:
                    write_once(attempt_directory / f"startup-{index}.json", evidence)

                try:
                    _, record = await rollout_evidence(
                        engine,
                        task,
                        startup_attempts=TEACHER_STARTUP_ATTEMPTS,
                        record_startup_failure=record_startup_failure,
                    )
                except RolloutContractError as error:
                    write_once(
                        contract_failure_path,
                        {
                            "plan_sha256": plan_sha256,
                            "slot": f"{family_index:02d}-{attempt}",
                            **identity,
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        },
                    )
                    raise
                write_once(rollout_path, record)
            record = json.loads(rollout_path.read_text())
            status = {**identity, "status": "failed", "rollout_sha256": compact_json_sha256(record)}
            if (
                record["execution_error"] is None
                and record["grade"]["status"] == Outcome.GRADED
                and record["grade"]["reward"] == 1
                and record["stop_reason"] not in {LENGTH_STOP_REASON, MAX_TURNS_STOP_REASON}
            ):
                row = student_row(record["messages"], options, tokenizer)
                status.update(tokens=len(row.input_ids), assistant_targets=sum(row.assistant_mask))
                row_sha = compact_json_sha256(row.example)
                if len(row.input_ids) > context_tokens:
                    status["status"] = "student_context_overflow"
                elif not any(row.assistant_mask):
                    status["status"] = "no_assistant_targets"
                elif row_sha in row_hashes:
                    status["status"] = "duplicate_student_row"
                else:
                    status["status"] = "accepted"
                    row_hashes.add(row_sha)
                    accepted.append({**identity, "row_sha256": row_sha, "row": row.example})
                    write_once(slot / "student-row.json", asdict(row))
            write_once(slot / "qualification.json", status)
            attempts.append(status)
            if status["status"] == "accepted":
                break
        if len(accepted) == STUDENT_ROWS:
            break
    result = {
        "status": "passed" if len(accepted) == STUDENT_ROWS else "insufficient_rows",
        "preflight_sha256": compact_json_sha256(preflight),
        "accepted": accepted,
        "attempts": attempts,
    }
    write_once(directory / "collection.json", result)
    return result
