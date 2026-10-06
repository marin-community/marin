# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a bounded development cohort with the shared Shellbox engine."""

import asyncio
import base64
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import traceback
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import httpx
from marin.datakit.download.opencode import opencode_protocol_messages
from marin.external_dependencies import MARIN_SKYRL
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.serve import local_inference
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle
from taskcompendium.environment import ArtifactKind, ShellVerifierSpec, VerifierArtifact
from taskcompendium.models import TaskSpec, VerifierKind

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.evaluation_journal import ACTIVE_ATTEMPT, EvaluationJournal
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.settings import (
    CHAT_TEMPLATE_KWARGS,
    CONTEXT_TOKENS,
    PROMPT_TOKENS,
    RESPONSE_TOKENS,
    ROLLOUT_CONCURRENCY,
    STOP_TOKEN_IDS,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256

DEVELOPMENT_MAX_TURNS = 16
DEVELOPMENT_COMMAND_TIMEOUT = 120
SUPPLEMENTARY_TASKS = 4
CALIBRATION_TASKS = 32
CALIBRATION_SAMPLES = 8
CALIBRATION_STARTUP_ATTEMPTS = 3


def require_journal_submission(task: TaskSpec) -> None:
    if task.stages:
        raise ValueError("Supplementary journals do not support staged submissions")
    if task.verifier.kind != VerifierKind.SHELL:
        return
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    if verifier.environment is None or len(verifier.artifacts) != 1 or verifier.artifacts[0].kind != ArtifactKind.FILE:
        raise ValueError("Supplementary journals require one collected file and a private grader")


async def preserve_supplementary_submission(artifact: VerifierArtifact, path: Path) -> None:
    """Persist the collected submission in its reserved attempt journal."""
    attempt = ACTIVE_ATTEMPT.get()
    assert attempt is not None
    attempt.save_submission(artifact.model_dump(mode="json"), path.read_bytes())


@dataclass(frozen=True)
class DevelopmentEvaluationConfig:
    model_uri: str
    model_identity: str
    tasks_identity: str
    tokenizer: str
    tokenizer_revision: str
    tasks_path: str
    output_path: str
    runtime_bundle: RuntimeBundle
    limit: int
    samples_per_task: int = 1
    temperature: float = 0.0
    require_reward_variation: bool = False
    startup_attempts: int = 1


@dataclass(frozen=True)
class SupplementaryEvaluationConfig:
    evaluation: DevelopmentEvaluationConfig
    journal_path: str
    checkpoint_index: int
    model_identities: tuple[str, str]
    panel_manifest_path: str
    panel_manifest_sha256: str


def supplementary_evaluation_journal(config: SupplementaryEvaluationConfig) -> EvaluationJournal:
    """Bind one owner per checkpoint to the fixed two-checkpoint comparison."""
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415

    from experiments.post_training.russell_rsi.token_preflight import (  # noqa: PLC0415
        PREFLIGHT_PROBES,
        preflight_task,
    )

    evaluation = config.evaluation
    manifest = json.loads(pinned_bytes(config.panel_manifest_path, config.panel_manifest_sha256))
    parquet = StoragePath(evaluation.tasks_path).read_bytes()
    tasks = list(read_tasks(evaluation.tasks_path))
    if (
        config.checkpoint_index not in (0, 1)
        or len(config.model_identities) != 2
        or evaluation.model_identity != config.model_identities[config.checkpoint_index]
        or len({task.id for task in tasks}) != SUPPLEMENTARY_TASKS
        or evaluation.require_reward_variation
        or evaluation.limit != SUPPLEMENTARY_TASKS
        or len(tasks) != SUPPLEMENTARY_TASKS
        or evaluation.samples_per_task != 1
        or evaluation.temperature != 0.0
        or evaluation.startup_attempts != 3
        or manifest["parquet_sha256"] != hashlib.sha256(parquet).hexdigest()
        or [row["task_sha256"] for row in manifest["tasks"]] != [digest(task.model_dump(mode="json")) for task in tasks]
        or manifest["runtime_bundle"] != asdict(evaluation.runtime_bundle)
    ):
        raise ValueError("Supplementary evaluation differs from the frozen panel or matched attempt protocol")
    for task in tasks:
        require_journal_submission(task)
    root = StoragePath(config.journal_path)
    comparison = {
        "model_identities": list(config.model_identities),
        "panel_manifest_sha256": config.panel_manifest_sha256,
        "tasks_identity": evaluation.tasks_identity,
        "parquet_sha256": manifest["parquet_sha256"],
        "runtime_bundle": asdict(evaluation.runtime_bundle),
        "samples_per_task": 1,
        "temperature": 0.0,
        "startup_attempts": 3,
        "context_tokens": CONTEXT_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "response_tokens": RESPONSE_TOKENS,
        "stop_token_ids": list(STOP_TOKEN_IDS),
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "max_turns": DEVELOPMENT_MAX_TURNS,
        "command_timeout": DEVELOPMENT_COMMAND_TIMEOUT,
        "tokenizer": evaluation.tokenizer,
        "tokenizer_revision": evaluation.tokenizer_revision,
    }
    write_once(root / "comparison.json", comparison)
    probes = [
        preflight_task(index, instruction, value) for index, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)
    ]
    journal = EvaluationJournal(
        root / str(config.checkpoint_index),
        {
            "comparison": comparison,
            "config": asdict(evaluation),
            "attempts": {
                "task": {f"{task.id}/0": digest(task.model_dump(mode="json")) for task in tasks},
                "preflight": {
                    str(index): compact_json_sha256(task.model_dump(mode="json")) for index, task in enumerate(probes, 1)
                },
            },
        },
    )
    journal.seal()
    return journal


def calibration_evaluation_journal(config: DevelopmentEvaluationConfig) -> EvaluationJournal:
    """Bind the complete 32-task, eight-sample calibration before inference."""
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415

    from experiments.post_training.russell_rsi.token_preflight import (  # noqa: PLC0415
        PREFLIGHT_PROBES,
        preflight_task,
    )

    provenance = development_worker_provenance()
    parquet = StoragePath(config.tasks_path).read_bytes()
    tasks = list(read_tasks(config.tasks_path))
    if (
        len(tasks) != CALIBRATION_TASKS
        or len({task.id for task in tasks}) != CALIBRATION_TASKS
        or config.limit != CALIBRATION_TASKS
        or config.samples_per_task != CALIBRATION_SAMPLES
        or config.temperature != 1.0
        or config.startup_attempts != CALIBRATION_STARTUP_ATTEMPTS
        or config.require_reward_variation
    ):
        raise ValueError("Calibration differs from the frozen 256-slot protocol")
    for task in tasks:
        require_journal_submission(task)
    probes = [
        preflight_task(index, instruction, value) for index, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)
    ]
    journal = EvaluationJournal(
        StoragePath(prefix_join(config.output_path, "journal")),
        {
            "protocol": "russell-calibration-journal-v1",
            "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "worker_provenance": {
                "modules": {name: value["sha256"] for name, value in provenance["modules"].items()},
                "skyrl": provenance["skyrl"],
            },
            "config": asdict(config),
            "parquet_sha256": hashlib.sha256(parquet).hexdigest(),
            "settings": {
                "context_tokens": CONTEXT_TOKENS,
                "prompt_tokens": PROMPT_TOKENS,
                "response_tokens": RESPONSE_TOKENS,
                "stop_token_ids": list(STOP_TOKEN_IDS),
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "max_turns": DEVELOPMENT_MAX_TURNS,
                "command_timeout": DEVELOPMENT_COMMAND_TIMEOUT,
            },
            "attempts": {
                "task": {
                    f"{task.id}/{index}": digest(task.model_dump(mode="json"))
                    for task in tasks
                    for index in range(config.samples_per_task)
                },
                "preflight": {
                    str(index): compact_json_sha256(task.model_dump(mode="json")) for index, task in enumerate(probes, 1)
                },
            },
        },
    )
    journal.seal()
    return journal


def run_calibration_evaluation(config: DevelopmentEvaluationConfig) -> None:
    journal = calibration_evaluation_journal(config)
    run_development_evaluation(config, journal=journal)


def run_supplementary_evaluation(config: SupplementaryEvaluationConfig) -> None:
    journal = supplementary_evaluation_journal(config)
    run_development_evaluation(config.evaluation, journal=journal)


class HermesSyntaxError(ValueError):
    """The Hermes parser rejected a received completion."""


def completion_message(text: str, tools: list[dict], message_index: int) -> dict:
    """Serialize call arguments as JSON and assign IDs from the assistant message position."""
    parsed = opencode_protocol_messages([{"role": "assistant", "content": text}], tools)
    if parsed is None:
        raise HermesSyntaxError("The model emitted invalid Hermes tool syntax")
    message = parsed[0][0]
    # The parser starts message numbering at zero for each completion.
    for call_index, call in enumerate(message.get("tool_calls", [])):
        call["id"] = f"call_{call['function']['name']}_{message_index}_{call_index}"
        call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    return message


def qemu_factory(manifest: dict, runtime_bundle: RuntimeBundle):
    from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory  # noqa: PLC0415

    return QemuMachineFactory(
        Acceleration.TCG,
        prepared_registry_bundles={
            manifest["source_image"]: Path(runtime_bundle.installation_parent) / manifest["directory_name"],
        },
    )


async def rollout_evidence(
    engine,
    task,
    *,
    startup_attempts: int = 1,
    record_startup_failure: Callable[[int, dict], None] | None = None,
):
    """Return the rollout and a record with execution failure details."""
    from rolloutengine.contracts import RolloutInterrupted, RolloutOperation  # noqa: PLC0415
    from shellbox.machine import MachineStartupError  # noqa: PLC0415

    if startup_attempts < 1 or (startup_attempts > 1 and record_startup_failure is None):
        raise ValueError("Startup retries require a positive bound and durable failure records")
    for attempt in range(1, startup_attempts + 1):
        operation = None
        execution_error: dict | None = None
        retry_start = False
        try:
            rollout = await engine.run(task)
        except RolloutInterrupted as error:
            rollout = error.rollout
            operation = error.operation.value
            cause = error.__cause__ or error
            execution_error = {
                "type": type(cause).__name__,
                "message": str(cause),
                "traceback": "".join(traceback.format_exception(error)),
            }
            retry_start = (
                error.operation == RolloutOperation.START
                and isinstance(cause, MachineStartupError)
                and not rollout.steps
                and not rollout.response_token_ids
                and (rollout.failure is None or "pending_turn" not in rollout.failure.diagnostics)
            )
        evidence: dict[str, Any] = {
            **asdict(rollout),
            "interrupted_operation": operation,
            "execution_error": execution_error,
            "startup_attempt": attempt,
        }
        if retry_start and record_startup_failure is not None:
            record_startup_failure(attempt, evidence)
        if not retry_start or attempt == startup_attempts:
            return rollout, evidence
    raise AssertionError("Startup attempt bound was not applied")


async def evaluate_development(
    config: DevelopmentEvaluationConfig,
    base_url: str,
    model: str,
    runtime_manifest: dict,
    *,
    journal: EvaluationJournal | None = None,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    from rolloutengine.contracts import (  # noqa: PLC0415
        GenerationLimitReached,
        ModelRequest,
        ModelResponseRejected,
        ModelTurn,
        RejectedModelResponse,
        RolloutContractError,
    )
    from rolloutengine.engine import ShellboxRolloutEngine  # noqa: PLC0415
    from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: PLC0415
    from skyrl_train.inference_engines.chat_continuation import (  # noqa: PLC0415
        render_exact_chat_continuation,
    )
    from taskcompendium.environment import EnvironmentKind  # noqa: PLC0415
    from taskcompendium.grading_result import Outcome  # noqa: PLC0415
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415
    from taskcompendium.submission import (  # noqa: PLC0415
        AnswerFormat,
        SubmissionConvention,
    )

    from experiments.post_training.russell_rsi.token_preflight import run_token_preflight  # noqa: PLC0415

    tasks = list(islice(read_tasks(config.tasks_path), config.limit))
    if not tasks:
        raise ValueError("The frozen development cohort is empty")
    if journal is not None:
        for task in tasks:
            require_journal_submission(task)
    categories: Counter[str] = Counter()
    startup_counts: Counter[str] = Counter()
    failed_ids: set[str] = set()
    group_rewards: dict[str, list[float]] = {}
    async with httpx.AsyncClient(timeout=600, transport=http_transport or httpx.AsyncHTTPTransport(retries=0)) as client:

        async def tokenize(request: dict) -> dict:
            url = base_url.removesuffix("/v1") + "/tokenize"
            attempt = ACTIVE_ATTEMPT.get()
            response = (
                await client.post(url, json=request["json"])
                if attempt is None
                else await attempt.post(client, url, request["json"])
            )
            response.raise_for_status()
            return response.json()

        async def turn(request: ModelRequest, temperature: float = config.temperature) -> ModelTurn:
            body = {
                "model": model,
                "messages": list(request.messages),
                **request.options,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "add_generation_prompt": True,
            }
            if request.assistant_message_index is None:
                prompt_ids = (await tokenize({"json": body}))["tokens"]
            else:
                prompt_ids = await render_exact_chat_continuation(
                    tokenize,
                    {"json": body},
                    assistant_message_index=request.assistant_message_index,
                    served_prefix_token_ids=list(request.prefix_token_ids),
                )
                if prompt_ids is None:
                    raise RolloutContractError("The renderer cannot retain the served token prefix")
            if len(prompt_ids) > PROMPT_TOKENS:
                raise GenerationLimitReached(tuple(prompt_ids))
            completion = {
                "model": model,
                "prompt": prompt_ids,
                "max_tokens": RESPONSE_TOKENS,
                "temperature": temperature,
                "return_token_ids": True,
                # Preserve the special <tool_call> markers for the Hermes parser.
                "skip_special_tokens": False,
                "stop_token_ids": list(STOP_TOKEN_IDS),
                "include_stop_str_in_output": False,
            }
            attempt = ACTIVE_ATTEMPT.get()
            response = (
                await client.post(base_url + "/completions", json=completion)
                if attempt is None
                else await attempt.post(client, base_url + "/completions", completion)
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            response_ids = choice.get("token_ids")
            if not isinstance(response_ids, list) or not all(type(token) is int for token in response_ids):
                raise RolloutContractError("The server did not return exact response tokens")
            # A parse rejection bypasses the engine's normal ModelTurn token checks.
            if not response_ids:
                raise RolloutContractError("The received completion has no response tokens")
            text = choice["text"]
            finish_reason = choice["finish_reason"]
            if not isinstance(text, str) or not isinstance(finish_reason, str):
                raise RolloutContractError("The server did not return a completion text and finish reason")
            try:
                message = completion_message(text, request.options.get("tools", []), len(request.messages))
            except HermesSyntaxError as error:
                if not isinstance(prompt_ids, list) or not all(type(token) is int for token in prompt_ids):
                    raise RolloutContractError("The rejected completion has no exact prompt tokens") from error
                evidence = RejectedModelResponse(
                    request=completion,
                    request_sha256=compact_json_sha256(completion),
                    response_body_base64=base64.b64encode(response.content).decode("ascii"),
                    response_sha256=hashlib.sha256(response.content).hexdigest(),
                )
                raise ModelResponseRejected(str(error), evidence) from error
            return ModelTurn(
                message=message,
                prompt_token_ids=tuple(prompt_ids),
                response_token_ids=tuple(response_ids),
                logprobs=None,
                stop_reason=finish_reason,
                text=text,
            )

        await run_token_preflight(
            lambda request: turn(request, temperature=0.0), client, config.output_path, journal=journal
        )
        engine = (
            None
            if journal is not None and journal.complete()
            else ShellboxRolloutEngine(
                turn,
                {
                    EnvironmentKind.DOCKER: qemu_factory(runtime_manifest, config.runtime_bundle),
                    EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
                },
                max_turns=DEVELOPMENT_MAX_TURNS,
                command_timeout=DEVELOPMENT_COMMAND_TIMEOUT,
                convention=SubmissionConvention(id="russell-dev", answer_format=AnswerFormat.PLAIN),
                submission_sink=preserve_supplementary_submission if journal is not None else None,
            )
        )
        semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)
        records = {}
        with (
            nullcontext(None)
            if journal is not None
            else StoragePath(prefix_join(config.output_path, "traces.jsonl")).open("w")
        ) as traces:

            async def run_task(task, sample_index: int) -> None:
                async with semaphore:
                    slot_counts: Counter[str] = Counter()

                    def record_startup_failure(attempt: int, evidence: dict) -> None:
                        task_sha256 = compact_json_sha256(task.model_dump(mode="json"))
                        path = prefix_join(
                            config.output_path,
                            f"startup-failures/{task_sha256}/{sample_index}-{attempt}.json",
                        )
                        StoragePath(path).write_text(
                            json.dumps(
                                {
                                    "task_id": task.id,
                                    "task_sha256": task_sha256,
                                    "sample_index": sample_index,
                                    "model_identity": config.model_identity,
                                    "model_requests_issued": 0,
                                    "evidence": evidence,
                                }
                            )
                            + "\n"
                        )
                        slot_counts["failed_starts"] += 1
                        if attempt == config.startup_attempts:
                            slot_counts["exhausted_samples"] += 1

                    async def rollout_attempt() -> dict:
                        assert engine is not None
                        _, record = await rollout_evidence(
                            engine,
                            task,
                            startup_attempts=config.startup_attempts,
                            record_startup_failure=record_startup_failure,
                        )
                        slot_counts["retries"] += record["startup_attempt"] - 1
                        return {"record": record, "startup_counts": dict(slot_counts)}

                    if journal is None:
                        saved = await rollout_attempt()
                    else:
                        saved = await journal.attempt(
                            "task", f"{task.id}/{sample_index}", digest(task.model_dump(mode="json"))
                        ).run(rollout_attempt)
                    record: dict[str, Any] = saved["record"]
                    startup_counts.update(saved["startup_counts"])
                    record = {**record, "sample_index": sample_index}
                    records[(task.id, sample_index)] = record
                    if traces is not None:
                        traces.write(json.dumps(record) + "\n")
                    operation = record["interrupted_operation"]
                    if record["grade"]["status"] != Outcome.GRADED:
                        category = f"execution_{operation or 'ungraded'}"
                    elif record["grade"]["reward"] is not None and record["grade"]["reward"] > 0:
                        category = "passed"
                    else:
                        category = "incorrect"
                    categories[category] += 1
                    if record["grade"]["status"] == Outcome.GRADED and record["grade"]["reward"] is not None:
                        group_rewards.setdefault(task.id, []).append(record["grade"]["reward"])
                    if category != "passed":
                        failed_ids.add(task.id)

            async with asyncio.TaskGroup() as group:
                for task in tasks:
                    for sample_index in range(config.samples_per_task):
                        group.create_task(run_task(task, sample_index))

    if journal is not None:
        path = StoragePath(prefix_join(config.output_path, "traces.jsonl"))
        content = "".join(
            json.dumps(records[(task.id, index)], sort_keys=True) + "\n"
            for task in tasks
            for index in range(config.samples_per_task)
        )
        if path.exists() and path.read_text() != content:
            raise ValueError("Completed supplementary traces differ from their saved records")
        if not path.exists():
            path.write_text(content)
    informative_groups = sum(len(set(rewards)) > 1 for rewards in group_rewards.values())
    summary = {
        "model_identity": config.model_identity,
        "tasks_path": config.tasks_path,
        "tasks_identity": config.tasks_identity,
        "count": len(tasks),
        "samples_per_task": config.samples_per_task,
        "startup_attempts": config.startup_attempts,
        "startup_counts": dict(startup_counts),
        "informative_groups": informative_groups,
        "task_rewards": group_rewards,
        "categories": dict(categories),
        "failed_task_ids": sorted(failed_ids),
    }
    if journal is None:
        StoragePath(prefix_join(config.output_path, "failure_summary.json")).write_text(json.dumps(summary) + "\n")
    else:
        write_once(StoragePath(prefix_join(config.output_path, "failure_summary.json")), summary)
    if config.require_reward_variation and informative_groups == 0:
        raise ValueError("No sampled task group has reward variation; do not allocate the policy")


def development_worker_provenance() -> dict[str, Any]:
    """Return checked import paths and hashes for the branch worker and pinned SkyRL renderer."""
    root = Path(__file__).resolve().parents[3]
    for package in ("rolloutengine", "taskcompendium", "shellbox"):
        spec = importlib.util.find_spec(package)
        expected = root / "lib" / package / "src" / package
        if spec is None or list(spec.submodule_search_locations or ()) != [str(expected)]:
            raise ValueError(f"Development worker imported {package} outside the packaged branch: {spec}")
    required = {
        "rolloutengine.contracts": (
            "GenerationLimitReached",
            "ModelRequest",
            "ModelResponseRejected",
            "ModelTurn",
            "RejectedModelResponse",
            "RolloutContractError",
        ),
        "rolloutengine.engine": ("ShellboxRolloutEngine",),
        "rolloutengine.task_session": ("session_start",),
        "taskcompendium.models": ("TaskSpec",),
        "shellbox.backends.qemu.machine": ("QemuMachineFactory",),
        "shellbox.backends.shellsim.machine": ("ShellSimMachineFactory",),
    }
    files = {}
    for name, symbols in required.items():
        package = name.split(".")[0]
        expected = root / "lib" / package / "src" / Path(*name.split(".")).with_suffix(".py")
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
            raise ValueError(f"Development worker imported {name} outside the packaged branch: {spec}")
        module = importlib.import_module(name)
        for symbol in symbols:
            getattr(module, symbol)
        files[name] = {"path": str(expected), "sha256": hashlib.sha256(expected.read_bytes()).hexdigest()}
    distribution = importlib.metadata.distribution(MARIN_SKYRL.distribution)
    direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    if not isinstance(direct_url, dict) or direct_url.get("vcs_info", {}).get("commit_id") != MARIN_SKYRL.commit:
        raise ValueError("Development worker SkyRL installation differs from the pinned revision")
    name = "skyrl_train.inference_engines.chat_continuation"
    expected = Path(distribution.locate_file("skyrl_train/inference_engines/chat_continuation.py")).resolve()
    module = importlib.import_module(name)
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise ValueError("Development worker renderer differs from the pinned SkyRL installation")
    if not callable(module.render_exact_chat_continuation):
        raise ValueError("Development worker renderer is not callable")
    files[name] = {"path": str(expected), "sha256": hashlib.sha256(expected.read_bytes()).hexdigest()}
    return {
        "branch_root": str(root),
        "modules": files,
        "skyrl": {"version": distribution.version, "direct_url": direct_url},
    }


def run_development_evaluation(config: DevelopmentEvaluationConfig, *, journal: EvaluationJournal | None = None) -> None:
    """Own the model server for one fixed development evaluation."""
    provenance = development_worker_provenance()
    write_once(StoragePath(prefix_join(config.output_path, "worker-import-provenance.json")), provenance)
    if journal is not None and journal.complete():

        def reject_saved_request(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("Completed supplementary replay must not make HTTP requests")

        asyncio.run(
            evaluate_development(
                config,
                "https://saved.invalid/v1",
                "russell-dev",
                {},
                journal=journal,
                http_transport=httpx.MockTransport(reject_saved_request),
            )
        )
        return
    runtime_manifest = install_runtime_bundle(config.runtime_bundle)
    with local_inference(
        ServedModelConfig(
            weights=config.model_uri,
            api_model="russell-dev",
            tokenizer=config.tokenizer,
            tokenizer_revision=config.tokenizer_revision,
            max_model_len=CONTEXT_TOKENS,
            tensor_parallel_size=1,
        ),
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            extra_args=(
                "--data-parallel-size",
                "8",
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
            ),
        ),
        num_chips=8,
    ) as server:
        asyncio.run(
            evaluate_development(
                config, server.model.endpoint.base_url, server.model.endpoint.model, runtime_manifest, journal=journal
            )
        )
