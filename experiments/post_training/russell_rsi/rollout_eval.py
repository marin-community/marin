# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a bounded development cohort with the shared Shellbox engine."""

import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import httpx
from marin.datakit.download.opencode import opencode_protocol_messages
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.serve import local_inference
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle

from experiments.post_training.russell_rsi.settings import (
    CHAT_TEMPLATE_KWARGS,
    CONTEXT_TOKENS,
    PROMPT_TOKENS,
    RESPONSE_TOKENS,
    ROLLOUT_CONCURRENCY,
    STOP_TOKEN_IDS,
)


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


def qemu_factory(manifest: dict, runtime_bundle: RuntimeBundle):
    from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory  # noqa: PLC0415

    return QemuMachineFactory(
        Acceleration.TCG,
        prepared_registry_bundles={
            manifest["source_image"]: Path(runtime_bundle.installation_parent) / manifest["directory_name"],
        },
    )


async def evaluate_development(
    config: DevelopmentEvaluationConfig, base_url: str, model: str, runtime_manifest: dict
) -> None:
    from rolloutengine.contracts import (  # noqa: PLC0415
        GenerationLimitReached,
        ModelRequest,
        ModelTurn,
        RolloutContractError,
        RolloutInterrupted,
    )
    from rolloutengine.engine import ShellboxRolloutEngine  # noqa: PLC0415
    from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: PLC0415
    from skyrl_train.inference_engines.chat_continuation import (  # noqa: PLC0415
        render_exact_chat_continuation,
    )
    from taskcompendium.environment import EnvironmentKind  # noqa: PLC0415
    from taskcompendium.grading import Outcome  # noqa: PLC0415
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415
    from taskcompendium.submission import (  # noqa: PLC0415
        AnswerFormat,
        SubmissionConvention,
    )

    from experiments.post_training.russell_rsi.token_preflight import run_token_preflight  # noqa: PLC0415

    tasks = list(islice(read_tasks(config.tasks_path), config.limit))
    if not tasks:
        raise ValueError("The frozen development cohort is empty")
    categories: Counter[str] = Counter()
    failed_ids: set[str] = set()
    group_rewards: dict[str, list[float]] = {}
    async with httpx.AsyncClient(timeout=600) as client:

        async def tokenize(request: dict) -> dict:
            response = await client.post(base_url.removesuffix("/v1") + "/tokenize", json=request["json"])
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
                "skip_special_tokens": True,
                "stop_token_ids": list(STOP_TOKEN_IDS),
                "include_stop_str_in_output": False,
            }
            response = await client.post(base_url + "/completions", json=completion)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            response_ids = choice.get("token_ids")
            if not isinstance(response_ids, list) or not all(type(token) is int for token in response_ids):
                raise RolloutContractError("The server did not return exact response tokens")
            parsed = opencode_protocol_messages(
                [{"role": "assistant", "content": choice["text"]}], request.options.get("tools", [])
            )
            if parsed is None:
                raise ValueError("The model emitted invalid Hermes tool syntax")
            return ModelTurn(
                message=parsed[0][0],
                prompt_token_ids=tuple(prompt_ids),
                response_token_ids=tuple(response_ids),
                logprobs=None,
                stop_reason=choice["finish_reason"],
                text=choice["text"],
            )

        await run_token_preflight(lambda request: turn(request, temperature=0.0), client, config.output_path)
        engine = ShellboxRolloutEngine(
            turn,
            {
                EnvironmentKind.DOCKER: qemu_factory(runtime_manifest, config.runtime_bundle),
                EnvironmentKind.SHELLSIM: ShellSimMachineFactory(),
            },
            max_turns=16,
            command_timeout=120,
            convention=SubmissionConvention(id="russell-dev", answer_format=AnswerFormat.PLAIN),
        )
        semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)
        with StoragePath(prefix_join(config.output_path, "traces.jsonl")).open("w") as traces:

            async def run_task(task) -> None:
                async with semaphore:
                    operation = None
                    try:
                        rollout = await engine.run(task)
                    except RolloutInterrupted as error:
                        rollout = error.rollout
                        operation = error.operation.value
                    record = asdict(rollout)
                    record["interrupted_operation"] = operation
                    traces.write(json.dumps(record) + "\n")
                    if rollout.grade.status != Outcome.GRADED:
                        category = f"execution_{operation or 'ungraded'}"
                    elif rollout.grade.reward is not None and rollout.grade.reward > 0:
                        category = "passed"
                    else:
                        category = "incorrect"
                    categories[category] += 1
                    if rollout.grade.status == Outcome.GRADED and rollout.grade.reward is not None:
                        group_rewards.setdefault(task.id, []).append(rollout.grade.reward)
                    if category != "passed":
                        failed_ids.add(task.id)

            async with asyncio.TaskGroup() as group:
                for task in tasks:
                    for _ in range(config.samples_per_task):
                        group.create_task(run_task(task))

    informative_groups = sum(len(set(rewards)) > 1 for rewards in group_rewards.values())
    summary = {
        "model_identity": config.model_identity,
        "tasks_path": config.tasks_path,
        "tasks_identity": config.tasks_identity,
        "count": len(tasks),
        "samples_per_task": config.samples_per_task,
        "informative_groups": informative_groups,
        "task_rewards": group_rewards,
        "categories": dict(categories),
        "failed_task_ids": sorted(failed_ids),
    }
    StoragePath(prefix_join(config.output_path, "failure_summary.json")).write_text(json.dumps(summary) + "\n")
    if config.require_reward_variation and informative_groups == 0:
        raise ValueError("No sampled task group has reward variation; do not allocate the policy")


def run_development_evaluation(config: DevelopmentEvaluationConfig) -> None:
    """Own the model server for one fixed development evaluation."""
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
            evaluate_development(config, server.model.endpoint.base_url, server.model.endpoint.model, runtime_manifest)
        )
