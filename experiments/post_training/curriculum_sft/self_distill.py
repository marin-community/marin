# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the base model on curriculum problems and keep its own verified solutions as SFT rows.

Self-distilled rows keep the student's reasoning style, so an SFT run on them isolates the effect
of the selected problems from the effect of imitating a teacher's traces. The step serves the model
with vLLM through Marin's remote inference, samples each accepted problem several times in thinking
mode, and keeps the first samples whose think block closes, whose last ``\\boxed{}`` answer is
math-verify-equivalent to the problem's reference answer, and whose rendered row fits the SFT
sequence length.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import requests
from fray.types import ResourceConfig
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from rigging.filesystem.storage_path import StoragePath
from tasktrove_verify.modes.extract import extract_boxed
from zephyr.readers import load_parquet

from experiments.post_training.curriculum_sft.generation import (
    MANIFEST_FILENAME,
    PROBLEMS_FILENAME,
    REASONING_CHAT_SCHEMA,
    answers_match,
    reasoning_chat_row,
    sequence_token_counter,
    write_table,
)

logger = logging.getLogger(__name__)

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
REQUEST_TIMEOUT = 1800
CONCURRENT_REQUESTS = 128
SAMPLES_FILENAME = "samples/{capability_id}.parquet"
SELF_CHAT_FILENAME = "chat/{capability_id}.parquet"

SAMPLE_SCHEMA = pa.schema(
    [
        ("problem_request_id", pa.string()),
        ("sample", pa.int64()),
        ("finish_reason", pa.string()),
        ("extracted", pa.string()),
        ("correct", pa.bool_()),
        ("selected", pa.bool_()),
        ("rejection_reason", pa.string()),
        ("output", pa.string()),
    ]
)


@dataclass(frozen=True)
class SelfDistillConfig:
    problems_paths: dict[str, str]
    output_path: str
    model: ModelConfig
    accelerator: AcceleratorChoice
    samples_per_problem: int
    solutions_per_problem: int
    temperature: float
    max_completion_tokens: int
    max_sequence_tokens: int
    seed: int


def split_thinking(output: str) -> tuple[str, str] | None:
    """Split a raw Snowball completion into reasoning and answer, or None when the think block is unclosed."""
    text = output.removeprefix(START_THINK)
    if END_THINK not in text:
        return None
    reasoning, _, content = text.partition(END_THINK)
    return reasoning.strip(), content.strip()


def grade_samples(
    problem: dict[str, Any],
    outputs: list[tuple[str, str]],
    config: SelfDistillConfig,
    sequence_tokens: Callable[[dict[str, Any]], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grade one problem's (finish_reason, output) samples and keep the first verified solutions."""
    records: list[dict[str, Any]] = []
    chat_rows: list[dict[str, Any]] = []
    for sample, (finish_reason, output) in enumerate(outputs):
        request_id = f"{problem['request_id']}-self{sample:02d}"
        reason: str | None = None
        extracted: str | None = None
        correct = False
        row: dict[str, Any] | None = None
        split = split_thinking(output)
        if finish_reason == "length":
            reason = "truncated"
        elif split is None:
            reason = "unclosed_think"
        else:
            reasoning, content = split
            extracted = extract_boxed(content)
            if not reasoning:
                reason = "no_reasoning"
            elif extracted is None:
                reason = "no_boxed_answer"
            elif not (correct := answers_match(problem["answer"], extracted)):
                reason = "wrong_answer"
            else:
                row = reasoning_chat_row(request_id, problem["problem"], content, reasoning)
                if sequence_tokens(row) > config.max_sequence_tokens:
                    reason = "too_long"
        selected = reason is None and len(chat_rows) < config.solutions_per_problem
        if selected:
            assert row is not None
            chat_rows.append(row)
        records.append(
            {
                "problem_request_id": problem["request_id"],
                "sample": sample,
                "finish_reason": finish_reason,
                "extracted": extracted,
                "correct": correct,
                "selected": selected,
                "rejection_reason": reason,
                "output": output,
            }
        )
    return records, chat_rows


def _sample(endpoint: OpenAIEndpoint, problem: dict[str, Any], config: SelfDistillConfig) -> list[tuple[str, str]]:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    body = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": problem["problem"]}],
        "n": config.samples_per_problem,
        "temperature": config.temperature,
        "max_tokens": config.max_completion_tokens,
        "seed": config.seed,
        # Snowball's think delimiters are special tokens; keep them so the reasoning can be split out.
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    response = requests.post(endpoint.url("chat/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return [(choice["finish_reason"], choice["message"]["content"] or "") for choice in response.json()["choices"]]


def _serve_and_sample(config: SelfDistillConfig) -> None:
    problems = {
        capability_id: [row for row in load_parquet(str(StoragePath(path) / PROBLEMS_FILENAME)) if row["accepted"]]
        for capability_id, path in config.problems_paths.items()
    }
    flat = [(capability_id, problem) for capability_id, rows in problems.items() for problem in rows]
    inference = inference_config_for_model(
        config.model,
        config.accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
            outputs = list(pool.map(lambda item: _sample(endpoint, item[1], config), flat))

    # math-verify arms its timeout with SIGALRM, which only works on the main thread.
    assert config.model.tokenizer is not None and config.model.tokenizer_revision is not None
    sequence_tokens = sequence_token_counter(config.model.tokenizer, config.model.tokenizer_revision)
    output = StoragePath(config.output_path)
    output.mkdirs()
    manifest: dict[str, Any] = {"model": config.model.location, "revision": config.model.revision, "capabilities": {}}
    for capability_id in problems:
        records: list[dict[str, Any]] = []
        chat_rows: list[dict[str, Any]] = []
        for (item_capability, problem), samples in zip(flat, outputs, strict=True):
            if item_capability != capability_id:
                continue
            problem_records, problem_rows = grade_samples(problem, samples, config, sequence_tokens)
            records += problem_records
            chat_rows += problem_rows
        write_table(output, SAMPLES_FILENAME.format(capability_id=capability_id), records, SAMPLE_SCHEMA)
        write_table(output, SELF_CHAT_FILENAME.format(capability_id=capability_id), chat_rows, REASONING_CHAT_SCHEMA)
        reasons: dict[str, int] = {}
        for record in records:
            key = record["rejection_reason"] or ("selected" if record["selected"] else "extra_correct")
            reasons[key] = reasons.get(key, 0) + 1
        manifest["capabilities"][capability_id] = {
            "problems": len(problems[capability_id]),
            "chat_rows": len(chat_rows),
            "samples": reasons,
        }
        logger.info("kept %s self-distilled rows for %s: %s", len(chat_rows), capability_id, reasons)
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def self_distill_step(
    problems: dict[str, ArtifactStep[Artifact]],
    *,
    name: str,
    version: str,
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    samples_per_problem: int,
    solutions_per_problem: int,
    temperature: float,
    max_completion_tokens: int,
    max_sequence_tokens: int,
    seed: int,
) -> ArtifactStep[Artifact]:
    """Build one step that samples ``model`` on every accepted problem and writes per-capability chat rows."""
    if accelerator.platform is not Platform.GPU:
        raise ValueError("self-distillation serves Snowball on GPU vLLM")

    def build_config(ctx: StepContext) -> SelfDistillConfig:
        return SelfDistillConfig(
            problems_paths={capability_id: ctx.artifact_path(step) for capability_id, step in problems.items()},
            output_path=ctx.output_path,
            model=model,
            accelerator=accelerator,
            samples_per_problem=samples_per_problem,
            solutions_per_problem=solutions_per_problem,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_sequence_tokens=max_sequence_tokens,
            seed=seed,
        )

    def run(config: SelfDistillConfig) -> Artifact:
        # The orchestrator is a small CPU job; remote inference launches the GPU serving job beside it.
        remote(
            _serve_and_sample,
            name=name.replace("/", "-"),
            resources=ResourceConfig.with_cpu(cpu=4, ram="32g", target_cluster=accelerator.target_cluster),
            env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        )(config)
        return Artifact(path=config.output_path)

    return ArtifactStep(
        name=user_owned_name(name),
        version=version,
        artifact_type=Artifact,
        run=run,
        build_config=build_config,
        deps=tuple(problems.values()),
    )
