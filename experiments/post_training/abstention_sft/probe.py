# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure what the student knows: sample it 8 times per question and judge each final answer.

The probe serves the v1 SFT model with vLLM, samples with the AbstentionBench evaluation's settings, and
grades each final answer against the Nemotron answer key with the SimpleQA Verified label judge. Short
factual answers ("CQ Press") need a judge: the self-distillation checks cover math, numbers, letters and
code only. A sample that is truncated or still thinking is wrong and never reaches the judge.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
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
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.importers.evalchemy.judge_prompts import simpleqa_verified_spec
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from zephyr.readers import load_parquet

from experiments.post_training.abstention_sft.knowledge import SAMPLES, held_out, knowledge_label
from experiments.post_training.curriculum_sft.generation import MANIFEST_FILENAME, PROBLEMS_FILENAME, write_table
from experiments.post_training.curriculum_sft.self_distill import (
    CONCURRENT_REQUESTS,
    sample_completions,
    split_thinking,
)

SAMPLES_FILENAME = "samples/part-00000-of-00001.parquet"
LABELS_FILENAME = "labels/part-00000-of-00001.parquet"
# The AbstentionBench evaluation's sampling, so the probe measures the behaviour the eval scores.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 16384

SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("sample", pa.int64(), nullable=False),
        pa.field("finish_reason", pa.string(), nullable=False),
        pa.field("output", pa.string(), nullable=False),
        pa.field("correct", pa.bool_(), nullable=False),
    ]
)
LABEL_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("correct_count", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("held_out", pa.bool_(), nullable=False),
    ]
)


def sample_correct(problem: dict[str, Any], finish_reason: str, output: str, connection: JudgeConnection) -> bool:
    """Judge one sample's final answer; a truncated or unfinished reply is wrong without a judge call."""
    if finish_reason == "length":
        return False
    split = split_thinking(output)
    if split is None:
        return False
    _reasoning, content = split
    verdict = grade_judge_candidate(
        simpleqa_verified_spec(problem["problem"], problem["answer"]), content, connection=connection
    )
    return verdict.reward == 1.0


@dataclass(frozen=True)
class ProbeConfig:
    problems_path: str
    output_path: str
    model: ModelConfig
    accelerator: AcceleratorChoice
    judge_endpoint: str
    judge_api_key_env: str
    judge_model: str
    seed: int


def _serve_and_probe(config: ProbeConfig) -> None:
    problems = list(load_parquet(str(StoragePath(config.problems_path) / PROBLEMS_FILENAME)))
    inference = inference_config_for_model(
        config.model,
        config.accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
            outputs = list(
                pool.map(
                    lambda problem: sample_completions(
                        endpoint,
                        problem["problem"],
                        n=SAMPLES,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS,
                        seed=config.seed,
                        enable_thinking=True,
                    ),
                    problems,
                )
            )

    connection = JudgeConnection(
        config.judge_endpoint, require_secret(config.judge_api_key_env), default_model=config.judge_model
    )
    jobs = [
        (problem, index, finish_reason, output)
        for problem, samples in zip(problems, outputs, strict=True)
        for index, (finish_reason, output) in enumerate(samples)
    ]
    with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
        correct = list(pool.map(lambda job: sample_correct(job[0], job[2], job[3], connection), jobs))
    samples = [
        {"request_id": problem["request_id"], "sample": index, "finish_reason": reason, "output": output, "correct": ok}
        for (problem, index, reason, output), ok in zip(jobs, correct, strict=True)
    ]
    counts: dict[str, int] = {problem["request_id"]: 0 for problem in problems}
    for sample in samples:
        counts[sample["request_id"]] += sample["correct"]
    labels = [
        {
            "request_id": request_id,
            "correct_count": count,
            "label": knowledge_label(count).value,
            "held_out": held_out(request_id),
        }
        for request_id, count in counts.items()
    ]
    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, SAMPLES_FILENAME, samples, SAMPLE_SCHEMA)
    write_table(output, LABELS_FILENAME, labels, LABEL_SCHEMA)
    manifest = {
        "model": config.model.location,
        "revision": config.model.revision,
        "judge_model": config.judge_model,
        "samples_per_question": SAMPLES,
        "labels": {label: sum(row["label"] == label for row in labels) for label in ("answer", "abstain", "drop")},
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def probe_step(
    problems: ArtifactStep[Artifact],
    *,
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    judge_endpoint: str,
    judge_api_key_env: str,
    judge_model: str,
    seed: int,
    version: str,
) -> ArtifactStep[Artifact]:
    """Sample ``model`` 8 times per problem and label each problem by how often the judge marks it correct."""
    if accelerator.platform is not Platform.GPU:
        raise ValueError("the probe serves the student on GPU vLLM")

    def build_config(ctx: StepContext) -> ProbeConfig:
        return ProbeConfig(
            problems_path=ctx.artifact_path(problems),
            output_path=ctx.output_path,
            model=model,
            accelerator=accelerator,
            judge_endpoint=judge_endpoint,
            judge_api_key_env=judge_api_key_env,
            judge_model=judge_model,
            seed=seed,
        )

    def run(config: ProbeConfig) -> Artifact:
        remote(
            _serve_and_probe,
            name="abstention-sft-probe",
            resources=ResourceConfig.with_cpu(cpu=4, ram="32g"),
            env_vars=env_vars_from_keys((*EVAL_RUNTIME_ENV_KEYS, judge_api_key_env)),
        )(config)
        return Artifact(path=config.output_path)

    return ArtifactStep(
        name=user_owned_name("documents/abstention-sft/probe"),
        version=version,
        artifact_type=Artifact,
        run=run,
        build_config=build_config,
        deps=(problems,),
    )
