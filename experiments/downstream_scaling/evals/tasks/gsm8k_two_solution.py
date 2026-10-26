# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-solution GSM8K advisor prompts for downstream-scaling evals."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import fsspec
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.dataset import ShardInfo

from experiments.downstream_scaling.evals.framework.schema import prompts_file
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, _load_gsm8k_task
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, localize_mirror_path, version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}

NUM_EXEMPLARS = 5
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
N_SAMPLES = 128
TEMPERATURE = 0.4
TOP_K = 16
MAX_TOKENS = 512
MAX_MODEL_LEN = 8192
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")
LEDGER_PREFIX = "gs://marin-us-central2"

ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TwoSolutionGSM8KTaskConfig:
    advisor_model_path: str | InputName | MirroredValue
    worker_pools: tuple[WorkerPoolConfig, ...]
    heartbeat_timeout: float
    poll_backoff: float
    n_problems: int | None = None
    grade_workers: int = 32


@dataclass(frozen=True)
class TwoSolutionGSM8KPromptsConfig:
    output_path: str
    advisor_model_path: str
    n_problems: int | None
    num_exemplars: int
    num_fewshot: int
    fewshot_seed: int
    n_samples: int
    temperature: float
    top_k: int
    max_tokens: int
    max_model_len: int
    seed: int
    stop: tuple[str, ...]
    worker_pools: tuple[WorkerPoolConfig, ...]
    heartbeat_timeout: float
    poll_backoff: float


@dataclass(frozen=True)
class TwoSolutionExemplar:
    question: str
    reference: str
    second_solution: str


def select_second_solution(reference: str, samples: list[str]) -> str:
    """Select the first correct sample whose text differs from the reference."""
    reference_match = ANSWER_PATTERN.search(reference)
    if reference_match is None:
        raise ValueError("GSM8K reference has no '####' answer marker")
    answer = reference_match.group(1).replace(",", "")
    reference_words = reference.split()
    for sample in samples:
        match = ANSWER_PATTERN.search(sample)
        if match is not None and match.group(1).replace(",", "") == answer and sample.split() != reference_words:
            return sample.strip()
    raise ValueError(f"No correct, distinct second solution among {len(samples)} samples")


def render_exemplar_block(exemplars: list[TwoSolutionExemplar]) -> str:
    """Render two-solution exemplars, each followed by a blank line."""
    return "".join(
        f"Question: {exemplar.question}\n"
        f"Solution 1: {exemplar.reference}\n"
        f"Solution 2: {exemplar.second_solution}\n\n"
        for exemplar in exemplars
    )


@dataclass(frozen=True)
class TwoSolutionGSM8KTask:
    config: TwoSolutionGSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/gsm8k_two_solution",
            fn=remote(
                run_gsm8k_two_solution_prompts,
                pip_dependency_groups=["vllm", "lm_eval"],
                env_vars=VLLM_TPU_ENV_VARS,
            ),
            config=TwoSolutionGSM8KPromptsConfig(
                output_path=this_output_path(),
                advisor_model_path=version_path(self.config.advisor_model_path),  # type: ignore[arg-type]
                n_problems=versioned(self.config.n_problems),  # type: ignore[arg-type]
                num_exemplars=versioned(NUM_EXEMPLARS),  # type: ignore[arg-type]
                num_fewshot=versioned(NUM_FEWSHOT),  # type: ignore[arg-type]
                fewshot_seed=versioned(FEWSHOT_SEED),  # type: ignore[arg-type]
                n_samples=versioned(N_SAMPLES),  # type: ignore[arg-type]
                temperature=versioned(TEMPERATURE),  # type: ignore[arg-type]
                top_k=versioned(TOP_K),  # type: ignore[arg-type]
                max_tokens=versioned(MAX_TOKENS),  # type: ignore[arg-type]
                max_model_len=versioned(MAX_MODEL_LEN),  # type: ignore[arg-type]
                seed=versioned(SEED),  # type: ignore[arg-type]
                stop=versioned(STOP_TOKENS),  # type: ignore[arg-type]
                worker_pools=self.config.worker_pools,
                heartbeat_timeout=self.config.heartbeat_timeout,
                poll_backoff=self.config.poll_backoff,
            ),
        )


def run_gsm8k_two_solution_prompts(config: TwoSolutionGSM8KPromptsConfig) -> None:
    """Dispatch the shared prompts artifact across the configured TPU pools."""
    ledger_path = ledger.convert_mirror_path(ledger_prefix=LEDGER_PREFIX, output_path=config.output_path)
    ledger.ensure_manifest(ledger_path, [{"chunk_id": 0}])

    def make_process_shard(pool: WorkerPoolConfig):
        return functools.partial(
            run_two_solution_prompts_worker,
            config=config,
            ledger_path=ledger_path,
            pool_id=pool.pool_id,
        )

    xregion_pool.run_worker_pools(
        worker_pools=config.worker_pools,
        ledger_path=ledger_path,
        make_process_shard=make_process_shard,
        poll_backoff=config.poll_backoff,
        heartbeat_timeout=config.heartbeat_timeout,
    )


def run_two_solution_prompts_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: TwoSolutionGSM8KPromptsConfig,
    ledger_path: str,
    pool_id: str,
) -> Iterator[dict[str, object]]:
    """Load the advisor before claiming and writing the shared prompts."""
    llm = _load_advisor(config)
    owner = f"{pool_id}/shard-{shard_info.shard_idx}"
    while True:
        with ledger.claim_next_chunk(ledger_path, owner) as claim:
            if claim is None:
                summary = ledger.summarize(ledger_path)
                if summary.done == summary.total:
                    break
                time.sleep(config.poll_backoff)
                continue

            write_gsm8k_two_solution_prompts(config, llm)
            ledger.mark_done(claim)
            break

    yield {"status": "done", "pool_id": pool_id, "shard_idx": shard_info.shard_idx}


def _load_advisor(config: TwoSolutionGSM8KPromptsConfig) -> Any:
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    from vllm import LLM  # noqa: PLC0415  # optional dep: vllm

    model_path = localize_mirror_path(discover_hf_checkpoints(config.advisor_model_path)[-1])
    logger.info("Loading advisor from %s", model_path)
    return LLM(
        model=model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=config.seed,
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_model_len,
        enable_prefix_caching=False,
    )


def write_gsm8k_two_solution_prompts(config: TwoSolutionGSM8KPromptsConfig, llm: Any) -> None:
    """Sample the exemplar solutions and write the shared advisor prompts."""
    from vllm import SamplingParams  # noqa: PLC0415  # optional dep: vllm

    task = _load_gsm8k_task()
    task.set_fewshot_seed(config.fewshot_seed)
    exemplar_docs = list(islice(task.training_docs(), config.num_exemplars))
    sampling_prompts = [task.fewshot_context(doc, num_fewshot=config.num_fewshot) for doc in exemplar_docs]

    outputs = llm.generate(
        [prompt for prompt in sampling_prompts for _ in range(config.n_samples)],
        SamplingParams(
            n=1,
            temperature=config.temperature,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            stop=list(config.stop),
        ),
    )
    exemplars = []
    for i, doc in enumerate(exemplar_docs):
        samples = [output.outputs[0].text for output in outputs[i * config.n_samples : (i + 1) * config.n_samples]]
        exemplars.append(
            TwoSolutionExemplar(
                question=doc["question"],
                reference=doc["answer"],
                second_solution=select_second_solution(doc["answer"], samples),
            )
        )
    exemplar_block = render_exemplar_block(exemplars)

    docs = list(task.test_docs())
    if config.n_problems is not None:
        docs = docs[: config.n_problems]

    path = prompts_file(config.output_path)
    with fsspec.open(path, "wt", compression="gzip") as f:
        for i, doc in enumerate(docs):
            match = ANSWER_PATTERN.search(doc["answer"])
            if match is None:
                raise ValueError(f"GSM8K problem {i} has no '####' answer marker")
            row = {
                "id": f"gsm8k/test/{i}",
                "prompt": f"{exemplar_block}Question: {doc['question']}\nSolution 1: {doc['answer']}\nSolution 2:",
                "ground_truth": match.group(1).replace(",", ""),
                "metadata": {
                    "problem": doc["question"],
                    "solution": doc["answer"],
                    "split": "test",
                },
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d two-solution GSM8K prompts to %s", len(docs), path)
