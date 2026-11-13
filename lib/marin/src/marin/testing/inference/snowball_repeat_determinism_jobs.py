# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote Iris job measuring whether identical vLLM scoring requests repeat exactly.

Retest for https://github.com/marin-community/marin/issues/10699. One launch serves
Snowball at DP8 with expert parallelism and scores a fixed set of prompts on every
data-parallel rank: two serial passes with one request in flight on the whole serve,
then one pass with one request in flight per rank. Each response is reduced to a
fingerprint over the emitted token, the next-token top log-probs, and every prompt
position's log-prob, so equal fingerprints mean bit-identical scores.

The job measures and reports; it does not gate. Each launch compiles from an empty
cache, so launches of one arm can be compared through their logged fingerprints.
"""

import dataclasses
import hashlib
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

import requests
from rigging.filesystem.s3_compat import configure_coreweave_s3

from marin.inference.backend import ModelSpec
from marin.inference.config import VllmCompilationCacheMode, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_vllm import iris_vllm_followers, iris_vllm_launch
from marin.inference.vllm_backend import VllmBackend
from marin.testing.inference.snowball import (
    SNOWBALL,
    RepresentativeCase,
    RepresentativeGolden,
    read_prompt_fixture,
)
from marin.testing.inference.snowball_backend_parity_jobs import (
    GPU_COUNT,
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    RETURNED_LOGPROBS,
    _wait_for_exact_vllm_topology,
)

logger = logging.getLogger(__name__)

RESULT_MARKER = "REPEAT_DETERMINISM_RESULT"
RESULT_FILENAME = "repeat_determinism.json"
# Two full buckets of short prompts plus the parity job's sentinel, which is longer than
# the 2,048-token sliding window and spans several 512-token prefill chunks.
SHORT_BUCKET_MAX_TOKENS = (256, 1024)
SENTINEL_CASE_ID = "knowledge-longbench-02"
# vLLM enables the combo-kernel benchmark by default, and TORCHINDUCTOR_DETERMINISTIC=1
# rejects it at startup. This is the override reported in the issue.
DETERMINISTIC_COMPILATION_CONFIG = json.dumps(
    {"inductor_compile_config": {"combo_kernels": True, "benchmark_combo_kernel": False}}
)


class CompileMode(StrEnum):
    """How the served graph is compiled. The job environment sets the matching env var."""

    DEFAULT = "default"
    DETERMINISTIC = "deterministic"


class ScoringPass(StrEnum):
    SERIAL_FIRST = "serial-1"
    SERIAL_SECOND = "serial-2"
    ONE_PER_RANK = "one-per-rank"


@dataclasses.dataclass(frozen=True)
class ScoreObservation:
    """One scoring response reduced to an exact fingerprint."""

    case_id: str
    rank: int
    scoring_pass: ScoringPass
    fingerprint: str
    summed_prompt_logprob: float
    prompt_tokens: int


def _token_id(key: object) -> int:
    return int(str(key).removeprefix("token_id:"))


def _score_case(
    completions_url: str,
    model_id: str,
    case: RepresentativeCase,
    rank: int,
    scoring_pass: ScoringPass,
    prompt_logprobs: int,
) -> ScoreObservation:
    request_id = f"{scoring_pass}-{case.id}-rank-{rank}"
    response = requests.post(
        completions_url,
        headers={"X-data-parallel-rank": str(rank), "X-Request-Id": request_id},
        json={
            "model": model_id,
            "prompt": list(case.prompt_token_ids),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": RETURNED_LOGPROBS,
            "prompt_logprobs": prompt_logprobs,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    assert choice["prompt_token_ids"] == list(case.prompt_token_ids), request_id
    (emitted_token_id,) = choice["token_ids"]
    (next_token_top,) = choice["logprobs"]["top_logprobs"]
    positions = choice["prompt_logprobs"]
    assert positions is not None and len(positions) == len(case.prompt_token_ids), request_id
    assert positions[0] is None, request_id

    prompt_scores = []
    summed = 0.0
    for token_id, entry in zip(case.prompt_token_ids[1:], positions[1:], strict=True):
        by_token = {_token_id(key): float(value["logprob"]) for key, value in entry.items()}
        summed += by_token[token_id]
        prompt_scores.append(sorted((key, value.hex()) for key, value in by_token.items()))
    canonical = json.dumps(
        {
            "emitted": int(emitted_token_id),
            "next": sorted((_token_id(key), float(value).hex()) for key, value in next_token_top.items()),
            "prompt": prompt_scores,
        },
        separators=(",", ":"),
    )
    return ScoreObservation(
        case_id=case.id,
        rank=rank,
        scoring_pass=scoring_pass,
        fingerprint=hashlib.sha256(canonical.encode()).hexdigest(),
        summed_prompt_logprob=summed,
        prompt_tokens=len(case.prompt_token_ids),
    )


def _supported_prompt_logprobs(completions_url: str, model_id: str, case: RepresentativeCase) -> int:
    """Prefer the gate's ``prompt_logprobs=0``; use 1 if the server returns nothing for 0."""
    response = requests.post(
        completions_url,
        json={
            "model": model_id,
            "prompt": list(case.prompt_token_ids),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    return 0 if choice["prompt_logprobs"] else 1


def _select_cases(goldens: tuple[RepresentativeGolden, ...]) -> tuple[RepresentativeCase, ...]:
    fixture = read_prompt_fixture(goldens)
    cases = []
    for max_tokens in SHORT_BUCKET_MAX_TOKENS:
        batch = next(batch for batch in fixture.batches if batch.max_tokens == max_tokens)
        cases.extend(batch.cases)
    sentinel = next(case for case in fixture.cases if case.id == SENTINEL_CASE_ID)
    assert len(sentinel.prompt_token_ids) > 2048
    cases.append(sentinel)
    return tuple(cases)


def _matching_fingerprints(first: list[ScoreObservation], second: list[ScoreObservation]) -> int:
    return sum(a.fingerprint == b.fingerprint for a, b in zip(first, second, strict=True))


def measure_endpoint(
    cases: tuple[RepresentativeCase, ...],
    completions_url: str,
    model_id: str,
    *,
    data_parallel_size: int,
) -> dict:
    """Score every case on every rank in three passes and summarize exact agreement."""
    prompt_logprobs = _supported_prompt_logprobs(completions_url, model_id, cases[0])
    logger.info("Scoring with prompt_logprobs=%d", prompt_logprobs)
    ranks = range(data_parallel_size)
    observations: list[ScoreObservation] = []
    for scoring_pass in (ScoringPass.SERIAL_FIRST, ScoringPass.SERIAL_SECOND):
        for case in cases:
            for rank in ranks:
                observations.append(_score_case(completions_url, model_id, case, rank, scoring_pass, prompt_logprobs))
        logger.info("Finished pass %s", scoring_pass)
    with ThreadPoolExecutor(max_workers=data_parallel_size) as executor:
        for case in cases:
            futures = [
                executor.submit(
                    _score_case, completions_url, model_id, case, rank, ScoringPass.ONE_PER_RANK, prompt_logprobs
                )
                for rank in ranks
            ]
            observations.extend(future.result() for future in futures)
    logger.info("Finished pass %s", ScoringPass.ONE_PER_RANK)

    by_key = {(o.case_id, o.rank, o.scoring_pass): o for o in observations}
    case_reports = []
    for case in cases:
        first = [by_key[case.id, rank, ScoringPass.SERIAL_FIRST] for rank in ranks]
        second = [by_key[case.id, rank, ScoringPass.SERIAL_SECOND] for rank in ranks]
        concurrent = [by_key[case.id, rank, ScoringPass.ONE_PER_RANK] for rank in ranks]
        sums = [o.summed_prompt_logprob for o in first]
        scored_positions = len(case.prompt_token_ids) - 1
        case_reports.append(
            {
                "case_id": case.id,
                "prompt_tokens": len(case.prompt_token_ids),
                "same_rank_serial_repeat_matches": _matching_fingerprints(first, second),
                "same_rank_concurrent_matches": _matching_fingerprints(first, concurrent),
                "distinct_fingerprints_across_ranks": len({o.fingerprint for o in first}),
                "rank_groups": sorted(Counter(o.fingerprint for o in first).values(), reverse=True),
                "summed_logprob_range_across_ranks": max(sums) - min(sums),
                "mean_logprob_range_across_ranks": (max(sums) - min(sums)) / scored_positions,
                "fingerprint_by_rank": [o.fingerprint[:16] for o in first],
                "summed_logprob_by_rank": [float(value).hex() for value in sums],
            }
        )
    pairs = len(cases) * data_parallel_size
    return {
        "prompt_logprobs": prompt_logprobs,
        "cases": len(cases),
        "ranks": data_parallel_size,
        "same_rank_serial_repeat_matches": sum(r["same_rank_serial_repeat_matches"] for r in case_reports),
        "same_rank_concurrent_matches": sum(r["same_rank_concurrent_matches"] for r in case_reports),
        "same_rank_pairs": pairs,
        "cases_identical_across_ranks": sum(r["distinct_fingerprints_across_ranks"] == 1 for r in case_reports),
        "max_summed_logprob_range_across_ranks": max(r["summed_logprob_range_across_ranks"] for r in case_reports),
        "max_mean_logprob_range_across_ranks": max(r["mean_logprob_range_across_ranks"] for r in case_reports),
        "launch_digest": (
            hashlib.sha256(json.dumps([r["fingerprint_by_rank"] for r in case_reports]).encode()).hexdigest()[:16]
        ),
        "case_reports": case_reports,
    }


def measure_vllm_repeat_determinism(
    goldens: tuple[RepresentativeGolden, ...],
    attention_backend: str,
    compile_mode: CompileMode,
) -> dict:
    """Serve Snowball at DP8 with expert parallelism from a fresh compile and measure repeats."""
    configure_coreweave_s3()
    cases = _select_cases(goldens)
    if compile_mode is CompileMode.DETERMINISTIC:
        assert os.environ.get("TORCHINDUCTOR_DETERMINISTIC") == "1"
        compile_args: tuple[str, ...] = ("--compilation-config", DETERMINISTIC_COMPILATION_CONFIG)
    else:
        assert "TORCHINDUCTOR_DETERMINISTIC" not in os.environ
        compile_args = ()
    assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"

    spec = ModelSpec(
        weights=SNOWBALL.export_uri,
        api_model=SNOWBALL.model_name,
        num_chips=GPU_COUNT,
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            # No restore and no publish: every launch compiles from an empty task-local cache,
            # which is the condition under which kernels were reported to differ by launch.
            compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=1,
            extra_args=(
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--max-logprobs",
                str(RETURNED_LOGPROBS),
                "--attention-backend",
                attention_backend,
                "--no-enable-prefix-caching",
                *compile_args,
            ),
        ),
    )
    with iris_vllm_launch(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=GPU_COUNT,
    ) as launch:
        assert launch.is_leader
        with backend.start(
            spec,
            extra_args=launch.extra_cli_args,
            subprocess_env=launch.subprocess_env,
        ) as environment:
            environment.wait_until_ready()
            _wait_for_exact_vllm_topology(
                environment.logs,
                check_alive=environment.check_alive,
                task_index=launch.task_index,
                pipeline_parallel_size=1,
            )
            with iris_vllm_followers(launch):
                report = measure_endpoint(
                    cases,
                    f"{environment.server_url}/completions",
                    spec.api_model,
                    data_parallel_size=GPU_COUNT,
                )
    report = {"compile_mode": str(compile_mode), "attention_backend": attention_backend, **report}
    summary = {key: value for key, value in report.items() if key != "case_reports"}
    logger.info("%s %s", RESULT_MARKER, json.dumps(summary, sort_keys=True))
    for case_report in report["case_reports"]:
        logger.info("%s_CASE %s", RESULT_MARKER, json.dumps(case_report, sort_keys=True))
    output_dir = os.environ.get("IRIS_OUTPUT_DIR")
    if output_dir:
        Path(output_dir, RESULT_FILENAME).write_text(json.dumps(report, sort_keys=True, indent=1))
    return report
