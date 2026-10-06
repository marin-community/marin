# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one context-checked coding analysis on a bounded regional CPU worker."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import httpx
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.rpc import job_pb2
from marin.evaluation.eval_env import env_vars_from_keys
from marin.external_dependencies import MARIN_SKYRL
from marin.training.run_environment import dependency_groups_for_resources
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    PartitionedCodingAnalysisConfig,
    analyze_partitioned_coding_eval_failures,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingAnalysisConfig, analyze_coding_failures
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal
from experiments.post_training.russell_rsi.interrupted_calibration import PACKAGED_PYTHONPATH
from experiments.post_training.russell_rsi.launch import CLUSTER
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256

WORKER_SETTINGS = {
    "cluster": CLUSTER,
    "cpu": 2,
    "ram": "8GB",
    "disk": "16GB",
    "timeout_minutes": 20,
    "priority": "batch",
    "failure_retries": 0,
    "preemption_retries": 0,
    "max_task_failures": 0,
}
SOURCE_FILES = (
    "experiments/post_training/russell_rsi/coding_analysis_worker.py",
    "experiments/post_training/russell_rsi/coding_eval_feedback.py",
    "experiments/post_training/russell_rsi/coding_analysis_recovery.py",
    "experiments/post_training/russell_rsi/completed_partitioned_coding_analysis.py",
    "experiments/post_training/russell_rsi/coding_analysis_response_recovery.py",
    "experiments/post_training/russell_rsi/feedback.py",
    "experiments/post_training/glm.py",
)


@dataclass(frozen=True)
class RegionalCodingAnalysisConfig:
    analysis: CodingAnalysisConfig
    input_pin: PinnedFile
    decision: PinnedFile
    evidence: PinnedFile
    source_files: dict[str, str]


@dataclass(frozen=True)
class RegionalPartitionedCodingAnalysisConfig:
    analysis: CodingAnalysisConfig
    input_pin: PinnedFile
    failure: PinnedFile
    evidence: PinnedFile
    source_files: dict[str, str]
    manifest: PinnedFile

    def partitioned(self) -> PartitionedCodingAnalysisConfig:
        return PartitionedCodingAnalysisConfig(self.analysis, self.manifest.uri, self.manifest.sha256)


def worker_source_files() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


async def context_preflight(config: RegionalCodingAnalysisConfig, base_url: str, request: dict) -> None:
    """Measure the exact served chat prompt before the canonical issuance marker."""
    decision = config.decision.read_json()
    settings = decision["analysis"]
    if compact_json_sha256(request) != decision["request_sha256"]:
        raise ValueError("Analyst request differs from the frozen decision")
    directory = StoragePath(config.analysis.output_path) / "context-preflight"
    async with httpx.AsyncClient(timeout=30, headers={"Authorization": f"Bearer {os.environ[GLM_TOKEN_ENV]}"}) as client:

        async def probe(name: str, method: str, url: str, body: dict | None = None) -> dict:
            record = {
                "method": method,
                "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                "request_sha256": None if body is None else compact_json_sha256(body),
            }
            try:
                response = await client.request(method, url, json=body)
                record.update(
                    status_code=response.status_code, response_sha256=hashlib.sha256(response.content).hexdigest()
                )
                write_once(directory / f"{name}.json", record)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as error:
                if "status_code" not in record:
                    write_once(directory / f"{name}.json", {**record, "error_type": type(error).__name__})
                raise

        models = await probe("models", "GET", base_url.rstrip("/") + "/models")
        cards = [card for card in models["data"] if card["id"] == request["model"]]
        if len(cards) != 1 or cards[0]["max_model_len"] < settings["context_limit"]:
            raise ValueError("Analyst relay does not advertise the exact model and required context")
        tokenize_request = {
            "model": request["model"],
            "messages": request["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": request["extra_body"]["chat_template_kwargs"],
        }
        service_root = base_url.rstrip("/").removesuffix("/v1")
        try:
            tokenized = await probe("tokenize", "POST", service_root + "/tokenize", tokenize_request)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            document = await probe("openapi", "GET", service_root + "/openapi.json")
            paths = document["paths"]
            write_once(directory / "documented-routes.json", {"paths": sorted(paths)})
            if "/v1/tokenize" not in paths or "post" not in paths["/v1/tokenize"]:
                raise ValueError("Relay has no documented exact chat tokenization route") from error
            tokenized = await probe("documented-tokenize", "POST", service_root + "/v1/tokenize", tokenize_request)
        tokens = tokenized["tokens"]
        count = tokenized["count"]
        if (
            type(count) is not int
            or count < 0
            or len(tokens) != count
            or any(type(token) is not int or token < 0 for token in tokens)
            or tokenized["max_model_len"] < settings["context_limit"]
            or count + request["max_tokens"] > settings["context_limit"]
        ):
            raise ValueError("Exact served analyst chat prompt does not fit the frozen context")
        write_once(
            directory / "result.json",
            {
                "status": "passed",
                "request_sha256": decision["request_sha256"],
                "model": request["model"],
                "prompt_tokens": count,
                "max_output_tokens": request["max_tokens"],
                "context_limit": settings["context_limit"],
                "token_ids_sha256": compact_json_sha256(tokens),
                "tokenize_request_sha256": compact_json_sha256(tokenize_request),
            },
        )


def require_regional_worker_source(
    output_path: str, expected_files: dict[str, str], inputs: tuple[PinnedFile, ...]
) -> None:
    for pin in inputs:
        pin.read_bytes()
    distribution = importlib.metadata.distribution(MARIN_SKYRL.distribution)
    direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    source_files = worker_source_files()
    if not isinstance(direct_url, dict) or "vcs_info" not in direct_url:
        raise ValueError("Regional analyst runtime has no pinned VCS provenance")
    if source_files != expected_files or direct_url["vcs_info"]["commit_id"] != MARIN_SKYRL.commit:
        raise ValueError("Regional analyst loaded different source or runtime")
    write_once(
        StoragePath(output_path) / "analysis-worker-provenance.json",
        {
            "source_files": source_files,
            "branch_root": str(Path(__file__).resolve().parents[3]),
            "skyrl": {"version": distribution.version, "direct_url": direct_url},
            "worker": WORKER_SETTINGS,
        },
    )


def run_regional_coding_analysis(config: RegionalCodingAnalysisConfig) -> None:
    require_regional_worker_source(
        config.analysis.output_path, config.source_files, (config.input_pin, config.evidence, config.decision)
    )
    asyncio.run(analyze_coding_failures(config.analysis, before_issue=partial(context_preflight, config)))


def run_regional_partitioned_coding_analysis(config: RegionalPartitionedCodingAnalysisConfig) -> None:
    require_regional_worker_source(
        config.analysis.output_path,
        config.source_files,
        (config.input_pin, config.evidence, config.failure, config.manifest),
    )
    analyze_partitioned_coding_eval_failures(config.partitioned())


def regional_worker_request(output_path: str, entrypoint: Entrypoint) -> JobRequest:
    if not os.environ.get(GLM_TOKEN_ENV):
        raise ValueError("Regional analyst requires the approved GLM token in its private environment")
    resources = ResourceConfig.with_cpu(
        cpu=WORKER_SETTINGS["cpu"],
        ram=WORKER_SETTINGS["ram"],
        disk=WORKER_SETTINGS["disk"],
        target_cluster=WORKER_SETTINGS["cluster"],
    )
    return JobRequest(
        name=f"russell-coding-analysis-{hashlib.sha256(output_path.encode()).hexdigest()[:12]}",
        entrypoint=entrypoint,
        resources=resources,
        environment=create_environment(
            extras=dependency_groups_for_resources(resources, None),
            pip_packages=[MARIN_SKYRL.requirement()],
            env_vars={
                "UV_PRERELEASE": "allow",
                "PYTHONPATH": PACKAGED_PYTHONPATH,
                **env_vars_from_keys((GLM_TOKEN_ENV, "CW_KEY_ID", "CW_KEY_SECRET")),
            },
        ),
        max_retries_failure=WORKER_SETTINGS["failure_retries"],
        max_retries_preemption=WORKER_SETTINGS["preemption_retries"],
        max_task_failures=WORKER_SETTINGS["max_task_failures"],
        priority=job_pb2.PRIORITY_BAND_BATCH,
        timeout=Duration.from_minutes(WORKER_SETTINGS["timeout_minutes"]),
    )


def regional_analysis_request(config: RegionalCodingAnalysisConfig) -> JobRequest:
    return regional_worker_request(
        config.analysis.output_path, Entrypoint.from_callable(run_regional_coding_analysis, args=(config,))
    )


def submit_regional_worker(output_path: str, binding: dict, request_factory: Callable[[], JobRequest]) -> None:
    attempt = AttemptJournal(StoragePath(output_path) / "worker-submission", binding)
    if attempt.saved_result() is not None:
        return
    request = request_factory()

    async def submit() -> dict:
        current_client().submit(request, adopt_existing=True).wait(raise_on_failure=True)
        return {"worker_completed": True}

    asyncio.run(attempt.run(submit))


def submit_regional_coding_analysis(config: RegionalCodingAnalysisConfig) -> None:
    submit_regional_worker(config.analysis.output_path, asdict(config), partial(regional_analysis_request, config))


def regional_partitioned_analysis_request(config: RegionalPartitionedCodingAnalysisConfig) -> JobRequest:
    return regional_worker_request(
        config.analysis.output_path, Entrypoint.from_callable(run_regional_partitioned_coding_analysis, args=(config,))
    )


def submit_regional_partitioned_coding_analysis(config: RegionalPartitionedCodingAnalysisConfig) -> None:
    submit_regional_worker(
        config.analysis.output_path, asdict(config), partial(regional_partitioned_analysis_request, config)
    )
