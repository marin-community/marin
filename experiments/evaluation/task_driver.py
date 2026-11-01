# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run published evaluation corpora through the standard TaskSpec driver."""

import argparse
import asyncio
import logging
import uuid
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, Job
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.evaluation.likelihood import VllmLikelihood
from marin.evaluation.task_driver import EvaluationInfrastructureError, EvaluationRunConfig, run_evaluation
from openai import AsyncOpenAI
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret, require_secret
from rigging.timing import Duration, retry_with_backoff
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, NetworkPolicy
from transformers import AutoTokenizer
from verifyit.modes.grade_judge import JudgeConnection

DEFAULT_CLUSTER = "cw-us-east-02a"
DEFAULT_MAX_ATTEMPTS = 3
WORKER_TIMEOUT = 36 * 60 * 60
SOLVER_ENDPOINT = "EVALUATION_SOLVER_ENDPOINT"
SOLVER_API_KEY = "EVALUATION_SOLVER_API_KEY"
JUDGE_ENDPOINT = "EVALUATION_JUDGE_ENDPOINT"
JUDGE_API_KEY = "EVALUATION_JUDGE_API_KEY"


def execution_config(config: EvaluationRunConfig) -> EvaluationRunConfig:
    """Select network-denied Shellbox bubblewrap for published package-lock graders."""
    verifier = config.verifier_machine or MachineRuntimeSpec(
        backend=Backend.LOCAL,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=config.limits.cleanup_timeout,
    )
    backends = {selection.backend for selection in (config.task_machine, verifier) if selection is not None}
    if unsupported := backends - {Backend.LOCAL}:
        raise ValueError(f"This entrypoint has no factories for {sorted(unsupported)}; supply them to run_evaluation")
    return config.model_copy(update={"verifier_machine": verifier})


async def execute(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str,
    api_key: str,
    judge_connection: JudgeConnection | None,
) -> None:
    """Resolve the pinned tokenizer and endpoint transports once for the run."""
    configure_coreweave_s3()
    config = execution_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False
    )
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("Likelihood scoring needs a tokenizer BOS or EOS prefix token")
    machines = {Backend.LOCAL: LocalMachineFactory()}
    async with AsyncOpenAI(
        base_url=endpoint, api_key=api_key, max_retries=0, timeout=config.limits.model_turn_timeout
    ) as client:
        await run_evaluation(
            datasets,
            output,
            config,
            ChatModel(client, config.model.name, config.max_tokens),
            completion_model=CompletionModel(client, config.model.name),
            likelihood=VllmLikelihood(client, config.model.name, tokenizer, config.model.context_window, prefix),
            machine_factories=machines,
            tokenizer=tokenizer,
            judge_connection=judge_connection,
        )


def execute_with_retries(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str,
    api_key: str,
    judge_connection: JudgeConnection | None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> None:
    """Resume infrastructure failures with the same run identity and Marin's standard backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    retry_with_backoff(
        lambda: asyncio.run(execute(datasets, output, config, endpoint, api_key, judge_connection)),
        retryable=lambda error: isinstance(error, EvaluationInfrastructureError),
        max_attempts=max_attempts,
        operation="TaskSpec evaluation",
    )


def run_worker(datasets: tuple[str, ...], output: str, config: EvaluationRunConfig, max_attempts: int) -> None:
    """Read connections from Iris secret files outside the grader machines."""
    logging.basicConfig(level=logging.INFO)
    # Iris endpoint URLs can contain scoped credentials.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    judge_endpoint = get_secret(JUDGE_ENDPOINT)
    judge = JudgeConnection(judge_endpoint, require_secret(JUDGE_API_KEY)) if judge_endpoint else None
    execute_with_retries(
        datasets,
        output,
        config,
        require_secret(SOLVER_ENDPOINT),
        require_secret(SOLVER_API_KEY),
        judge,
        max_attempts=max_attempts,
    )


def submit_evaluation(
    client: IrisClient,
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    *,
    name: str,
    resources: ResourceSpec,
    timeout: int,
    secrets: dict[str, str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Job:
    """Submit a disposable privileged CPU worker; Shellbox contains every grader command."""
    if any(not StoragePath(root).is_remote for root in (*datasets, output)):
        raise ValueError(
            "Iris evaluation requires shared object-storage corpus and output roots; use --local for local files"
        )
    return client.submit(
        entrypoint=Entrypoint.from_callable(run_worker, datasets, output, execution_config(config), max_attempts),
        name=name,
        resources=resources,
        environment=EnvironmentSpec(
            sync_packages=("marin-core", "marin-taskcompendium"),
            extras=("marin-taskcompendium:pipeline",),
            secrets=secrets,
        ),
        container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        timeout=Duration.from_seconds(timeout),
        max_retries_failure=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, help="Published corpus root; repeat for a mixed run."
    )
    parser.add_argument("--output", required=True, help="FineStore archive root, including an existing run to resume.")
    parser.add_argument(
        "--config", required=True, help="EvaluationRunConfig JSON, including immutable model/tokenizer IDs."
    )
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--endpoint", help="OpenAI-compatible base URL.")
    endpoint.add_argument(
        "--endpoint-file", type=Path, help="Read the base URL, including any scoped credential, from a file."
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-endpoint", help="Explicit judge endpoint, when the selected tasks use a judge.")
    parser.add_argument("--judge-api-key-env", default="VERIFYIT_JUDGE_API_KEY")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--cluster", default=DEFAULT_CLUSTER, help="Iris cluster for the CPU evaluation worker.")
    execution.add_argument(
        "--local", action="store_true", help="Run on this host, which must support Shellbox bubblewrap."
    )
    parser.add_argument("--job-name", default=None, help="Iris job name; generated when omitted.")
    parser.add_argument("--cpu", type=float, default=4)
    parser.add_argument("--memory", default="32g")
    parser.add_argument("--disk", default="64g")
    parser.add_argument("--timeout", type=int, default=WORKER_TIMEOUT, help="Iris worker lifetime in seconds.")
    parser.add_argument("--no-wait", action="store_true", help="Return after Iris submission.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Total passes for infrastructure failures; successful attempts are retained between passes.",
    )
    arguments = parser.parse_args()
    if arguments.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    config = EvaluationRunConfig.model_validate_json(StoragePath(arguments.config).read_text())
    base_url = arguments.endpoint_file.read_text().strip() if arguments.endpoint_file else arguments.endpoint
    datasets = tuple(arguments.dataset)
    api_key = require_secret(arguments.api_key_env)
    if arguments.local:
        judge = (
            JudgeConnection(arguments.judge_endpoint, require_secret(arguments.judge_api_key_env))
            if arguments.judge_endpoint
            else None
        )
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        execute_with_retries(
            datasets, arguments.output, config, base_url, api_key, judge, max_attempts=arguments.max_attempts
        )
        return
    secrets = {SOLVER_ENDPOINT: base_url, SOLVER_API_KEY: api_key}
    if arguments.judge_endpoint:
        secrets[JUDGE_ENDPOINT] = arguments.judge_endpoint
        secrets[JUDGE_API_KEY] = require_secret(arguments.judge_api_key_env)
    with open_iris_client(cluster_name=arguments.cluster, workspace=find_project_root()) as client:
        job = submit_evaluation(
            client,
            datasets,
            arguments.output,
            config,
            name=arguments.job_name or f"task-eval-{uuid.uuid4().hex[:8]}",
            resources=ResourceSpec(cpu=arguments.cpu, memory=arguments.memory, disk=arguments.disk),
            timeout=arguments.timeout,
            secrets=secrets,
            max_attempts=arguments.max_attempts,
        )
        print(job.job_id)
        if not arguments.no_wait:
            job.wait(timeout=arguments.timeout, stream_logs=True)


if __name__ == "__main__":
    main()
