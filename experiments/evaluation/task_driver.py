# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run published evaluation corpora through the standard TaskSpec driver."""

import argparse
import json
import logging
import uuid
from pathlib import Path

from finestore.reader import ReadView
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, Job
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.evaluation.task_worker import (
    DEFAULT_MAX_ATTEMPTS,
    JUDGE_API_KEY,
    JUDGE_ENDPOINT,
    SOLVER_API_KEY,
    SOLVER_ENDPOINT,
    WORKER_PYTHON,
    WORKER_TIMEOUT,
    execute_with_retries,
    execution_config,
    run_worker,
)
from marin.rollouts.evaluation import RUN_OBJECT, EvaluationRunConfig
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from rigging.timing import Duration
from verifyit.modes.grade_judge import JudgeConnection

DEFAULT_CLUSTER = "cw-us-east-02a"


def stored_config(root: str) -> EvaluationRunConfig:
    """The run configuration recorded in a result root."""
    configure_coreweave_s3()
    saved = ReadView(root).read_blob(RUN_OBJECT)
    if saved is None:
        raise ValueError(f"No evaluation run is stored in {root}")
    return EvaluationRunConfig.model_validate(json.loads(saved)["config"])


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
    responses_from: str | None = None,
) -> Job:
    """Submit a disposable privileged CPU worker; Shellbox contains every grader command."""
    roots = (*datasets, output, *((responses_from,) if responses_from else ()))
    if any(not StoragePath(root).is_remote for root in roots):
        raise ValueError(
            "Iris evaluation requires shared object-storage corpus, output and --responses-from roots; "
            "use --local for local files"
        )
    return client.submit(
        entrypoint=Entrypoint.from_callable(
            run_worker, datasets, output, execution_config(config), max_attempts, responses_from
        ),
        name=name,
        resources=resources,
        environment=EnvironmentSpec(
            setup_scripts=(default_setup_script(packages=("marin-core",), python_version=WORKER_PYTHON),),
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
        "--config",
        help="EvaluationRunConfig JSON, including immutable model/tokenizer IDs; defaults to --responses-from's.",
    )
    parser.add_argument(
        "--responses-from",
        help="Result root whose stored model responses answer identical requests, as when regrading a corrected corpus.",
    )
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--endpoint", help="OpenAI-compatible base URL; optional with --responses-from.")
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
    responses_from = arguments.responses_from
    base_url = (arguments.endpoint_file.read_text().strip() if arguments.endpoint_file else arguments.endpoint) or None
    if not responses_from and (arguments.config is None or base_url is None):
        parser.error("--config and an endpoint are required without --responses-from")
    config = (
        EvaluationRunConfig.model_validate_json(StoragePath(arguments.config).read_text())
        if arguments.config
        else stored_config(responses_from)
    )
    datasets = tuple(arguments.dataset)
    api_key = require_secret(arguments.api_key_env) if base_url else None
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
            datasets,
            arguments.output,
            config,
            base_url,
            api_key,
            judge,
            max_attempts=arguments.max_attempts,
            responses_from=responses_from,
        )
        return
    secrets = {SOLVER_ENDPOINT: base_url, SOLVER_API_KEY: api_key} if base_url and api_key else {}
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
            responses_from=responses_from,
        )
        print(job.job_id)
        if not arguments.no_wait:
            job.wait(timeout=arguments.timeout, stream_logs=True)


if __name__ == "__main__":
    main()
