#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual smoke for the Iris inference routing MVP.

This script intentionally does not launch vLLM or Levanter. Start an
OpenAI-compatible engine separately on each worker, then point this smoke at
the engine's worker-local API root, for example http://127.0.0.1:8000/v1.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from fray.client import Client
from fray.local_backend import LocalClient
from marin.evaluation.lm_eval import LmEvalAdapter, LmEvalRun, run_lm_eval
from marin.inference.iris_service import IrisInferenceLauncher, IrisInferenceLauncherConfig
from marin.inference.types import ModelDeployment


@dataclass(frozen=True)
class SmokeArgs:
    engine_base_url: str
    model: str
    task: str
    output_path: str
    tokenizer: str | None
    controller_url: str | None
    workspace: str | None
    service_name: str
    worker_count: int
    request_timeout: float
    apply_chat_template: bool
    limit: int | None
    num_fewshot: int | None
    dry_run: bool


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        _print_dry_run(args)
        return

    if args.controller_url is None:
        client = LocalClient()
        try:
            _run_smoke(client, args)
        finally:
            client.shutdown(wait=True)
        return

    _run_iris_smoke(args)


def _run_smoke(client: Client, args: SmokeArgs) -> None:
    deployment = ModelDeployment(
        model_name=args.model,
        model_path=args.model,
        tokenizer=args.tokenizer,
    )
    launcher = IrisInferenceLauncher(
        client=client,
        config=IrisInferenceLauncherConfig(
            engine_base_url=args.engine_base_url,
            worker_count=args.worker_count,
            request_timeout=args.request_timeout,
            service_name=args.service_name,
        ),
    )
    adapter = LmEvalAdapter.LOCAL_CHAT_COMPLETIONS if args.apply_chat_template else LmEvalAdapter.LOCAL_COMPLETIONS
    with launcher.launch(deployment) as running_model:
        run_lm_eval(
            running_model,
            LmEvalRun(
                tasks=[args.task],
                output_path=args.output_path,
                adapter=adapter,
                apply_chat_template=args.apply_chat_template,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
                batch_size=1,
                extra_model_args={
                    "tokenizer_backend": "huggingface",
                    "tokenized_requests": False,
                    "timeout": int(args.request_timeout),
                    "max_retries": 1,
                },
            ),
        )


def _run_iris_smoke(args: SmokeArgs) -> None:
    from fray.iris_backend import FrayIrisClient
    from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
    from iris.cluster.types import Entrypoint, ResourceSpec

    iris_client = IrisClient.remote(args.controller_url, workspace=args.workspace)
    client = FrayIrisClient.from_iris_client(iris_client)
    parent_job = iris_client.submit(
        entrypoint=Entrypoint.from_callable(_hold_parent_job),
        name=f"{args.service_name}-launcher",
        resources=ResourceSpec(cpu=1, memory="1g", disk="4g"),
    )
    try:
        with iris_ctx_scope(IrisContext(job_id=parent_job.job_id, client=iris_client)):
            _run_smoke(client, args)
    finally:
        iris_client.terminate(parent_job.job_id)
        client.shutdown(wait=True)


def _hold_parent_job() -> None:
    while True:
        time.sleep(3600)


def _parse_args() -> SmokeArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-base-url", required=True, help="OpenAI API root visible from workers, ending in /v1.")
    parser.add_argument("--model", required=True, help="Model id passed through to lm-eval and the engine.")
    parser.add_argument("--task", default="mmlu_sl_verb_5shot", help="lm-eval task to run.")
    parser.add_argument("--output-path", default="/tmp/marin-iris-inference-smoke")
    parser.add_argument("--tokenizer", help="Hugging Face tokenizer id/path for lm-eval.")
    parser.add_argument("--controller-url", help="Iris controller URL. Omit to run through LocalClient.")
    parser.add_argument("--workspace", help="Workspace path for IrisClient.remote().")
    parser.add_argument("--service-name", default="iris-inference-smoke")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    return SmokeArgs(**vars(parsed))


def _print_dry_run(args: SmokeArgs) -> None:
    backend = "FrayIrisClient" if args.controller_url is not None else "LocalClient"
    adapter = "local-chat-completions" if args.apply_chat_template else "local-completions"
    print(f"backend={backend}")
    print(f"engine_base_url={args.engine_base_url}")
    print(f"model={args.model}")
    print(f"task={args.task}")
    print(f"adapter={adapter}")
    print(f"worker_count={args.worker_count}")
    print("engine launch is external to this smoke; workers must reach the engine URL directly")


if __name__ == "__main__":
    main()
