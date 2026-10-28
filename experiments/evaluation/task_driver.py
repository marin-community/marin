# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run published evaluation corpora through the standard TaskSpec driver."""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from marin.evaluation.likelihood import VllmLikelihood
from marin.evaluation.task_driver import EvaluationRunConfig, run_evaluation
from openai import AsyncOpenAI
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from shellbox.backends.local.machine import LocalMachineFactory
from transformers import AutoTokenizer
from verifyit.modes.grade_judge import JudgeConnection


async def execute(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str,
    api_key: str,
    judge_connection: JudgeConnection | None,
) -> None:
    """Resolve the pinned tokenizer and endpoint transports once for the run."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False
    )
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("Likelihood scoring needs a tokenizer BOS or EOS prefix token")
    backends = {
        selection.backend for selection in (config.task_machine, config.verifier_machine) if selection is not None
    }
    if unsupported := backends - {"local"}:
        raise ValueError(f"This entrypoint has no factories for {sorted(unsupported)}; supply them to run_evaluation")
    machines = {"local": LocalMachineFactory()} if "local" in backends else {}
    async with AsyncOpenAI(base_url=endpoint, api_key=api_key, max_retries=0) as client:
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
    arguments = parser.parse_args()
    config = EvaluationRunConfig.model_validate_json(StoragePath(arguments.config).read_text())
    base_url = arguments.endpoint_file.read_text().strip() if arguments.endpoint_file else arguments.endpoint
    judge = (
        JudgeConnection(arguments.judge_endpoint, os.environ[arguments.judge_api_key_env])
        if arguments.judge_endpoint
        else None
    )
    logging.basicConfig(level=logging.INFO)
    # Iris endpoint URLs can contain scoped credentials.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    asyncio.run(
        execute(
            tuple(arguments.dataset),
            arguments.output,
            config,
            base_url,
            os.environ[arguments.api_key_env],
            judge,
        )
    )


if __name__ == "__main__":
    main()
