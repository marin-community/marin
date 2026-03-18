# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate completions from a proposal model, rerank by a scoring model's log-likelihood.

Launches two vLLM servers (proposal + scoring), generates K completions per prompt,
picks the best by scoring model log-likelihood, and saves results.
Compatible with Marin's executor framework.
"""

import atexit
import json
import logging
from dataclasses import dataclass

import fsspec
import openai

from experiments.rerank_decode.utils import rerank
from experiments.rerank_decode.serve import launch_vllm_servers, shutdown_servers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankDecodeConfig:
    proposal_model_path: str
    scoring_model_path: str
    output_path: str
    prompts_path: str  # JSONL file, each line has a "prompt" field

    num_samples: int = 16
    max_tokens: int = 2048
    chunk_size: int = 256
    temperature: float = 1.0

    proposal_gpus: tuple[int, ...] = (0,)
    scoring_gpus: tuple[int, ...] = (1,)
    proposal_port: int = 8000
    scoring_port: int = 8001


def run_rerank_decode(config: RerankDecodeConfig):
    logging.basicConfig(level=logging.INFO)

    # Load prompts
    prompts = []
    with fsspec.open(config.prompts_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])

    logger.info("Loaded %d prompts from %s", len(prompts), config.prompts_path)

    # Launch servers
    proc_a, proc_b = launch_vllm_servers(
        proposal_model=config.proposal_model_path,
        scoring_model=config.scoring_model_path,
        proposal_gpus=list(config.proposal_gpus),
        scoring_gpus=list(config.scoring_gpus),
        proposal_port=config.proposal_port,
        scoring_port=config.scoring_port,
    )
    atexit.register(shutdown_servers, proc_a, proc_b)

    try:
        proposal_client = openai.Client(
            base_url=f"http://localhost:{config.proposal_port}/v1", api_key="none"
        )
        scoring_client = openai.Client(
            base_url=f"http://localhost:{config.scoring_port}/v1", api_key="none"
        )

        results = []
        for i, prompt in enumerate(prompts):
            logger.info("Reranking prompt %d/%d", i + 1, len(prompts))
            result = rerank(
                proposal_client=proposal_client,
                scoring_client=scoring_client,
                proposal_model=config.proposal_model_path,
                scoring_model=config.scoring_model_path,
                prompt=prompt,
                num_samples=config.num_samples,
                max_tokens=config.max_tokens,
                chunk_size=config.chunk_size,
                temperature=config.temperature,
            )
            results.append(result)

        output = [
            {"prompt": prompt, "generation": generation}
            for prompt, generation in zip(prompts, results)
        ]

        with fsspec.open(f"{config.output_path}/results.json", "wt") as f:
            json.dump(output, f, indent=2)

        logger.info("Saved %d results to %s/results.json", len(results), config.output_path)

    finally:
        shutdown_servers(proc_a, proc_b)


if __name__ == "__main__":
    from marin.execution.executor import executor_main, this_output_path

    step_config = RerankDecodeConfig(
        proposal_model_path="meta-llama/Llama-3.2-1B",
        scoring_model_path="meta-llama/Llama-3.2-1B",
        output_path=this_output_path(),
        prompts_path="prompts.jsonl",
    )

    from marin.execution.executor import ExecutorStep

    rerank_decode_step = ExecutorStep(
        name="rerank_decode/example",
        fn=run_rerank_decode,
        config=step_config,
    )

    executor_main(steps=[rerank_decode_step], description="Generate-then-rerank decoding.")
