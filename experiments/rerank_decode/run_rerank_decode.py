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

Launches a vLLM proposal server, scores candidates with either VLLMLogprobScorer
or KVCacheScorer, and saves results. Compatible with Marin's executor framework.
"""

import atexit
import json
import logging
import time
from dataclasses import dataclass

import fsspec
import openai
from transformers import AutoTokenizer

from experiments.rerank_decode.scorer import KVCacheScorer, VLLMLogprobScorer
from experiments.rerank_decode.serve import launch_vllm_server, shutdown_servers, wait_for_server
from experiments.rerank_decode.utils import rerank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankDecodeConfig:
    proposal_model_path: str
    scoring_model_path: str
    output_path: str
    prompts_path: str  # JSONL file, each line has a "prompt" field

    scorer_type: str = "kv_cache"  # "kv_cache" or "vllm"
    num_samples: int = 16
    max_tokens: int = 2048
    chunk_size: int = 8
    temperature: float = 1.0

    proposal_gpus: tuple[int, ...] = (1,)
    scoring_gpus: tuple[int, ...] = (0,)
    proposal_port: int = 8000
    scoring_port: int = 8001


def run_rerank_decode(config: RerankDecodeConfig):
    logging.basicConfig(level=logging.INFO)

    eos_token = AutoTokenizer.from_pretrained(config.scoring_model_path).eos_token

    # Load prompts
    prompts = []
    with fsspec.open(config.prompts_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])

    logger.info("Loaded %d prompts from %s", len(prompts), config.prompts_path)

    # Launch proposal server
    procs = []
    proposal_proc = launch_vllm_server(
        model=config.proposal_model_path,
        port=config.proposal_port,
        gpu_ids=list(config.proposal_gpus),
    )
    procs.append(proposal_proc)

    # Launch scoring server only for vllm scorer
    if config.scorer_type == "vllm":
        scoring_proc = launch_vllm_server(
            model=config.scoring_model_path,
            port=config.scoring_port,
            gpu_ids=list(config.scoring_gpus),
        )
        procs.append(scoring_proc)

    atexit.register(shutdown_servers, *procs)

    try:
        wait_for_server(config.proposal_port)
        proposal_client = openai.Client(
            base_url=f"http://localhost:{config.proposal_port}/v1", api_key="none"
        )

        if config.scorer_type == "vllm":
            wait_for_server(config.scoring_port)
            scoring_client = openai.Client(
                base_url=f"http://localhost:{config.scoring_port}/v1", api_key="none"
            )
            scorer = VLLMLogprobScorer(client=scoring_client, model=config.scoring_model_path)
        elif config.scorer_type == "kv_cache":
            scorer = KVCacheScorer(model_name=config.scoring_model_path)
        else:
            raise ValueError(f"Unknown scorer_type: {config.scorer_type}")

        results = []
        timings = []
        for i, prompt in enumerate(prompts):
            start = time.monotonic()
            result = rerank(
                proposal_client=proposal_client,
                proposal_model=config.proposal_model_path,
                scorer=scorer,
                prompt=prompt,
                num_samples=config.num_samples,
                max_tokens=config.max_tokens,
                chunk_size=config.chunk_size,
                temperature=config.temperature,
                eos_token=eos_token,
            )
            elapsed = time.monotonic() - start
            results.append(result)
            timings.append(elapsed)
            logger.info("Prompt %d/%d: %.2fs, %d chars", i + 1, len(prompts), elapsed, len(result))

        output = [
            {"prompt": prompt, "generation": generation}
            for prompt, generation in zip(prompts, results)
        ]

        with fsspec.open(f"{config.output_path}/results.json", "wt") as f:
            json.dump(output, f, indent=2)

        with fsspec.open(f"{config.output_path}/timings.json", "wt") as f:
            json.dump(timings, f, indent=2)

        logger.info("Saved %d results to %s/results.json", len(results), config.output_path)

    finally:
        shutdown_servers(*procs)


if __name__ == "__main__":
    import argparse
    import sys

    from marin.execution.executor import ExecutorStep, executor_main, this_output_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["vllm", "kv_cache"], default="kv_cache")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=8)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    step_config = RerankDecodeConfig(
        proposal_model_path="meta-llama/Llama-3.2-1B",
        scoring_model_path="meta-llama/Llama-3.2-1B",
        output_path=this_output_path(),
        prompts_path="prompts.jsonl",
        scorer_type=args.scorer,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
    )

    rerank_decode_step = ExecutorStep(
        name="rerank_decode/example",
        fn=run_rerank_decode,
        config=step_config,
    )

    executor_main(steps=[rerank_decode_step], description="Generate-then-rerank decoding.")
