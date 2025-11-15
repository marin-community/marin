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

import logging
import tempfile
import traceback

from fray.cluster.ray.deps import build_runtime_env_for_packages

from marin.evaluation.evaluation_config import EvalTaskConfig, ModelConfig
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.utils import upload_to_gcs

logger = logging.getLogger(__name__)

# Default annotator for AlpacaEval using GPT-4 Turbo
# This has high agreement with human annotations and is cost-efficient
# Source: https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#quick-start
DEFAULT_ANNOTATOR_CONFIG: str = "weighted_alpaca_eval_gpt4_turbo"


class AlpacaEvaluator(Evaluator):
    """Evaluator that runs AlpacaEval: https://github.com/tatsu-lab/alpaca_eval"""

    # AlpacaEval has 805 examples: https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json
    # so if the number of instances is not specified, we will run on all of them.
    DEFAULT_MAX_INSTANCES: int = 805

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(pip_packages=["alpaca-eval", "datasets"])

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        openai_base_url: str,
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Runs AlpacaEval on the specified model.

        This evaluator runs in two phases:
        1. Generate model outputs by calling the vLLM server at openai_base_url
        2. Evaluate outputs using OpenAI's GPT-4 judge (requires OPENAI_API_KEY)

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[str]): Does nothing. We just run on the default eval set.
            openai_base_url (str): Base URL for the vLLM server to generate model outputs
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): Optional tags to add to the wandb run (unused).
        """
        results_tmp = tempfile.TemporaryDirectory()
        results_path = results_tmp.name

        try:
            import datasets
            from alpaca_eval import decoders, evaluate

            # Load the AlpacaEval dataset
            logger.info("Loading AlpacaEval dataset")
            eval_set = datasets.load_dataset(
                "tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True
            )

            # Limit instances if requested
            max_eval_instances = max_eval_instances or self.DEFAULT_MAX_INSTANCES
            if max_eval_instances < len(eval_set):
                eval_set = eval_set.select(range(max_eval_instances))
                logger.info(f"Limited to {max_eval_instances} instances")

            # Extract prompts from the dataset
            prompts = [example["instruction"] for example in eval_set]

            # Phase 1: Generate model outputs using vLLM server
            logger.info(f"Generating outputs for {len(prompts)} prompts using {openai_base_url}")
            generation_params = model.generation_params or {}

            openai_completions = decoders.get_fn_completions("openai_completions")

            # We set requires_chatml=False because vLLM handles chat formatting internally.
            # For max_tokens, default to 2048 for normal models, but cap at max_model_len - 256
            # to leave room for the input prompt
            max_tokens = generation_params.get("max_tokens", 2048)
            max_model_len = model.engine_kwargs.get("max_model_len")
            if max_model_len:
                # Reserve at least 256 tokens for input, or half the context for very small models
                reserved_for_input = min(256, max_model_len // 2)
                max_tokens = min(max_tokens, max_model_len - reserved_for_input)

            completions_result = openai_completions(
                prompts=prompts,
                model_name=model.name,
                max_tokens=max_tokens,
                temperature=generation_params.get("temperature", 0.7),
                top_p=generation_params.get("top_p", 1.0),
                openai_api_base=openai_base_url,
                requires_chatml=False,
            )

            # Prepare outputs in AlpacaEval format
            outputs = [
                {
                    "instruction": example["instruction"],
                    "output": completion,
                    "generator": model.name,
                }
                for example, completion in zip(eval_set, completions_result["completions"], strict=True)
            ]

            logger.info(f"Generated {len(outputs)} outputs")

            # Phase 2: Evaluate using GPT-4 judge
            logger.info(f"Evaluating outputs with {DEFAULT_ANNOTATOR_CONFIG}")
            evaluate(
                model_outputs=outputs,
                annotators_config=DEFAULT_ANNOTATOR_CONFIG,
                output_path=results_path,
            )

            upload_to_gcs(results_path, output_path)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("AlpacaEval failed. Please check the logs for more information.") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(filename)s:%(lineno)d:%(levelname)s:%(message)s")
    from pathlib import Path

    from fray.cluster import LocalCluster
    from fray.cluster.base import CpuConfig, Entrypoint, JobRequest, ResourceConfig, create_environment
    from fray.queue.file import FileQueue

    from marin.evaluation.backends.inference_pool import InferencePool
    from marin.evaluation.evaluation_config import InferencePoolConfig

    pool_config = InferencePoolConfig(
        resource_config=ResourceConfig(cpu=1, ram="4g", device=CpuConfig(), replicas=1),
        model_config=ModelConfig(
            name="timinar/baby-llama-58m",
            path="timinar/baby-llama-58m",
            engine_kwargs={
                "max_model_len": 128,
            },
            device="auto",
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir, LocalCluster() as cluster:
        from typing import Any

        request_queue = FileQueue[dict[str, Any]](Path(tmp_dir) / "requests")
        response_queue = FileQueue[dict[str, Any]](Path(tmp_dir) / "responses")

        with InferencePool(
            pool_config, cluster=cluster, request_queue=request_queue, response_queue=response_queue
        ) as pool:
            evaluator = AlpacaEvaluator()
            job_request = JobRequest(
                name="alpaca",
                entrypoint=Entrypoint(
                    callable=evaluator.evaluate,
                    function_args={
                        "model": pool_config.model_config,
                        "evals": [],
                        "openai_base_url": "http://localhost:9000/v1",
                        "output_path": "/tmp/alpaca/test",
                        "max_eval_instances": 10,
                    },
                ),
                resources=ResourceConfig(cpu=1, ram="4g", device=CpuConfig(), replicas=1),
                environment=create_environment(
                    pip_packages=["alpaca-eval", "datasets"],
                    extras=["eval"],
                ),
            )
            job_id = cluster.launch(job_request)
            logger.info("Started AlpacaEval task with job id %s", job_id)
            cluster.wait(job_id)
