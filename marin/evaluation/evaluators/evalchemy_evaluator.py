import logging
import os
import shutil
import subprocess
import traceback
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import is_remote_path, upload_to_gcs

logger = logging.getLogger(__name__)


class EvalchemyEvaluator(VllmTpuEvaluator):
    """
    Minimal Evalchemy evaluator that integrates the Evalchemy framework with Marin.

    Evalchemy (https://github.com/mlfoundations/evalchemy) is a comprehensive evaluation
    framework that supports multiple benchmarks including:

    - HumanEval: Code generation evaluation
    - MMLU: Multiple choice question answering
    - MTBench: Multi-turn dialogue evaluation
    - WildBench: Wild benchmark evaluation
    - RepoBench: Repository-level code generation
    - MixEval: Mixed evaluation tasks
    - AlpacaEval: Instruction following evaluation
    - IFEval: Instruction following evaluation
    - ZeroEval: Zero-shot evaluation
    - MBPP: Python programming evaluation
    - ARC: AI2 Reasoning Challenge
    - DROP: Reading comprehension evaluation

    This evaluator runs Evalchemy as a subprocess and handles:
    - Model downloading and configuration
    - Task execution with few-shot/zero-shot settings
    - Result upload to Google Cloud Storage
    - Resource cleanup

    Usage:
        evaluator = EvalchemyEvaluator()
        evaluator.evaluate(
            model=ModelConfig(name="your-model"),
            evals=[EvalTaskConfig(name="HumanEval", num_fewshot=0)],
            output_path="gs://your-bucket/results"
        )
    """

    CACHE_PATH: str = "/tmp/evalchemy"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "evalchemy_results")

    _pip_packages: ClassVar[list[Dependency]] = [
        *VllmTpuEvaluator.DEFAULT_PIP_PACKAGES,
        Dependency(name="evalchemy@git+https://github.com/mlfoundations/evalchemy.git"),
    ]
    _env_vars: ClassVar[dict[str, str]] = {
        # Human eval tests code from the model which requires permission to run
        "HF_ALLOW_CODE_EVAL": "1",
    }

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        """
        Runs Evalchemy evaluations on the specified model and tasks.

        Args:
            model: Model configuration including name, path, and engine kwargs
            evals: List of evaluation tasks to run (e.g., HumanEval, MMLU)
            output_path: Path to save results (local or GCS)
            max_eval_instances: Maximum number of instances to evaluate per task
        """
        try:
            model_name_or_path = self.download_model(model)

            # Prepare model arguments
            pretrained_args = f"pretrained={model_name_or_path}"
            if model.engine_kwargs:
                for key, value in model.engine_kwargs.items():
                    pretrained_args += f",{key}={value}"

            for eval_task in evals:
                result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

                # Create the output directory
                output_dir = os.path.dirname(result_filepath)
                os.makedirs(output_dir, exist_ok=True)

                # Build command
                cmd = [
                    "python",
                    "-m",
                    "eval.eval",
                    "--model",
                    "hf",
                    "--tasks",
                    eval_task.name,
                    "--model_args",
                    pretrained_args,
                    "--batch_size",
                    "auto",
                    "--output_path",
                    self.RESULTS_PATH,
                ]

                if eval_task.num_fewshot > 0:
                    cmd.extend(["--num_fewshot", str(eval_task.num_fewshot)])
                if max_eval_instances is not None:
                    cmd.extend(["--limit", str(max_eval_instances)])
                if model.apply_chat_template:
                    cmd.append("--apply_chat_template")

                # Add annotator model (default to auto for cost efficiency)
                cmd.append(["--annotator_model", "auto"])

                # Run evaluation
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Evalchemy failed: {result.stderr}")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("Evalchemy failed") from e
        finally:
            if is_remote_path(output_path):
                upload_to_gcs(self.RESULTS_PATH, output_path)
            self.cleanup(model)
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
