import logging
import os
import shutil
import traceback
import subprocess
from typing import ClassVar

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import upload_to_gcs, run_bash_command

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

    CACHE_PATH: str = VllmTpuEvaluator.CACHE_PATH
    LOCAL_RESULTS_PATH: str = os.path.join(CACHE_PATH, "evalchemy_results")

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
                # Dynamically check which parameters the model supports
                try:
                    from transformers import AutoModelForCausalLM
                    import inspect
                    
                    # Load the model config to inspect supported parameters
                    model_config = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
                    model_class = model_config.__class__
                    init_signature = inspect.signature(model_class.__init__)
                    supported_params = set(init_signature.parameters.keys())
                    
                    for key, value in model.engine_kwargs.items():
                        if key not in supported_params:
                            logger.warning(f"Skipping unsupported parameter '{key}' for model {model_name_or_path}")
                            continue
                        pretrained_args += f",{key}={value}"
                except Exception as e:
                    logger.warning(f"Could not inspect model parameters for {model_name_or_path}: {e}")
                    # Fallback to original behavior
                    for key, value in model.engine_kwargs.items():
                        pretrained_args += f",{key}={value}"

            # Install evalchemy (test)
            run_bash_command(["git", "clone", "https://github.com/mlfoundations/evalchemy.git"])
            os.chdir("evalchemy")
            
            # Surgery to remove "fschat @ file:eval/chat_benchmarks/MTBench" from pyproject.toml
            # We will install the MTBench package separately after installing evalchemy
            # See comments in `Quick Start::Installation` on `https://github.com/mlfoundations/evalchemy`
            with open("pyproject.toml", "r") as f:
                lines = f.readlines()
            with open("pyproject.toml", "w") as f:
                for line in lines:
                    if "fschat @ file:eval/chat_benchmarks/MTBench" not in line:
                        f.write(line)
            logger.info("Removed 'fschat @ file:eval/chat_benchmarks/MTBench' from pyproject.toml")
            
            logger.info("Installing evalchemy.")
            run_bash_command(["uv", "pip", "install", "-e", "."])
            
            logger.info("Installing evalchemy/chat_benchmarks/MTBench.")
            run_bash_command(["uv", "pip", "install", "-e", "eval/chat_benchmarks/MTBench"])

            for eval_task in evals:
                logger.info(f"Start evalchemy:{eval_task.name} ({eval_task.num_fewshot} shot).")
                result_filepath = os.path.join(self.LOCAL_RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")

                # Create the output directory
                output_dir = os.path.dirname(result_filepath)
                os.makedirs(output_dir, exist_ok=True)

                # Build command
                cmd = [
                    "python3",
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
                    self.LOCAL_RESULTS_PATH,
                ]

                if eval_task.num_fewshot > 0:
                    cmd.extend(["--num_fewshot", str(eval_task.num_fewshot)])
                if max_eval_instances is not None:
                    cmd.extend(["--limit", str(max_eval_instances)])
                if model.apply_chat_template:
                    cmd.append("--apply_chat_template")

                # Add annotator model (default to auto for cost efficiency)
                cmd.extend(["--annotator_model", "auto"])

                # Run evaluation
                try:
                    run_bash_command(cmd)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Evalchemy failed with exit code {e.returncode}: {e.stderr}")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("Evalchemy failed") from e
        finally:
            if not os.path.exists(self.LOCAL_RESULTS_PATH):
                logger.warning(f"No results found in {self.LOCAL_RESULTS_PATH}")
            else:
                upload_to_gcs(self.LOCAL_RESULTS_PATH, output_path)
                shutil.rmtree(self.LOCAL_RESULTS_PATH)
            self.cleanup(model)
