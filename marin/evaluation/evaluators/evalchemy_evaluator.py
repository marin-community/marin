import logging
import os
import shutil
import traceback
import tempfile
import subprocess
import ray

from typing import ClassVar

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import upload_to_gcs, run_bash_command
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

class EvalchemyEvaluator(VllmTpuEvaluator):
    """
    Minimal Evalchemy evaluator that integrates the Evalchemy framework with Marin.
    **Some notes:**
    1. Evalchemy needs to be installed in editable mode. The only way to do this is to
       to it as part of this script. We will then remove it after the evaluation is done.
       As such we don't add it in _pip_packages.
    2. Running evalchemy in command line mode (e.g., subprocess.check_call) causes
       TPUs to lock and cause the cluster to never run new jobs. This script
       tries to avoid the issue by calling the package directly. We still
       use subprocess.check_call, but only in 'safe' scenarios.
    3.
    """

    CACHE_PATH: str = VllmTpuEvaluator.CACHE_PATH
    LOCAL_RESULTS_PATH: str = os.path.join(CACHE_PATH, "evalchemy_results")

    _pip_packages: ClassVar[list[Dependency]] = [
        *VllmTpuEvaluator.DEFAULT_PIP_PACKAGES,
    ]
    _env_vars: ClassVar[dict[str, str]] = {
        # Human eval tests code from the model which requires permission to run
        "HF_ALLOW_CODE_EVAL": "1",
        "VLLM_USE_V1": "0",
    }

    _temp_dir: str | None = tempfile.mkdtemp()

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        resource_config: ResourceConfig | None = None,
    ) -> None:
        """
        Launches the evaluation run with Ray.
        """
        @ray.remote(
            scheduling_strategy=self._get_scheduling_strategy(resource_config),
            runtime_env=self.get_runtime_env(),
            max_calls=1,
            resources={"TPU": 1},
        )
        
        @remove_tpu_lockfile_on_exit
        def launch(
            model: ModelConfig,
            evals: list[EvalTaskConfig],
            output_path: str,
            max_eval_instances: int | None = None,
        ) -> None:
            self.evaluate(model, evals, output_path, max_eval_instances)

        ray.get(launch.remote(model, evals, output_path, max_eval_instances))

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
            # Clean up hack
            # run_bash_command(["rm", "-rf", "/tmp/*"])      
            
            # Clone the repository to a temporary directory
            logger.info(f"Cloning evalchemy to {self._temp_dir}")
            subprocess.check_call(["git", "clone", "https://github.com/chiheem/evalchemy.git", self._temp_dir])

            # Install evalchemy in editable mode
            logger.info("Installing evalchemy in editable mode...")
            subprocess.check_call(["pip", "install", "-e", self._temp_dir])
            logger.info("Evalchemy installed successfully")

            # Change to the cloned repository directory for proper data file resolution
            original_cwd = os.getcwd()
            os.chdir(self._temp_dir)
            logger.info(f"Changed working directory to: {self._temp_dir}")

            # Set HF_HUB_CACHE to the temporary directory
            os.environ["HF_HUB_CACHE"] = self._temp_dir

            # Create output directory
            os.makedirs(self.LOCAL_RESULTS_PATH, exist_ok=True)

            # Download model
            model_name_or_path = self.download_model(model)
            
            # Prepare model arguments for vllm backend
            # evalchemy's VLLM backend expects 'pretrained' parameter for all models
            model_args = f"pretrained={model_name_or_path}"
            
            # TPU-friendly defaults; merge user-provided engine kwargs if present
            if model.engine_kwargs:
                for key, value in model.engine_kwargs.items():
                    model_args += f",{key}={value}"
            else:
                model_args += ",tensor_parallel_size=1,trust_remote_code=True"

            for eval_task in evals:
                logger.info(f"Start evalchemy:{eval_task.name} ({eval_task.num_fewshot} shot).")

                # Prepare task list
                task_list = [eval_task.name]

                # Build the evalchemy command - try using vllm backend for TPU support
                command = [
                    "python", "-m", "eval.eval",
                    "--model", "vllm", # "vllm" or "hf"
                    "--tasks", ",".join(task_list),
                    "--model_args", model_args,
                    "--batch_size", "16",
                    "--output_path", self.LOCAL_RESULTS_PATH,
                ]

                # Decide whether to pass chat flag
                chat_tasks = {"MTBench", "WildBench", "AlpacaEval", "MixEval", "IFEval", "ZeroEval", "RepoBench"}
                if model.apply_chat_template and eval_task.name in chat_tasks:
                    command.append("--apply_chat_template")

                # Add optional parameters
                if max_eval_instances is not None:
                    command.extend(["--limit", str(max_eval_instances)])
                
                if eval_task.num_fewshot > 0:
                    command.extend(["--num_fewshot", str(eval_task.num_fewshot)])

                # Log the full command for debugging
                logger.info(f"Running evalchemy command: {' '.join(command)}")

                # Run evaluation using evalchemy command line interface
                try:
                    run_bash_command(command)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Evalchemy command failed with exit code {e.returncode}")
                    logger.error(f"Command was: {' '.join(command)}")
                    logger.error(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No stderr available'}")
                    raise

                logger.info(f"Completed evalchemy:{eval_task.name} evaluation")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("Evalchemy failed") from e
        finally:
            # Restore original working directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
                logger.info(f"Restored working directory to: {original_cwd}")

            # Upload to GCS if results are found
            if not os.path.exists(self.LOCAL_RESULTS_PATH) or not any(os.listdir(self.LOCAL_RESULTS_PATH)):
                logger.warning(f"No results found in {self.LOCAL_RESULTS_PATH}")
            else:
                upload_to_gcs(self.LOCAL_RESULTS_PATH, output_path)
                shutil.rmtree(self.LOCAL_RESULTS_PATH)
            # Clean up temporary directory and uninstall evalchemy
            if hasattr(self, '_temp_dir') and self._temp_dir:
                try:
                    # Uninstall evalchemy
                    try:
                        subprocess.check_call(["pip", "uninstall", "-y", "evalchemy"])
                        logger.info("Uninstalled evalchemy package")
                    except Exception as e:
                        logger.warning(f"Failed to uninstall evalchemy: {e}")

                    # Remove temporary directory
                    if os.path.exists(self._temp_dir):
                        shutil.rmtree(self._temp_dir)
                        logger.info(f"Removed temporary directory: {self._temp_dir}")
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
            # Remove HF_HUB_CACHE
            if "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            self.cleanup(model)
