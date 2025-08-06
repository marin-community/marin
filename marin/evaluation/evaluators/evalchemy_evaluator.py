import logging
import os
import shutil
import traceback
import tempfile
import subprocess
import argparse
import ray

from lm_eval.api.registry import get_model
from lm_eval.utils import sanitize_model_name

from typing import ClassVar

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator
from marin.evaluation.utils import upload_to_gcs

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
        # Dependency(name="lm-eval"),
        Dependency(name="evalchemy@git+https://github.com/chiheem/evalchemy.git"),
    ]
    _env_vars: ClassVar[dict[str, str]] = {
        # Human eval tests code from the model which requires permission to run
        "HF_ALLOW_CODE_EVAL": "1",
    }

    _temp_dir: str | None = tempfile.mkdtemp()

    def _initialize_model(self, model, model_args=None, device=None, batch_size=None):
        """Initialize a language model with proper chat template handling."""
        if isinstance(model, str):
            if model_args is None:
                model_args = ""
            config = {"device": device}
            if "batch_size" not in model_args and batch_size is not None:
                model_args += f",batch_size={batch_size}"
            lm = get_model(model).create_from_arg_string(model_args, config)
        else:
            lm = model
        lm.model_identifier = sanitize_model_name(f"model_{model}_model_args_{model_args}")

        # Add or override apply_chat_template method to handle models without chat template
        # Store the original method before overriding
        original_apply_chat_template = getattr(lm, 'apply_chat_template', None)

        def apply_chat_template(messages):
            """Fallback chat template for models without built-in chat template."""
            try:
                # Try the original method first
                if original_apply_chat_template is not None:
                    return original_apply_chat_template(messages)
                else:
                    # No original method, use fallback
                    raise AttributeError("No original apply_chat_template method")
            except (ValueError, AttributeError):
                # Fallback to simple concatenation
                result = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        result += f"System: {content}\n"
                    elif role == "user":
                        result += f"User: {content}\n"
                    elif role == "assistant":
                        result += f"Assistant: {content}\n"
                return result.strip()

        lm.apply_chat_template = apply_chat_template

        return lm

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
            # num_gpus=1,
            memory=64*1024*1024*1024, # 64GB
        )
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

            
            # Clone the repository to a temporary directory
            logger.info(f"Cloning evalchemy to {self._temp_dir}")
            subprocess.check_call(["git", "clone", "https://github.com/chiheem/evalchemy.git", self._temp_dir])

            # Install evalchemy in editable mode
            logger.info("Installing evalchemy in editable mode...")
            subprocess.check_call(["pip", "install", "-e", self._temp_dir])
            logger.info("Evalchemy installed successfully")

            # Verify installation and add to path if needed
            import sys
            sys.path.insert(0, self._temp_dir)
            logger.info(f"Added {self._temp_dir} to Python path")

            # Import evalchemy functions directly from the eval directory
            from eval.eval import evaluate
            from eval.task import TaskManager as InstructTaskManager
            from lm_eval.tasks import TaskManager as PretrainTaskManager

            # Change to the cloned repository directory for proper data file resolution
            original_cwd = os.getcwd()
            os.chdir(self._temp_dir)
            logger.info(f"Changed working directory to: {self._temp_dir}")

            # Set HF_HUB_CACHE to the temporary directory
            os.environ["HF_HUB_CACHE"] = self._temp_dir

            # Create output directory
            os.makedirs(self.LOCAL_RESULTS_PATH, exist_ok=True)

            for eval_task in evals:
                logger.info(f"Start evalchemy:{eval_task.name} ({eval_task.num_fewshot} shot).")

                # Prepare task list
                task_list = [eval_task.name]
                batch_sizes_list = ["auto"]

                # Initialize model with CPU device
                lm = self._initialize_model(
                    model="hf",
                    model_args=pretrained_args,
                    device="gpu", # `gpu` or `cpu`
                    batch_size="auto"
                )

                # Initialize task managers
                task_manager = InstructTaskManager(verbosity="INFO")
                pretrain_task_manager = PretrainTaskManager(verbosity="INFO")

                # Create args object for evalchemy
                args = argparse.Namespace()
                args.model = "hf"
                args.model_args = pretrained_args
                args.device = "cpu"
                args.batch_size = "auto"
                args.limit = max_eval_instances
                args.num_fewshot = eval_task.num_fewshot
                args.verbosity = "INFO"

                # Run evaluation using evalchemy framework
                results = evaluate(
                    lm=lm,
                    task_manager=task_manager,
                    pretrain_task_manager=pretrain_task_manager,
                    task_list=task_list,
                    batch_sizes_list=batch_sizes_list,
                    verbosity="INFO",
                    args=args,
                    limit=max_eval_instances,
                    num_fewshot=eval_task.num_fewshot,
                    device="cpu",
                )

                # Save results
                result_filepath = os.path.join(self.LOCAL_RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")
                import json
                with open(f"{result_filepath}.json", "w") as f:
                    json.dump(results, f, indent=2)

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
                    # Remove from sys.path if added
                    import sys
                    if self._temp_dir in sys.path:
                        sys.path.remove(self._temp_dir)
                        logger.info(f"Removed {self._temp_dir} from Python path")

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
