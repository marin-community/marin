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

import json
import logging
import os
import shutil
import contextlib

import fsspec
import jmp
import levanter
import levanter.eval_harness as eval_harness
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.evals.task_configs import convert_to_levanter_task_config
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator
from marin.run.ray_deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


"""
Note: to use this class, you will need to pass an additional flag `--extra eval` to ray_run.
This will ensure that the correct dependencies are installed at launch time.

e.g.,  
`python src/marin/run/ray_run.py --no_wait --extra eval -- python script.py`
"""

class LevanterLmEvalEvaluator(LevanterTpuEvaluator):
    """For `Evaluator`s that runs inference with Levanter's Lm Eval Harness on TPUs."""

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval", "tpu"],
            pip_packages=[
                "antlr4-python3-runtime==4.11", # Required by lm-eval[math]
                "haliax>=1.4.dev348",
                "immutabledict",
                "jax[tpu]",
                "langdetect",
                "lm-eval[math]@git+https://github.com/stanford-crfm/lm-evaluation-harness.git@18b376504e98aadcb985f9235c3e58ab2bf4c5bc",
                "math-verify", # Required by lm-eval[math]
                "ray==2.45",
                "statsmodels==0.14.4",
                "sympy>=1.12", # Required by lm-eval[math]
            ],
            env_vars={
                "TOKENIZERS_PARALLELISM": "false",
            },
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        show_logs_from_ray: bool = True,
        max_gen_toks: int | None = None,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): The tags to add to the wandb run.
            show_logs_from_ray (bool): Whether to show the logs from the ray run.
            max_gen_toks (int | None): Maximum number of tokens to generate during evaluation.
        """
        # Eval Harness code: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/eval_harness.py
        # Run the harness with the model and the specified evals

        try:
            # Download the model from GCS or HuggingFace
            print("before download")
            model_name_or_path: str = self.download_model(model)
            print(f"in lm_eval: {model_name_or_path}")
            name = model.name + "_lmeval_" + "-".join([eval_task.name for eval_task in evals])
            print(name)
            logger.info(f"WandB Run Name: {name}")
            logger.info(f"Running eval harness on model: {model_name_or_path}")

            # Set environment variable to allow code evaluation
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"

            print("after wandb log")
            # NOTE(chris): Before, the batch size was 16, but this is too large for the 8B model.
            # In the future, we should make this user-configurable.
            #
            # NOTE (chiheem 2025-09-30): We should make the users pass TrainerConfig so that they 
            # are forced to customize the TrainerConfig according to their device, device, and model.
            # Current config is customized for our 8B model on v6e-8.
            # 
            trainer_config = TrainerConfig(
                tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                per_device_eval_parallelism=1,
                ray=RayConfig(auto_start_cluster=False),
                model_axis_size=4,
                tensor_parallel_axes=["mlp", "heads", "kv_head", "vocab"],
            )
            print("after trainer?")

            model_config = HFCheckpointConverter.from_hf(model_name_or_path).LevConfigClass()

            # convert to the config that Levanter's eval_harness expects
            tasks = convert_to_levanter_task_config(evals)
            logger.info(f"Tasks: {tasks}")
            print("converted tasks")

            model_path = model_name_or_path

            logger.info(f"Model path: {model_path}")
            logger.info(f"Levanter Cache Path: {LevanterTpuEvaluator.CACHE_PATH}")
            logger.info(f"Model name: {model.name}")
            logger.info(f"model_name_or_path: {model_name_or_path}")

            # Create the eval harness config with max_gen_toks if specified
            harness_config_kwargs = {
                "task_spec": tasks,
                "max_examples": max_eval_instances,
                "log_samples": False,
            }

            # Add max_gen_toks to generation_kwargs if specified
            if max_gen_toks is not None:
                harness_config_kwargs["generation_kwargs"] = {"max_gen_toks": max_gen_toks}
                logger.info(f"Setting max_gen_toks={max_gen_toks} in LmEvalHarnessConfig generation_kwargs")

            print("starting harness")
            eval_config = eval_harness.EvalHarnessMainConfig(
                eval_harness=eval_harness.LmEvalHarnessConfig(**harness_config_kwargs),
                tokenizer=model_path,  # levanter picks up the tokenizer from the model path
                checkpoint_path=model_path,
                checkpoint_is_hf=True,
                trainer=trainer_config,
                model=model_config,
            )

            # Ensure unsafe tasks (e.g., humaneval) are allowed in lm-eval by
            # defaulting confirm_run_unsafe_code=True in its API.
            try:
                from lm_eval import evaluator as _lm_eval_evaluator  # type: ignore

                _original_evaluate = _lm_eval_evaluator.evaluate

                def _evaluate_with_unsafe(*args, **kwargs):
                    kwargs.setdefault("confirm_run_unsafe_code", True)
                    return _original_evaluate(*args, **kwargs)

                _lm_eval_evaluator.evaluate = _evaluate_with_unsafe  # type: ignore
            except Exception:
                pass

            # NOTE(chiheem 2025-10-01): We suppress the stdout/stderr from the harness (per-sample prints)
            # so that we don't overflow the buffer. The issue occurs at the levanter level.
            if not show_logs_from_ray:
                with contextlib.ExitStack() as stack:
                    devnull_out = stack.enter_context(open(os.devnull, "w"))
                    devnull_err = stack.enter_context(open(os.devnull, "w"))
                    stack.enter_context(contextlib.redirect_stdout(devnull_out))
                    stack.enter_context(contextlib.redirect_stderr(devnull_err))
                    results = eval_harness.run_eval_harness_main(eval_config)
            else:
                results = eval_harness.run_eval_harness_main(eval_config)
            print("finished harness")

            try:
                # add a results.json to output path
                output_path = os.path.join(output_path, "results.json")

                logger.info(f"Uploading results to GCS: {output_path}")

                # write output JSON directly to output_path on GCS
                fs = fsspec.filesystem("gcs")
                with fs.open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

                levanter.tracker.current_tracker().finish()
                logger.info("Upload completed successfully.")

            except Exception as upload_error:
                logger.info(f"Failed to upload results to GCS: {upload_error}")

        except Exception as e:
            logger.error(f"Error running eval harness: {e}")
            raise e

        finally:
            # Clean up resources
            self.cleanup(model)

            if os.path.exists(LevanterTpuEvaluator.CACHE_PATH) and "gcsfuse" not in LevanterTpuEvaluator.CACHE_PATH:
                shutil.rmtree(LevanterTpuEvaluator.CACHE_PATH)
