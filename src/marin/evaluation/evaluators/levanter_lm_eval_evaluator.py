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


class LevanterLmEvalEvaluator(LevanterTpuEvaluator):
    """For `Evaluator`s that runs inference with Levanter's Lm Eval Harness on TPUs."""

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval", "tpu"],
            pip_packages=["statsmodels==0.14.4"],
            env_vars={
                "TOKENIZERS_PARALLELISM": "false",
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            },
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): The tags to add to the wandb run.
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
            print("after wandb log")
            # NOTE(chris): Before, the batch size was 16, but this is too large for the 8B model.
            # In the future, we should make this user-configurable.
            trainer_config = TrainerConfig(
                tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                per_device_eval_parallelism=1,
                ray=RayConfig(auto_start_cluster=False),
            )
            print("after trainer?")

            model_config = HFCheckpointConverter.from_hf(model_name_or_path).LevConfigClass()

            # convert to the config that Levanter's eval_harness expects
            tasks = convert_to_levanter_task_config(evals)
            logger.info(f"Tasks: {tasks}")
            print("converted tasks")

            model_path = os.path.join(LevanterTpuEvaluator.CACHE_PATH, model.path)

            logger.info(f"Model path: {model_path}")
            logger.info(f"Levanter Cache Path: {LevanterTpuEvaluator.CACHE_PATH}")
            logger.info(f"Model name: {model.name}")
            logger.info(f"model_name_or_path: {model_name_or_path}")

            print("starting harness")
            eval_config = eval_harness.EvalHarnessMainConfig(
                eval_harness=eval_harness.LmEvalHarnessConfig(
                    task_spec=tasks,
                    max_examples=max_eval_instances,
                    log_samples=False,
                    max_eval_length=4096,
                    apply_chat_template=model.apply_chat_template,
                ),
                tokenizer=model_path,  # levanter picks up the tokenizer from the model path
                checkpoint_path=model_path,
                checkpoint_is_hf=True,
                trainer=trainer_config,
                model=model_config,
            )

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
