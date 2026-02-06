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

"""VLM evaluation using Levanter on TPU."""

import dataclasses
import json
import logging
import os
import shutil

import fsspec
import jmp
import levanter
from levanter.distributed import RayConfig
from levanter.main.eval_vlm import EvalVLMConfig, main as eval_vlm_main
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.vlm_eval_harness import VLMEvalHarnessConfig

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.levanter_tpu_evaluator import LevanterTpuEvaluator
from fray.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


class LevanterVLMEvaluator(LevanterTpuEvaluator):
    """VLM evaluation using Levanter's VLM eval harness on TPUs.

    This evaluator runs vision-language model benchmarks (MMMU, ChartQA, etc.)
    using Levanter's VLM evaluation infrastructure on TPU devices.
    """

    def get_runtime_env(self) -> dict:
        """Returns the runtime environment for Ray execution."""
        return build_runtime_env_for_packages(
            extra=["eval", "tpu"],
            pip_packages=["lmms-eval"],
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
        processor_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        """Run VLM evaluation on the specified model.

        Args:
            model: The model configuration containing path and name
            evals: List of evaluation tasks (e.g., mmmu_val_Art, chartqa)
            output_path: Path to save evaluation results
            max_eval_instances: Maximum number of examples per task
            wandb_tags: Optional tags for wandb logging
            processor_path: Path to processor (HF hub ID or GCS path). If None, uses default.
            tokenizer_path: Path to tokenizer (HF hub ID or GCS path). If None, uses default.
        """
        try:
            model_path = self.download_model_if_necessary(model)
            name = model.name + "_vlmeval_" + "-".join([e.name for e in evals])

            logger.info(f"WandB Run Name: {name}")
            logger.info(f"Running VLM eval on model: {model_path}")
            logger.info(f"Processor path: {processor_path or '(default GCS path)'}")
            logger.info(f"Tokenizer path: {tokenizer_path or '(default GCS path)'}")

            trainer_config = TrainerConfig(
                tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                per_device_eval_parallelism=1,
                ray=RayConfig(auto_start_cluster=False),
            )

            # Convert task names to VLM task spec
            task_spec = [e.name for e in evals]
            logger.info(f"VLM Tasks: {task_spec}")

            # Build eval config with optional processor/tokenizer paths
            # If not specified, EvalVLMConfig uses default GCS paths
            eval_config = EvalVLMConfig(
                hf_checkpoint=model_path,
                processor_path=processor_path,  # None uses default GCS path
                tokenizer_path=tokenizer_path,  # None uses default GCS path
                trainer=trainer_config,
                eval_harness=VLMEvalHarnessConfig(
                    task_spec=task_spec,
                    max_examples=max_eval_instances,
                ),
            )

            # Run VLM evaluation
            logger.info("Starting VLM evaluation...")
            results = eval_vlm_main(eval_config)
            logger.info("VLM evaluation completed.")

            # Save results to output path
            try:
                output_file = os.path.join(output_path, "results.json")
                logger.info(f"Uploading results to: {output_file}")

                fs = fsspec.filesystem("gcs")
                with fs.open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=_json_default)

                # Save sample outputs separately if available
                if results and "sample_outputs" in results:
                    samples_file = os.path.join(output_path, "sample_outputs.json")
                    logger.info(f"Uploading sample outputs to: {samples_file}")
                    with fs.open(samples_file, "w") as f:
                        json.dump(results["sample_outputs"], f, indent=2, default=_json_default)

                levanter.tracker.current_tracker().finish()
                logger.info("Results upload completed successfully.")

            except Exception as upload_error:
                logger.error(f"Failed to upload results: {upload_error}")

        except Exception as e:
            logger.error(f"Error running VLM eval: {e}")
            raise

        finally:
            # Clean up resources
            self.cleanup(model)

            if os.path.exists(LevanterTpuEvaluator.CACHE_PATH) and "gcsfuse" not in LevanterTpuEvaluator.CACHE_PATH:
                shutil.rmtree(LevanterTpuEvaluator.CACHE_PATH)


def _json_default(value):
    """Provide best-effort JSON serialization for evaluation results."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    if isinstance(value, set):
        return list(value)

    # Handle pandas DataFrames/Series before generic to_dict check
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            records = value.to_dict(orient='records')
            return records[0] if len(records) == 1 else records
        if isinstance(value, pd.Series):
            return value.to_dict()
    except ImportError:
        pass

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass

    # Handle numpy scalars and arrays
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()

    return repr(value)
