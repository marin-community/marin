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
import os
import shutil
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager

from fray.cluster import ResourceConfig
from fray.cluster.ray.deps import build_runtime_env_for_packages

import fsspec

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig, launch_evaluate_with_ray
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, VllmEnvironment, resolve_vllm_mode

logger = logging.getLogger(__name__)


# TODO: Multiple choice tasks currently don't work on TPUs: https://github.com/vllm-project/vllm/issues/8499
class LMEvaluationHarnessEvaluator(Evaluator):
    """
    Evaluator that runs lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
    """

    CACHE_PATH: str = "/tmp/lm-eval"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "eleuther_results")
    TOKENIZER_FILENAMES: tuple[str, ...] = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "config.json",
    )

    @classmethod
    @contextmanager
    def _stage_remote_tokenizer_dir(cls, remote_dir: str) -> Iterator[str | None]:
        # context manager so this deletes even with ray's process pooling
        with tempfile.TemporaryDirectory(prefix="marin-tokenizer-") as local_dir:
            copied_any = False
            for filename in cls.TOKENIZER_FILENAMES:
                remote_path = f"{remote_dir.rstrip('/')}/{filename}"
                if not is_remote_path(remote_path):
                    continue
                fs, fs_path = fsspec.core.url_to_fs(remote_path)
                if not fs.exists(fs_path):
                    continue
                local_path = os.path.join(local_dir, filename)
                with fsspec.open(remote_path, "rb") as src:
                    data = src.read()
                with open(local_path, "wb") as dst:
                    dst.write(data)
                copied_any = True
            if not copied_any:
                yield None
                return
            yield local_dir

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(
            extra=["eval"],
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
            # Human eval tests code from the model which requires permission to run.
        )

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        wandb_name: str | None = None,
        wandb_group: str | None = None,
    ) -> None:
        """Launch the evaluation run with Fray."""

        mode_str = resolve_vllm_mode(None)
        pip_packages = VLLM_NATIVE_PIP_PACKAGES if mode_str == "native" else ()
        launch_evaluate_with_ray(
            evaluator=self,
            job_name="lm-eval",
            model=model,
            evals=evals,
            output_path=output_path,
            resource_config=resource_config,
            max_eval_instances=max_eval_instances,
            wandb_tags=wandb_tags,
            wandb_name=wandb_name,
            wandb_group=wandb_group,
            extras=("eval", "tpu"),
            pip_packages=pip_packages,
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        wandb_name: str | None = None,
        wandb_group: str | None = None,
    ) -> None:
        """
        Runs EleutherAI's lm-eval harness on the specified model and set of  tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        # From https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers
        # Run lm_eval with the model and the specified evals
        env: VllmEnvironment | None = None
        resolved_model = model
        try:
            with VllmEnvironment(model) as env:
                resolved_model = env.model

                def _run_lm_eval(lm_eval_model_local: str, pretrained_args_local: str) -> None:
                    from lm_eval.evaluator import simple_evaluate
                    from lm_eval.loggers import EvaluationTracker, WandbLogger
                    from lm_eval.utils import simple_parse_args_string

                    for eval_task in evals:
                        result_filepath = os.path.join(
                            self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot"
                        )

                        # Create the output directory
                        output_dir = os.path.dirname(result_filepath)
                        os.makedirs(output_dir, exist_ok=True)

                        evaluation_tracker_args = simple_parse_args_string(f",output_path={result_filepath}")
                        evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

                        wandb_args_dict = {
                            "project": "marin",
                            "job_type": "eval",
                            "name": wandb_name or resolved_model.name,
                            "tags": wandb_tags,
                        }
                        if wandb_group is not None:
                            wandb_args_dict["group"] = wandb_group
                        # wandb_config_args_dict = simple_parse_args_string("")
                        wandb_logger = WandbLogger(init_args=wandb_args_dict)

                        results = simple_evaluate(
                            model=lm_eval_model_local,
                            tasks=[eval_task.name],
                            num_fewshot=eval_task.num_fewshot,
                            model_args=pretrained_args_local,
                            apply_chat_template=resolved_model.apply_chat_template,
                            batch_size="auto",
                            confirm_run_unsafe_code=True,
                            limit=max_eval_instances if max_eval_instances is not None else None,
                            evaluation_tracker=evaluation_tracker,
                            log_samples=True,
                        )
                        if results is not None:
                            samples = results.pop("samples")
                            evaluation_tracker.save_results_aggregated(results=results, samples=samples)
                            try:
                                wandb_logger.post_init(results)
                                wandb_logger.log_eval_result()
                                wandb_logger.log_eval_samples(samples)
                                wandb_logger.run.finish()
                            except Exception as e:
                                print(f"Logging to Weights and Biases failed due to {e}")

                            for task_name in results["configs"].keys():
                                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

                        assert os.path.exists(result_filepath), f"Results file {result_filepath} does not exist."

                if env.model_id is None:
                    raise RuntimeError("vLLM server did not report a model id.")

                def _run_with_tokenizer(tokenizer: str | None) -> None:
                    if resolved_model.apply_chat_template:
                        lm_eval_model_local = "local-chat-completions"
                        pretrained_args_local = (
                            f"model={env.model_id},"
                            f"base_url={env.server_url}/chat/completions,"
                            "tokenizer_backend=huggingface,"
                            "tokenized_requests=False"
                        )
                    else:
                        lm_eval_model_local = "local-completions"
                        pretrained_args_local = (
                            f"model={env.model_id},"
                            f"base_url={env.server_url}/completions,"
                            "tokenizer_backend=huggingface,"
                            "tokenized_requests=False"
                        )
                    if tokenizer is not None:
                        pretrained_args_local += f",tokenizer={tokenizer}"
                    if resolved_model.engine_kwargs:
                        for key, value in resolved_model.engine_kwargs.items():
                            if key == "tokenizer":
                                continue
                            pretrained_args_local += f",{key}={value}"

                    _run_lm_eval(lm_eval_model_local, pretrained_args_local)

                if isinstance(resolved_model.engine_kwargs.get("tokenizer"), str):
                    _run_with_tokenizer(resolved_model.engine_kwargs.get("tokenizer"))
                elif is_remote_path(env.model_name_or_path):
                    with self._stage_remote_tokenizer_dir(env.model_name_or_path) as staged_tokenizer_dir:
                        if staged_tokenizer_dir is None:
                            raise ValueError(
                                "lm-eval's `local-completions` model requires a Hugging Face tokenizer name/path, "
                                f"but the served model id is a remote object-store URI: {env.model_id!r}, and no "
                                f"tokenizer files were found under {env.model_name_or_path!r}. "
                                "Set `engine_kwargs['tokenizer']` to an HF tokenizer id (e.g. "
                                "'meta-llama/Llama-3.1-8B-Instruct') or a local tokenizer path."
                            )
                        _run_with_tokenizer(staged_tokenizer_dir)
                else:
                    _run_with_tokenizer(None)

                return

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("lm-eval failed. Please check the logs for more information.") from e

        finally:

            # this is in the finally block so even in the case of exceptions we will
            # write what has been saved
            if is_remote_path(output_path):
                try:
                    logger.info("Uploading eval results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, output_path)
                    logger.info("Upload completed successfully.")
                except Exception as upload_error:
                    logger.info(f"Failed to upload results to GCS: {upload_error}")

            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)
