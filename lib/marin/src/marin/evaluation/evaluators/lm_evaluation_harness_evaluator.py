# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EleutherAI lm-evaluation-harness driven via OpenAI-compatible HTTP.

Runs against a `RunningModel` endpoint; server lifecycle lives in the launcher.
"""

from __future__ import annotations

import logging
import os
import shutil

from marin.evaluation.api import LmEvalRun
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.inference.model_launcher import RunningModel

logger = logging.getLogger(__name__)


# Multiple-choice tasks currently don't work on TPU+vLLM:
# https://github.com/vllm-project/vllm/issues/8499
class LmEvalEvaluator:
    """Runs lm-eval against an OpenAI-compatible endpoint."""

    CACHE_PATH: str = "/tmp/lm-eval"
    RESULTS_PATH: str = os.path.join(CACHE_PATH, "eleuther_results")

    def __init__(self, run: LmEvalRun) -> None:
        self.run_config = run

    def run(self, model: RunningModel) -> None:
        run = self.run_config
        lm_eval_kind = _lm_eval_kind(run)
        model_args = _build_lm_eval_model_args(model, run)
        try:
            for eval_task in run.evals:
                result_filepath = os.path.join(self.RESULTS_PATH, f"{eval_task.name}_{eval_task.num_fewshot}shot")
                os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
                self._run_one_task(
                    eval_task=eval_task,
                    lm_eval_kind=lm_eval_kind,
                    model_args=model_args,
                    result_filepath=result_filepath,
                    wandb_run_name=run.base_eval_run_name or model.endpoint.model,
                )
                assert os.path.exists(result_filepath), f"Results file {result_filepath} does not exist."
        finally:
            # Log-and-swallow upload errors in the finally block so they don't
            # mask a primary eval exception; rmtree always runs.
            try:
                if is_remote_path(run.output_path):
                    logger.info("Uploading eval results to GCS...")
                    upload_to_gcs(self.RESULTS_PATH, run.output_path)
                    logger.info("Upload completed successfully.")
            except Exception:
                logger.exception("Failed to upload eval results to GCS")
            if os.path.exists(self.RESULTS_PATH):
                shutil.rmtree(self.RESULTS_PATH)

    def _run_one_task(
        self,
        *,
        eval_task,
        lm_eval_kind: str,
        model_args: str,
        result_filepath: str,
        wandb_run_name: str,
    ) -> None:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, WandbLogger

        run = self.run_config
        evaluation_tracker = EvaluationTracker(output_path=result_filepath)
        wandb_logger = WandbLogger(
            init_args={
                "project": "marin",
                "job_type": "eval",
                "name": wandb_run_name,
                "tags": run.wandb_tags,
            }
        )

        results = simple_evaluate(
            model=lm_eval_kind,
            tasks=[eval_task.name],
            num_fewshot=eval_task.num_fewshot,
            model_args=model_args,
            apply_chat_template=run.apply_chat_template,
            batch_size=run.batch_size,
            confirm_run_unsafe_code=True,
            limit=run.max_eval_instances,
            evaluation_tracker=evaluation_tracker,
            log_samples=True,
        )
        if results is None:
            return

        samples = results.pop("samples")
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)

        try:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(samples)
            wandb_logger.run.finish()
        except Exception as e:
            logger.warning(f"Logging to Weights and Biases failed due to {e}")

        for task_name in results["configs"].keys():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])


def _lm_eval_kind(run: LmEvalRun) -> str:
    return "local-chat-completions" if run.apply_chat_template else "local-completions"


def _build_lm_eval_model_args(model: RunningModel, run: LmEvalRun) -> str:
    completions_path = "chat/completions" if run.apply_chat_template else "completions"
    parts = [
        f"model={model.endpoint.model}",
        f"base_url={model.endpoint.url}/{completions_path}",
        f"tokenizer={model.tokenizer_ref}",
        "tokenizer_backend=huggingface",
        "tokenized_requests=False",
        *run.extra_model_args,
    ]
    return ",".join(parts)
