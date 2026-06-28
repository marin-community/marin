# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import math
import os

import jmp
import levanter
import levanter.eval_harness as eval_harness
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem import filesystem as marin_filesystem

from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import _resolve_task_kwargs

logger = logging.getLogger(__name__)


def _resolve_levanter_eval_tasks(evals: list[EvalTaskConfig]) -> list[EvalTaskConfig]:
    return [
        (
            dataclasses.replace(eval_task, task_kwargs=_resolve_task_kwargs(eval_task))
            if eval_task.task_kwargs is not None
            else eval_task
        )
        for eval_task in evals
    ]


def _target_index(target: object) -> int | None:
    if isinstance(target, list):
        if len(target) != 1:
            return None
        target = target[0]
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        return target
    return None


def _normalized_choice_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.strip().casefold().split())


def _letter_choice_index(value: str, response_count: int) -> int | None:
    normalized = value.strip().casefold()
    if len(normalized) != 1 or not normalized.isalpha():
        return None
    index = ord(normalized) - ord("a")
    if 0 <= index < response_count:
        return index
    return None


def _target_index_from_arguments(target: object, arguments: object, response_count: int) -> int | None:
    target_text = _normalized_choice_text(target)
    if target_text is None or not isinstance(arguments, list) or len(arguments) != response_count:
        return None
    matches: list[int] = []
    for index, argument in enumerate(arguments):
        if not isinstance(argument, (list, tuple)) or len(argument) < 2:
            continue
        continuation_text = _normalized_choice_text(argument[1])
        if continuation_text == target_text:
            matches.append(index)
    if len(matches) == 1:
        return matches[0]
    return None


def _target_index_from_doc(doc: object, response_count: int) -> int | None:
    if not isinstance(doc, dict):
        return None

    label = doc.get("label")
    if isinstance(label, int) and not isinstance(label, bool) and 0 <= label < response_count:
        return label

    answer = doc.get("answer")
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.isdigit():
            one_indexed = int(stripped) - 1
            if 0 <= one_indexed < response_count:
                return one_indexed
        return _letter_choice_index(stripped, response_count)
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= answer < response_count:
            return answer
        one_indexed = answer - 1
        if 0 <= one_indexed < response_count:
            return one_indexed

    answer_key = doc.get("answerKey")
    if isinstance(answer_key, str):
        return _letter_choice_index(answer_key, response_count)
    return None


def _sample_target_index(sample: dict, response_count: int) -> int | None:
    target_index = _target_index(sample.get("target"))
    if target_index is not None:
        return target_index
    # Single-continuation loglikelihood tasks, such as Lambada, use a string target.
    if isinstance(sample.get("target"), str) and response_count == 1:
        return 0
    target_index = _target_index_from_arguments(sample.get("target"), sample.get("arguments"), response_count)
    if target_index is not None:
        return target_index
    target_index = _target_index_from_doc(sample.get("doc"), response_count)
    if target_index is not None:
        return target_index
    return None


def _choice_logprob(response: object) -> float | None:
    if isinstance(response, (list, tuple)):
        if not response:
            return None
        return _choice_logprob(response[0])
    if isinstance(response, (int, float)) and math.isfinite(float(response)):
        return float(response)
    return None


def _continuation_byte_count(argument: object) -> int | None:
    if not isinstance(argument, (list, tuple)) or len(argument) < 2:
        return None
    continuation = argument[1]
    if not isinstance(continuation, str):
        return None
    return len(continuation.encode("utf-8"))


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_metric_values(sample: dict) -> dict[str, float]:
    responses = sample.get("filtered_resps")
    arguments = sample.get("arguments")
    if not isinstance(responses, list):
        return {}
    target_index = _sample_target_index(sample, len(responses))
    if target_index is None or target_index >= len(responses):
        return {}

    choice_logprobs = [_choice_logprob(response) for response in responses]
    if any(value is None for value in choice_logprobs):
        return {}
    logprobs = [float(value) for value in choice_logprobs if value is not None]
    if target_index >= len(logprobs):
        return {}

    gold_logprob = logprobs[target_index]
    incorrect_logprobs = [value for index, value in enumerate(logprobs) if index != target_index]
    best_incorrect_logprob = max(incorrect_logprobs) if incorrect_logprobs else math.nan
    max_logprob = max(logprobs)
    logsumexp = max_logprob + math.log(sum(math.exp(value - max_logprob) for value in logprobs))
    metrics = {
        "native_gold_logprob": gold_logprob,
        "native_margin": gold_logprob - best_incorrect_logprob if math.isfinite(best_incorrect_logprob) else math.nan,
        "native_choice_prob": math.exp(gold_logprob - logsumexp),
        "native_predicted_correct": 1.0 if gold_logprob == max_logprob else 0.0,
        "native_choice_count": float(len(logprobs)),
    }

    if isinstance(arguments, list) and len(arguments) == len(logprobs):
        byte_counts = [_continuation_byte_count(argument) for argument in arguments]
        if all(byte_count is not None and byte_count > 0 for byte_count in byte_counts):
            bytes_per_choice = [int(byte_count) for byte_count in byte_counts if byte_count is not None]
            normalized_logprobs = [
                logprob / byte_count for logprob, byte_count in zip(logprobs, bytes_per_choice, strict=True)
            ]
            gold_byte_count = bytes_per_choice[target_index]
            gold_normalized = normalized_logprobs[target_index]
            incorrect_normalized = [value for index, value in enumerate(normalized_logprobs) if index != target_index]
            metrics.update(
                {
                    "native_gold_bpb": -gold_logprob / (gold_byte_count * math.log(2.0)),
                    "native_gold_logprob_per_byte": gold_normalized,
                    "native_margin_per_byte": (
                        gold_normalized - max(incorrect_normalized) if incorrect_normalized else math.nan
                    ),
                }
            )
    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def add_sample_smooth_metrics(results: dict) -> None:
    """Add aggregate MCQ smooth metrics from lm-eval sample payloads in-place."""
    samples_by_task = results.get("samples")
    task_results = results.get("results")
    if not isinstance(samples_by_task, dict) or not isinstance(task_results, dict):
        return

    for task_name, samples in samples_by_task.items():
        if not isinstance(samples, list):
            continue
        values_by_metric: dict[str, list[float]] = {}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            for metric_name, value in _sample_metric_values(sample).items():
                values_by_metric.setdefault(metric_name, []).append(value)
        if not values_by_metric:
            continue
        task_result = task_results.setdefault(task_name, {})
        if not isinstance(task_result, dict):
            continue
        for metric_name, values in sorted(values_by_metric.items()):
            task_result[f"{metric_name},none"] = _mean(values)
            task_result[f"{metric_name}_stderr,none"] = _standard_error(values)
        task_result["native_sample_count,none"] = float(max(len(values) for values in values_by_metric.values()))


def drop_sample_payloads(results: dict) -> None:
    """Remove bulky sample payloads after aggregate metrics have been extracted."""
    results.pop("samples", None)
    task_results = results.get("results")
    if not isinstance(task_results, dict):
        return
    for task_result in task_results.values():
        if isinstance(task_result, dict):
            task_result.pop("outputs", None)


def _finish_current_tracker_best_effort() -> None:
    """Finish the active Levanter tracker without failing an otherwise complete eval."""
    try:
        tracker = levanter.tracker.current_tracker()
    except RuntimeError:
        logger.info("No active Levanter tracker to finish.")
        return
    try:
        tracker.finish()
    except Exception:
        logger.warning("Failed to finish Levanter tracker after writing results; continuing.", exc_info=True)


class LevanterLmEvalEvaluator(Evaluator):
    """Runs inference with Levanter's Lm Eval Harness on TPUs."""

    @staticmethod
    def model_name_or_path(model: ModelConfig) -> str:
        """Return a reference Levanter can read without staging to local disk."""
        if model.path is None:
            return model.name
        return model.path

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        use_wandb_tracker: bool = True,
        eval_datasets_cache_path: str | None = None,
        log_samples: bool = False,
        sample_log_all: bool = False,
        max_logged_samples_per_task: int | None = None,
        sample_smooth_metrics: bool = False,
        drop_samples_after_metrics: bool = False,
    ) -> None:
        """
        Runs Levanter's lm-eval harness on the specified model and set of tasks.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            wandb_tags (list[str] | None): The tags to add to the wandb run.
            use_wandb_tracker (bool): Whether to create a W&B tracker for this eval.
            eval_datasets_cache_path (str | None): Accepted for compatibility with
                EvaluationConfig. Levanter's evaluator currently uses lm-eval's own
                task loading path, so this cache path is not consumed here.
            log_samples (bool): Whether lm-eval should persist per-document sample payloads.
            sample_log_all (bool): Whether Levanter's extra sample logging payload should keep every sample.
            max_logged_samples_per_task (int | None): Optional cap for Levanter's extra sample logging payload.
            sample_smooth_metrics (bool): Whether to derive aggregate smooth metrics from lm-eval samples.
            drop_samples_after_metrics (bool): Whether to remove bulky sample payloads after extraction.
        """
        # Eval Harness code: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/eval_harness.py
        # Run the harness with the model and the specified evals
        model_name_or_path: str = self.model_name_or_path(model)
        name = model.name + "_lmeval_" + "-".join([eval_task.name for eval_task in evals])
        logger.info(
            "Running eval harness on model: %s, tracker run name: %s, use_wandb_tracker=%s",
            model_name_or_path,
            name,
            use_wandb_tracker,
        )
        if eval_datasets_cache_path is not None:
            logger.info("Levanter lm-eval ignores eval_datasets_cache_path: %s", eval_datasets_cache_path)

        # NOTE(chris): Before, the batch size was 16, but this is too large for the 8B model.
        # In the future, we should make this user-configurable.
        tracker_config = WandbConfig(project="marin", tags=wandb_tags, name=name) if use_wandb_tracker else NoopConfig()
        trainer_config = TrainerConfig(
            tracker=tracker_config,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=1,
        )

        model_config = HFCheckpointConverter.from_hf(model_name_or_path).LevConfigClass()

        # convert to the config that Levanter's eval_harness expects
        tasks = convert_to_levanter_task_config(_resolve_levanter_eval_tasks(evals))
        logger.info(f"Tasks: {tasks}")

        eval_config = eval_harness.EvalHarnessMainConfig(
            eval_harness=eval_harness.LmEvalHarnessConfig(
                task_spec=tasks,
                max_examples=max_eval_instances,
                log_samples=log_samples,
                max_length=4096,
                apply_chat_template=model.apply_chat_template,
                confirm_run_unsafe_code=True,
                sample_logging=eval_harness.SampleLoggingConfig(
                    log_all=sample_log_all,
                    max_samples_per_benchmark=max_logged_samples_per_task,
                ),
            ),
            tokenizer=model_name_or_path,  # levanter picks up the tokenizer from the model path
            checkpoint_path=model_name_or_path,
            checkpoint_is_hf=True,
            trainer=trainer_config,
            model=model_config,
        )

        results = eval_harness.run_eval_harness_main(eval_config)
        if results is None:
            results = {}
        if sample_smooth_metrics:
            add_sample_smooth_metrics(results)
        if drop_samples_after_metrics:
            drop_sample_payloads(results)

        # Upload is best-effort: a transient GCS failure should not throw away an
        # otherwise successful (and very expensive) eval run.
        results_path = os.path.join(output_path, "results.json")
        logger.info(f"Uploading results to GCS: {results_path}")
        try:
            fs = marin_filesystem("gcs")
            with fs.open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=_json_default)
            logger.info("Upload completed successfully.")
        except Exception:
            logger.warning("Failed to upload results to GCS: %s", results_path, exc_info=True)
        finally:
            _finish_current_tracker_best_effort()


def _json_default(value):
    """
    Provide a best-effort JSON serialization for objects returned by the eval harness.
    """
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    if isinstance(value, set):
        return list(value)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass

    return repr(value)
