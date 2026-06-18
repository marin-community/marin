# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import socket
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from rigging.filesystem import open_url, url_to_fs

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "experiments").exists():
            return parent
    return Path.cwd()


def _resolve_repo_relative_path(path: str) -> str:
    if is_remote_path(path) or os.path.isabs(path):
        return path
    return str(_repo_root() / path)


def _resolve_data_files(data_files: object) -> object:
    if isinstance(data_files, str):
        return _resolve_repo_relative_path(data_files)
    if isinstance(data_files, list):
        return [_resolve_data_files(value) for value in data_files]
    if isinstance(data_files, tuple):
        return tuple(_resolve_data_files(value) for value in data_files)
    if isinstance(data_files, Mapping):
        return {key: _resolve_data_files(value) for key, value in data_files.items()}
    return data_files


def _resolve_task_kwargs(eval_task: EvalTaskConfig) -> dict:
    task_kwargs = deepcopy(eval_task.task_kwargs or {})
    dataset_kwargs = task_kwargs.get("dataset_kwargs")
    if isinstance(dataset_kwargs, dict) and "data_files" in dataset_kwargs:
        dataset_kwargs["data_files"] = _resolve_data_files(dataset_kwargs["data_files"])
    return task_kwargs


def _lm_eval_task_spec(eval_task: EvalTaskConfig) -> dict:
    """Return a lm-eval task spec preserving aliases and inline task fields."""
    spec = {
        "task": eval_task.name,
        "num_fewshot": eval_task.num_fewshot,
    }
    if eval_task.task_alias is not None:
        spec["task_alias"] = eval_task.task_alias
    spec.update(_resolve_task_kwargs(eval_task))
    return spec


def _patch_lm_eval_vllm_compat() -> None:
    """Patch lm-eval/vLLM API drift before lm-eval imports its vLLM backend."""
    try:
        import vllm.utils
    except ImportError:
        return

    if hasattr(vllm.utils, "get_open_port"):
        return

    def get_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    vllm.utils.get_open_port = get_open_port


def _patch_lm_eval_none_alias_compat(alias_fallbacks: Mapping[str, str] | None = None) -> None:
    """Patch lm-eval result formatting when task configs carry ``task_alias=None``.

    Some lm-eval Python tasks dump ``task_alias`` with a ``None`` value. Current
    lm-eval treats the key's presence as an explicit alias and later concatenates
    it with a string while building the result table, which crashes successful
    generation evals after metrics have been computed. Normalize only the
    returned alias values; do not change task loading or metric computation.
    """
    from lm_eval import evaluator, evaluator_utils

    fallback_map = dict(getattr(evaluator, "_marin_alias_fallbacks", {}))
    if alias_fallbacks is not None:
        fallback_map.update(alias_fallbacks)
    evaluator._marin_alias_fallbacks = fallback_map

    if getattr(evaluator, "_marin_none_alias_compat", False):
        return

    evaluate_fn = getattr(evaluator.evaluate, "__wrapped__", evaluator.evaluate)
    evaluate_globals = evaluate_fn.__globals__
    original_consolidate_results = evaluate_globals.get("consolidate_results", evaluator.consolidate_results)
    original_prepare_print_tasks = evaluate_globals.get("prepare_print_tasks", evaluator_utils.prepare_print_tasks)

    def alias_for(task_name: object, alias: object) -> str:
        if isinstance(alias, str) and alias:
            return alias
        task_name_str = str(task_name)
        return getattr(evaluator, "_marin_alias_fallbacks", {}).get(task_name_str, task_name_str)

    def consolidate_results_without_none_alias(eval_tasks):
        results, samples, configs, versions, num_fewshot, higher_is_better = original_consolidate_results(eval_tasks)
        alias_fallbacks_local = getattr(evaluator, "_marin_alias_fallbacks", {})
        for task_name, metrics in results.items():
            alias = metrics.get("alias")
            alias_fallback = alias_fallbacks_local.get(task_name)
            if alias is None:
                metrics["alias"] = alias_fallback or task_name
            elif alias == task_name and alias_fallback is not None:
                metrics["alias"] = alias_fallback
        for task_name, config in configs.items():
            if isinstance(config, dict):
                alias = config.get("task_alias")
                alias_fallback = alias_fallbacks_local.get(task_name)
                if alias is None and alias_fallback is not None:
                    config["task_alias"] = alias_fallback
        return results, samples, configs, versions, num_fewshot, higher_is_better

    def prepare_print_tasks_without_none_alias(task_dict, results, task_depth=0, group_depth=0):
        sanitized_results = {}
        for task_name, metrics in results.items():
            if isinstance(metrics, dict):
                sanitized_metrics = dict(metrics)
                sanitized_metrics["alias"] = alias_for(task_name, sanitized_metrics.get("alias"))
                sanitized_results[task_name] = sanitized_metrics
            else:
                sanitized_results[task_name] = metrics

        def infer_task_name(task_obj: object, used_names: set[str]) -> str:
            for attr in ("task_name", "task"):
                value = getattr(task_obj, attr, None)
                if isinstance(value, str) and value in sanitized_results:
                    return value
            for attr in ("config", "_config"):
                config = getattr(task_obj, attr, None)
                if isinstance(config, dict):
                    for key in ("task", "task_name"):
                        value = config.get(key)
                        if isinstance(value, str) and value in sanitized_results:
                            return value
            remaining = [task_name for task_name in sanitized_results if task_name not in used_names]
            if len(remaining) == 1:
                return remaining[0]
            raise KeyError("Could not infer lm-eval task name for None task_dict key")

        def sanitize_task_dict(node: dict) -> dict:
            sanitized_node = {}
            used_names: set[str] = set()
            for task_or_group_name, task_or_group_obj in node.items():
                sanitized_obj = (
                    sanitize_task_dict(task_or_group_obj) if isinstance(task_or_group_obj, dict) else task_or_group_obj
                )
                sanitized_name = task_or_group_name
                if sanitized_name is None:
                    sanitized_name = infer_task_name(sanitized_obj, used_names)
                if isinstance(sanitized_name, str):
                    used_names.add(sanitized_name)
                if getattr(sanitized_obj, "task_name", object()) is None:
                    sanitized_obj = object()
                sanitized_node[sanitized_name] = sanitized_obj
            return sanitized_node

        return original_prepare_print_tasks(sanitize_task_dict(task_dict), sanitized_results, task_depth, group_depth)

    evaluator.consolidate_results = consolidate_results_without_none_alias
    evaluator_utils.consolidate_results = consolidate_results_without_none_alias
    evaluate_globals["consolidate_results"] = consolidate_results_without_none_alias
    evaluator_utils.prepare_print_tasks = prepare_print_tasks_without_none_alias
    evaluate_globals["prepare_print_tasks"] = prepare_print_tasks_without_none_alias
    evaluator._marin_none_alias_compat = True


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
        with tempfile.TemporaryDirectory(prefix="marin-tokenizer-") as local_dir:
            copied_any = False
            for filename in cls.TOKENIZER_FILENAMES:
                remote_path = f"{remote_dir.rstrip('/')}/{filename}"
                if not is_remote_path(remote_path):
                    continue
                fs, fs_path = url_to_fs(remote_path)
                if not fs.exists(fs_path):
                    continue
                local_path = os.path.join(local_dir, filename)
                with open_url(remote_path, "rb") as src:
                    data = src.read()
                with open(local_path, "wb") as dst:
                    dst.write(data)
                copied_any = True
            if not copied_any:
                yield None
                return
            yield local_dir

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
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
        try:
            with VllmEnvironment(model) as env:
                resolved_model = env.model

                def _run_lm_eval(lm_eval_model_local: str, pretrained_args_local: str) -> None:
                    _patch_lm_eval_vllm_compat()

                    from lm_eval.evaluator import simple_evaluate
                    from lm_eval.loggers import EvaluationTracker, WandbLogger
                    from lm_eval.utils import simple_parse_args_string

                    for eval_task in evals:
                        task_label = eval_task.task_alias or eval_task.name
                        _patch_lm_eval_none_alias_compat({eval_task.name: task_label})
                        result_filepath = os.path.join(self.RESULTS_PATH, f"{task_label}_{eval_task.num_fewshot}shot")

                        # Create the output directory
                        output_dir = os.path.dirname(result_filepath)
                        os.makedirs(output_dir, exist_ok=True)

                        evaluation_tracker_args = simple_parse_args_string(f",output_path={result_filepath}")
                        evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

                        wandb_args_dict = {
                            "project": "marin",
                            "job_type": "eval",
                            "name": resolved_model.name,
                            "tags": wandb_tags,
                        }
                        # wandb_config_args_dict = simple_parse_args_string("")
                        wandb_logger = WandbLogger(init_args=wandb_args_dict)

                        results = simple_evaluate(
                            model=lm_eval_model_local,
                            tasks=[_lm_eval_task_spec(eval_task)],
                            model_args=pretrained_args_local,
                            apply_chat_template=resolved_model.apply_chat_template,
                            batch_size="auto",
                            confirm_run_unsafe_code=True,
                            limit=max_eval_instances if max_eval_instances is not None else None,
                            gen_kwargs=resolved_model.generation_params or None,
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
