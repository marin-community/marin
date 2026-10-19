import asyncio
from eval.execution.driver import source_plans, run_evaluations
from eval.execution.run_spec import RunSpec
import argparse
import difflib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models

try:
    import torch.distributed as dist
except ModuleNotFoundError:
    dist = None
import yaml
import numpy as np
from lm_eval import utils

# Upstream EleutherAI lm-eval (>=0.4.8) refactored the CLI into lm_eval._cli and
# removed lm_eval.__main__.{setup_parser,parse_eval_args} as well as the
# module-level lm_eval.utils.eval_logger singleton. We vendor compatible
# versions in eval.lm_eval_compat to stay decoupled from lm-eval CLI churn.
from eval.lm_eval_compat import parse_eval_args, setup_parser

if not hasattr(utils, "eval_logger"):
    from eval.lm_eval_compat import eval_logger as _eval_logger

    utils.eval_logger = _eval_logger
from lm_eval.api.model import LM
from lm_eval.models.api_models import TemplateAPI
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager as PretrainTaskManager
from lm_eval.utils import sanitize_model_name
from evalchemy_config.limits import MAX_OUTPUT_ALIASES, MODEL_LENGTH_ALIASES

# Register the async-batch robustness patch before any model adapter is built.
from eval import robust_api  # noqa: F401
from eval.serve_eval.local_api import RetryingLocalChatCompletions, RetryingLocalCompletions

from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.contracts.benchmark_metadata import (
    canonicalize_results,
    describe_benchmarks,
    infer_benchmark_metadata,
)
from eval.contracts.conformance import build_task_contract_registry
from eval.contracts.finestore_output import FineStoreOutput, read_finestore_output
from eval.contracts.finestore_resume import ResumeRefused
from eval.contracts.preflight import prepare_requested_tasks
from eval.contracts.result_serialization import handle_non_serializable_extended
from eval.contracts.task_outcome import TaskRoute, validate_result_document
from eval.limits import parse_key_value_args, resolve_evaluation_limits
from eval.resume.wiring import build_resume_wiring
from eval.task import TaskManager as InstructTaskManager

# Evalchemy-owned lm-eval task overrides (marin-community/evalchemy#31). include_path
# task configs take precedence over lm-eval's packaged defaults, so shipping
# eval/lm_eval_tasks/drop/drop.yaml here transparently overrides upstream `drop` with the
# answer-extraction filter_list — no run-config change required. A user-supplied
# --include_path still wins (it is appended last).
DEFAULT_LM_EVAL_INCLUDE_DIR = os.path.join(os.path.dirname(__file__), "lm_eval_tasks")
CHAT_BENCHMARK_ROUTE = TaskRoute.CUSTOM
LM_EVAL_ROUTE = TaskRoute.LM_EVAL


def _attach_benchmark_metadata(results: dict[str, Any], args: argparse.Namespace) -> None:
    """Attach source descriptions and canonical scores to the shared result."""
    metadata = list(getattr(args, "resolved_benchmark_metadata", ()))
    described = {description.task for description in metadata}
    for task, metrics in results["results"].items():
        if task in described:
            continue
        count = results["task_outcomes"].get(task, {}).get("expected_count")
        metadata.append(infer_benchmark_metadata(task, metrics, n_attempted=count))
    results["benchmark_metadata"] = {description.task: description.to_dict() for description in metadata}
    results["canonical_results"] = canonicalize_results(results["results"], metadata)


def resolve_task_routes(
    task_list: List[str], task_manager: InstructTaskManager, pretrain_task_manager: PretrainTaskManager
) -> Dict[str, str]:
    """Resolve each selected task to the registry that will evaluate it."""
    custom_tasks = set(task_manager.tasks) | set(getattr(task_manager, "load_failures", {}))
    contracts = build_task_contract_registry(custom_tasks, pretrain_task_manager.all_tasks)
    contracts_by_name = {contract.task_name: contract for contract in contracts}
    task_routes: Dict[str, str] = {}
    unknown_tasks: List[str] = []
    available_tasks = sorted(contracts_by_name)

    for task in dict.fromkeys(task_list):
        if contract := contracts_by_name.get(task):
            task_routes[task] = contract.route
        else:
            unknown_tasks.append(task)

    if unknown_tasks:
        suggestions = {
            task: difflib.get_close_matches(task, available_tasks, n=3) for task in unknown_tasks
        }
        suggestion_text = "; ".join(
            f"{task} (did you mean: {', '.join(matches)})"
            for task, matches in suggestions.items()
            if matches
        )
        message = f"Unknown evaluation tasks: {', '.join(unknown_tasks)}."
        if suggestion_text:
            message = f"{message} Close matches: {suggestion_text}."
        raise ValueError(message)

    for task, route in task_routes.items():
        utils.eval_logger.info("%s -> %s", task, route)

    return task_routes


def setup_custom_parser():
    """
    Create a custom argument parser that extends lm-eval-harness parser.
    """
    parser = setup_parser()
    parser.add_argument(
        "--annotator_model",
        type=str,
        default=AUTO_ANNOTATOR_MODEL,
        help="Judge model used to evaluate generations. Example: gpt-4o-mini-2024-07-18",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Canonical maximum generated tokens for every Evalchemy benchmark.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Canonical total model context length for every Evalchemy benchmark.",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml. Sets tasks, batch sizes, and annotator; explicit CLI generation limits take precedence.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run evalutaions in debug mode on a few examples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of completions to generate per problem. 1 (default) reproduces today's "
        "single-sample score byte-identically; >1 enables native pass@k for supported benchmarks.",
    )
    parser.add_argument(
        "--pass_at_k",
        type=str,
        default="1,8,32,128",
        help="Comma-separated k-list for pass@k aggregation (only consulted when --num_samples > 1).",
    )
    parser.add_argument(
        "--resume-mode",
        "--resume_mode",
        dest="resume_mode",
        type=str,
        default="auto",
        choices=["auto", "force-fresh"],
        help="Restore completed requests from FineStore by default. force-fresh requires an empty archive.",
    )
    return parser


def evaluate(
    lm: LM,
    task_manager: InstructTaskManager,
    pretrain_task_manager: PretrainTaskManager,
    task_list: List[str],
    task_routes: Dict[str, str],
    batch_sizes_list: List[int],
    verbosity: str = "INFO",
    args=None,
    output_sink: FineStoreOutput | None = None,
    **eval_kwargs,
) -> Dict[str, Dict]:
    """Normalize every source into EvalSpec items and execute one shared flow."""
    run = RunSpec.from_args(args)
    robust_api.configure_generation_overrides(lm, run.generation)
    if run.seeds[0] is not None:
        random.seed(run.seeds[0])
    if run.seeds[1] is not None:
        np.random.seed(run.seeds[1])
    if run.seeds[2] is not None:
        utils.set_torch_seed(run.seeds[2])
    plans = source_plans(task_list, task_routes, task_manager, pretrain_task_manager, run, lm)
    factory = build_resume_wiring(args, lm, run, store=output_sink.store, judge_configs={
        task: plan.judge_config for plan in plans if plan.judge_config is not None for task in plan.task_names
    }) if output_sink is not None and getattr(args, "finestore_output_path", None) else None
    checkpoints = {
        task: factory(task) for plan in plans for task in plan.task_names
    } if factory is not None else {}
    model_options = parse_key_value_args(getattr(args, "model_args", ""))
    workers = int(model_options.get("num_concurrent", 1))
    local_batches = {}
    if not isinstance(lm, TemplateAPI):
        local_batches = {task: size for plan, size in zip(plans, batch_sizes_list, strict=True) for task in plan.task_names}
        workers = max(workers, *(int(size) if size != "auto" else (getattr(args, "max_batch_size", None) or 64)
                                 for size in batch_sizes_list))
    results = asyncio.run(run_evaluations(lm, plans, workers=workers, sink=output_sink, checkpoints=checkpoints,
                                         predict_only=run.predict_only, seeds=run.seeds,
                                         supports_request_seed=args.model == "vllm",
                                         batch_sizes=local_batches, batch_size_attribute="batch_size_per_gpu" if args.model == "hf" else "batch_size"))
    results["task_preparations"] = getattr(args, "task_preparations", {})
    _attach_benchmark_metadata(results, args)
    return results


def _apply_config_limit_defaults(args: argparse.Namespace, tasks_yaml: dict) -> None:
    """Use YAML limits only when the caller supplied no limit in any CLI spelling."""
    model_limit_keys = parse_key_value_args(args.model_args)
    generation_limit_keys = parse_key_value_args(args.gen_kwargs)
    if "max_length" in tasks_yaml and args.max_length is None and not any(
        key in model_limit_keys for key in MODEL_LENGTH_ALIASES
    ):
        args.max_length = tasks_yaml["max_length"]
    if "max_tokens" in tasks_yaml and args.max_tokens is None and not any(
        key in generation_limit_keys for key in MAX_OUTPUT_ALIASES
    ):
        args.max_tokens = tasks_yaml["max_tokens"]


def cli_evaluate(args: Optional[argparse.Namespace] = None) -> None:
    """
    Command-line interface for evaluating language models.

    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    # Parse arguments if not provided
    if not args:
        parser = setup_custom_parser()
        args = parse_eval_args(parser)

    if not args.finestore_output_path:
        raise ValueError("--finestore_output_path is required")
    if args.resume_mode == "off":
        raise ValueError("FineStore output requires --resume-mode auto or force-fresh")
    if args.config is not None:
        # This overwrites `--tasks` and `--batch_size`
        with open(args.config, "r") as file:
            tasks_yaml = yaml.safe_load(file)
        args.tasks = ",".join([t["task_name"] for t in tasks_yaml["tasks"]])
        batch_sizes_list = [int(t["batch_size"]) if t["batch_size"] != "auto" else "auto" for t in tasks_yaml["tasks"]]
        args.annotator_model = tasks_yaml.get("annotator_model", args.annotator_model)
        _apply_config_limit_defaults(args, tasks_yaml)
    else:
        batch_sizes_list = [
            int(args.batch_size) if args.batch_size != "auto" else args.batch_size
            for _ in range(len(args.tasks.split(",")))
        ]

    prior_result = read_finestore_output(args.finestore_output_path, args.finestore_output_prefix)
    if prior_result is not None:
        if args.resume_mode == "force-fresh":
            raise ResumeRefused("FineStore output already exists; use a new output path for a fresh evaluation")
        if prior_result.get("config", {}).get("tasks") != args.tasks:
            raise ResumeRefused("FineStore result task list changed; use a new output path")

    general_config_tracker = GeneralConfigTracker()

    task_list = args.tasks.split(",")

    # Resolve once before either benchmark family or the model is initialized.
    # Native lm-eval receives ``model_args`` / ``gen_kwargs``; custom chat
    # benchmarks receive the same values from the task manager below.
    limits = resolve_evaluation_limits(args)
    utils.eval_logger.info(
        "Evaluation limits: max_length=%s max_tokens=%s",
        limits.max_length if limits.max_length is not None else "default",
        limits.max_tokens if limits.max_tokens is not None else "default",
    )
    utils.eval_logger.setLevel(getattr(logging, args.verbosity))
    logging.basicConfig(level=getattr(logging, args.verbosity), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Initialize tasks
    task_manager = InstructTaskManager(
        annotator_model=args.annotator_model,
        max_length=limits.max_length,
        max_tokens=limits.max_tokens,
        limit=getattr(args, "limit", None),
        num_fewshot=args.num_fewshot,
        debug=args.debug,
        seed=args.seed,
        task_list=task_list,
        system_instruction=args.system_instruction,
        num_samples=getattr(args, "num_samples", 1),
        pass_at_k=getattr(args, "pass_at_k", None),
    )
    # Always scan the evalchemy-owned overrides first; a user --include_path is appended
    # last so it still wins (later include paths take precedence).
    _include_paths = [DEFAULT_LM_EVAL_INCLUDE_DIR]
    if args.include_path:
        _include_paths.append(args.include_path)
    pretrain_task_manager = PretrainTaskManager(args.verbosity, include_path=_include_paths)
    task_routes = resolve_task_routes(task_list, task_manager, pretrain_task_manager)
    preparations = prepare_requested_tasks(
        task_list,
        task_routes,
        task_manager,
        pretrain_task_manager,
    )
    args.task_preparations = {
        preparation.task_name: preparation.to_dict() for preparation in preparations
    }
    benchmark_metadata = describe_benchmarks(
        task_list,
        task_routes,
        task_manager,
        pretrain_task_manager,
        limit=getattr(args, "limit", None),
    )
    args.resolved_benchmark_metadata = benchmark_metadata

    utils.eval_logger.info(f"Selected Tasks: {[task for task in task_list]}")

    # Shared equivalence judges use credentials isolated from the candidate endpoint.
    judge_tasks = [
        task
        for task in task_list
        if task_routes[task] == CHAT_BENCHMARK_ROUTE
        and task_manager.requires_judge_credentials(task)
    ]
    if judge_tasks and not os.getenv("JUDGE_API_KEY"):
        raise ValueError(
            "Please set JUDGE_API_KEY to evaluate the following tasks: "
            f"{judge_tasks}"
        )

    # Initialize model
    try:
        lm = initialize_model(args.model, args.model_args, batch_size=args.batch_size)
    except Exception as e:
        utils.eval_logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    # Log experiment configuration
    general_config_tracker.log_experiment_args(
        model_source=args.model,
        model_args=args.model_args,
        system_instruction=args.system_instruction,
        chat_template=lm.chat_template(args.apply_chat_template),
        fewshot_as_multiturn=args.fewshot_as_multiturn,
    )

    # Initialize logging and environment
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with FineStoreOutput(args.finestore_output_path, args.finestore_output_prefix, task_list) as output_sink:
        results = evaluate(
            lm=lm,
            task_manager=task_manager,
            pretrain_task_manager=pretrain_task_manager,
            task_list=task_list,
            task_routes=task_routes,
            batch_sizes_list=batch_sizes_list,
            verbosity=args.verbosity,
            args=args,
            output_sink=output_sink,
        )
        output_sink.finish_task()
        if lm.world_size > 1:
            lm.barrier()

        if lm.rank == 0:
            add_results_metadata(results, batch_sizes_list, args, lm)
            handle_evaluation_output(results, args, general_config_tracker, output_sink=output_sink)

    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


def initialize_model(
    model: Union[str, LM],
    model_args: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> LM:
    """
    Initialize the language model based on provided configuration.

    Args:
        model (Union[str, LM]):
            Either a string identifier for the model to load from registry,
            or an already instantiated LM object.
        model_args (Optional[str], optional):
            Additional arguments for model initialization as a string.
            Only used if model is provided as a string. Defaults to None.
        device (Optional[str], optional):
            Device to load the model on (e.g., 'cuda', 'cpu'). Defaults to None.

    Returns:
        LM:
            Initialized language model instance with configured parameters
            and a sanitized model identifier.
    """
    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        config = {
            "device": device,
        }

        if "batch_size" not in model_args:
            if batch_size is not None:
                model_args += f",batch_size={batch_size}"

        endpoint_models = {
            "local-chat-completions": RetryingLocalChatCompletions,
            "local-completions": RetryingLocalCompletions,
        }
        model_class = endpoint_models.get(model) or lm_eval.api.registry.get_model(model)
        lm = model_class.create_from_arg_string(
            model_args,
            config,
        )
    else:
        lm = model

    lm.model_identifier = sanitize_model_name(f"model_{model}_model_args_{model_args}")
    return lm


def add_results_metadata(results: Dict, batch_sizes_list: List[int], args: argparse.Namespace, lm: LM) -> None:
    """
    Add metadata and configuration to results.

    Args:
        results (Dict):
            Dictionary of evaluation results to be augmented with metadata.
            The function will modify this dictionary in-place to add
            configuration and runtime information.
        batch_sizes_list (List[int]):
            List of batch sizes for each task.
        args (argparse.Namespace):
            Command line arguments containing runtime configuration
            and parameters used during evaluation.
        lm (LM):
            Language model instance, used to extract model-specific
            configuration and parameters.

    Returns:
        None:
            The function modifies the results dictionary in-place.
    """
    results["config"] = {
        "model": (
            args.model
            if isinstance(args.model, str)
            else args.model.config._name_or_path if hasattr(args.model, "config") else type(args.model).__name__
        ),
        "model_args": args.model_args,
        "tasks": args.tasks,
        "batch_sizes": batch_sizes_list,
        "device": args.device,
        "limit": args.limit,
        "annotator_model": args.annotator_model,
        "max_length": getattr(args, "max_length", None) if getattr(args, "max_length", None) is not None else "default",
        "max_tokens": getattr(args, "max_tokens", None) if getattr(args, "max_tokens", None) is not None else "default",
        "evaluation_limits": {
            "max_length": (
                getattr(args, "max_length", None) if getattr(args, "max_length", None) is not None else "default"
            ),
            "max_tokens": (
                getattr(args, "max_tokens", None) if getattr(args, "max_tokens", None) is not None else "default"
            ),
        },
        # "bootstrap_iters": args.bootstrap_iters,
        "gen_kwargs": args.gen_kwargs,
        "random_seed": args.seed[0],
        "numpy_seed": args.seed[1],
        "torch_seed": args.seed[2],
        "fewshot_seed": args.seed[3],
    }

    # `import lm_eval.models` does NOT import the huggingface submodule (it is only
    # registered lazily via the model registry), so `lm_eval.models.huggingface`
    # raises AttributeError for a non-HF (e.g. API/endpoint) model -- which would
    # crash this metadata step and discard an already-computed score. Resolve HFLM
    # defensively; `isinstance(lm, ())` is a safe False on minimal installs.
    try:
        from lm_eval.models.huggingface import HFLM
    except Exception:  # pragma: no cover - minimal installs may lack HF deps
        HFLM = ()
    if isinstance(lm, HFLM):
        results["config"].update(lm.get_model_info())

    results["git_hash"] = get_git_commit_hash()
    results["date"] = time.time()
    add_env_info(results)
    add_tokenizer_info(results, lm)


def handle_evaluation_output(
    results: Dict,
    args: argparse.Namespace,
    general_config_tracker: GeneralConfigTracker,
    *,
    output_sink: FineStoreOutput | None = None,
) -> None:
    """Commit aggregate results and run metadata to FineStore."""
    if not args.finestore_output_path:
        raise ValueError("--finestore_output_path is required")
    validate_result_document(results)
    if output_sink is None:
        raise ValueError("FineStore output sink must be opened before evaluation")
    if results.get("samples"):
        raise ValueError("sample records must be staged before aggregate output")
    results.pop("samples", None)

    dumped = json.dumps(
        results,
        indent=2,
        default=handle_non_serializable_extended,
        ensure_ascii=False,
    )
    if args.show_config:
        print(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

    general_config_tracker.log_end_time()
    results.update(asdict(general_config_tracker))
    output_sink.finish(results)

    utils.eval_logger.info(
        f"Eval arugments: {args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
        f"max_length: {getattr(args, 'max_length', None)}, max_tokens: {getattr(args, 'max_tokens', None)}, "
        f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, annotator_model: {args.annotator_model}, "
        f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )


if __name__ == "__main__":
    cli_evaluate()
