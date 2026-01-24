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

"""
Canonical set of evals.
"""

import logging
from typing import Sequence

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    output_path_of,
    this_output_path,
    versioned,
)

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
from experiments.evals.evalchemy_task_configs import EVALCHEMY_CORE_TASKS
from experiments.evals.task_configs import (
    BASE_GENERATION_TASKS,
    CORE_TASKS,
    CORE_TASKS_PLUS_LEADERBOARD,
    KEY_GENERATION_TASKS,
    KEY_MULTIPLE_CHOICE_TASKS,
    MMLU_0_SHOT,
    MMLU_5_SHOT,
    MMLU_PRO_5_SHOT,
    OPEN_LM_LEADERBOARD_GEN,
    OPEN_LM_LEADERBOARD_MCQ,
)

logger = logging.getLogger(__name__)

# Wandb project name for evaluations
# Note: Also defined in evalchemy_evaluator.py to avoid circular imports.
WANDB_PROJECT = "marin"


def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (list[EvalTaskConfig]): List of evaluations to run with LM Evaluation Harness.
    """
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            launch_with_ray=True,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )


def _infer_model_name_for_path(model_path: str) -> str:
    """
    Infer model name from model path.
    """
    # path names are like gs://marin-us-central2/checkpoints/dclm_7b2x/hf/dclm_7b0828/dclm_7b0828/step-479999/
    # we want something like: dclm_7b0828_step-479999
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    return "_".join(model_path.split("/")[-2:])


def extract_model_name_and_path(step: ExecutorStep | InputName | str) -> tuple[str, InputName | str]:
    """
    Extract the model name and path from a step.
    """
    if isinstance(step, ExecutorStep):
        model_step_path = output_path_of(step, "hf" if "gcsfuse" not in step.name else "")
        name = step.name
    elif isinstance(step, InputName):
        # `InputName.hardcoded(...)` has `step.step is None`; treat it as a direct path.
        if step.step is None:
            if step.name is None:
                raise ValueError("Invalid InputName: both `step` and `name` are None.")
            model_step_path = step.name
            name = _infer_model_name_for_path(step.name)
        else:
            # If `name` is already set, the InputName refers to a specific subpath under the step's output.
            # Otherwise default to the HF export directory (except for gcsfuse mounts).
            model_step_path = (
                step
                if step.name is not None
                else output_path_of(step.step, "hf" if "gcsfuse" not in step.step.name else "")
            )
            name = step.step.name
    elif isinstance(step, str):
        model_step_path = step
        name = _infer_model_name_for_path(step)
    else:
        raise ValueError(f"Invalid step type: {step}")

    return name, model_step_path


def evaluate_levanter_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Levanter LM Evaluation Harness.
    """
    logger.info(f"Running evals on the following tasks: {evals}")
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed automatically
            model_path=model_path,  # type: ignore
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=discover_latest_checkpoint,
            max_eval_instances=versioned(max_eval_instances),
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
        ),
    )


def default_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v4-8"),
    evals: list[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using LM Evaluation Harness on a step.

    Args:
        step (ExecutorStep | InputName): step to evaluate.
        evals (list[EvalTaskConfig]): List of evals to run- defaults to a set of CORE_TASKS defined in task_configs.py
        max_eval_instances (int): Maximum number of evaluation instances to run.
    """

    # this logic extracts the `ExecutorStep` corresponding to the training step, and get the model path
    name, model_step_path = extract_model_name_and_path(step)

    logger.info(f"Creating default evaluation step for {name}")

    # Default to CORE_TASKS
    if evals is None:
        evals = CORE_TASKS

    logger.info(f"Running evals on the following tasks: {evals}")

    return evaluate_levanter_lm_evaluation_harness(
        name,
        model_step_path,
        evals,
        resource_config,
        max_eval_instances=max_eval_instances,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


def default_base_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    discover_latest_checkpoint: bool = True,
):
    # Add GPQA to CORE_TASKS
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    core_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=CORE_TASKS_PLUS_LEADERBOARD,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(core_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.
    mmlu_0shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_0_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_0shot)

    mmlu_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step,
        resource_config=resource_config,
        evals=(MMLU_PRO_5_SHOT,),
        discover_latest_checkpoint=discover_latest_checkpoint,
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        generation = evaluate_lm_evaluation_harness(
            name,
            model_step_path,
            BASE_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
            discover_latest_checkpoint=discover_latest_checkpoint,
        )

        eval_jobs.append(generation)
    return eval_jobs


def default_sft_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v6e-8"),
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
    run_generation_evals: bool = True,
    apply_chat_template: bool = True,
    use_levanter_inference: bool = False,
):
    # Set up evaluations for core tasks (including GPQA)
    eval_jobs = []
    leaderboard_grouped = default_eval(
        step=step,
        resource_config=resource_config,
        evals=OPEN_LM_LEADERBOARD_MCQ,
        apply_chat_template=apply_chat_template,
    )
    eval_jobs.append(leaderboard_grouped)

    # Run tasks where we report Macro_Avg separately to make sure the macro avg gets computed correctly.

    mmlu_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_5shot)

    mmlu_pro_5shot = default_eval(
        step=step, resource_config=resource_config, evals=(MMLU_PRO_5_SHOT,), apply_chat_template=apply_chat_template
    )
    eval_jobs.append(mmlu_pro_5shot)

    name, model_step_path = extract_model_name_and_path(step)
    if run_generation_evals:
        if use_levanter_inference:
            leaderboard_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                KEY_GENERATION_TASKS,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_levanter_lm_evaluation_harness(
                name,
                model_step_path,
                OPEN_LM_LEADERBOARD_GEN,
                resource_config,
                max_eval_instances=max_eval_instances,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
        else:
            leaderboard_generation = evaluate_lm_evaluation_harness(
                name,
                model_step_path,
                KEY_GENERATION_TASKS,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )

            eval_jobs.append(leaderboard_generation)

            olmo_generation = evaluate_lm_evaluation_harness(
                name,
                model_step_path,
                OPEN_LM_LEADERBOARD_GEN,
                max_eval_instances=max_eval_instances,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=apply_chat_template,
            )
            eval_jobs.append(olmo_generation)
    return eval_jobs


def default_key_evals(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig,
    model_name: str | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = DEFAULT_LM_EVAL_MODEL_KWARGS,
) -> list[ExecutorStep]:
    """
    Create a list of ExecutorSteps to evaluate the model using LM Evaluation Harness on a step.
    """
    name, model_step_path = extract_model_name_and_path(step)

    if model_name is None:
        model_name = name

    stop_token_ids = []
    if "llama3" in model_name:
        stop_token_ids.append(128009)
    elif "olmo" in model_name:
        stop_token_ids.append(100257)

    return [
        evaluate_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_GENERATION_TASKS,
            max_eval_instances=max_eval_instances,
            engine_kwargs=engine_kwargs,
            resource_config=resource_config,
        ),
        evaluate_levanter_lm_evaluation_harness(
            model_name,
            model_step_path,
            KEY_MULTIPLE_CHOICE_TASKS,
            resource_config,
            max_eval_instances=max_eval_instances,
        ),
    ]


def evaluate_evalchemy(
    model_name: str,
    model_path: str,
    evals: Sequence[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Evalchemy.

    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model.
        evals (Sequence[EvalTaskConfig]): Evaluations to run with Evalchemy.
        max_eval_instances (int | None): Maximum number of evaluation instances to run.
        engine_kwargs (dict | None): Additional engine kwargs for vLLM.
        generation_params (dict | None): Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seeds: list[int] for multiple runs with different seeds
        resource_config (ResourceConfig | None): Resource configuration for the job.
        apply_chat_template (bool): Whether to apply chat template.
        wandb_tags (list[str] | None): Tags to add to the WandB run.
        discover_latest_checkpoint (bool): Whether to discover the latest checkpoint.
    """
    # Include task names in the step name to ensure different tasks get different output paths
    task_names = "_".join(sorted(e.name for e in evals))
    return ExecutorStep(
        name=f"evaluation/evalchemy/{model_name}/{task_names}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="evalchemy",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            launch_with_ray=True,
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,
            generation_params=generation_params,
            resource_config=resource_config,
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )


def default_evalchemy_eval(
    step: ExecutorStep | InputName | str,
    resource_config: ResourceConfig = ResourceConfig.with_tpu("v5p-8"),
    evals: Sequence[EvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    generation_params: dict | None = None,
    apply_chat_template: bool = False,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    """
    Create an ExecutorStep to evaluate the model using Evalchemy reasoning benchmarks.

    Args:
        step (ExecutorStep | InputName | str): Step to evaluate.
        resource_config (ResourceConfig): Resource configuration (defaults to v5p-8 TPU).
        evals (list[EvalTaskConfig] | None): List of evals to run. Defaults to EVALCHEMY_CORE_TASKS.
        max_eval_instances (int | None): Maximum number of evaluation instances to run.
        engine_kwargs (dict | None): Additional vLLM engine kwargs (optional for evalchemy).
        generation_params (dict | None): Generation parameters including:
            - temperature: float (e.g., 0.7)
            - top_p: float (e.g., 1.0)
            - max_gen_toks: int (e.g., 32768)
            - seed: int for reproducibility
        apply_chat_template (bool): Whether to apply chat template.
        discover_latest_checkpoint (bool): Whether to discover the latest checkpoint.
    """
    name, model_step_path = extract_model_name_and_path(step)

    logger.info(f"Creating Evalchemy evaluation step for {name}")

    if evals is None:
        evals = EVALCHEMY_CORE_TASKS

    logger.info(f"Running Evalchemy evals on the following tasks: {evals}")

    return evaluate_evalchemy(
        name,
        model_step_path,
        evals,
        max_eval_instances=max_eval_instances,
        engine_kwargs=engine_kwargs or {},  # Pass empty dict to avoid warning
        generation_params=generation_params,
        resource_config=resource_config,
        apply_chat_template=apply_chat_template,
        discover_latest_checkpoint=discover_latest_checkpoint,
    )


def compile_evalchemy_results(steps: list[ExecutorStep], seeds: list[int] | None = None) -> ExecutorStep:
    """
    Compile results from multiple Evalchemy evaluation steps into aggregated metrics.

    Takes a list of ExecutorSteps for evalchemy tasks and compiles the results into a
    single DataFrame, then logs averaged results to wandb.

    Args:
        steps: List of ExecutorSteps from evalchemy evaluations (one per seed).
        seeds: List of seeds used for the evaluations (for wandb config).

    Returns:
        ExecutorStep that compiles and logs aggregated results.
    """
    from marin.execution.executor import OutputName

    def _compile_results_fn(config) -> None:
        """Function that will be executed by the ExecutorStep to compile results."""
        import json
        import re
        import fsspec
        import pandas as pd

        all_results = []
        input_paths = config["input_paths"]
        output_path = config["output_path"]
        seeds_config = config.get("seeds", [])

        logger.info(f"Compiling evalchemy results from {len(input_paths)} input paths")

        if not input_paths:
            raise Exception("No input paths found!")

        fs = fsspec.filesystem("gcs")

        for input_path in input_paths:
            # Normalize path
            base_dir = input_path
            if base_dir.endswith("results.json"):
                base_dir = base_dir.rsplit("/", 1)[0]

            logger.info(f"Loading evalchemy samples from root {base_dir}")

            # Normalize to GCS URL
            if base_dir.startswith("gs://"):
                gcs_root = base_dir
            else:
                gcs_root = "gs://" + base_dir.lstrip("/")

            # Pattern for sample files (evalchemy uses same structure as lm-eval)
            pattern = gcs_root.rstrip("/") + "/*/*/samples_*.jsonl"
            sample_files = fs.glob(pattern)

            if not sample_files:
                logger.warning(f"No samples_*.jsonl files found for input root {base_dir}")
                continue

            for sample_file in sample_files:
                logger.info(f"Reading samples from {sample_file}")
                path_parts = sample_file.split("/")

                # Infer dataset_name from the task directory
                if len(path_parts) >= 3:
                    task_dir = path_parts[-3]
                    if "_" in task_dir:
                        dataset_name = task_dir.rsplit("_", 1)[0]
                    else:
                        dataset_name = task_dir
                else:
                    dataset_name = "unknown_dataset"

                # Infer model_name from directory structure
                if len(path_parts) >= 4:
                    model_dir = path_parts[-4]
                elif len(path_parts) >= 2:
                    model_dir = path_parts[-2]
                else:
                    model_dir = "unknown_model"

                # Strip hash suffix from model_dir
                if "-" in model_dir:
                    model_name = model_dir.rsplit("-", 1)[0]
                else:
                    model_name = model_dir

                # Read JSONL samples
                with fs.open(sample_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            logger.warning(f"Failed to parse JSON line in {sample_file}")
                            continue

                        record["dataset_name"] = dataset_name.lower()
                        record["model_name"] = model_name.lower()
                        all_results.append(record)

        if not all_results:
            raise Exception("No results found in any of the provided steps")

        df = pd.DataFrame(all_results)

        # Extract base model name and seed from model_name
        def extract_base_model_and_seed(model_name):
            """Extract base model name and seed from model_name like 'model-task-seed42'"""
            match = re.search(r'-seed(\d+)(?=-|$)', model_name)
            if match:
                seed = int(match.group(1))
                base_model = model_name[:match.start()]
                return base_model, seed
            return model_name, None

        df[['base_model_name', 'seed']] = df['model_name'].apply(
            lambda x: pd.Series(extract_base_model_and_seed(x))
        )

        # Save compiled results
        results_file = f"{output_path}/compiled_results.json"
        with fsspec.open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        csv_file = f"{output_path}/compiled_results.csv"
        with fsspec.open(csv_file, "w") as f:
            df.to_csv(f, index=False)

        logger.info(f"Compiled results saved to: {results_file}")

        # Compute averaged results across seeds
        accuracy_cols = [col for col in df.columns if col in ['exact_match', 'acc', 'accuracy', 'correct']]

        if accuracy_cols and 'base_model_name' in df.columns and 'dataset_name' in df.columns:
            avg_results = []
            for (base_model, dataset), group in df.groupby(['base_model_name', 'dataset_name']):
                per_seed_accuracies = {}
                for col in accuracy_cols:
                    if col in group.columns:
                        seed_accs = group.groupby('seed')[col].mean()
                        per_seed_accuracies[col] = seed_accs

                result = {
                    'base_model_name': base_model,
                    'dataset_name': dataset,
                    'num_seeds': group['seed'].nunique(),
                    'seeds': sorted(group['seed'].dropna().unique().tolist()),
                }

                for col in accuracy_cols:
                    if col in per_seed_accuracies:
                        seed_accs = per_seed_accuracies[col]
                        result[f'{col}_mean'] = seed_accs.mean()
                        result[f'{col}_std'] = seed_accs.std()
                        result[f'{col}_per_seed'] = seed_accs.to_dict()

                avg_results.append(result)

            avg_df = pd.DataFrame(avg_results)

            # Save averaged results
            avg_results_file = f"{output_path}/averaged_results.json"
            with fsspec.open(avg_results_file, "w") as f:
                json.dump(avg_results, f, indent=2)

            avg_csv_file = f"{output_path}/averaged_results.csv"
            with fsspec.open(avg_csv_file, "w") as f:
                avg_df.to_csv(f, index=False)

            logger.info(f"Averaged results saved to: {avg_results_file}")
            logger.info(f"Averaged results:\n{avg_df.to_string()}")

            # Log averaged results to wandb - one run per model
            try:
                import wandb

                num_seeds = len(seeds_config) if seeds_config else avg_df['num_seeds'].max()

                for base_model in avg_df['base_model_name'].unique():
                    model_df = avg_df[avg_df['base_model_name'] == base_model]

                    # Wandb run name: evalchemy-{model}-averaged-{n}seeds (lowercase)
                    wandb_run_name = f"evalchemy-{base_model.lower()}-averaged-{num_seeds}seeds"

                    wandb.init(
                        project=WANDB_PROJECT,
                        name=wandb_run_name,
                        job_type="eval",
                        tags=["evalchemy", "averaged-results", base_model.lower()[:64]],
                        config={
                            "base_model_name": base_model,
                            "num_seeds": num_seeds,
                            "seeds": seeds_config,
                        },
                        reinit=True,
                    )

                    # Log averaged metrics for each dataset
                    for _, row in model_df.iterrows():
                        dataset = row['dataset_name']
                        for col in accuracy_cols:
                            mean_col = f'{col}_mean'
                            std_col = f'{col}_std'
                            if mean_col in row and std_col in row:
                                wandb.log({
                                    f"{dataset}/{col}_mean": row[mean_col],
                                    f"{dataset}/{col}_std": row[std_col],
                                })

                    wandb.log({"averaged_results": wandb.Table(dataframe=model_df)})
                    wandb.finish()
                    logger.info(f"Averaged results for {base_model} logged to wandb as '{wandb_run_name}'")

            except Exception as e:
                logger.warning(f"Failed to log averaged results to wandb: {e}")
        else:
            logger.warning("Could not compute averaged results: missing accuracy columns or grouping columns")

    # Create input paths from steps
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")

    return ExecutorStep(
        name="evaluation/evalchemy/compile_results",
        fn=_compile_results_fn,
        config={"input_paths": input_paths, "output_path": output_path, "seeds": seeds or []},
        description="Compile results from multiple evalchemy evaluation steps",
    )
