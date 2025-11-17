import logging
import os

# Disable HuggingFace logging
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.WARNING)

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig

# from experiments.models import llama_3_1_8b_instruct, tulu_3_1_8b_instruct
# from experiments.tootsie.exp1237_starling_sft import mixture_sft_deeper_starling
from marin.execution.executor import executor_main, ExecutorStep
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import this_output_path
from experiments.evals.resource_configs import ResourceConfig
from marin.utils import fsspec_exists, fsspec_glob, fsspec_mtime

resource_config = ResourceConfig(num_tpu=4, tpu_type="TPU-v4-8", strategy="STRICT_PACK")

"""
Note for people trying to do evals:
- The difference between evaluate_lm_evaluation_harness and evaluate_levanter_lm_evaluation_harness is that the latter uses the vLLM engine and the former uses the Levanter engine.
- The levanter engine can only compute loglikelihoods and not completions. So, we have to use use lm_evaluation_harness for typical evals.

"""
EVAL_TASKS = [
    # >>> 3-shot tests in legal domain
    # EvalTaskConfig("agieval_lsat_ar", num_fewshot=3, task_alias="agieval_lsat_ar_3shot"),
    # >>> AIME dataset
    # EvalTaskConfig("aime", num_fewshot=0, task_alias="aime_0shot"),
    EvalTaskConfig("aime24", num_fewshot=0, task_alias="aime24_0shot"),
    EvalTaskConfig("aime25", num_fewshot=0, task_alias="aime25_0shot"),
    # >>> 10-shot, four-way MCQ questions involving grade 3-9 basic science
    # EvalTaskConfig("arc_easy", num_fewshot=10, task_alias="arc_easy_10shot"),
    # >>> Harder version of arc_easy
    # EvalTaskConfig("arc_challenge", num_fewshot=10, task_alias="arc_challenge_10shot"),
    # >>> answer yes/no questions based on a passage
    # EvalTaskConfig("boolq", num_fewshot=10, task_alias="boolq_10shot"),
    # >>> 5-way multiple-choice questions based on common-sense, everyday scenarios
    # EvalTaskConfig("commonsense_qa", num_fewshot=10, task_alias="commonsense_qa_10shot"),
    # >>> use causal reasoning to predict the correct outcome of a given scenario
    # EvalTaskConfig("copa", num_fewshot=0, task_alias="copa_0shot"),
    # EvalTaskConfig(name="drop", num_fewshot=0, task_alias="drop_0shot"),
    # EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot"),
    # >>> 4-way multiple choice commonsense reasoning dataset
    # EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    # EvalTaskConfig("hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),
    # >>> 4-way MCQ commonsense reasoning dataset
    # EvalTaskConfig(name="humaneval_instruct", num_fewshot=0, task_alias="humanevalInstruct_0shot"),
    # EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh_3shot"),
    # EvalTaskConfig(name="ifeval", num_fewshot=0, task_alias="ifeval_0shot"),
    # >>> predict the endings of text passages
    # EvalTaskConfig("lambada_openai", num_fewshot=0, task_alias="lambada_openai_0shot"),
    # >>> Leaderboard tasks
    # EvalTaskConfig("leaderboard_gpqa", num_fewshot=0, task_alias="gpqa_0shot"),
    # EvalTaskConfig("leaderboard_ifeval", num_fewshot=0, task_alias="lb_ifeval_0shot"),
    # EvalTaskConfig("leaderboard_math_hard", num_fewshot=4, task_alias="lb_math_4shot"),
    # EvalTaskConfig("leaderboard_mmlu_pro", num_fewshot=5, task_alias="mmlu_5shot"),
    # EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
    # EvalTaskConfig("mmlu", num_fewshot=0, task_alias="mmlu_0shot"),
    # EvalTaskConfig("mmlu", num_fewshot=1, task_alias="mmlu_1shot"),
    # >>> 4-way multiple choice question answering task that requires multi-step reasoning
    # EvalTaskConfig("openbookqa", num_fewshot=0, task_alias="openbookqa_0shot"),
    # >>> answer questions based on a passage
    # EvalTaskConfig("piqa", num_fewshot=10, task_alias="piqa_10shot"),
    # EvalTaskConfig("truthfulqa_mc2", num_fewshot=6, task_alias="truthqa_6shot"),
    # >>> Winograd Schema Challenge
    # EvalTaskConfig("wsc273", num_fewshot=0, task_alias="wsc273_0shot"),
    # >>> Winograd challenge, extended to more domains
    # EvalTaskConfig("winogrande", num_fewshot=0, task_alias="winogrande_0shot"),
]


def discover_checkpoints(base_path: str, initial_glob_pattern: str, is_checkpoint_dir_pattern: list[str]) -> list[str]:
    """
    Discover the checkpoints in the given path, sorted by the last modified time. (Most recent last)
    Args:
        base_path:  Fsspec Path to the directory containing the checkpoints, possibly in nested directories.
        initial_glob_pattern:  Initial glob pattern to use to find the checkpoints.
        is_checkpoint_dir_pattern:  List of patterns to check if a directory is a checkpoint directory.
    """
    def _is_checkpoint_dir(path):
        for checkpoint_dir_pattern in is_checkpoint_dir_pattern:
            if not fsspec_exists(os.path.join(path, checkpoint_dir_pattern)):
                return False
        return True

    paths = fsspec_glob(os.path.join(base_path, initial_glob_pattern))
    paths.sort(key=lambda path: fsspec_mtime(path))
    checkpoint_paths = [os.path.dirname(path) for path in paths if _is_checkpoint_dir(os.path.dirname(path))]
    return checkpoint_paths


def list_checkpoint_paths(root_path: str) -> list[str]:
    """
    List all checkpoint models in the given root path.

    Example:
    >> input: gs://marin-us-central2/checkpoints/sft/qwen25_7b_sft_nemotron_and_openthoughts3
    >> output: [
    "gs://marin-us-central2/checkpoints/sft/qwen25_7b_sft_nemotron_and_openthoughts3/hf/step-1540000",
    "gs://marin-us-central2/checkpoints/sft/qwen25_7b_sft_nemotron_and_openthoughts3/hf/step-1550000",
    """
    # For HuggingFace checkpoints, look for config.json and tokenizer_config.json
    # instead of metadata.json and model (which are for Levanter checkpoints)
    checkpoint_paths = discover_checkpoints(root_path, "**/config.json", ["config.json", "tokenizer_config.json"])
    logging.info(f"Checkpoint paths: {checkpoint_paths}")
    return checkpoint_paths
    
def get_model_checkpoints(model_configs: list[dict]) -> list[dict]:
    """
    Get all models in the given root path.
    """
    model_checkpoints = []
    for model_config in model_configs:
        checkpoint_paths = list_checkpoint_paths(model_config["path"])
        for checkpoint_path in checkpoint_paths:
            model_checkpoints.append({
                "name": f"{model_config['name'].replace('/', '_')}-{checkpoint_path.split('/')[-1].replace('-', '_')}",
                "path": checkpoint_path,
                "apply_chat_template": model_config["apply_chat_template"],
                "max_length": model_config["max_length"],
            })
    return model_checkpoints

MODEL_CONFIGS = [
    {
        "name": "sft/qwen25_7b_sft_openthoughts3",
        "path": "gs://marin-us-central2/checkpoints/sft/qwen25_7b_sft_openthoughts3",
        "apply_chat_template": True,
        "max_length": 8192,
    },
    {
        "name": "sft/qwen25_7b_sft_nemotron_and_openthoughts3",
        "path": "gs://marin-us-central2/checkpoints/sft/qwen25_7b_sft_nemotron_and_openthoughts3",
        "apply_chat_template": True,
        "max_length": 8192,
    }
]

def compile_results(steps: list[ExecutorStep]) -> ExecutorStep:
    """
    Takes in a list of ExecutorSteps for lm-eval tasks and compiles the results into a single DataFrame.
    Reads the results.json files from each step's output path, extracts the 'results' field,
    combines them into a DataFrame, and logs the results.
    """
    import json
    import logging
    import pandas as pd
    import fsspec
    from marin.execution.executor import InputName, OutputName
    
    logger = logging.getLogger(__name__)
    
    def _compile_results_fn(config) -> None:
        """Function that will be executed by the ExecutorStep to compile results."""
        all_results = []
        
        # Extract parameters from config
        input_paths = config["input_paths"]
        output_path = config["output_path"]
        
        logger.info(f"Input paths: {input_paths}")
        
        if not input_paths:
            raise Exception("No input paths found!")
        
        # Read results from each step's output path
        for i, input_path in enumerate(input_paths):
            logger.info(f"Loading results from {input_path}")
            
            # Read the JSON file from the step's output path
            with fsspec.open(input_path, "r") as f:
                data = json.load(f)
            
            # Extract the 'results' field
            if "results" in data:
                results = data["results"]
                # Add step identifier and model name
                for task_name, task_results in results.items():
                    task_results["task_name"] = task_name
                    
                    # Extract model name from the input path
                    # The path format should be: gs://bucket/evaluation/lm_evaluation_harness_levanter/levanter_lmeval_{task}_{model_name}-{hash}/results.json
                    path_parts = input_path.split('/')
                    if len(path_parts) > 1:
                        # Get the directory name (second to last part)
                        dir_name = path_parts[-2]
                        if '-' in dir_name:
                            # Remove the hash part
                            model_name = dir_name.rsplit('-', 1)[0]
                            # Remove the task prefix to get just the model name
                            if '_' in model_name:
                                # Split on the first underscore to remove 'levanter_lmeval_{task}_' prefix
                                parts = model_name.split('_', 2)
                                if len(parts) >= 3:
                                    model_name = parts[2]  # Get everything after 'levanter_lmeval_{task}_'
                                else:
                                    model_name = model_name
                        else:
                            model_name = dir_name
                    else:
                        model_name = f"unknown_model_{i}"
                    
                    task_results["model_name"] = model_name
                    all_results.append(task_results)
            else:
                logger.warning(f"No 'results' field found in {input_path}")
        
        if not all_results:
            raise Exception("No results found in any of the provided steps")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save compiled results
        results_file = f"{output_path}/compiled_results.json"
        with fsspec.open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        csv_file = f"{output_path}/compiled_results.csv"
        with fsspec.open(csv_file, "w") as f:
            df.to_csv(f, index=False)
        
        logger.info(f"Compiled results saved to: {results_file}")
    
    # Create input paths and output path
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")
    
    return ExecutorStep(
        name="scratch/chiheem/compile_results",
        fn=_compile_results_fn,
        config={"input_paths": input_paths, "output_path": output_path},
        description="Compile results from multiple lm-eval steps into a single DataFrame"
    )




if __name__ == "__main__":
    # Quiet ray logs for this experiment
    # logging.getLogger("ray").setLevel(logging.WARNING)

    # NOTE(chiheem 2025-10-01): We may want to run the lm-eval tasks as separate steps so that we can avoid
    # `out of pages` error.
    all_steps = []

    MODELS = get_model_checkpoints(MODEL_CONFIGS)

    for model in MODELS[0:1]:
        for eval_task in EVAL_TASKS:
            lm_eval_task_step = evaluate_levanter_lm_evaluation_harness(
                model["name"],
                model["path"],
                evals=[eval_task],
                max_eval_instances=None,
                resource_config=resource_config,
                apply_chat_template=model["apply_chat_template"] if "apply_chat_template" in model else False,
                max_length=model["max_length"] if "max_length" in model else 2**14,  # Total length of the prompt + max_gen_toks
                generation_kwargs={"max_gen_toks": max(1, model["max_length"]-2048), "temperature": 0.0},  # Override lm-eval generation parameters
            )
            all_steps.append(lm_eval_task_step)

    # Collect all steps properly
    # all_steps.extend(default_sft_eval(mixture_sft_deeper_starling))
    # all_steps.extend(default_sft_eval(llama_3_1_8b_instruct))
    # all_steps.extend(default_sft_eval(tulu_3_1_8b_instruct))
    # all_steps.append(helm_eval)  # Commented out to avoid TPU resource conflict

    # Add compile results step
    compile_step = compile_results(all_steps)
    all_steps.append(compile_step)

    executor_main(steps=all_steps)

    print(f"Output to: {compile_step.output_path}")
