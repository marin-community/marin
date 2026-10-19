"""
Usage:
python gen_ground_truth_judgment.py --bench-name live_bench --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import os
import re
import glob

import nltk
import numpy as np
from tqdm import tqdm

# todo: find a better solution than all these imports.
from eval.chat_benchmarks.LiveBench.livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.data_analysis.cta.utils import cta_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
from eval.chat_benchmarks.LiveBench.livebench.process_results.reasoning.spatial.utils import spatial_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.math.math_competitions.utils import mathcontest_process_results, aime_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.writing.typos.utils import typos_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
from eval.chat_benchmarks.LiveBench.livebench.process_results.coding.utils import LCB_generation_process_results
from eval.chat_benchmarks.LiveBench.livebench.process_results.instruction_following.utils import instruction_following_process_results

from eval.chat_benchmarks.LiveBench.livebench.common import (
    LIVE_BENCH_RELEASES,
    load_questions,
    load_questions_jsonl,
    load_model_answers,
    check_data,
    get_model_list,
    make_match_single,
    MatchSingle,
    get_categories_tasks,
    LIVE_BENCH_DATA_SUPER_PATH,
    model_display_name,
)


def reorg_output_file(output_file):
    """De-duplicate and sort by question id and model"""
    judgments = {}
    with open(output_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            model = json.loads(l)["model"]
            key = (qid, model)
            judgments[key] = l

    keys = sorted(list(judgments.keys()))
    with open(output_file, "w") as fout:
        for key in keys:
            fout.write(judgments[key])


def play_a_match_gt(match: MatchSingle, output_file: str, debug=False):
    """
    Evaluate a model's answer to a question.

    Args:
        match: An object containing the question, model name, and model answer.
        output_file: The path to which the judgment should be outputted.

    Returns:
        result: The judgment, containing the question id, task name, model name, score, turn, timestamp, and category name
    """
    question, model, answer = (
        match.question,
        match.model,
        match.answer,
    )
    coding_test_case_tasks = ["coding_completion", "LCB_generation"]
    if (
        "ground_truth" not in question
        and "reference" not in question
        and question["task"] not in coding_test_case_tasks
        and question["category"] != "instruction_following"
    ):
        # aside from coding and instruction following tasks, all questions should contain the ground truth answer
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    llm_answer = answer["choices"][0]["turns"][-1]
    score = 0
    category = None

    # todo: find a better solution than a long if statement.
    splits = task_or_subtask.split("_")
    try:
        if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
            score = mathcontest_process_results(ground_truth, llm_answer, question_text, debug)
            category = "math"
        elif splits[0] == "aime":
            score = aime_process_results(ground_truth, llm_answer, debug)
            category = "math"
        elif splits[0] in ["imo", "usamo"]:
            score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True, debug=debug)
            category = "math"
        elif task_or_subtask == "cta":
            score = cta_process_results(ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablereformat":
            score = table_process_results(question_text, ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablejoin":
            score = joinmap_process_results(question_text, ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif "amps_hard" in task_or_subtask:
            score = amps_hard_process_results(ground_truth, llm_answer, debug)
            category = "math"
        elif task_or_subtask == "web_of_lies_v2":
            score = web_of_lies_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "house_traversal":
            score = house_traversal_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "zebra_puzzle":
            zebra_evaluator = get_zebra_puzzle_evaluator(question["livebench_release_date"])
            score = zebra_evaluator(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "spatial":
            score = spatial_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "typos":
            score = typos_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "connections":
            connections_evaluator = get_connections_puzzle_evaluator(question["livebench_release_date"])
            score = connections_evaluator(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "plot_unscrambling":
            score = plot_unscrambling_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask in coding_test_case_tasks:
            # use entire question object, because there are test cases inside.
            try:
                score = LCB_generation_process_results(question, llm_answer, debug)
            except Exception as e:
                if debug:
                    print(f"Warning: Error evaluating coding question {question['question_id']}: {str(e)}")
                # Return score of 0 for failed evaluation but continue processing other questions
                score = 0
            category = "coding"
        else:
            raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")
    except Exception as e:
        # Log the error but continue with other questions
        print(f"Error evaluating question {question['question_id']}: {str(e)}")
        score = 0
        # Try to determine category even if evaluation failed
        if task_or_subtask in coding_test_case_tasks:
            category = "coding"
        elif any(x in task_or_subtask for x in ["amc", "smc", "aime", "imo", "usamo", "amps_hard"]):
            category = "math"
        elif task_or_subtask in ["web_of_lies_v2", "house_traversal", "zebra_puzzle", "spatial"]:
            category = "reasoning"
        else:
            category = "unknown"

    if not category:
        category = "unknown"  # Fallback category instead of raising error

    question_id = question["question_id"]
    turn = 1
    result = {
        "question_id": question_id,
        "task": task,
        "model": model,
        "score": score,
        "turn": turn,
        "tstamp": time.time(),
        "category": category,
    }
    if "subtask" in question.keys():
        result["subtask"] = question["subtask"]
    print(f"question: {question_id}, turn: {turn}, model: {model}, " f"score: {score}, ")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def gen_judgments(
    parallel: int,
    questions: list[dict],
    output_file: str,
    answer_dir: str,
    model_list: list[str] | None,
    remove_existing_file: bool,
    bench_name: str,
    debug=False,
):
    """
    Evaluate answers to questions for all the given models.
    """
    # Convert to absolute paths
    output_file = os.path.abspath(output_file)
    answer_dir = os.path.abspath(answer_dir)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    print("models:", model_answers.keys())

    if model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = model_list

    play_a_match_func = play_a_match_gt

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file and os.path.exists(output_file) and remove_existing_file:
        try:
            os.remove(output_file)
        except FileNotFoundError:
            # File doesn't exist, which is fine since we want to remove it anyway
            pass

    make_match_func = make_match_single
    check_data(questions, model_answers, models)

    # Make matches
    matches = []
    matches += make_match_func(
        questions,
        models,
        model_answers,
    )

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    if "instruction_following" in bench_name:
        # instruction following tasks are evaluated differently from all other tasks
        nltk.download("punkt")
        nltk.download("punkt_tab")
        task_name = matches[0].question["task"]

        if model_list is None:
            models = get_model_list(answer_dir)
        else:
            models = model_list

        for model_id in models:
            scores = instruction_following_process_results(questions, model_answers, task_name, model_id, debug)
            for item in scores:
                question_id = item["question_id"]
                score = item["score"]
                turn = 1
                result = {
                    "question_id": question_id,
                    "task": task_name,
                    "model": model_id,
                    "score": score,
                    "turn": turn,
                    "tstamp": time.time(),
                    "category": "instruction_following",
                }
                print(f"question: {question_id}, turn: {turn}, model: {model_id}, " f"score: {score}, ")

                if output_file:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    else:
        # Play matches
        if parallel == 1:
            for match in tqdm(matches):
                results = play_a_match_func(match, output_file=output_file, debug=debug)
        else:

            def play_a_match_wrapper(match):
                return play_a_match_func(match, output_file=output_file, debug=debug)

            np.random.seed(0)
            np.random.shuffle(matches)

            with ThreadPoolExecutor(parallel) as executor:
                for match in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                    pass

    # De-duplicate and sort judgment file
    reorg_output_file(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model answers compared to ground truths")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set. Defaults to 'live_bench', or all tasks in the benchmark. Specify e.g. live_bench/reasoning/web_of_lies_v2 to generate only for that task.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument("--parallel", type=int, default=1, help="The number of concurrent API calls.")
    parser.add_argument("--first-n", type=int, help="A debug option. Only run the first `n` judgments.")
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument("--question-end", type=int, help="A debug option. The end index of questions.")
    parser.add_argument(
        "--remove-existing-file", action="store_true", default=False, help="Remove existing judgment file."
    )
    parser.add_argument(
        "--question-source",
        type=str,
        default="huggingface",
        help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will gather local jsonl files at data/{bench_name}/**/question.jsonl to permit tweaking or writing custom questions.",
    )
    parser.add_argument(
        "--livebench-release-option",
        type=str,
        default=max(LIVE_BENCH_RELEASES),
        choices=sorted(LIVE_BENCH_RELEASES),
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Print debug information.")
    parser.add_argument(
        "--question-id", type=str, nargs="+", default=None, help="A debug option. The question id to be evaluated."
    )
    parser.add_argument(
        "--model-display-name",
        type=str,
        nargs="+",
        default=None,
        help="The display name of the model(s). If provided, will be used to name the output file. Will match order to --model-list. If not provided, will be generated from --model-list.",
    )
    args = parser.parse_args()

    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")

    release_set = set([r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option])

    if args.model is None:
        model_list = None
    else:
        # model_list = [get_model(model_name).display_name for model_name in args.model_list]
        model_list = []
        for i, model_name in enumerate(args.model):
            if args.model_display_name is not None:
                model_list.append(args.model_display_name[i])
            else:
                model_list.append(model_display_name(model_name))

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(
                    categories[category_name], release_set, args.livebench_release_option, task_name, args.question_id
                )
                if args.first_n:
                    questions = questions[: args.first_n]
                questions = questions[args.question_begin : args.question_end]

                task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                output_file = f"data/{task_full_name}/model_judgment/ground_truth_judgment.jsonl"

                answer_dir = f"data/{task_full_name}/model_answer/"  # expected location of model answers

                gen_judgments(
                    parallel=args.parallel,
                    questions=questions,
                    output_file=output_file,
                    answer_dir=answer_dir,
                    model_list=model_list,
                    remove_existing_file=args.remove_existing_file,
                    bench_name=task_full_name,
                    debug=args.debug,
                )

    elif args.question_source == "jsonl":
        list_of_question_files = []
        original_question_file = f"data/{args.bench_name}/question.jsonl"
        if os.path.exists(original_question_file):
            list_of_question_files = [original_question_file]
        else:
            list_of_question_files = glob.glob(f"data/{args.bench_name}/**/question.jsonl", recursive=True)
        for question_file in list_of_question_files:
            print("questions from", question_file)
            questions = load_questions_jsonl(
                question_file, release_set, args.livebench_release_option, args.question_id
            )
            if args.first_n:
                questions = questions[: args.first_n]

            questions = questions[args.question_begin : args.question_end]

            bench_name = os.path.dirname(question_file).replace("data/", "")

            output_file = f"data/{bench_name}/model_judgment/ground_truth_judgment.jsonl"
            answer_dir = f"data/{bench_name}/model_answer/"  # expected location of model answers
            if len(questions) > 0:
                gen_judgments(
                    parallel=args.parallel,
                    questions=questions,
                    output_file=output_file,
                    answer_dir=answer_dir,
                    model_list=model_list,
                    remove_existing_file=args.remove_existing_file,
                    bench_name=bench_name,
                    debug=args.debug,
                )

    else:
        raise ValueError(f"Bad question source {args.question_source}.")
