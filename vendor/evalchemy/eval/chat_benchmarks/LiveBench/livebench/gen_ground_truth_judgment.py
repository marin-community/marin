"""Grade a single LiveBench response against its reference."""

import json
import os
import time

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

from eval.chat_benchmarks.LiveBench.livebench.common import MatchSingle

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
