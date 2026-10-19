"""
Common data structures and utilities.
"""

import dataclasses
from datasets import load_dataset, Dataset
from datetime import datetime
import glob
import json
import os

import re
from typing import Optional

from eval.chat_benchmarks.LiveBench.livebench.model.api_models import get_model


# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Huggingface and dataset-related constants
LIVE_BENCH_HF_ORGANIZATION = "livebench"
LIVE_BENCH_DATA_SUPER_PATH = "live_bench"
LIVE_BENCH_CATEGORIES = [
    "coding",
    "data_analysis",
    "instruction_following",
    "math",
    "reasoning",
    "language",
]
LIVE_BENCH_RELEASES = {"2024-07-26", "2024-06-24", "2024-08-31", "2024-11-25"}


def model_display_name(model_name: str) -> str:
    return get_model(model_name).display_name


@dataclasses.dataclass
class MatchSingle:
    """
    A helper dataclass for storing a question, model name, and model answer all together.
    Also stores a reference answer, if provided, and whether the question involves multiple turns.
    """

    question: dict
    model: str
    answer: dict
    ref_answer: dict = None
    multi_turn: bool = False


def get_categories_tasks(bench_name: str):
    """
    Retrieve task categories and tasks themselves for a subset of LiveBench from HuggingFace.
    If bench_name='live_bench', will include all categories (coding, data_analysis, math, etc.).
    If bench_name='live_bench/{category_name}', will only include tasks in category category_name.
    If bench_name='live_bench/{category_name}/{task_name}', will only include the task task_name.

    Args:
        bench_name: The 'path' to the desired subset.

    Returns:
        categories: A dictionary mapping each category name to a corresponding HuggingFace dataset

        tasks: A dictionary mapping each category name to the list of tasks in that category
    """
    split_bench_name = bench_name.rstrip("/").split("/")
    if len(split_bench_name) == 1:
        # specify entire bench

        categories = {category_name: get_hf_dataset(category_name) for category_name in LIVE_BENCH_CATEGORIES}

        tasks = {
            category_name: get_tasks_from_hf_category(categories[category_name])
            for category_name in LIVE_BENCH_CATEGORIES
        }

    else:
        # specify a category or task
        category_name = split_bench_name[1]

        categories = {category_name: get_hf_dataset(category_name)}

        if len(split_bench_name) == 2:
            tasks = {category_name: get_tasks_from_hf_category(categories[category_name])}
        else:
            assert len(split_bench_name) == 3
            task_name = split_bench_name[2]
            tasks = {category_name: [task_name]}
    return categories, tasks


def get_hf_dataset(dataset_name: str, split="test"):
    """Load a dataset from HuggingFace using the given split."""
    return load_dataset(f"{LIVE_BENCH_HF_ORGANIZATION}/{dataset_name}", split=split)


def get_tasks_from_hf_category(category: Dataset):
    """Retrieve the set of task names for a category."""
    return list(set(category["task"]))


def load_answers_judgments():
    model_judgment_dataset = get_hf_dataset("model_judgment", split="leaderboard")
    model_answer_dataset = get_hf_dataset("model_answer", split="leaderboard")

    model_judgment = {
        category_name: [
            example for example in model_judgment_dataset.filter(lambda row: row["category"] == category_name)
        ]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    model_answer = {
        category_name: [
            example for example in model_answer_dataset.filter(lambda row: row["category"] == category_name)
        ]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    return model_answer, model_judgment


def load_questions(
    category: Dataset,
    livebench_releases: set = LIVE_BENCH_RELEASES,
    livebench_release: Optional[str] = None,
    task_name: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Load questions from a huggingface dataset.
    Filter based on question release date and task and limit results to a certain range of indices.

    Args:
        category: The Dataset from which to parse questions
        livebench_releases: A set of valid release dates. Questions with other release dates will be filtered out.
        livebench_release: The current livebench release. If specified, questions that have been removed prior to this release will be filtered out.
        task_name: The desired task within the category. If specified, only questions for this task will be returned.
        question_ids: A list of question ids to include. If None, all questions will be included.
    """
    if task_name is not None:
        questions = [example for example in category.filter(lambda row: row["task"] == task_name)]
    else:
        questions = list(category)
    for q in questions:
        if "livebench_release_date" in q.keys() and isinstance(q["livebench_release_date"], datetime):
            q["livebench_release_date"] = datetime.strftime(q["livebench_release_date"], "%Y-%m-%d")
        if "livebench_removal_date" in q.keys() and isinstance(q["livebench_removal_date"], datetime):
            q["livebench_removal_date"] = datetime.strftime(q["livebench_removal_date"], "%Y-%m-%d")
        if "release_date" in q.keys() and isinstance(q["release_date"], datetime):
            q["release_date"] = datetime.strftime(q["release_date"], "%Y-%m-%d")
        if (
            "original_json" in q.keys()
            and "contest_date" in q["original_json"].keys()
            and isinstance(q["original_json"]["contest_date"], datetime)
        ):
            q["original_json"]["contest_date"] = datetime.strftime(
                q["original_json"]["contest_date"], "%Y-%m-%d %H:%M:%S"
            )
    questions = [q for q in questions if q["livebench_release_date"] in livebench_releases]
    if livebench_release is not None:
        questions = [
            q for q in questions if q["livebench_removal_date"] == "" or q["livebench_removal_date"] > livebench_release
        ]

    if question_ids is not None:
        questions = [q for q in questions if q["question_id"] in question_ids]
    return questions


def load_questions_jsonl(
    question_file: str,
    livebench_releases: set = LIVE_BENCH_RELEASES,
    livebench_release: Optional[str] = None,
    question_ids: Optional[list[str]] = None,
):
    """
    Load questions from a jsonl file.
    Filter based on question release date and limit results to a certain range of indices.

    Args:
        question_file: The filename of the question file
        livebench_releases: A set of valid release dates. Questions with other release dates will be filtered out.
        livebench_release: The current livebench release. If specified, questions that have been removed prior to this release will be filtered out.\
        question_ids: A list of question ids to include. If None, all questions will be included.
    """
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))

    questions = [q for q in questions if q["livebench_release_date"] in livebench_releases]
    if livebench_release is not None:
        questions = [
            q for q in questions if q["livebench_removal_date"] == "" or q["livebench_removal_date"] > livebench_release
        ]
    if question_ids is not None:
        questions = [q for q in questions if q["question_id"] in question_ids]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers from answer_dir.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[: -len(".jsonl")]
        model_name = model_display_name(model_name)
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def reorg_answer_file(answer_file):
    """Sort the entires in the file answer_file by question id and de-duplicate"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def make_match_single(
    questions: list[dict],
    models: list[str],
    model_answers,
    multi_turn=False,
):
    """
    Create MatchSingle objects {question, model_name, model_answer, multi_turn} for each question
    in questions and each model in models.
    """
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]

            matches.append(MatchSingle(dict(q), m, a, multi_turn=multi_turn))
    return matches


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def check_data(questions, model_answers, models):
    """Check that all models have answers for all questions"""
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert q["question_id"] in m_answer, f"Missing model {m}'s answer to Question {q['question_id']}"


def get_model_list(answer_dir):
    """Get list of models for which there are answer files in answer_dir"""
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names


def find_last_question_id(answer_file):
    id = None
    with open(answer_file, "r") as fin:
        for line in fin:
            qid = json.loads(line)["question_id"]
            id = qid
    return id
