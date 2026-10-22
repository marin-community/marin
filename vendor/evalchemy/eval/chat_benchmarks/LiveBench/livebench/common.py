"""LiveBench dataset loading and per-question grading records."""

import dataclasses
import json
from datetime import datetime
from typing import Optional

from datasets import Dataset, load_dataset

LIVE_BENCH_HF_ORGANIZATION = "livebench"


LIVE_BENCH_CATEGORIES = [
    "coding",
    "data_analysis",
    "instruction_following",
    "math",
    "reasoning",
    "language",
]


LIVE_BENCH_RELEASES = {"2024-07-26", "2024-06-24", "2024-08-31", "2024-11-25"}


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
