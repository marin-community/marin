"""
Usage:
python3 show_livebench_result.py
"""

import argparse
import pandas as pd
import glob
import os

from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions, load_questions_jsonl
from livebench.model.api_models import get_model


def display_result_single(args):
    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
    print(f"Using release {args.livebench_release_option}")
    release_set = set([r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option])

    if args.input_file is None:
        # read all judgments for bench_name
        input_files = []
        for bench in args.bench_name:
            files = glob.glob(f"data/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
            input_files += files
    else:
        # read only the judgments in input_file
        input_files = args.input_file

    # categories, tasks = get_categories_tasks(args.bench_name)
    categories = {}
    tasks = {}
    for bench in args.bench_name:
        bench_cats, bench_tasks = get_categories_tasks(bench)
        categories.update(bench_cats)
        for k, v in bench_tasks.items():
            if k in tasks and isinstance(tasks[k], list):
                tasks[k].extend(v)
            else:
                tasks[k] = v
    print(tasks)

    tasks_set = set([task for task_list in tasks.values() for task in task_list])

    questions_all = []
    if args.question_source == "huggingface":
        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(
                    categories[category_name], release_set, args.livebench_release_option, task_name, None
                )
                questions_all.extend(questions)
    elif args.question_source == "jsonl":
        for bench in args.bench_name:
            list_of_question_files = []
            original_question_file = f"data/{bench}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
                questions_all.extend(questions)

    print("loaded ", len(questions_all), " questions")
    question_id_set = set([q["question_id"] for q in questions_all])

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "task", "category", "question_id"]]
    df = df[df["score"] != -1]
    df = df[df["question_id"].isin(question_id_set)]
    df["model"] = df["model"].str.lower()
    df["score"] *= 100

    if args.model_list is not None:
        model_list = [get_model(x).display_name for x in args.model_list]
        df = df[df["model"].isin([x.lower() for x in model_list])]
        model_list_to_check = model_list
    else:
        model_list_to_check = set(df["model"])
    for model in model_list_to_check:
        df_model = df[df["model"] == model]

        if len(df_model) < len(questions_all):
            print(
                "removing model",
                model,
                "has missing",
                len(questions_all) - len(df_model),
                "judgments - has ",
                len(df_model),
            )
            missing_tasks = set()
            for task in tasks_set:
                if len(df_model[df_model["task"] == task]) != len([q for q in questions_all if q["task"] == task]):
                    missing_tasks.add(task)
            print("missing judgments in ", missing_tasks)
            df = df[df["model"] != model]
            # raise ValueError(f'Invalid result, missing judgments (and possibly completions) for {len(questions_all) - len(df_model)} questions for model {model}.')

    df.to_csv("df_raw.csv")

    print("\n########## All Tasks ##########")
    df_1 = df[["model", "score", "task"]]
    df_1 = df_1.groupby(["model", "task"]).mean()
    df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["task"], aggfunc="sum")
    if args.show_average:
        df_1.loc["average"] = df_1.mean()
    df_1 = df_1.round(3)
    print(df_1.sort_values(by="model")[:60])
    df_1.to_csv("all_tasks.csv")

    print("\n########## All Groups ##########")
    df_1 = df[["model", "score", "category", "task"]]
    df_1 = df_1.groupby(["model", "task", "category"]).mean().groupby(["model", "category"]).mean()
    df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["category"], aggfunc="sum")

    df_1["average"] = df_1.mean(axis=1)
    first_col = df_1.pop("average")
    df_1.insert(0, "average", first_col)
    df_1 = df_1.sort_values(by="average", ascending=False)
    df_1 = df_1.round(1)
    if args.show_average:
        df_1.loc["average"] = df_1.mean()
    print(df_1[:60])
    df_1.to_csv("all_groups.csv")

    for column in df_1.columns[1:]:
        max_value = df_1[column].max()
        df_1[column] = df_1[column].apply(lambda x: f"\\textbf{{{x}}}" if x == max_value else x)
    df_1.to_csv("latex_table.csv", sep="&", lineterminator="\\\\\n", quoting=3, escapechar=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display benchmark results for the provided models")
    parser.add_argument("--bench-name", type=str, default=["live_bench"], nargs="+")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--questions-equivalent",
        action=argparse.BooleanOptionalAction,
        help="""Use this argument to treat all questions with the same weight. 
        If this argument is not included, all categories have the same weight.""",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--question-source",
        type=str,
        default="huggingface",
        help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions.",
    )
    parser.add_argument(
        "--livebench-release-option",
        type=str,
        default=max(LIVE_BENCH_RELEASES),
        choices=LIVE_BENCH_RELEASES,
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release.",
    )
    parser.add_argument(
        "--show-average", default=False, help="Show the average score for each task", action="store_true"
    )
    args = parser.parse_args()

    display_result_func = display_result_single

    display_result_func(args)
