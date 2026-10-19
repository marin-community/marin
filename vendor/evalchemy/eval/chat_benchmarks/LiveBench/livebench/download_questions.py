import json
import os
from tqdm import tqdm
from collections import defaultdict

from livebench.common import (
    LIVE_BENCH_RELEASES,
    get_categories_tasks,
    load_questions,
    LIVE_BENCH_DATA_SUPER_PATH,
)

categories, tasks = get_categories_tasks(LIVE_BENCH_DATA_SUPER_PATH)

for category_name, task_names in tqdm(tasks.items()):
    # load questions from all livebench releases
    questions = load_questions(categories[category_name], livebench_releases=LIVE_BENCH_RELEASES, task_name=None)
    task_questions = defaultdict(list)
    for question in questions:
        task = question["task"]
        task_questions[task].append(question)

    category_path = f"data/{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}"
    os.makedirs(category_path, exist_ok=True)

    for task in task_questions.keys():
        task_path = category_path + "/" + task
        os.makedirs(task_path, exist_ok=True)

        question_file_path = task_path + "/question.jsonl"

        with open(question_file_path, "w") as f:
            f.writelines([json.dumps(example, default=str) + "\n" for example in task_questions[task]])
