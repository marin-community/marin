import math
from pathlib import Path

import datasets
import pytest
from lm_eval.tasks import TaskManager

TASK_CATEGORIES = (
    ("uncheatable_eval_wikipedia_english", "wikipedia_english"),
    ("uncheatable_eval_wikipedia_nonenglish", "wikipedia_nonenglish"),
    ("uncheatable_eval_github_python", "github_python"),
    ("uncheatable_eval_github_cpp", "github_cpp"),
    ("uncheatable_eval_github_javascript", "github_javascript"),
    ("uncheatable_eval_github_markdown", "github_markdown"),
    ("uncheatable_eval_github_other", "github_other"),
    ("uncheatable_eval_bbc_news", "bbc_news"),
    ("uncheatable_eval_arxiv_physics", "arxiv_physics"),
    ("uncheatable_eval_arxiv_computer_science", "arxiv_cs"),
    ("uncheatable_eval_arxiv_math", "arxiv_math"),
    ("uncheatable_eval_arxiv_other", "arxiv_other"),
    ("uncheatable_eval_biorxiv_all", "biorxiv_all"),
    ("uncheatable_eval_ao3_english", "ao3_english"),
    ("uncheatable_eval_ao3_nonenglish", "ao3_nonenglish"),
)


@pytest.fixture
def uncheatable_tasks(monkeypatch: pytest.MonkeyPatch):
    rows = [
        {
            "content": f"document from {category}",
            "untruncated_content": f"document from {category}",
            "category": category,
            "date": "2026-07-01",
            "url": f"https://example.com/{category}",
            "metadata": "{}",
        }
        for _, category in TASK_CATEGORIES
    ]
    dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)

    task_path = Path(__file__).parents[2] / "eval" / "lm_eval_tasks"
    loaded = TaskManager(include_path=task_path, include_defaults=False).load("uncheatable_eval")
    return loaded["tasks"]


def test_uncheatable_eval_tasks_load_pinned_dataset(uncheatable_tasks) -> None:
    assert set(uncheatable_tasks) == {task_name for task_name, _ in TASK_CATEGORIES}
    for task in uncheatable_tasks.values():
        assert task.config.dataset_path == "Jellyfish042/UncheatableEval-2026-07"
        assert task.config.dataset_kwargs == {"revision": "65889535d56aa38d448ce7e07b08e6e36c031545"}


def test_uncheatable_eval_tasks_filter_categories(uncheatable_tasks) -> None:
    for task_name, category in TASK_CATEGORIES:
        documents = list(uncheatable_tasks[task_name].eval_docs)
        assert [document["category"] for document in documents] == [category]


def test_uncheatable_eval_tasks_construct_rolling_requests(uncheatable_tasks) -> None:
    for task in uncheatable_tasks.values():
        document = next(iter(task.eval_docs))
        request = task.construct_requests(document, ctx="")
        assert request.request_type == "loglikelihood_rolling"
        assert request.args == (document["content"],)


def test_uncheatable_eval_tasks_score_bpb(uncheatable_tasks) -> None:
    for task in uncheatable_tasks.values():
        document = next(iter(task.eval_docs))
        log_likelihood = -12.0
        result = task.process_results(document, [log_likelihood])
        aggregate_bpb = task.aggregation()["bits_per_byte"]([result["bits_per_byte"]])
        expected_bpb = -log_likelihood / len(document["content"].encode()) / math.log(2)
        assert aggregate_bpb == pytest.approx(expected_bpb)
