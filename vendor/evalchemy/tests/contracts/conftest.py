"""Keep source contracts offline at the Hugging Face file boundary."""

import csv
import json
from pathlib import Path

import pytest

from eval.chat_benchmarks.IFEval import eval_instruct as ifeval
from eval.chat_benchmarks.MATH500 import eval_instruct as math500
from eval.chat_benchmarks.MixEval import eval_instruct as mixeval
from eval.chat_benchmarks.SimpleQA import eval_instruct as simpleqa
from eval.chat_benchmarks.SimpleQAVerified import eval_instruct as simpleqa_verified


@pytest.fixture(scope="session")
def simpleqa_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("simpleqa") / "simpleqa.csv"
    with path.open("w", newline="", encoding="utf-8") as data:
        writer = csv.DictWriter(data, fieldnames=["problem", "answer", "metadata"])
        writer.writeheader()
        for index in range(simpleqa.DATASET_SIZE):
            writer.writerow({"problem": f"question {index}", "answer": f"answer {index}", "metadata": "{'topic': 'fixture'}"})
    return path


@pytest.fixture(scope="session")
def simpleqa_verified_csv(tmp_path_factory):
    """Two rows in the publisher's column layout, keyed by their original SimpleQA indices."""
    path = tmp_path_factory.mktemp("simpleqa-verified") / "simpleqa_verified.csv"
    columns = ["original_index", "problem", "answer", "topic", "answer_type", "multi_step", "requires_reasoning", "urls"]
    with path.open("w", newline="", encoding="utf-8") as data:
        writer = csv.DictWriter(data, fieldnames=columns)
        writer.writeheader()
        for original_index, multi_step in ((5, "True"), (8, "False")):
            writer.writerow({
                "original_index": original_index, "problem": f"question {original_index}",
                "answer": f"answer {original_index}", "topic": "Politics", "answer_type": "Person",
                "multi_step": multi_step, "requires_reasoning": "False", "urls": "https://example.org/a,https://example.org/b",
            })
    return path


@pytest.fixture(scope="session")
def math500_jsonl(tmp_path_factory):
    path = tmp_path_factory.mktemp("math500") / "math500.jsonl"
    path.write_text("".join(json.dumps({"problem": f"Compute {index} + 0.", "answer": str(index)}) + "\n" for index in range(500)))
    return path


@pytest.fixture(autouse=True)
def offline_other_hf_files(monkeypatch, simpleqa_csv, simpleqa_verified_csv, math500_jsonl, request):
    if request.node.get_closest_marker("data_integration"):
        return
    fixtures = Path(__file__).parent / "fixtures" / "hf-other"

    def fetch_fixture(repo_id, filename, *, revision, cache_prefix=None):
        if repo_id == "basicv8vc/SimpleQA":
            return str(simpleqa_csv)
        if repo_id == "google/simpleqa-verified":
            return str(simpleqa_verified_csv)
        if repo_id == "HuggingFaceH4/MATH-500":
            return str(math500_jsonl)
        if repo_id == "MixEval/MixEval":
            return str(fixtures / Path(filename).relative_to("data/mixeval-jsonl"))
        return str(fixtures / filename)

    for module in (simpleqa, simpleqa_verified, mixeval, ifeval, math500):
        monkeypatch.setattr(module, "fetch_hf", fetch_fixture)
