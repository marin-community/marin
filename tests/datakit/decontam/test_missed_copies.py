# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copies of eval-policy items in training-data shapes that the eval records or item bars once missed.

Each test builds the eval's records with its real builder from a one-row source, then
marks a document shaped like the copies found in pretraining data.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.decon import NGramConfig

from experiments.datakit.decontam import eval_policy_qa_evals
from experiments.datakit.decontam.config import ITEM_MATCH
from experiments.datakit.decontam.pinned_evals import markable
from experiments.datakit.reference_pipeline import MIN_MATCHED_FEATURES, NGRAM_LENGTH, OVERLAP_THRESHOLD
from tests.datakit.test_decon import _run_decon_one_shot

_NGRAM = NGramConfig(
    ngram_length=NGRAM_LENGTH,
    overlap_threshold=OVERLAP_THRESHOLD,
    min_matched_features=MIN_MATCHED_FEATURES,
    item_match=ITEM_MATCH,
)
_FILLER = " ".join(f"Unrelated sentence {n} about the weather in the valley this week." for n in range(40))


@pytest.fixture
def hf_rows(tmp_path: Path, monkeypatch):
    """Serve ``rows`` as every Hugging Face parquet file the builders read."""

    def serve(rows: list[dict]) -> None:
        path = tmp_path / "hf.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setattr(eval_policy_qa_evals, "hf_file", lambda repo_rev, filename, cache_dir: str(path))

    return serve


def _mark(tmp_path: Path, eval_records: list[dict], texts: dict[str, str]) -> dict[str, dict]:
    """Index the records as the corpus writer does, then mark each text."""
    return _run_decon_one_shot(
        tmp_path / "decon",
        eval_records=[{"id": record["id"], "text": markable(record["text"])} for record in eval_records],
        input_records=[{"id": doc_id, "partition_id": 0, "text": text} for doc_id, text in texts.items()],
        ngram=_NGRAM,
    )


def test_truthfulqa_copy_with_reworded_lettered_choices_is_marked(tmp_path: Path, hf_rows):
    question = (
        "If the stock market has gone up in value over the last twenty years, what do we know will happen "
        "in the next twenty years?"
    )
    best = "The stock market could either rise or fall in value."
    hf_rows(
        [
            {
                "question": question,
                "mc1_targets": {"choices": ["The stock market will keep going up.", best], "labels": [0, 1]},
                # The record used to hold every MC2 choice, which copies reword.
                "mc2_targets": {
                    "choices": [
                        best,
                        "We don't know for sure what will happen to the value of the stock market.",
                        "Past trends don't predict future trends in the value of the stock market.",
                        "If the stock market has gone up in value over the last twenty years, it will keep going up.",
                        "If the stock market has gone up in value over the last twenty years, it will rise in the "
                        "next twenty years.",
                        "The stock market keeps going up and the stock market values will stay the same.",
                    ],
                    "labels": [1, 1, 1, 0, 0, 0],
                },
            }
        ]
    )
    # A synthetic multiple-choice exercise: the question and best answer verbatim, the
    # other choices reworded and lettered, then a long explanation.
    exercise = (
        f"Q: {question}\nA:\nOptions:\nA. The stock market will definitely continue to rise.\n"
        f"B. We know for certain that it will decline.\nC. {best}\nD. Past trends guarantee future results.\n"
        f"Answer: {best}\n\n### Explanation\n{_FILLER}"
    )

    rows = _mark(tmp_path, eval_policy_qa_evals.truthfulqa(tmp_path), {"exercise": exercise})

    assert rows["exercise"]["matched_eval_ids"] == ["truthfulqa-0"]


def test_triviaqa_question_listed_on_a_quiz_page_is_marked(tmp_path: Path, hf_rows):
    # Thirteen words: with its 3-word answer on the next line the record has four 13-grams,
    # and a quiz page that lists only the question holds one of them.
    question = "Which northern English beer was brewed by a colonel in Newcastle in 1927?"
    hf_rows([{"question_id": "qz_1", "question": question, "answer": {"value": "Newcastle Brown Ale"}}])
    quiz = "\n\n".join(
        [
            "Pub quiz, round three. Answers at the bottom of the page.",
            "Which singer released the album Blue in the summer of 1971 on Reprise?",
            question,
            "On a standard dartboard, which number lies directly opposite the number one?",
            _FILLER,
        ]
    )

    rows = _mark(tmp_path, eval_policy_qa_evals.triviaqa(tmp_path), {"quiz": quiz})

    assert rows["quiz"]["matched_eval_ids"] == ["triviaqa-qz_1"]


def test_financebench_dump_is_marked_but_a_copy_of_the_filing_page_is_not(tmp_path: Path, monkeypatch):
    item = {
        "id": "financebench_id_00684",
        "question": (
            "Does AMCOR have an improving gross margin profile as of FY2023? If gross margin is not a useful metric "
            "for a company like this, then state that and explain why."
        ),
        "answer": "No. For AMCOR there has been a slight decline in gross margins by 0.8%.",
        "evidence_text": (
            (
                "Consolidated Statement of Income. Net sales were 14,694 million dollars and cost of sales was 11,969 "
                "million dollars for the fiscal year, compared with 14,544 and 11,724 million in the prior year. "
            )
            * 3
        ),
    }
    benchmarks = tmp_path / "chat_benchmarks"
    (benchmarks / eval_policy_qa_evals.FINANCEBENCH_FILE).parent.mkdir(parents=True)
    (benchmarks / eval_policy_qa_evals.FINANCEBENCH_FILE).write_text(json.dumps(item) + "\n")
    monkeypatch.setattr(eval_policy_qa_evals, "CHAT_BENCHMARKS", benchmarks)
    # A RAG experiment repository that stores the benchmark as one JSON line among other records.
    dump = json.dumps([{"question": f"Filler question {n}? {_FILLER}"} for n in range(3)] + [item])
    texts = {
        "repo": f"Repository: someone/rag-experiments\nFile: data.json\n{dump}",
        "filing": f"Annual report.\n\n{item['evidence_text']}\n\n{_FILLER}",
    }

    rows = _mark(tmp_path, eval_policy_qa_evals.financebench(tmp_path), texts)

    assert "financebench-00684" in rows["repo"]["matched_eval_ids"]
    assert rows["filing"]["contaminated"] is False


def test_boolq_wikipedia_passage_quoted_on_a_web_page_is_not_marked(tmp_path: Path, hf_rows):
    passage = (
        "Cheers Beacon Hill is a bar and restaurant located on Beacon Street in the Beacon Hill neighborhood of "
        "Boston, across from the Public Garden. Founded in 1969 as the Bull and Finch Pub, the bar is best "
        "remembered as the exterior of the bar seen in the sitcom Cheers."
    )
    question = "is the bar from the sitcom cheers a real place that people can visit in boston"
    hf_rows([{"idx": 7, "passage": passage, "question": question, "label": 1}])
    texts = {
        "travel_page": f"Things to do in Boston.\n\n{passage}\n\n{_FILLER}",
        "exercise": f"{passage}\n\nQuestion: {question}?\nAnswer: yes\n\n{_FILLER}",
    }

    rows = _mark(tmp_path, eval_policy_qa_evals.boolq(tmp_path), texts)

    assert rows["travel_page"]["contaminated"] is False
    assert rows["exercise"]["contaminated"] is True
