# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinanceBench's exact bundled sample and canonical SimpleQA classification tasks.

SimpleQAMini's reader emits the seeded sample in benchmark order. Source locators
index that emitted stream; source_index retains the original CSV identity for
paired comparisons, independently of task IDs and sampled order.
"""

import ast
import csv
import random
from functools import partial

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import answer_task
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.evaluation import EvaluationItem, EvaluationProtocol, Metric, MetricKind, Sampling, TrialPolicy
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Converter, IntendedUse, RawRow
from verifyit.adapters.evalchemy_judges import financebench_spec, simpleqa_spec
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SOURCE_REVISION = "623134de90723e7f189036da005067f36817b6e7"
FINANCEBENCH_SHA256 = "0df2b2080400693d98eeccac6970432af72d7e092287b08bcc9bd013d2c3232e"
SIMPLEQA_REVISION = "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"
FINANCE_PROMPT = (
    "Based on the following financial document excerpt, answer the question.\n\nDocument:\n{evidence_text}"
    "\n\nQuestion: {question}\n\nProvide a direct, concise answer."
)
CLASSIFICATION_METRICS = ("accuracy", "incorrect", "not_attempted", "judge_failed")


def read_simpleqa(file: StoragePath, _context: ConversionContext, *, mini: bool):
    with file.open("r", newline="", encoding="utf-8") as stream:
        rows = [
            {
                "source_index": index,
                "question": row["problem"],
                "answer": row["answer"],
                "metadata": ast.literal_eval(row["metadata"]),
            }
            for index, row in enumerate(csv.DictReader(stream))
        ]
    yield from random.Random(0).sample(rows, 500) if mini else rows


def convert_finance(row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = answer_task(
        row,
        prompt=FINANCE_PROMPT.format_map(row.data),
        spec=financebench_spec(row.data["question"], str(row.data["answer"])),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:FinanceBench")})


def convert_simpleqa(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    task = answer_task(
        row,
        prompt=row.data["question"],
        spec=simpleqa_spec(row.data["question"], row.data["answer"]),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:" + benchmark)})


def judged_item(task: TaskSpec) -> EvaluationItem:
    evidence = source_evidence(task)
    groups = {"source_index": str(evidence["source_index"])} if "source_index" in evidence else {}
    return chat_item(task).model_copy(update={"groups": groups})


def judged_source(
    name: str,
    benchmark: str,
    source: HfSource | UrlSource,
    converter: Converter,
    rows: int,
    max_tokens: int,
) -> RlDataSource[EvalDataSet]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="judged-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "llm-judge"),
            count=rows,
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=converter,
                version="3",
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="maintained-classification-v4",
                operation="chat_generation",
                sampling=Sampling(
                    temperature=0,
                    top_p=1,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": benchmark != "FinanceBench"},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(name=name, kind=MetricKind.BINARY, higher_is_better=name == "accuracy")
                    for name in CLASSIFICATION_METRICS
                ),
                primary_metric="accuracy",
                reducer="classification_v1",
                expected_population=rows,
            ),
            judged_item,
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        judged_source(
            "financebench",
            "FinanceBench",
            UrlSource(
                f"https://raw.githubusercontent.com/marin-community/marin/{SOURCE_REVISION}/"
                "vendor/evalchemy/eval/chat_benchmarks/FinanceBench/data/financebench.jsonl",
                FINANCEBENCH_SHA256,
                "financebench.jsonl",
                SourceFormat.JSONL,
            ),
            convert_finance,
            50,
            4096,
        ),
        *(
            judged_source(
                "simpleqa-mini" if mini else "simpleqa",
                benchmark,
                HfSource(
                    "basicv8vc/SimpleQA",
                    SIMPLEQA_REVISION,
                    ("simple_qa_test_set.csv",),
                    SourceFormat.CSV,
                    read=partial(read_simpleqa, mini=mini),
                ),
                partial(convert_simpleqa, benchmark=benchmark),
                500 if mini else 4326,
                512,
            )
            for benchmark, mini in (("SimpleQA", False), ("SimpleQAMini", True))
        ),
    ]
