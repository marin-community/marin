# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full pinned long-context populations; the run records eligibility before selecting attempts."""

import json
from functools import cache

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import answer_task
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.evaluation import (
    ChatGeneration,
    ContextPolicy,
    EvaluationItem,
    EvaluationProtocol,
    GraphWalksContextSelection,
    Metric,
    MetricKind,
    MRCRContextSelection,
    Sampling,
    TrialPolicy,
)
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow
from verifyit.adapters.evalchemy_long_context import FINAL_ANSWER_PREFIX, mrcr_bin
from verifyit.spec import GraphWalksSpec, MRCRSpec

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MRCR_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
LONG_CONTEXT_RESOURCE_BYTES = 64 * 1024 * 1024
MRCR_FILES = (
    "2needle/2needle_0.parquet",
    "2needle/2needle_1.parquet",
    "4needle/4needle_0.parquet",
    "4needle/4needle_1.parquet",
    "8needle/8needle_0.parquet",
    "8needle/8needle_1.parquet",
)
GRAPHWALKS_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
GRAPHWALKS_FILES = ("graphwalks_128k_and_shorter.parquet", "graphwalks_256k_to_1mil.parquet")


@cache
def official_tokenizer():
    # o200k is a curation dependency; inference uses the saved official counts.
    from tiktoken import get_encoding  # noqa: PLC0415

    return get_encoding("o200k_base")


def official_token_count(row: dict) -> int:
    encode = official_tokenizer().encode
    return sum(len(encode(message["content"])) for message in json.loads(row["prompt"])) + len(encode(row["answer"]))


def read_mrcr(path: StoragePath, _context: ConversionContext):
    with path.open("rb") as file:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                tokens = official_token_count(row)
                yield {**row, "official_tokens": tokens, "mrcr_bin_upper": mrcr_bin(tokens)}


def convert_mrcr(row: RawRow, _context: ConversionContext) -> TaskSpec:
    data = row.data
    messages = json.loads(data["prompt"])
    task = answer_task(
        row,
        prompt=messages[-1]["content"],
        spec=MRCRSpec(expected=data["answer"], nonce=data["random_string_to_prepend"]),
        evidence=dict(data),
    )
    return task.model_copy(
        update={
            "context": ConversationInput(
                events=tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
            ),
            "tags": ("eval", "source:evalchemy", "benchmark:MRCR", "long-context"),
        }
    )


def mrcr_item(task: TaskSpec) -> EvaluationItem:
    row = source_evidence(task)
    return chat_item(task).model_copy(
        update={
            "operation": ChatGeneration(historical_assistant_reasoning="empty"),
            "context_selection": MRCRContextSelection(
                official_tokens=row["official_tokens"],
                bin_upper=row["mrcr_bin_upper"],
                needles=row["n_needles"],
            ),
            "groups": {"cell": f"{row['mrcr_bin_upper']}_{row['n_needles']}needle"},
        }
    )


def convert_graphwalks(row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = answer_task(
        row,
        prompt=row.data["prompt"],
        spec=GraphWalksSpec(expected=tuple(row.data["answer_nodes"])),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:GraphWalks", "long-context")})


def graphwalks_item(task: TaskSpec) -> EvaluationItem:
    row = source_evidence(task)
    return chat_item(task).model_copy(
        update={
            "context_selection": GraphWalksContextSelection(
                prompt_chars=row["prompt_chars"],
                gold_output=FINAL_ANSWER_PREFIX + " [" + ", ".join(row["answer_nodes"]) + "]",
            ),
            "groups": {"problem_type": row["problem_type"]},
        }
    )


def mrcr_source(name: str, max_context: int) -> RlDataSource[EvalDataSet]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name.lower(),
            title=name,
            origin="Evalchemy",
            family="long-context",
            tags=("eval", "source:evalchemy", "benchmark", "long-context", "retrieval"),
            count=2400,
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="evalchemy-" + name.lower(),
                source=HfSource("openai/mrcr", MRCR_REVISION, MRCR_FILES, SourceFormat.PARQUET, read=read_mrcr),
                convert=convert_mrcr,
                version="1",
                intended_use=IntendedUse.EVAL,
                resource_budget_bytes=LONG_CONTEXT_RESOURCE_BYTES,
            ),
            EvaluationProtocol(
                benchmark=name,
                version="1",
                operation="chat_generation",
                sampling=Sampling(
                    temperature=0,
                    top_p=1,
                    top_k=None,
                    max_tokens=4096,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(
                    Metric(name="accuracy", kind=MetricKind.CONTINUOUS, higher_is_better=True),
                    Metric(name="prefix_hit_rate", kind=MetricKind.BINARY, higher_is_better=True),
                ),
                primary_metric="accuracy",
                reducer="mrcr_cell_means_v1",
                expected_population=2400,
                context_policy=ContextPolicy(kind="mrcr", max_context=max_context),
            ),
            mrcr_item,
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        mrcr_source("MRCR", 73728),
        mrcr_source("MRCR-32k", 32704),
        mrcr_source("MRCR-65k", 65472),
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:graphwalks",
                title="GraphWalks",
                origin="Evalchemy",
                family="long-context",
                tags=("eval", "source:evalchemy", "benchmark", "long-context", "graph"),
                count=1150,
            ),
            config=EvalDataSet(
                CurationRecipe(
                    name="evalchemy-graphwalks",
                    source=HfSource("openai/graphwalks", GRAPHWALKS_REVISION, GRAPHWALKS_FILES, SourceFormat.PARQUET),
                    convert=convert_graphwalks,
                    version="1",
                    intended_use=IntendedUse.EVAL,
                    resource_budget_bytes=LONG_CONTEXT_RESOURCE_BYTES,
                ),
                EvaluationProtocol(
                    benchmark="GraphWalks",
                    version="1",
                    operation="chat_generation",
                    sampling=Sampling(
                        temperature=0,
                        top_p=1,
                        top_k=None,
                        max_tokens=131072,
                        stop=(),
                        chat_template_kwargs={"enable_thinking": True},
                    ),
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=tuple(
                        Metric(
                            name=name,
                            kind=MetricKind.BINARY if name == "exact_match" else MetricKind.CONTINUOUS,
                            higher_is_better=True,
                        )
                        for name in ("f1", "precision", "recall", "exact_match")
                    ),
                    primary_metric="f1",
                    reducer="mean_v1",
                    expected_population=1150,
                    context_policy=ContextPolicy(kind="graphwalks", max_context=None),
                ),
                graphwalks_item,
            ),
            pipeline=curate,
        ),
    ]
