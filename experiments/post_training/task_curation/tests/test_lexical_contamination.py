# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical scans use every public conversation and never private grader resources."""

import json
from contextlib import ExitStack

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.decon import NGramConfig, build_eval_bloom, decon_to_parquet
from marin.datakit.normalize import NormalizedData
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.lexical_contamination import write_text_population


def lexical_task(identifier: str, public: str, private: str) -> dict:
    return TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="pin", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=public),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="The fixture has no executable grader"),
        resources=ResourceGroups(oracle=(inline_resource("private.txt", private.encode()),)),
    ).model_dump(mode="json")


def test_full_task_text_marks_public_eval_copy_and_excludes_private_reference(tmp_path):
    leaked = " ".join(f"DistinctiveEvalWord{index}" for index in range(40))
    unrelated = " ".join(f"IndependentTaskWord{index}" for index in range(40))
    first = lexical_task("public-copy", leaked, "private witness")
    second = lexical_task("private-only", unrelated, leaked)
    population = write_text_population([(first, {}), (second, {})], str(tmp_path / "normalized"))
    assert population.task_count == 2
    texts = pq.read_table(tmp_path / "normalized/outputs/main/part-000000.parquet").to_pylist()
    assert {r["id"]: r["text"] for r in texts} == {
        "public-copy": "User:\n" + leaked,
        "private-only": "User:\n" + unrelated,
    }
    specs = pq.read_table(tmp_path / "normalized/specs/part-000000.parquet").to_pylist()
    assert {json.loads(r["task_spec"])["id"] for r in specs} == {"public-copy", "private-only"}
    evals = tmp_path / "evals"
    evals.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": "eval-control", "text": leaked}]), evals / "eval.parquet")
    config = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
    with ExitStack() as cleanup:
        client = LocalClient()
        cleanup.callback(client.shutdown)
        resources = ResourceConfig(cpu=1, ram="1g")
        with ZephyrContext(client=client, max_workers=1, resources=resources) as context:
            bloom = build_eval_bloom(
                eval_data_sources=[str(evals)],
                output_path=str(tmp_path / "bloom"),
                ngram=config,
                estimated_doc_count=10000,
                max_workers=1,
                worker_resources=resources,
                zephyr_context=context,
            )
            result = decon_to_parquet(
                normalized_data=NormalizedData(
                    main_output_dir=population.main_output_dir, dup_output_dir="", counters={}
                ),
                prebuilt_bloom_dir=bloom.bloom_dir,
                output_path=str(tmp_path / "marks"),
                ngram=config,
                max_workers=1,
                worker_resources=resources,
                zephyr_context=context,
            )
    flags = pq.read_table(tmp_path / "marks/outputs/main/part-000000.parquet").to_pylist()
    assert {r["id"]: r["contaminated"] for r in flags} == {"public-copy": True, "private-only": False}
    assert result.counters["decon/contaminated"] == result.counters["decon/clean"] == 1
