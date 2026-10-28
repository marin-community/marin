# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen long-context scoring, official selection policy, and complete publication."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from taskcompendium.evaluation import GraphWalksContextSelection, MRCRContextSelection
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.submission import conversation_messages
from verifyit.adapters.evalchemy_long_context import graphwalks_output_budget, interleave_mrcr_indices, mrcr_bin
from verifyit.candidate import grade_candidate
from verifyit.spec import GraphWalksSpec, MRCRSpec, parse_spec, render_spec
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.long_context import official_token_count, sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_long_context_oracle.json").read_text())


@pytest.mark.parametrize("case", ORACLE["mrcr"]["grades"])
def test_mrcr_nonce_gate_and_similarity_match_legacy(case):
    row = ORACLE["mrcr"]["row"]
    spec = parse_spec(render_spec(MRCRSpec(expected=row["answer"], nonce=row["random_string_to_prepend"])))
    verdict = grade_candidate(spec, case["response"], {})
    assert verdict.detail["metrics"] == case["grade"]["metrics"]


@pytest.mark.parametrize("case", ORACLE["graph"]["grades"])
def test_graphwalks_last_line_node_metrics_match_legacy(case):
    spec = parse_spec(render_spec(GraphWalksSpec(expected=tuple(case["nodes"]))))
    verdict = grade_candidate(spec, case["response"], {})
    assert verdict.detail["metrics"] == case["grade"]["metrics"]
    assert verdict.detail["model_answer"] == case["grade"]["audit"]["model_answer"]


def test_mrcr_official_tokenizer_and_published_bin_boundaries_match_legacy():
    assert official_token_count(ORACLE["mrcr"]["row"]) == ORACLE["mrcr"]["official_tokens"]
    assert [mrcr_bin(case["tokens"]) for case in ORACLE["mrcr"]["bins"]] == [
        case["bin"] for case in ORACLE["mrcr"]["bins"]
    ]


@pytest.mark.parametrize("case", ORACLE["mrcr"]["orders"])
def test_mrcr_alias_contexts_preserve_official_interleaving_before_limit(case):
    cells = {(cell["bin"], cell["needles"]): cell["indices"] for cell in ORACLE["mrcr"]["cells"]}
    assert interleave_mrcr_indices(cells, case["context"], case["limit"]) == case["indices"]


def test_graphwalks_source_output_budgets_keep_reference_space_without_shortening():
    reference = ORACLE["graph"]
    for row in reference["selection"]:
        answer = "Final Answer: [" + ", ".join(row["answer_nodes"]) + "]"
        assert graphwalks_output_budget(len(answer.split())) == row["max_output_tokens"]
    over_cap = reference["rows"][2]
    answer = "Final Answer: [" + ", ".join(over_cap["answer_nodes"]) + "]"
    assert graphwalks_output_budget(len(answer.split())) > reference["cap"]


@pytest.mark.parametrize("name", ["MRCR", "MRCR-32k", "MRCR-65k", "GraphWalks"])
def test_long_context_publication_retains_ineligible_rows_and_private_references(name, tmp_path):
    source = next(source for source in sources() if source.config.protocol.benchmark == name)
    dataset = source.config
    assert dataset is not None
    rows = ORACLE["graph"]["rows"] if name == "GraphWalks" else [ORACLE["mrcr"]["row"]]
    root = tmp_path / "source"
    table = pa.Table.from_pylist(rows)
    for index, filename in enumerate(dataset.files):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table if index == 0 else table.slice(0, 0), path)
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=1,
                chunk_storage_prefix=str(tmp_path / "chunks"),
            ) as context,
        ):
            manifest = publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, None),
                dataset.protocol.model_copy(update={"expected_population": len(rows)}),
                dataset.item_builder,
                context,
                str(root),
                output,
            )
        published = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert manifest.rows == len(rows)
    assert len(published) == len(rows)
    if name == "GraphWalks":
        assert [conversation_messages(item.task.context.events) for item in published] == [
            [{"role": "user", "content": row["prompt"]}] for row in rows
        ]
        assert all(isinstance(item.item.context_selection, GraphWalksContextSelection) for item in published)
    else:
        expected_messages = [
            {key: message[key] for key in ("role", "content")} for message in ORACLE["mrcr"]["messages"]
        ]
        assert conversation_messages(published[0].task.context.events) == expected_messages
        selection = published[0].item.context_selection
        assert isinstance(selection, MRCRContextSelection)
        assert selection.bin_upper is None
    assert all(not item.task.resources.worker for item in published)
