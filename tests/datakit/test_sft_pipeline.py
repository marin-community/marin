# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.decon import DeconAttributes, NGramConfig, decon_to_parquet
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    prepare_verification_text,
    verify_prepared_candidate,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)
from openai_harmony import Message, Role

from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData
from experiments.datakit.reference_pipeline import SMOKE_SCALE
from experiments.datakit.sft_pipeline import SFT_VERIFICATION_PARAMS, chat_comparison_record, filter_sft_source


@pytest.mark.parametrize("keep_clean", [True, False])
def test_filter_sft_source_removes_marked_rows_and_preserves_clean_text(tmp_path: Path, keep_clean: bool):
    shard = "part-00000-of-00001.parquet"
    rows = [
        {
            "id": name,
            "text": f"<user>{name}</user><assistant>answer</assistant>",
            "source_id": f"chat-{name}",
            "rendered_text": f"full rendering of {name}",
        }
        for name in ("clean", "contaminated", "exact", "fuzzy")
    ]
    directories = {name: tmp_path / name for name in ("input", "decon", "exact", "verified")}
    for directory in directories.values():
        directory.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), directories["input"] / shard)
    pq.write_table(
        pa.table({"id": [row["id"] for row in rows], "contaminated": [not keep_clean, True, False, False]}),
        directories["decon"] / shard,
    )
    for name, record_id in (("exact", "exact"), ("verified", "fuzzy")):
        pq.write_table(pa.table({"id": [record_id], "dup_doc": [True]}), directories[name] / shard)
    source = NormalizedData(main_output_dir=str(directories["input"]), dup_output_dir="", counters={})
    source_key = datakit_source_key(source.main_output_dir)
    result = filter_sft_source(
        output_path=str(tmp_path / "filtered"),
        normalized=source,
        exact=GlobalExactDedupData(
            sources={source_key: ExactDupsPerSource(attr_dir=str(directories["exact"]))}, counters={}
        ),
        verified=VerifiedFuzzyDupsAttrData(
            verification=FuzzyVerificationParams(),
            local_representatives=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            sources={source_key: VerifiedFuzzyDupsPerSource(attr_dir=str(directories["verified"]), source_tag="input")},
            counters={},
        ),
        decontam=DeconAttributes(
            main_output_dir=str(directories["decon"]),
            flagged_output_dir="",
            num_partitions=1,
            eval_hash_index_path="",
            counters={},
        ),
        scale=SMOKE_SCALE,
    )
    actual = [row for path in Path(result.main_output_dir).glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert actual == (
        [{"id": "clean", "source_id": "chat-clean", "text": "full rendering of clean"}] if keep_clean else []
    )


def test_chat_comparison_excludes_instructions_but_retains_training_rendering():
    messages = [
        Message.from_role_and_content(Role.SYSTEM, "Shared system instructions."),
        Message.from_role_and_content(Role.DEVELOPER, "Shared developer instructions."),
        Message.from_role_and_content(Role.USER, "Solve the unique problem."),
        Message.from_role_and_content(Role.ASSISTANT, "Work through the solution.").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "The unique answer.").with_channel("final"),
    ]
    record = {
        "id": "full-chat-hash",
        "messages": [message.to_dict() for message in messages],
        "chat_template_kwargs": json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Shared tool schema.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            }
        ),
    }
    comparison = chat_comparison_record(record)
    assert comparison["text"] == "Solve the unique problem.\n\nWork through the solution.\n\nThe unique answer."
    assert comparison["id"] == "full-chat-hash"
    assert "Shared system instructions." in comparison["rendered_text"]
    assert "Shared developer instructions." in comparison["rendered_text"]
    assert "Shared tool schema." in comparison["rendered_text"]
    assert "<|start_think|>Work through the solution.<|end_think|>" in comparison["rendered_text"]


@pytest.mark.parametrize("eval_role", [Role.SYSTEM, Role.USER, Role.ASSISTANT])
def test_sft_decontamination_scans_message_bodies_without_template_instructions(tmp_path: Path, eval_role: Role):
    eval_text = (
        "Which sequence of seventeen distinct positive integers satisfies "
        "all the constraints in this mathematical problem?"
    )
    messages = [
        Message.from_role_and_content(
            Role.SYSTEM, eval_text if eval_role == Role.SYSTEM else "Follow the instructions."
        ),
        Message.from_role_and_content(Role.USER, eval_text if eval_role == Role.USER else "Explain the solution."),
        Message.from_role_and_content(
            Role.ASSISTANT, eval_text if eval_role == Role.ASSISTANT else "The answer is forty two."
        ).with_channel("final"),
    ]
    comparison = chat_comparison_record({"id": "chat", "messages": [message.to_dict() for message in messages]})
    input_dir = tmp_path / "input"
    eval_dir = tmp_path / "eval"
    input_dir.mkdir()
    eval_dir.mkdir()
    repeated = [{**comparison, "id": f"chat-{index}"} for index in range(3)]
    pq.write_table(pa.Table.from_pylist(repeated), input_dir / "part-00000-of-00001.parquet")
    pq.write_table(pa.table({"id": ["eval"], "text": [eval_text]}), eval_dir / "eval.parquet")
    result = decon_to_parquet(
        normalized_data=NormalizedData(main_output_dir=str(input_dir), dup_output_dir="", counters={}),
        eval_data_sources=str(eval_dir),
        output_path=str(tmp_path / "decontam"),
        ngram=NGramConfig(ngram_length=13, overlap_threshold=0.5),
        estimated_doc_count=100,
        max_workers=1,
    )
    marked = pq.read_table(result.main_output_dir).to_pylist()
    assert [(row["id"], row["contaminated"]) for row in marked] == [
        (f"chat-{index}", eval_role != Role.SYSTEM) for index in range(3)
    ]


@pytest.mark.parametrize(
    "answers",
    [
        ("Use Dijkstra with a priority queue.", "Use Bellman Ford to handle negative edge weights."),
        ("Use Dijkstra with a priority queue.", "Use Dijkstra with a priority queue. Stop when the target is settled."),
        ("Visit node A then node B then node A.", "Visit node B then node A then node B."),
    ],
)
def test_sft_fuzzy_verification_keeps_distinct_answers_to_a_shared_prompt(answers: tuple[str, str]):
    prompt = "Explain the properties of this graph and derive an algorithm to find the shortest path. " * 100
    texts = []
    for answer in answers:
        messages = [
            Message.from_role_and_content(Role.USER, prompt),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel("final"),
        ]
        texts.append(
            chat_comparison_record({"id": answer, "messages": [message.to_dict() for message in messages]})["text"]
        )
    params = SFT_VERIFICATION_PARAMS
    shorter, longer = sorted(texts, key=len)
    result = verify_prepared_candidate(
        prepare_verification_text(shorter, params), prepare_verification_text(longer, params), params
    )
    assert not result.accepted


def test_sft_fuzzy_verification_removes_case_and_whitespace_variants():
    params = SFT_VERIFICATION_PARAMS
    result = verify_prepared_candidate(
        prepare_verification_text("Explain the graph.\n\nVisit node A then node B.", params),
        prepare_verification_text("Explain  the graph.\n\nVisit node a then node b.", params),
        params,
    )
    assert result.accepted
