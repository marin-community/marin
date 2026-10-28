# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen old/new raw-completion prompts and native source override grading."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from taskcompendium.grading import grade_answer
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    compile_generation_documents,
    convert_generation_document,
    generation_item,
)
from taskcompendium.models import ConversationTrace, GradingAttempt, Source, TaskSpec, TextMessage, verifyit_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from verifyit.grade import grade
from verifyit.spec import parse_spec, render_spec

from experiments.post_training.task_curation.datasets.evalchemy.harness import generation_sources

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_harness_generation_oracle.json").read_text())


@pytest.mark.parametrize("reference", ORACLE["cases"], ids=lambda case: f"{case['name']}-{case['shots']}shot")
def test_native_generation_matches_source_prompt_and_all_filter_grades(reference, tmp_path):
    name = reference["name"]
    benchmark = "gsm8k-0shot" if name == "gsm8k" and not reference["shots"] else name
    declared = next(source.config for source in generation_sources() if source.config.protocol.benchmark == benchmark)
    compiled = list(
        compile_generation_documents(
            [reference["document"]] * 2,
            reference["pool"],
            HarnessTemplate(name, reference["shots"]),
        )
    )
    for ordinal, (document, expected) in enumerate(zip(compiled, reference["specs"], strict=True)):
        row = RawRow(
            id=f"fixture-{name}-{ordinal}",
            source=Source(
                dataset=declared.recipe.source.repo,
                revision=declared.recipe.source.revision,
                row=f"fixture.parquet:{ordinal}",
                importer_revision="1",
            ),
            data=document,
        )
        task = convert_generation_document(row, ConversionContext({}, None), benchmark=benchmark)
        # Exercise the persisted typed grader parameters, including grouped DROP spans.
        task = TaskSpec.model_validate_json(task.model_dump_json())
        item = generation_item(task)
        assert item.operation.context == expected["requests"][0]["prompt"]
        source_stops = expected["requests"][0]["generation"]["until"]
        semantic_stops = [stop for stop in source_stops if not stop.startswith(("<|", "</s>"))]
        assert declared.protocol.sampling.stop == tuple(semantic_stops[:4])
        for grading in reference["grades"]:
            result = grade_answer(
                task,
                GradingAttempt(
                    ConversationTrace(
                        events=(
                            *task.context.events,
                            TextMessage(role="assistant", content=grading["response"]),
                        )
                    )
                ),
            )
            assert result.detail["metrics"] == grading["metrics"]
            assert result.reward == grading["metrics"][declared.protocol.primary_metric]
        if ordinal == 0:
            answer = tmp_path / "answer.txt"
            answer.write_text(reference["grades"][0]["response"])
            verifier = replace(verifyit_spec(task.grader), output=str(answer))
            persisted = parse_spec(render_spec(verifier))
            assert grade(persisted, tmp_path, tmp_path).detail["metrics"] == reference["grades"][0]["metrics"]
