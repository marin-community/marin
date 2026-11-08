# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.post_training.capability_driven_envs.biotasks_library.notebooks_v1 import (
    GeneratedIdea,
    GeneratedPortfolio,
    SourceRecord,
    extract_source_text,
    hydrate_ideas,
    source_group_id,
)
from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import Assessment, IdeaResource


def _source(raw: bytes, *, source_format: str = "Jupyter") -> SourceRecord:
    return SourceRecord(
        canonical_document_key="github:example/project:notebook.ipynb",
        document_key="github:example/project:notebook.ipynb",
        fields=("genomics",),
        format=source_format,
        modalities=("sequences",),
        object_uri="s3://private/object",
        operations=("analysis",),
        path="notebook.ipynb",
        repo="Example/Project",
        source_bytes=len(raw),
        source_ids=("github:wrong/group",),
        source_role="analysis_or_teaching_document",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        title="Example",
    )


def _assessment(action: str = "accept") -> Assessment:
    return Assessment(
        information_sufficiency=5,
        biological_substance=4,
        scientific_validity=4,
        objective_verifiability=5,
        hack_resistance=4,
        practical_usefulness=4,
        recommended_action=action,
        rationale="The task is complete and objectively graded.",
        confidence=0.9,
    )


def test_extract_notebook_excludes_outputs_and_checks_digest():
    raw = json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": ["Biological context"]},
                {"cell_type": "code", "source": ["value = 3"], "outputs": [{"text": ["SECRET OUTPUT"]}]},
            ]
        }
    ).encode()
    source = _source(raw)

    text = extract_source_text(source, raw)

    assert "Biological context" in text
    assert "value = 3" in text
    assert "SECRET OUTPUT" not in text
    with pytest.raises(ValueError, match="digest mismatch"):
        extract_source_text(source, raw + b"changed")


def test_hydrated_idea_has_content_derived_stable_identity():
    raw = b"source"
    source = _source(raw, source_format="R Markdown")
    generated = GeneratedIdea(
        family="Allele Frequency Audit",
        parameters_json='{"threshold": 0.2}',
        scientific_question="Which variants exceed the declared frequency threshold?",
        instruction="Read /workspace/data/variants.csv and return one JSON object.",
        expected_json='{"variants": ["v2"]}',
        resources=(IdeaResource(path="workspace/data/variants.csv", content="variant,frequency\nv1,0.1\nv2,0.3\n"),),
        author_assessment=_assessment(),
    )
    portfolio = GeneratedPortfolio(
        source_summary="A compact variant-analysis example.",
        disposition="ideas_proposed",
        ideas=(generated,),
    )

    first = hydrate_ideas(source, portfolio)[0]
    second = hydrate_ideas(source, portfolio)[0]

    assert first.idea_id == second.idea_id
    assert first.family == "allele_frequency_audit"
    assert first.source_group_id == "github:example/project"
    assert first.expected == {"variants": ["v2"]}
    assert source_group_id(source) == "github:example/project"


def test_null_portfolio_creates_one_explicit_non_task_record():
    raw = b"generic infrastructure"
    source = _source(raw, source_format="R Markdown")
    portfolio = GeneratedPortfolio(
        source_summary="Infrastructure only.",
        disposition="no_objective_task",
        null_reason="The source contains no biological data or result.",
        null_assessment=_assessment("reject"),
        ideas=(),
    )

    result = hydrate_ideas(source, portfolio)

    assert len(result) == 1
    assert result[0].instruction is None
    assert result[0].null_reason == "The source contains no biological data or result."
