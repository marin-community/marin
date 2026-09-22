# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
from marin.execution.lazy import materialized_config

from experiments.post_training.curriculum_rl.pool import QWEN3_MODEL
from experiments.post_training.curriculum_sft.pipeline import (
    QWEN_EXECUTION_CLUSTER,
    CurriculumSftRequest,
    build_pipeline,
    completion_body,
    parse_batch_output,
    subject_and_capabilities,
)
from experiments.post_training.task_curriculum.models import CurriculumCatalog


def _catalog() -> CurriculumCatalog:
    return CurriculumCatalog.model_validate(
        {
            "catalog_version": "test",
            "curricula": [
                {
                    "routing_facet": "subject_domain",
                    "curriculum": {
                        "version": "test-d27",
                        "subject_id": "D27",
                        "subject_name": "Finance, Accounting & Audit",
                        "sections": [
                            {
                                "id": "d27.reporting.analysis",
                                "parent_id": None,
                                "name": "Calculate Reported Financial Measures",
                                "kind": "capability",
                                "outcome": "Calculate reported measures from supplied disclosures.",
                                "includes": ["ratio and subtotal calculations"],
                                "excludes": ["investment recommendations"],
                                "prerequisites": [],
                                "sample_tasks": [
                                    {"kind": "entry", "instruction": "Calculate one disclosed subtotal."},
                                    {
                                        "kind": "representative",
                                        "instruction": "Reconcile two reported measures from a note.",
                                    },
                                ],
                            }
                        ],
                    },
                }
            ],
        }
    )


def _batch_response(arguments: dict) -> str:
    row = {
        "custom_id": "examples-000000-01",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_examples",
                                        "arguments": json.dumps(arguments),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        },
    }
    return json.dumps(row)


def test_subject_and_capabilities_preserves_requested_order() -> None:
    subject, capabilities = subject_and_capabilities(_catalog(), "D27", ("d27.reporting.analysis",))

    assert subject == "Finance, Accounting & Audit"
    assert [capability.id for capability in capabilities] == ["d27.reporting.analysis"]


def test_subject_and_capabilities_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unknown capabilities"):
        subject_and_capabilities(_catalog(), "D27", ("d27.reporting.missing",))


def test_completion_body_requests_low_reasoning_effort() -> None:
    _, capabilities = subject_and_capabilities(_catalog(), "D27", ("d27.reporting.analysis",))

    body = completion_body(
        subject_name="Finance, Accounting & Audit",
        capabilities=capabilities,
        task="financial disclosure calculation",
        evaluation_area="financial question answering",
        expected_examples=1,
        seed=17,
    )

    assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_parse_batch_output_emits_canonical_messages_with_provenance() -> None:
    output = _batch_response(
        {
            "examples": [
                {
                    "id": 0,
                    "task_summary": "Compute a disclosed margin.",
                    "user": "A fictional issuer reports revenue of 10 and cost of 6.",
                    "assistant": "Gross margin is (10 - 6) / 10 = 40%.",
                    "verification": "Recompute (10 - 6) / 10.",
                }
            ]
        }
    )

    rows = parse_batch_output(
        output,
        {"examples-000000-01": (0, 1)},
        request=CurriculumSftRequest(
            subject_id="D27",
            capability_ids=("d27.reporting.analysis",),
            task="financial disclosure calculation",
            evaluation_area="financial question answering",
        ),
    )

    assert [message["role"] for message in rows[0]["messages"]] == ["user", "assistant"]
    assert rows[0]["metadata"]["capability_ids"] == ["d27.reporting.analysis"]
    assert rows[0]["metadata"]["sample_index"] == 0


def test_parse_batch_output_rejects_non_array_examples() -> None:
    output = _batch_response({"examples": '[{"id":0,"user":"Question","assistant":"Answer"}]'})

    with pytest.raises(ValueError, match="examples as str, expected array"):
        parse_batch_output(
            output,
            {"examples-000000-01": (0, 1)},
            request=CurriculumSftRequest(
                subject_id="D27",
                capability_ids=("d27.reporting.analysis",),
                task="financial disclosure calculation",
                evaluation_area="financial question answering",
            ),
        )


def test_pipeline_connects_generation_training_and_reevaluation(tmp_path: Path) -> None:
    pipeline = build_pipeline(
        request=CurriculumSftRequest(
            subject_id="D27",
            capability_ids=("d27.reporting.analysis",),
            task="financial disclosure calculation",
            evaluation_area="financial question answering",
        ),
        eval_config=Path("experiments/evaluation/configs/evalchemy/financebench.yaml"),
        sample_count=8,
        version="2026.09.21",
    )

    assert pipeline.data in pipeline.sft.deps
    assert pipeline.sft in pipeline.reevaluation.deps
    baseline_config = materialized_config(pipeline.baseline, "gs://test-prefix")
    assert baseline_config.evals is None
    assert baseline_config.evalchemy_config_path == "experiments/evaluation/configs/evalchemy/financebench.yaml"
    assert baseline_config.model.tokenizer == QWEN3_MODEL
    assert baseline_config.submission_cluster == QWEN_EXECUTION_CLUSTER
    assert baseline_config.federated_cluster is None

    checkpoint_root = tmp_path / pipeline.sft.name / pipeline.sft.version / "hf"
    older = checkpoint_root / "step-7"
    latest = checkpoint_root / "step-15"
    for checkpoint, timestamp in ((older, 1), (latest, 2)):
        checkpoint.mkdir(parents=True)
        config_path = checkpoint / "config.json"
        config_path.write_text("{}")
        (checkpoint / "tokenizer_config.json").write_text("{}")
        os.utime(config_path, (timestamp, timestamp))

    reevaluation_config = materialized_config(pipeline.reevaluation, str(tmp_path))
    assert reevaluation_config.model.location == str(latest)
