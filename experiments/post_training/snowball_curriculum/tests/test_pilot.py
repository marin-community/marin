# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import pyarrow.parquet as pq
import pytest
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review_requests import RequestObservation, RequestOutput

from experiments.post_training.snowball_curriculum.annotations import AnnotationModel
from experiments.post_training.snowball_curriculum.pilot import annotate_pilot, embed_pilot, freeze_pilot, load_frozen


def test_pilot_freeze_filter_quality_denominator_and_public_descriptions(tmp_path):
    sources = []
    for source in ("small", "large", "eval"):
        path = tmp_path / f"{source}.jsonl"
        tasks = []
        for index in range(2 if source != "large" else 6):
            prompt = "Add 2 and 3." if index == 0 else f"Add {index + 10} and {len(source)}."
            task = numeric_answer_task(
                RawRow(
                    id=f"{source}-{index}",
                    source=Source(dataset=source, revision="1", row=str(index), importer_revision="1"),
                    data={},
                ),
                prompt=prompt,
                answer="987654321",
                tolerance_abs=0,
                tolerance_rel=0,
            )
            task = cast(TaskSpec, task)
            tasks.append({"task_json": task.model_dump_json()})
        path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
        sources.append(
            {
                "source_id": source,
                "intended_use": "eval" if source == "eval" else "train",
                "population": "converted",
                "task_paths": [str(path)],
            }
        )
    source_manifest = tmp_path / "sources.json"
    source_manifest.write_text(json.dumps({"sources": sources}))
    frozen = tmp_path / "frozen"
    manifest = freeze_pilot(
        str(source_manifest), output=frozen, target=4, seed=7, heldout_contract_ids=[], heldout_public_content_ids=[]
    )
    assert manifest["selected"] == 4
    assert manifest["shortfall"] == 0
    assert manifest["rejected_counts"]["heldout_overlap"] > 0
    original, rows = load_frozen(frozen)
    assert all(row["source_id"] != "eval" for row in rows)
    identifiers = [row["contract_id"] for row in rows]
    keep = set(identifiers[:2])
    annotation_inputs = []

    def submit(requests):
        observations = []
        for request in requests:
            key = request["custom_id"]
            tool = request["body"]["tools"][0]["function"]["name"]
            if tool == "quality_decision":
                value = {
                    "verdict": "keep" if key in keep else "reject",
                    "reason_codes": ["fixture"],
                    "evidence": ["Fixture review outcome"],
                    "confidence": 0.8,
                }
            else:
                annotation_inputs.append(request["body"]["messages"][1]["content"])
                value = {
                    "summary": (
                        "Elementary arithmetic addition of two integers, obtaining the sum of the operands "
                        "and returning a single numerical answer under the public contract."
                    ),
                    "domain_tags": ["arithmetic"],
                    "skill_tags": ["addition"],
                    "interaction_mode": "text",
                    "difficulty": "easy",
                    "difficulty_rationale": "Single operation",
                    "difficulty_confidence": 0.8,
                }
            body = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {"name": tool, "arguments": json.dumps(value)},
                                }
                            ],
                        },
                    }
                ]
            }
            observations.append(
                RequestObservation(
                    (key,), json.dumps({"custom_id": key, "response": {"status_code": 200, "body": body}})
                )
            )
        return RequestOutput(tuple(requests), observations)

    model = AnnotationModel("fixture", "rev-1", "http://fixture/v1")
    output = tmp_path / "annotations"
    result = annotate_pilot(
        frozen,
        output=output,
        quality_model=model,
        annotation_model=model,
        quality_submit=submit,
        annotation_submit=submit,
        cache_root=str(tmp_path / "cache"),
        execute=True,
    )
    assert result["original_denominator"] == 4
    assert result["quality_counts"] == {"keep": 2, "reject": 2}
    assert not result["refill_after_quality"]
    sidecars = [json.loads(line) for line in (output / "sidecars.jsonl").read_text().splitlines()]
    assert len(sidecars) == 4
    descriptions = pq.read_table(output / "descriptions.parquet").to_pylist()
    assert len(descriptions) == 2
    kept_hashes = {row["content_hash"] for row in sidecars if row["annotation"] is not None}
    assert kept_hashes == {row["content_hash"] for row in descriptions}
    assert any(row["content_hash"] != row["public_content_id"] for row in sidecars)
    embedding_plan = embed_pilot(
        output / "descriptions.parquet",
        output=tmp_path / "embedding-plan",
        cache_root=str(tmp_path / "vectors-cache"),
        execute=False,
        device="cpu",
    )
    embedding_inputs = pq.read_table(embedding_plan["embedding_inputs"]).to_pylist()
    assert kept_hashes == {row["content_hash"] for row in embedding_inputs}
    # Different task summaries can share a vector, with every task retaining an explicit join.
    assert len({row["embedding_key"] for row in embedding_inputs}) == 1
    assert not embedding_plan["executed"]
    assert all("987654321" not in value for value in annotation_inputs)
    assert load_frozen(frozen)[0] == original
    (frozen / "tasks.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="Frozen task selection"):
        load_frozen(frozen)
