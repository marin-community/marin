# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from finestore.mismatch import (
    LAYERS_TABLE,
    MANIFEST_TABLE,
    PROBE_TABLE,
    SCORES_TABLE,
    LayerRow,
    ManifestRow,
    ProbeRow,
    ScoreRow,
    register_mismatch_tables,
)
from finestore.reader import ReadView
from finestore.store import DataStore


def test_mismatch_archive_round_trip_retains_tokens_routes_float32_scores_and_completion(tmp_path):
    root = str(tmp_path / "mismatch")
    probe = ProbeRow(
        probe_hash="hash",
        sample_id="sample-0",
        prompt_id="prompt-0",
        prompt_token_ids=[3, 4],
        trainer_prompt_ids=[3, 4],
        vllm_output_ids=[7, 9],
        trainer_input_ids=[7, 9],
        response_mask=[True, True],
        loss_mask=[True, True],
        reward=1.0,
        advantage=0.5,
        request_seed=91,
        batch_position=0,
        routed_experts=b"\x00\x03\x00\x00",
        routed_experts_shape=[2, 1, 2],
        routed_experts_dtype="uint8",
        route_valid_mask=[[True], [False]],
    )
    with pytest.raises(ValueError, match="route validity must match"):
        ProbeRow.model_validate(probe.model_dump() | {"routed_experts": b"\x00\x03", "routed_experts_shape": [1, 1, 2]})
    score = ScoreRow(
        probe_hash="hash",
        sample_id="sample-0",
        scoring="trainer@0:native",
        update=0,
        weights_hash="weights",
        logprobs=[-0.125, -2.75],
        expert_choices=b"\x00\x03\x00\x00",
        expert_choices_shape=[2, 1, 2],
        expert_choices_dtype="uint8",
        replacement_mask=b"\x00\x01\x00\x00",
    )
    layer = LayerRow(
        probe_hash="hash",
        sample_id="sample-0",
        scoring="trainer@0:native",
        layer=0,
        output_bytes=b"\x00\x3c\x00\x40",
        shape=[2, 1],
        dtype="bfloat16-little-endian",
        token_positions=[2, 3],
    )
    manifest = ManifestRow(
        archive=root,
        status="building",
        probe_hash="hash",
        starting_weights_hash="weights",
        architecture="GrugMoeForCausalLM",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=1,
        seed=91,
        bootstrap_seed=92,
        created_at_utc="2026-09-26T00:00:00Z",
        config_json=json.dumps({"score_after_updates": [0, 1, 2]}),
        software_json="{}",
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json="{}",
        step_metrics_json="{}",
    )

    with DataStore.open(root, writer_id="test") as store:
        register_mismatch_tables(store)
        with store.transaction() as transaction:
            for table, row in (
                (PROBE_TABLE, probe),
                (SCORES_TABLE, score),
                (LAYERS_TABLE, layer),
                (MANIFEST_TABLE, manifest),
            ):
                transaction.table(table).add(row.model_dump())
        with store.transaction() as transaction:
            transaction.table(MANIFEST_TABLE).add(manifest.model_copy(update={"status": "complete"}).model_dump())

    view = ReadView(root)
    observed_probe = ProbeRow.model_validate(view.scan(PROBE_TABLE).to_pylist()[0])
    observed_score = ScoreRow.model_validate(view.scan(SCORES_TABLE).to_pylist()[0])
    observed_layer = LayerRow.model_validate(view.scan(LAYERS_TABLE).to_pylist()[0])
    observed_manifest = ManifestRow.model_validate(view.scan(MANIFEST_TABLE).to_pylist()[0])

    assert observed_probe == probe
    assert observed_score == score
    assert observed_layer == layer
    assert observed_manifest.status == "complete"
    assert view.scan(SCORES_TABLE).schema.field("logprobs").type == pa.list_(pa.float32())
    assert view.scan(SCORES_TABLE, columns=["probe_hash", "sample_id", "logprobs"]).to_pylist() == [
        {"probe_hash": "hash", "sample_id": "sample-0", "logprobs": [-0.125, -2.75]}
    ]
