# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from itertools import batched

import numpy as np
import pytest
from finestore.rl.mismatch_probe import (
    MANIFEST_TABLE,
    PROBE_TABLE,
    SCORES_TABLE,
    ManifestRow,
    ProbeRow,
    ScoreRow,
    register_mismatch_tables,
)
from finestore.store import DataStore

from experiments.post_training.hero_rl.report import GateStatus, main, qualify_archive

MODEL_SOURCE = "s3://test-completed-exports/hero-step144000"


def test_missing_archive_writes_an_invalid_report(tmp_path):
    output = tmp_path / "report"
    exit_code = main(
        [
            str(tmp_path / "missing-archive"),
            "--expected-model-source",
            MODEL_SOURCE,
            "--expected-starting-step",
            "0",
            "--output-dir",
            str(output),
        ]
    )
    report = json.loads((output / "report.json").read_text())
    assert exit_code == 2
    assert report["status"] == GateStatus.INVALID
    assert report["counts"] is None
    assert "no manifest table" in report["reasons"][0]


def _archive(root, *, tokens_per_response=2500, outliers=0, corruption=None):
    probes = []
    scores = []
    width = tokens_per_response + 1
    layers = 1 if corruption == "partial_routes" else 48
    routes = np.tile(np.arange(8, dtype=np.uint16)[None, None, :], (width, layers, 1))
    for index in range(20):
        sample = f"p{index // 2}:{index % 2}"
        prompt = [1, index // 2 + 2]
        response = [10 + index] * width
        mask = [True] * tokens_per_response + [False]
        probe = ProbeRow(
            probe_hash="frozen-bank",
            sample_id=sample,
            prompt_id=f"p{index // 2}",
            prompt_token_ids=prompt,
            trainer_prompt_ids=prompt,
            vllm_output_ids=response,
            trainer_input_ids=response,
            response_mask=mask,
            loss_mask=mask,
            request_seed=index,
            batch_position=index,
            routed_experts=routes.tobytes(),
            routed_experts_shape=list(routes.shape),
            routed_experts_dtype=str(routes.dtype),
            route_valid_mask=[[True] * layers] * width,
        )
        if index == 0 and corruption == "token":
            probe.trainer_input_ids = [99, *response[1:]]
        if index == 1 and corruption == "request":
            probe.request_seed = 0
        probes.append(probe)
        generation = [-2.0] * width
        replay = generation.copy()
        # Masked positions have a large mismatch and must never inflate either count.
        replay[-1] = -0.01
        if index == 0:
            replay[:outliers] = [-2.0 + math.log(0.5)] * outliers
            if corruption == "nonfinite":
                replay[0] = math.nan
        for scorer, mode, update, values in (
            ("vllm.generate", "", 0, generation),
            ("trainer", "native", 0, generation),
            ("trainer", "router_replay", 0, replay),
            ("trainer", "repeat", 0, replay),
            ("trainer", "router_replay", 1, [-1.0] * width),
        ):
            if corruption == "missing_replay" and mode == "router_replay" and update == 0:
                continue
            has_routes = mode == "router_replay" and update == 0
            observed = routes.copy() if has_routes else None
            if index == 0 and corruption == "route" and observed is not None:
                observed[0, 0, 0] = 3
            scores.append(
                ScoreRow(
                    probe_hash="frozen-bank",
                    sample_id=sample,
                    scorer=scorer,
                    mode=mode,
                    update=update,
                    global_step=(1 if corruption == "stale" and mode == "router_replay" and update == 0 else update),
                    logprobs=values,
                    expert_choices=observed.tobytes() if observed is not None else None,
                    expert_choices_shape=list(routes.shape) if observed is not None else None,
                    expert_choices_dtype=str(routes.dtype) if observed is not None else None,
                )
            )
    model = {"path": "/staged/hero", "source_uri": MODEL_SOURCE, "source_identity": "completed-export@sha256:abc"}
    manifest = ManifestRow(
        archive=str(root),
        status="complete",
        probe_hash="frozen-bank",
        checkpoint_path="/other/checkpoint" if corruption == "checkpoint" else model["path"],
        runtime_commit="pinned-runtime",
        tokenizer_fingerprint="pinned-tokenizer",
        starting_global_step=0,
        scored_updates=[0, 1],
        scored_global_steps=[0, 1],
        architecture="GrugMoeForCausalLM",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=1,
        seed=42,
        bootstrap_seed=17,
        created_at_utc="2026-11-03T12:00:00Z",
        config_json=json.dumps({"trainer": {"policy": {"model": model}}}),
        software_json="{}",
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json="{}",
        step_metrics_json="{}",
    )
    with DataStore.open(str(root), writer_id="hero-gate-test") as store:
        register_mismatch_tables(store)
        for table, rows in ((PROBE_TABLE, probes), (SCORES_TABLE, scores)):
            for batch in batched(rows, 8):
                with store.transaction() as transaction:
                    for row in batch:
                        transaction.table(table).add(row.model_dump())
        with store.transaction() as transaction:
            transaction.table(MANIFEST_TABLE).add(manifest.model_dump())


def test_gate_counts_masked_initial_tokens_once_and_writes_report(tmp_path):
    root = tmp_path / "archive"
    _archive(root, outliers=49)
    output = tmp_path / "report"
    exit_code = main(
        [
            str(root),
            "--expected-model-source",
            MODEL_SOURCE,
            "--expected-starting-step",
            "0",
            "--output-dir",
            str(output),
        ]
    )
    result = json.loads((output / "report.json").read_text())
    assert exit_code == 0
    assert result["status"] == "PASS"
    assert result["counts"] == {
        "independent_responses": 20,
        "prompts": 10,
        "response_tokens": 50_000,
        "ratios_outside_interval": 49,
        "outside_fraction": 49 / 50_000,
    }
    assert result["provenance"]["model_source"] == MODEL_SOURCE
    assert result["provenance"]["archive_commit_token"]
    assert (output / "report.md").is_file()


@pytest.mark.parametrize("tokens_per_response,outliers", [(2499, 0), (2500, 50)])
def test_gate_rejects_underpowered_bank_and_outlier_threshold(tmp_path, tokens_per_response, outliers):
    root = tmp_path / "archive"
    _archive(root, tokens_per_response=tokens_per_response, outliers=outliers)
    result = qualify_archive(str(root), expected_model_source=MODEL_SOURCE, expected_starting_step=0)
    assert result.status == GateStatus.FAIL
    assert result.counts is not None
    assert result.counts.response_tokens == 20 * tokens_per_response
    assert result.counts.ratios_outside_interval == outliers


@pytest.mark.parametrize(
    "corruption", ["token", "stale", "checkpoint", "request", "route", "partial_routes", "nonfinite", "missing_replay"]
)
def test_gate_marks_nonindependent_or_mismatched_evidence_invalid(tmp_path, corruption):
    root = tmp_path / "archive"
    _archive(root, corruption=corruption)
    result = qualify_archive(str(root), expected_model_source=MODEL_SOURCE, expected_starting_step=0)
    assert result.status == GateStatus.INVALID
    assert result.reasons
    assert result.counts is None


def test_gate_cli_returns_nonzero_and_preserves_expected_source_for_invalid_archive(tmp_path):
    root = tmp_path / "archive"
    _archive(root)
    output = tmp_path / "report"
    exit_code = main(
        [
            str(root),
            "--expected-model-source",
            "s3://test-completed-exports/different-policy",
            "--expected-starting-step",
            "0",
            "--output-dir",
            str(output),
        ]
    )
    result = json.loads((output / "report.json").read_text())
    assert exit_code != 0
    assert result["status"] == "INVALID"
    assert result["provenance"]["model_source"] == MODEL_SOURCE
    assert result["expected_model_source"] != MODEL_SOURCE
