# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared local native-suite fixtures for format handoff and serving tests."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import draccus
import jax
import numpy as np
import pytest
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from tokenizers import Tokenizer

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Candidate, Interface, Panel, PanelCase, Serialization
from experiments.grug.moe_hero_ep.format_handoff import CandidateHandoffRequest
from experiments.grug.moe_hero_ep.format_measure import PtPanel, PtRow, token_bytes_digest
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, native_evaluation_model
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest, PanelPurpose, SuitePanel, run_suite
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, digest


@pytest.fixture
def suite_inputs(tmp_path):
    stage_gpt2_tokenizer(Path("lib/levanter/tests"), tmp_path)
    underlying = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    underlying.add_tokens([f"<unused-{index}>" for index in range(EOT_ID - underlying.get_vocab_size())])
    underlying.add_special_tokens(["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"])
    underlying.save(str(tmp_path / "tokenizer.json"))
    tokenizer = load_tokenizer(str(tmp_path))
    config = native_evaluation_model(
        GrugModelConfig(
            vocab_size=len(tokenizer),
            hidden_dim=16,
            intermediate_dim=8,
            shared_expert_intermediate_dim=8,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=16384,
            sliding_window=8,
            attention_implementation="reference",
        )
    )
    panel = Panel(
        release="suite-test",
        cases=(
            PanelCase(
                id="greeting",
                interface=Interface.SIMPLE,
                messages=({"role": "user", "content": "Hi"},),
                legacy_prompt="Hi",
                seed=0,
                source="local-fixture",
                expected="A human-written free-form example",
            ),
        ),
        tokenizer="pinned-local-tokenizer",
        tokenizer_revision="a" * 40,
        source_revision="b" * 40,
        backend=NATIVE_BACKEND,
        max_new_tokens=2,
        temperature=0,
    )
    content_panel = panel.model_copy(
        update={"cases": (panel.cases[0].model_copy(update={"id": "content", "expected": "x"}),)}
    )
    tokens, weights, segments = (2, 3, 4, 5), (1.0, 0.5, 1.0, 0.0), (0, 0, 0, 0)
    data = {"heldout-source": "pinned-pt-cache"}
    pt = PtPanel(
        source_data_config=data,
        source_data_config_digest=digest(data),
        component="pt",
        tags=("heldout",),
        tokenizer=panel.tokenizer,
        tokenizer_revision=panel.tokenizer_revision,
        sequence_length=4,
        rows=(
            PtRow(
                row_index=0,
                tokens=tokens,
                loss_weight=weights,
                segment_ids=segments,
                arrays_sha256=token_bytes_digest(tokens, weights, segments),
            ),
        ),
    )
    pt_path = tmp_path / "pt.json"
    pt_path.write_text(pt.model_dump_json() + "\n")
    request = NativeSuiteRequest(
        checkpoint=Checkpoint(
            uri="unread-full-model",
            run_id="suite-test",
            step=294000,
            timestamp="2026-10-20",
            metadata_digest="c" * 64,
        ),
        model=draccus.encode(config),
        source_revision=panel.source_revision,
        tokenizer=panel.tokenizer,
        tokenizer_revision=panel.tokenizer_revision,
        target_cluster="explicit-test-target",
        expert_axis_size=1,
        replica_axis_size=1,
        panels=(
            SuitePanel(panel=panel, serialization=Serialization.MARIN, purpose=PanelPurpose.FORMAT),
            SuitePanel(panel=content_panel, serialization=Serialization.MARIN, purpose=PanelPurpose.CONTENT),
        ),
        pt_repetitions=1,
        pt_panel_path=str(pt_path),
        pt_panel_sha256=hashlib.sha256(pt_path.read_bytes()).hexdigest(),
    )
    return request, tokenizer


@pytest.fixture
def completed_suite(tmp_path, suite_inputs):
    request, tokenizer = suite_inputs
    checkpoint = tmp_path / "A"
    checkpoint.mkdir()
    metadata = {"is_temporary": False, "step": 294000}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    reference = request.checkpoint.model_copy(update={"uri": str(checkpoint), "metadata_digest": digest(metadata)})
    content = request.panels[1]
    request = request.model_copy(
        update={
            "checkpoint": reference,
            "panels": (
                *request.panels,
                SuitePanel(panel=content.panel, serialization=Serialization.LEGACY, purpose=PanelPurpose.CONTENT),
            ),
        }
    )
    answer = tokenizer.encode("x")[0]

    def logits_backend(model, mesh):
        def logits(tokens, positions):
            scores = np.full((len(tokens), len(tokenizer)), -100.0)
            for row in range(len(tokens)):
                stop = 128009 if tokens[row, positions[row]] == answer else answer
                # Legacy does not stop on EOT: emitting it still supplies a wrong,
                # fully retained answer rather than fabricating an exact success.
                scores[row, stop] = 0
            return scores

        return logits

    def pt_backend(model, panel, mesh):
        return {
            "rows": [
                {
                    "row_index": 0,
                    "arrays_sha256": panel.rows[0].arrays_sha256,
                    "loss_sum": 5.0,
                    "loss_weight": 2.5,
                    "loss": 2.0,
                    "valid_tokens": 4,
                }
            ],
            "routing_batches": [
                {
                    "row_indices": [0],
                    "filler_rows": 0,
                    "valid_tokens": 4,
                    "routing_counts_per_layer": [[4, 4, 0, 0]],
                    "routing_entropy_mean": 0.5,
                    "router_z_loss": 0.2,
                }
            ],
            "batch_size": 1,
            "forward_passes": 1,
            "loss_sum": 5.0,
            "loss_weight": 2.5,
            "loss": 2.0,
            "valid_tokens": 4,
            "routing_counts_per_layer": [[4, 4, 0, 0]],
            "routing_entropy_mean": 0.5,
            "router_z_loss": 0.2,
            "routing_policy": "all non-padding inputs",
            "routing_aggregation": "valid-token-weighted means",
        }

    destination = tmp_path / "suite"
    with (
        patch("experiments.grug.moe_hero_ep.format_suite.load_tokenizer", return_value=tokenizer),
        patch(
            "experiments.grug.moe_hero_ep.format_suite.restore_weights",
            side_effect=lambda uri, metadata_digest, config, mesh: Transformer.init(config, key=jax.random.key(0)),
        ),
        patch("experiments.grug.moe_hero_ep.format_suite.native_logits", side_effect=logits_backend),
        patch("experiments.grug.moe_hero_ep.format_suite.measure_pt_model", side_effect=pt_backend),
    ):
        manifest = run_suite(request, str(destination))
    handoff = CandidateHandoffRequest(
        candidate=Candidate.A,
        parent=None,
        suite_root=str(destination),
        suite_manifest_digest=manifest["manifest_digest"],
        checkpoint=reference,
        model_config_digest=manifest["model_digest"],
        tokenizer=request.tokenizer,
        tokenizer_revision=request.tokenizer_revision,
        format_panel_id=request.panels[0].panel.panel_id,
        content_panel_id=content.panel.panel_id,
        pt_panel_path=request.pt_panel_path,
        pt_panel_sha256=request.pt_panel_sha256,
    )
    return handoff, manifest
