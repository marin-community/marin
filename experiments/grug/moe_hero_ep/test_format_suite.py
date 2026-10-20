# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from tokenizers import Tokenizer

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Interface, Panel, PanelCase, Serialization
from experiments.grug.moe_hero_ep.format_measure import PtPanel, PtRow, token_bytes_digest
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, native_evaluation_model
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest, PanelPurpose, SuitePanel, run_suite
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, SampleResult, digest


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


@pytest.mark.parametrize("pt_repetitions", (1, 3))
def test_suite_reads_weights_once_and_persists_real_generations_and_purpose_specific_grades(
    tmp_path, suite_inputs, pt_repetitions
):
    request, tokenizer = suite_inputs
    request = request.model_copy(update={"pt_repetitions": pt_repetitions})
    weight_reads = []
    execution_models = []
    measured_models = []
    measured_losses = []
    answer_id = tokenizer.encode("x")[0]

    def restore(uri, metadata_digest, config, mesh):
        # Replace the expensive checkpoint-storage boundary with real tiny weights.
        weight_reads.append((uri, metadata_digest))
        return Transformer.init(config, key=jax.random.key(0))

    def backend_logits(model, mesh):
        # Only model execution is replaced; rendering/generation/decoding/grading
        # and every result write run through the actual evaluation code.
        assert model.token_embed.dtype == jnp.bfloat16
        assert model.stacked_blocks.stacked.mlp.router_bias.dtype == jnp.float32
        assert measured_models == [model] * pt_repetitions
        assert (tmp_path / "suite-output/pt/report.json").exists()
        execution_models.append(model)

        def logits(tokens, positions):
            scores = np.full((len(tokens), len(tokenizer)), -100.0)
            for row in range(len(tokens)):
                selected = EOT_ID if tokens[row, positions[row]] == answer_id else answer_id
                scores[row, selected] = 0
            return scores

        return logits

    def pt_backend(model, panel, mesh):
        assert not execution_models
        if measured_models:
            assert model is measured_models[0]
        measured_models.append(model)
        loss = (2.0, 2.25, 1.75)[len(measured_models) - 1]
        measured_losses.append(loss)
        return {"loss": loss, "valid_tokens": 4, "routing_counts_per_layer": [[4, 4, 0, 0]]}

    destination = tmp_path / "suite-output"
    with (
        patch("experiments.grug.moe_hero_ep.format_suite.load_tokenizer", return_value=tokenizer),
        patch("experiments.grug.moe_hero_ep.format_suite.restore_weights", side_effect=restore),
        patch("experiments.grug.moe_hero_ep.format_suite.native_logits", side_effect=backend_logits),
        patch("experiments.grug.moe_hero_ep.format_suite.measure_pt_model", side_effect=pt_backend),
    ):
        manifest = run_suite(request, str(destination))
        prior_bytes = (destination / "manifest.json").read_bytes()
        with pytest.raises(FileExistsError):
            run_suite(request, str(destination))
        assert (destination / "manifest.json").read_bytes() == prior_bytes
    assert weight_reads == [(request.checkpoint.uri, request.checkpoint.metadata_digest)]
    assert manifest["model_digest"] == digest(request.model)
    assert manifest["checkpoint"] == request.checkpoint.model_dump()
    assert manifest["manifest_digest"] == digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    for entry, output in zip(request.panels, manifest["panels"], strict=True):
        root = destination / output["result_key"]
        samples = SampleResult.model_validate_json((root / "samples.json").read_bytes())
        assert samples.completions[0].samples[0].token_ids == (answer_id, EOT_ID)
        report = json.loads((root / "report.json").read_bytes())
        assert report["metrics"]["simple"]["successes"] == 1
        assert report["checkpoint"] == request.checkpoint.model_dump()
        assert report["model_config_digest"] == manifest["model_digest"]
        for name, expected_sha256 in output["file_sha256"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_sha256
        if entry.purpose == PanelPurpose.CONTENT:
            assert output["content_panel_id"] == entry.panel.panel_id
            assert json.loads((root / "content.json").read_bytes())["outcomes"] == {"content": True}
        else:
            assert output["content_panel_id"] is None
            # The free-form example must never become an exact-answer accuracy label.
            assert not (root / "content.json").exists()
    pt_report = json.loads((destination / "pt/report.json").read_bytes())
    assert pt_report["loss"] == 2.0
    assert [repeat["loss"] for repeat in pt_report["repeat_measurements"]] == measured_losses
    assert len(measured_models) == pt_repetitions
    assert pt_report["variability"] == {
        "repetitions": pt_repetitions,
        "primary_loss": measured_losses[0],
        "losses": measured_losses,
        "loss_min": min(measured_losses),
        "loss_max": max(measured_losses),
        "loss_range": max(measured_losses) - min(measured_losses),
    }
    assert pt_report["request"]["checkpoint"] == request.checkpoint.model_dump()
    assert manifest["pt"]["panel_sha256"] == request.pt_panel_sha256


@pytest.mark.parametrize("failure", ("second-panel-identity", "second-panel-context", "pt-bytes", "vocabulary"))
def test_suite_rejects_all_input_mismatches_before_checkpoint_io(tmp_path, suite_inputs, failure):
    request, tokenizer = suite_inputs
    if failure == "second-panel-identity":
        second = request.panels[1].model_copy(
            update={"panel": request.panels[1].panel.model_copy(update={"tokenizer_revision": "d" * 40})}
        )
        request = request.model_copy(update={"panels": (request.panels[0], second)})
    elif failure == "second-panel-context":
        second = request.panels[1].model_copy(
            update={"panel": request.panels[1].panel.model_copy(update={"context_length": 65536})}
        )
        request = request.model_copy(update={"panels": (request.panels[0], second)})
    elif failure == "pt-bytes":
        Path(request.pt_panel_path).write_text("{}")
    else:
        config = draccus.decode(GrugModelConfig, request.model)
        request = request.model_copy(update={"model": draccus.encode(dataclasses.replace(config, vocab_size=16))})
    destination = tmp_path / "must-not-create"
    with (
        patch("experiments.grug.moe_hero_ep.format_suite.load_tokenizer", return_value=tokenizer),
        patch(
            "experiments.grug.moe_hero_ep.format_suite.restore_weights",
            side_effect=AssertionError("Validation must precede opening the full-model checkpoint"),
        ),
        pytest.raises(ValueError),
    ):
        run_suite(request, str(destination))
    assert not destination.exists()
