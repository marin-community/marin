# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from finestore.mismatch import (
    MANIFEST_TABLE,
    PROBE_TABLE,
    SCORES_TABLE,
    ManifestRow,
    ProbeRow,
    ScoreRow,
    register_mismatch_tables,
)
from finestore.store import DataStore

from experiments.post_training.mismatch_probe.report import (
    analyze_archive,
    compare_archives,
    render_markdown,
    write_plots,
)


def _archive(
    root,
    *,
    corrupt_token=False,
    with_routes=False,
    partial_route_mask=False,
    cache_mode="off",
    invalid_generation=False,
    native_offset=0.0,
    weight_tag="w",
    reread_weight_override=None,
    tokenizer_fingerprint="toy-tokenizer",
    probe_hash="frozen",
    response_shift=0,
):
    rows = []
    scores = []
    for position in range(4):
        prompt = f"p{position // 2}"
        sample = f"{prompt}:{position % 2}"
        response = [10 + position + response_shift, 20 + position + response_shift]
        captured = np.array([[[1, 2], [2, 3]], [[1, 2], [2, 3]]], dtype=np.uint8)
        rows.append(
            ProbeRow(
                probe_hash=probe_hash,
                sample_id=sample,
                prompt_id=prompt,
                prompt_token_ids=[1, position // 2 + 2],
                trainer_prompt_ids=[1, position // 2 + 2],
                vllm_output_ids=response,
                trainer_input_ids=[response[0], 99] if corrupt_token and position == 0 else response,
                response_mask=[True, True],
                loss_mask=[True, True],
                reward=float(position % 2),
                advantage=1.0 if position % 2 else -1.0,
                request_seed=position,
                batch_position=position,
                routed_experts=captured.tobytes() if with_routes else None,
                routed_experts_shape=list(captured.shape) if with_routes else None,
                routed_experts_dtype=str(captured.dtype) if with_routes else None,
                route_valid_mask=(
                    ([[False, True], [True, True]] if partial_route_mask else [[True, True], [True, True]])
                    if with_routes
                    else None
                ),
            )
        )
        values = {
            "vllm.generate@0": [-3.0, -4.0] if invalid_generation else [-2.0, -3.0],
            "vllm.rescore@0": [-2.0, -3.0],
            "trainer@0:native": [-2.1, -3.1],
            "trainer@0:repeat": [-2.1, -3.1],
            "trainer@0:router_replay": [-2.02, -3.02],
            "trainer@0:router_replay_filtered": [-2.01, -3.01],
            "vllm.rescore@1": [-1.99, -2.99],
            "trainer@1:native": [-2.09, -3.09],
            "vllm.rescore@2": [-1.97, -2.97],
            "trainer@2:native": [-2.07, -3.07],
        }
        if cache_mode == "on":
            values = {
                f"{name}:on" if name.startswith("vllm.rescore@") else name: scores for name, scores in values.items()
            }
        elif cache_mode == "both":
            values.update({f"{name}:on": scores for name, scores in values.items() if name.startswith("vllm.rescore@")})
        for name in values:
            if name.startswith("trainer@"):
                values[name] = [score + native_offset for score in values[name]]
        for name, logprobs in values.items():
            update = int(name.split("@", 1)[1].split(":", 1)[0])
            observed = captured.astype(np.int32).copy() if with_routes and name.startswith("trainer@") else None
            replaced = np.zeros_like(captured, dtype=np.bool_)
            if observed is not None and ":native" in name:
                observed[:, 0] = [0, 1]
            if observed is not None and ":router_replay_filtered" in name:
                observed[1, 0] = [0, 1]
                replaced[1, 0] = [True, True]
            scores.append(
                ScoreRow(
                    probe_hash=probe_hash,
                    sample_id=sample,
                    scoring=name,
                    update=update,
                    weights_hash=(
                        reread_weight_override
                        if reread_weight_override is not None and name.startswith("vllm.rescore@1")
                        else f"{weight_tag}{update}"
                    ),
                    logprobs=logprobs,
                    expert_choices=observed.tobytes() if observed is not None else None,
                    expert_choices_shape=list(observed.shape) if observed is not None else None,
                    expert_choices_dtype=str(observed.dtype) if observed is not None else None,
                    replacement_mask=replaced.tobytes() if observed is not None else None,
                )
            )
    manifest = ManifestRow(
        archive=str(root),
        status="complete",
        probe_hash=probe_hash,
        starting_weights_hash=f"{weight_tag}0",
        architecture="GrugMoeForCausalLM",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=1,
        seed=7,
        bootstrap_seed=8,
        created_at_utc="2026-09-26T00:00:00Z",
        config_json="{}",
        software_json=json.dumps({"tokenizer_fingerprint": tokenizer_fingerprint}),
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json=json.dumps({"trainer@0:native/seconds": 0.1}),
        step_metrics_json="{}",
    )
    with DataStore.open(str(root), writer_id="report-test") as store:
        register_mismatch_tables(store)
        with store.transaction() as transaction:
            for row in rows:
                transaction.table(PROBE_TABLE).add(row.model_dump())
            for row in scores:
                transaction.table(SCORES_TABLE).add(row.model_dump())
            transaction.table(MANIFEST_TABLE).add(manifest.model_dump())


def test_report_recovers_same_weight_modes_paired_intervals_and_drift(tmp_path):
    root = tmp_path / "archive"
    _archive(root)
    report = analyze_archive(str(root), bootstrap_draws=80)
    assert report["token_identity"]["fraction"] == 1.0
    assert report["comparisons"]["implementation_mismatch"]["metrics"]["abs_p99"] > 0.09
    assert report["comparisons"]["trainer_floor"]["metrics"]["abs_p99"] == 0.0
    assert report["paired_improvements"]["router_replay"]["abs_p99"]["ci95"][0] > 0
    assert report["paired_improvements"]["router_replay_filtered"]["abs_p99"]["ci95"][0] > 0
    assert report["checks"]["drift_identity_after_2"]["pass"]
    assert report["checks"]["generation_ratio_sanity"]["pass"]
    assert "router_replay_filtered" in render_markdown(report)
    assert report["input_commit_token"] != "None"
    output = tmp_path / "figures"
    output.mkdir()
    write_plots(report, output)
    assert (output / "mismatch.png").stat().st_size > 0

    cached_root = tmp_path / "cached"
    _archive(cached_root, cache_mode="on")
    cached = analyze_archive(str(cached_root), bootstrap_draws=20)
    assert cached["comparisons"]["reread_mismatch"]["reference"] == "vllm.rescore@0:on"
    assert "mismatch_after_2" in cached["comparisons"]

    both_root = tmp_path / "both-cache-modes"
    _archive(both_root, cache_mode="both")
    both = analyze_archive(str(both_root), bootstrap_draws=20)
    assert "prefix_cache_effect_at_0" in both["comparisons"]


def test_report_stops_numerical_analysis_after_token_mutation(tmp_path):
    root = tmp_path / "archive"
    _archive(root, corrupt_token=True)
    report = analyze_archive(str(root), bootstrap_draws=20)
    assert report["token_identity"]["fraction"] < 1
    assert report["comparisons"] == {}


def test_report_route_agreement_excludes_missing_routes_and_tracks_replacements(tmp_path):
    root = tmp_path / "archive"
    _archive(root, with_routes=True)
    report = analyze_archive(str(root), bootstrap_draws=20)
    native = report["route_diagnostics"]["trainer@0:native"]["metrics"]
    replay = report["route_diagnostics"]["trainer@0:router_replay"]["metrics"]
    filtered = report["route_diagnostics"]["trainer@0:router_replay_filtered"]["metrics"]
    assert native["set_agreement"] == 0.5
    assert replay["set_agreement"] == 1.0
    assert filtered["set_agreement"] == 0.75
    assert filtered["replacement_fraction"] == 0.25
    assert "layer_0/set_agreement" in report["route_diagnostics"]["trainer@0:native"]["ci95"]

    masked_root = tmp_path / "partially-masked"
    _archive(masked_root, with_routes=True, partial_route_mask=True)
    masked = analyze_archive(str(masked_root), bootstrap_draws=20)
    masked_native = masked["route_diagnostics"]["trainer@0:native"]["metrics"]
    assert masked_native["set_agreement"] == pytest.approx(2 / 3)


def test_report_refuses_invalid_generation_distribution(tmp_path):
    root = tmp_path / "archive"
    _archive(root, invalid_generation=True)
    with pytest.raises(ValueError, match="sampling-distribution check failed"):
        analyze_archive(str(root), bootstrap_draws=20)


def test_report_refuses_same_update_comparison_with_different_weights(tmp_path):
    root = tmp_path / "different-reread-weights"
    _archive(root, reread_weight_override="wrong-update-1")
    with pytest.raises(ValueError, match="comparison mismatch_after_1 scores different weights"):
        analyze_archive(str(root), bootstrap_draws=20)


def test_configuration_comparison_pairs_prompts_and_requires_same_starting_weights(tmp_path):
    left, right, changed_weights, changed_tokenizer, independent = (
        tmp_path / name for name in ("left", "right", "changed-weights", "changed-tokenizer", "independent")
    )
    _archive(left)
    _archive(right, native_offset=0.04)
    _archive(changed_weights, weight_tag="other")
    _archive(changed_tokenizer, tokenizer_fingerprint="other-tokenizer")
    _archive(independent, probe_hash="another-answer-set", response_shift=1)
    paired = compare_archives(str(left), str(right), bootstrap_draws=40)
    assert paired["kind"] == "shared_tokens"
    effect = paired["comparisons"]["implementation_mismatch"]
    assert effect["metrics"]["abs_p99_left_minus_right"] > 0
    assert effect["ci95"]["abs_p99_left_minus_right"][0] > 0
    with pytest.raises(ValueError, match="same starting weight hash"):
        compare_archives(str(left), str(changed_weights), bootstrap_draws=20)
    with pytest.raises(ValueError, match="same tokenizer fingerprint"):
        compare_archives(str(left), str(changed_tokenizer), bootstrap_draws=20)
    assert compare_archives(str(left), str(independent), bootstrap_draws=20)["kind"] == "independent_generation"
