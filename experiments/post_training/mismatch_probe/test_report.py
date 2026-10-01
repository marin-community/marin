# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from finestore.mismatch_probe import (
    MANIFEST_TABLE,
    PROBE_TABLE,
    SCORES_TABLE,
    ManifestRow,
    ProbeRow,
    ScoreRow,
    register_mismatch_tables,
)
from finestore.reader import ReadView
from finestore.store import DataStore

from experiments.post_training.mismatch_probe import report as report_module
from experiments.post_training.mismatch_probe.report import (
    analyze_archive,
    compare_archives,
    render_archive_comparison,
    render_markdown,
    write_plots,
)


def _archive(
    root,
    *,
    corrupt_token=False,
    corrupt_prompt=False,
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
    forward_seconds=0.1,
    matching_native_routes=False,
    missing_route_layer=False,
    frozen_reread=False,
    logged_reread=False,
):
    rows = []
    scores = []
    for position in range(4):
        prompt = f"p{position // 2}"
        sample = f"{prompt}:{position % 2}"
        response = [10 + position + response_shift, 20 + position + response_shift]
        captured = np.array([[[1, 2], [2, 3]], [[1, 2], [2, 3]]], dtype=np.uint8)
        if missing_route_layer:
            captured[:, 0] = 0
        # Re-read routes cover the two prompt inputs and response token 0; response token 1 is never an input.
        reread_routes = np.array([[[1, 2], [2, 3]]] * 3, dtype=np.uint8)
        reread_routes[2] = [[[1, 2], [2, 3]], [[1, 2], [2, 3]], [[1, 2], [3, 4]], [[0, 1], [2, 3]]][position]
        again_routes = reread_routes.copy()
        if position == 3:
            again_routes[2, 1] = [4, 5]
        frozen_routes = reread_routes.copy()
        frozen_routes[2] = [[0, 1], [2, 3]]
        vllm_routes = {
            "vllm.rescore@0": reread_routes,
            "vllm.rescore_again@0": again_routes,
            "vllm.rescore_frozen@0": frozen_routes,
        }
        rows.append(
            ProbeRow(
                probe_hash=probe_hash,
                sample_id=sample,
                prompt_id=prompt,
                prompt_token_ids=[1, position // 2 + 2],
                trainer_prompt_ids=(
                    [99, position // 2 + 2] if corrupt_prompt and position == 0 else [1, position // 2 + 2]
                ),
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
                    ([[False, True], [True, True]] if partial_route_mask else (captured != 0).any(-1).tolist())
                    if with_routes
                    else None
                ),
            )
        )
        values = {
            "vllm.generate@0": [-3.0, -4.0] if invalid_generation else [-2.0, -3.0],
            "vllm.rescore@0": [-2.0, -3.0],
            "vllm.rescore_again@0": [-2.0, -3.004],
            "trainer@0:native": [-2.1, -3.1],
            "trainer@0:repeat": [-2.1, -3.1],
            "trainer@0:native_again": [-2.1, -3.1],
            "trainer@0:router_replay": [-2.02, -3.02],
            "trainer@0:repeat_replay": [-2.02, -3.02],
            "trainer@0:router_replay_response": [-2.05, -3.05],
            "trainer@0:router_replay_filtered": [-2.01, -3.01],
            "trainer@0:fp32_head": [-2.03, -3.03],
            "trainer@0:reread_replay": [-2.02, -3.02],
            "trainer@0:repeat_reread_replay": [-2.02, -3.02],
            "trainer@0:reread_replay+stack": [-2.02, -3.03],
            # Under the stack the batch layout moves the second token's score by 0.02.
            "trainer@0:repeat_reread_replay+stack": [-2.02, -3.05],
            # Closer to the re-read than the kept stack on prompt p0 and farther on p1.
            "trainer@0:prompt_dependent": [-2.01, -3.01] if position < 2 else [-2.03, -3.03],
            "vllm.rescore@1": [-1.99, -2.99],
            "trainer@1:native": [-2.09, -3.09],
            "vllm.rescore@2": [-1.97, -2.97],
            "trainer@2:native": [-2.07, -3.07],
        }
        if frozen_reread:
            values["vllm.rescore_frozen@0"] = [-1.99, -2.99]
        if logged_reread:
            # Byte-equal to the re-read on prompt p0, whose prefixes ran in CUDA-graph steps, and not on p1.
            values["trainer@0:steps"] = [-2.0, -3.0] if position < 2 else [-2.0, -3.01]
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
            if observed is not None and ":native" in name and not matching_native_routes:
                observed[:, 0] = [0, 1]
            if observed is not None and ":router_replay_filtered" in name:
                observed[1, 0] = [0, 1]
                replaced[1, 0] = [True, True]
            if observed is not None and name.endswith("reread_replay"):
                observed[0] = vllm_routes["vllm.rescore_frozen@0" if frozen_reread else "vllm.rescore@0"][2]
            routes = observed if observed is not None else vllm_routes.get(name) if with_routes else None
            scores.append(
                ScoreRow(
                    probe_hash=probe_hash,
                    sample_id=sample,
                    scorer="trainer" if name.startswith("trainer@") else name.split("@", 1)[0],
                    mode=name.split(":", 1)[1] if name.startswith("trainer@") else "",
                    cache_mode=("on" if name.endswith(":on") else "off") if name.startswith("vllm.rescore") else None,
                    update=update,
                    weights_hash=(
                        reread_weight_override
                        if reread_weight_override is not None and name.startswith("vllm.rescore@1")
                        else f"{weight_tag}{update}"
                    ),
                    logprobs=logprobs,
                    expert_choices=routes.tobytes() if routes is not None else None,
                    expert_choices_shape=list(routes.shape) if routes is not None else None,
                    expert_choices_dtype=str(routes.dtype) if routes is not None else None,
                    replacement_mask=replaced.tobytes() if observed is not None else None,
                )
            )
    manifest = ManifestRow(
        archive=str(root),
        status="complete",
        probe_hash=probe_hash,
        starting_weights_hash=f"{weight_tag}0",
        tokenizer_fingerprint=tokenizer_fingerprint,
        starting_global_step=0,
        scored_updates=[0, 1, 2],
        scored_global_steps=[0, 1, 2],
        architecture="GrugMoeForCausalLM",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=1,
        seed=7,
        bootstrap_seed=8,
        created_at_utc="2026-09-26T00:00:00Z",
        config_json="{}",
        software_json="{}",
        hardware_json=json.dumps(_logged_hardware() if logged_reread else {}),
        batch_layout_json="{}",
        timing_json=json.dumps({"trainer@0:native/seconds": forward_seconds}),
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


def _logged_hardware() -> dict:
    """A logged re-read: prompt p0's prefixes in 300-token CUDA-graph steps, p1's in 700-token eager steps."""
    steps = {
        f"p{prompt}:{repetition}": {"tokens": tokens, "rows": rows, "tokens_across_dp": None}
        for prompt, (tokens, rows) in enumerate(((300, 304), (700, 700)))
        for repetition in range(2)
    }
    kernels = {"launches": {"rms_norm": {"kwargs": {"R0_BLOCK": 2048, "XBLOCK": 1}}}, "gate_columns": [24, 20]}
    return {
        "vllm_reread": {
            "engine": 0,
            "steps": {"0": steps},
            "step_logs": {
                "vllm.rescore@0": [[{"steps": [{}] * 4, "dummy_steps": 0}], [{"steps": [], "dummy_steps": 4}]]
            },
            "kernels": kernels,
        },
        "prefill_reference": {"archive": "here", "dp_ranks": [0] * 4, "steps": steps, "kernels": kernels},
    }


def test_report_splits_byte_equality_by_the_step_kind_of_the_logged_reread(tmp_path):
    root = tmp_path / "archive"
    _archive(root, logged_reread=True)

    report = analyze_archive(str(root), bootstrap_draws=20)

    by_step = {
        mode: {kind: (item["byte_equal_fraction"], item["tokens"]) for kind, item in kinds.items()}
        for mode, kinds in report["byte_equal_by_step"].items()
    }
    assert by_step["steps"] == {"cuda_graph": (1.0, 4), "eager": (0.5, 4)}
    assert by_step["reread_replay"] == {"cuda_graph": (0.0, 4), "eager": (0.0, 4)}
    assert report["logged_reread"]["updates"]["0"] == {
        "prefixes": 4,
        "cuda_graph_steps": 2,
        "padded_rows": 8,
        "max_tokens": 700,
    }
    markdown = render_markdown(report)
    assert "| steps | 100.00% | 4 | 50.00% | 4 |" in markdown
    assert "each decoder layer: `[24, 20]`" in markdown


def test_report_recovers_same_weight_modes_paired_intervals_and_drift(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    _archive(root)
    report = analyze_archive(str(root), bootstrap_draws=80)
    assert report["comparisons"]["implementation_mismatch"]["metrics"]["abs_p99"] > 0.09
    assert report["comparisons"]["trainer_floor"]["metrics"]["abs_p99"] == 0.0
    assert report["comparisons"]["trainer_determinism"]["metrics"]["byte_equal_fraction"] == 1.0
    assert report["comparisons"]["replay_layout_floor"]["metrics"]["abs_p99"] == 0.0
    assert report["comparisons"]["reread_layout_floor"]["metrics"]["abs_mean"] == 0.0
    assert report["comparisons"]["reread_layout_floor+stack"]["metrics"]["abs_mean"] == pytest.approx(0.01)
    assert report["comparisons"]["prompt_replay_effect"]["metrics"]["abs_p99"] == pytest.approx(0.03)
    assert report["paired_improvements"]["router_replay"]["abs_p99"]["ci95"][0] > 0
    assert report["paired_improvements"]["router_replay_filtered"]["abs_p99"]["ci95"][0] > 0
    assert report["comparisons"]["fp32_head_vs_generation"]["metrics"]["abs_p99"] == pytest.approx(0.03)
    assert report["paired_improvements"]["fp32_head"]["abs_p99"]["ci95"] == pytest.approx([0.07, 0.07])
    assert report["prefill_reference"] == "vllm.rescore@0"
    # The second re-read differs from the first on each answer's last token, so half the tokens are stable.
    assert report["vllm_stable_token_fraction"] == 0.5
    assert report["comparisons"]["native_vs_reread_stable"]["metrics"]["tokens"] == 4
    assert "router_replay_filtered" in report["paired_vs_reread_stable"]["reread_replay"]
    prefill = report["comparisons"]["native_vs_reread"]
    assert prefill["reference"] == "vllm.rescore@0"
    assert prefill["metrics"]["abs_mean"] == pytest.approx(0.1)
    assert report["comparisons"]["repeat_vs_reread"]["metrics"]["abs_mean"] == pytest.approx(0.1)
    assert report["comparisons"]["reread_replay_vs_reread"]["metrics"]["abs_mean"] == pytest.approx(0.02)
    noise = report["comparisons"]["reread_noise"]["metrics"]
    assert noise["abs_mean"] == pytest.approx(0.002, rel=1e-3)
    assert noise["byte_equal_fraction"] == 0.5
    assert "reread_vs_frozen" not in report["comparisons"]
    kept = report["paired_vs_reread"]["reread_replay"]
    assert "reread_replay" not in kept
    # Native sits 0.1 from the re-read against the kept stack's 0.02, so it is worse on every metric.
    assert kept["native"]["metrics"]["abs_mean"]["ci95"] == pytest.approx([-0.08, -0.08], rel=1e-3)
    assert kept["native"]["metrics"]["k3"]["ci95"][1] < 0
    assert kept["native"]["regresses"] is True
    assert kept["router_replay_filtered"]["metrics"]["abs_p99"]["ci95"] == pytest.approx([0.01, 0.01], rel=1e-3)
    assert kept["router_replay_filtered"]["regresses"] is False
    assert kept["repeat_reread_replay"]["metrics"]["abs_mean"]["ci95"] == [0.0, 0.0]
    assert kept["repeat_reread_replay"]["regresses"] is False
    straddling = kept["prompt_dependent"]["metrics"]["abs_mean"]["ci95"]
    assert straddling[0] == pytest.approx(-0.01, rel=1e-3) and straddling[1] == pytest.approx(0.01, rel=1e-3)
    assert kept["prompt_dependent"]["regresses"] is False
    rendered = render_markdown(report)
    comparison_line = next(
        line for line in rendered.splitlines() if line.startswith("| router_replay_filtered_vs_generation |")
    )
    assert float(comparison_line.split("|")[2].strip()) == pytest.approx(
        report["comparisons"]["router_replay_filtered_vs_generation"]["metrics"]["abs_p99"], abs=0.005
    )
    kept_section = rendered.split("## Prefill metric: `reread_replay` minus candidate", 1)[1].split("\n## ", 1)[0]
    regression_rows = {
        line.split("|")[1].strip(): line.split("|")[-2].strip()
        for line in kept_section.splitlines()
        if line.startswith("| ")
    }
    assert regression_rows["native"] == "**yes**" and regression_rows["router_replay_filtered"] == "no"

    frozen_root = tmp_path / "frozen"
    _archive(frozen_root, frozen_reread=True)
    frozen = analyze_archive(str(frozen_root), bootstrap_draws=20)
    assert frozen["prefill_reference"] == "vllm.rescore_frozen@0"
    assert frozen["comparisons"]["native_vs_reread"]["reference"] == "vllm.rescore_frozen@0"
    # Generation against the re-read of the engine run that generated it: the source's frozen re-read.
    assert frozen["comparisons"]["decode_vs_prefill"]["reference"] == "vllm.rescore_frozen@0"
    assert frozen["comparisons"]["native_vs_reread"]["metrics"]["abs_mean"] == pytest.approx(0.11)
    assert frozen["comparisons"]["reread_vs_frozen"]["metrics"]["abs_mean"] == pytest.approx(0.01)
    assert frozen["paired_vs_reread"]["reread_replay"]["native"]["metrics"]["abs_mean"][
        "baseline_minus_candidate"
    ] == pytest.approx(-0.08)
    snapshot = ReadView(str(root))
    expected_token = str(snapshot.token)
    advanced = False

    def open_then_advance(uri):
        nonlocal advanced
        view = ReadView(uri)
        if not advanced:
            advanced = True
            manifest = report_module.ManifestRow.model_validate(view.scan(MANIFEST_TABLE).to_pylist()[0]).model_copy(
                update={"timing_json": "{}"}
            )
            with DataStore.open(uri, writer_id="concurrent-writer") as store:
                register_mismatch_tables(store)
                with store.transaction() as transaction:
                    transaction.table(MANIFEST_TABLE).add(manifest.model_dump())
        return view

    monkeypatch.setattr(report_module, "ReadView", open_then_advance)
    concurrent_report = analyze_archive(str(root), bootstrap_draws=20)
    assert concurrent_report["input_commit_token"] == expected_token
    assert concurrent_report["timing"] == report["timing"]


@pytest.mark.parametrize("corrupt_field", ["response", "prompt"])
@pytest.mark.parametrize("probe_hash", ["frozen", "independent"])
def test_report_stops_numerical_analysis_after_token_mutation(tmp_path, corrupt_field, probe_hash):
    root, valid = tmp_path / "archive", tmp_path / "valid"
    _archive(
        root, corrupt_token=corrupt_field == "response", corrupt_prompt=corrupt_field == "prompt", probe_hash=probe_hash
    )
    _archive(valid)
    report = analyze_archive(str(root), bootstrap_draws=20)
    assert report["token_identity"]["fraction"] < 1
    assert report["comparisons"] == {}
    rendered = render_markdown(report)
    assert "token_identity" in rendered and "failed" in rendered
    for left, right in ((root, valid), (valid, root)):
        with pytest.raises(ValueError, match="trainer and sampler token identity"):
            compare_archives(str(left), str(right), bootstrap_draws=20)


def test_report_route_agreement_against_generation_and_reread(tmp_path):
    root = tmp_path / "archive"
    _archive(root, with_routes=True)
    report = analyze_archive(str(root), bootstrap_draws=20)
    native = report["route_diagnostics"]["trainer@0:native"]["metrics"]
    replay = report["route_diagnostics"]["trainer@0:router_replay"]["metrics"]
    filtered = report["route_diagnostics"]["trainer@0:router_replay_filtered"]["metrics"]
    assert native["set_agreement"] == 0.5
    assert replay["set_agreement"] == 1.0
    assert filtered["set_agreement"] == 0.75

    # Response token 0 is the only response input: native differs from the re-read at layer 0 in three
    # samples and at layer 1 in one; generation-route replay differs at layer 1 once and at layer 0 once.
    reread = report["route_diagnostics_vs_reread"]
    native_reread = reread["trainer@0:native"]
    assert native_reread["reference"] == "vllm.rescore@0"
    assert native_reread["layers"]["0"]["metrics"]["set_agreement"] == 0.25
    assert native_reread["layers"]["1"]["metrics"]["set_agreement"] == 0.75
    assert native_reread["first_disagreeing_layer"] == {
        "tokens": 4,
        "no_disagreement": 1,
        "layers": {"0": 3, "1": 0},
    }
    assert reread["trainer@0:router_replay"]["first_disagreeing_layer"] == {
        "tokens": 4,
        "no_disagreement": 2,
        "layers": {"0": 1, "1": 1},
    }
    assert reread["trainer@0:reread_replay"]["metrics"]["set_agreement"] == 1.0
    assert report["comparisons"]["native_vs_reread"]["route_set_agreement"]["value"] == 0.5
    noise = report["reread_route_agreement"]["reread_noise"]
    assert noise["layers"]["0"]["metrics"]["set_agreement"] == 1.0
    assert noise["layers"]["1"]["metrics"]["set_agreement"] == pytest.approx(11 / 12)
    rendered = render_markdown(report)
    assert "| trainer@0:native | 0 | 3 | 75.0% |" in rendered

    frozen_root = tmp_path / "frozen"
    _archive(frozen_root, with_routes=True, frozen_reread=True)
    frozen = analyze_archive(str(frozen_root), bootstrap_draws=20)
    frozen_native = frozen["route_diagnostics_vs_reread"]["trainer@0:native"]
    assert frozen_native["reference"] == "vllm.rescore_frozen@0"
    assert frozen_native["first_disagreeing_layer"]["no_disagreement"] == 4
    assert frozen["reread_route_agreement"]["reread_vs_frozen"]["metrics"]["set_agreement"] == pytest.approx(20 / 24)
    output = tmp_path / "figures"
    output.mkdir()
    write_plots(report, output)
    assert (output / "routes.png").stat().st_size > 0

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
    _archive(left, with_routes=True)
    _archive(right, native_offset=0.04, with_routes=True, forward_seconds=0.3, matching_native_routes=True)
    _archive(changed_weights, weight_tag="other")
    _archive(changed_tokenizer, tokenizer_fingerprint="other-tokenizer")
    _archive(independent, probe_hash="another-answer-set", response_shift=1)
    paired = compare_archives(str(left), str(right), bootstrap_draws=40)
    assert paired["timing"]["trainer@0:native/seconds"]["right_minus_left_seconds"] == pytest.approx(0.2)
    route_effect = paired["routes"]["trainer@0:native"]
    assert route_effect["right_minus_left"]["set_agreement"] == 0.5
    assert route_effect["ci95"]["set_agreement"] == pytest.approx([0.5, 0.5])
    rendered = render_archive_comparison(paired)
    assert "0.2" in rendered and "| 50 | [50, 50] |" in rendered
    missing_left, missing_right = tmp_path / "missing-left", tmp_path / "missing-right"
    _archive(missing_left, with_routes=True, missing_route_layer=True)
    _archive(missing_right, with_routes=True, missing_route_layer=True)
    missing = compare_archives(str(missing_left), str(missing_right), bootstrap_draws=20)
    assert "| layer_0 | nonfinite (nan) | - |" in render_archive_comparison(missing)
    output = tmp_path / "missing-route-figures"
    output.mkdir()
    write_plots(analyze_archive(str(missing_left), bootstrap_draws=20), output)
    assert (output / "routes.png").stat().st_size > 0
    cached = tmp_path / "cached"
    _archive(cached, cache_mode="on")
    with pytest.raises(ValueError, match="prefix-cache"):
        compare_archives(str(left), str(cached), bootstrap_draws=20)
    effect = paired["comparisons"]["implementation_mismatch"]
    assert effect["metrics"]["abs_p99_left_minus_right"] > 0
    assert effect["ci95"]["abs_p99_left_minus_right"][0] > 0
    with pytest.raises(ValueError, match="same starting weight hash"):
        compare_archives(str(left), str(changed_weights), bootstrap_draws=20)
    with pytest.raises(ValueError, match="same tokenizer fingerprint"):
        compare_archives(str(left), str(changed_tokenizer), bootstrap_draws=20)
    with pytest.raises(ValueError, match="same frozen probe hash"):
        compare_archives(str(left), str(independent), bootstrap_draws=20)
