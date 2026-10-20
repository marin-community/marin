# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from marin.evaluation.eval_stats import wilson_interval
from tokenizers import Tokenizer

from experiments.grug.moe_hero_ep.format_eval import (
    Candidate,
    CandidateEvaluation,
    GatePolicy,
    Interface,
    Panel,
    PanelCase,
    PilotBudget,
    Serialization,
    SmokeRecord,
    SmokeUpdate,
    assemble_smoke,
    compare_candidates,
    compare_native_candidates,
    completion_failures,
    grade_panel,
    matched_throughput,
    paired_content_interval,
    plan_pilot,
    render_panel,
    score_content_panel,
    summarize_smoke,
    validate_smoke_checkpoint,
    validated_format_metrics,
)
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, NativePanelRequest, sample_panel
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Completion,
    Prompt,
    PromptCompletions,
    SampleRequest,
    SampleResult,
    SamplingSpec,
    StopReason,
    digest,
)
from experiments.grug.moe_hero_ep.sft import SftConfig


def test_native_evaluation_rounding_preserves_expert_selection_near_qb_ties():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from levanter.grug.sharding import compact_grug_mesh
        from experiments.grug.moe_hero_ep.weights import cast_weights
        from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, _router_top_k

        config = GrugModelConfig(vocab_size=16, hidden_dim=32, intermediate_dim=16,
            shared_expert_intermediate_dim=16, num_experts=4, num_experts_per_token=2,
            num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, attention_implementation="reference",
            moe_implementation="fixed_all_to_all")
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
        with jax.set_mesh(mesh):
            base = Transformer.init(config, key=jax.random.key(0))
            bias = jnp.tile(jnp.array([1.001, 1.003, 1.002, 1.0005], jnp.float32), (2, 1))
            base = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, base, bias)
            rounded = cast_weights(base, jnp.bfloat16)
            actual = rounded.stacked_blocks.stacked.mlp.router_bias
            np.testing.assert_array_equal(_router_top_k(actual, 2), [[1, 2], [1, 2]])
            np.testing.assert_array_equal(actual, bias)
            assert rounded.token_embed.dtype == jnp.bfloat16
            assert not np.array_equal(_router_top_k(bias.astype(jnp.bfloat16), 2), _router_top_k(actual, 2))
        """,
        device_count=1,
    )


def test_raw_generations_retain_termination_tool_and_reasoning_failures():
    simple = PanelCase(
        id="qa",
        interface=Interface.SIMPLE,
        messages=({"role": "user", "content": "Hi"},),
        legacy_prompt="Hi",
        seed=1,
        source="test",
    )
    assert completion_failures(simple, "Hello<|eot_id|>", [5, 128009], StopReason.EOS) == ()
    assert "termination" in completion_failures(simple, "Hello", [5], StopReason.MAX_NEW_TOKENS)
    assert "role_or_observation_continuation" in completion_failures(
        simple, "Hello<|eot_id|><|start_header_id|>user<|end_header_id|>", [5, 128009, 6], StopReason.MAX_NEW_TOKENS
    )
    reasoning = simple.model_copy(update={"interface": Interface.REASONING_ON})
    assert (
        completion_failures(
            reasoning, "<|start_think|>Check<|end_think|>Yes<|eot_id|>", [1, 2, 3, 128009], StopReason.EOS
        )
        == ()
    )
    assert "reasoning_mode" in completion_failures(reasoning, "Yes<|eot_id|>", [3, 128009], StopReason.EOS)
    tool = simple.model_copy(update={"interface": Interface.TOOL_CALL, "expected_tool_names": ("lookup",)})
    assert (
        completion_failures(
            tool, '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call><|eot_id|>', [7, 128009], StopReason.EOS
        )
        == ()
    )
    assert "tool_call" in completion_failures(
        tool, '<tool_call>{"name":"lookup","arguments":"bad"}</tool_call><|eot_id|>', [7, 128009], StopReason.EOS
    )
    observed = simple.model_copy(update={"interface": Interface.OBSERVATION, "expected_observation_answer": "seven"})
    assert "observation_handling" in completion_failures(
        observed, "I did not read it<|eot_id|>", [7, 128009], StopReason.EOS
    )


def _bound_report(report: dict) -> dict:
    report = {**report, "generations": [], "metrics": {}}
    for interface, total in (
        (("simple", 500), ("context16k", 3)) if report["context_length"] == 16384 else (("context65k", 1),)
    ):
        interval = wilson_interval(total, total)
        report["metrics"][interface] = {
            "successes": total,
            "failures": 0,
            "total": total,
            "observed_rate": 1,
            "confidence_interval_95": [interval.low, interval.high],
            "interval_method": "Wilson score",
        }
        report["generations"].extend(
            {"id": f"{interface}-{index}", "interface": interface, "failures": []} for index in range(total)
        )
    report["report_digest"] = digest({key: value for key, value in report.items() if key != "report_digest"})
    return report


def test_candidate_gates_keep_uncertain_content_and_route_failures_visible():
    correct = {str(i): True for i in range(500)}
    baseline = CandidateEvaluation(
        candidate=Candidate.A,
        parent=None,
        checkpoint="A",
        checkpoint_metadata_digest="a",
        model_config_digest="config-16k",
        panel_id="pinned-panel",
        content_panel_id="pinned-content-panel",
        tokenizer_revision="pinned-tokenizer",
        backend="native",
        evaluation_dtype="bf16",
        heldout_loss_panel="pt-heldout",
        heldout_loss=2.0,
        routing={"max_load": 1.1},
        content={Serialization.LEGACY: correct, Serialization.MARIN: correct},
        format_report={},
    )
    policy = GatePolicy(
        baseline_evaluation_id=baseline.evaluation_id,
        baseline_variability_record="baseline-repeat.json",
        heldout_loss_increase=0.02,
        content_accuracy_decline=0.02,
        routing_maximums={"max_load": 1.2},
        enabled_interfaces=(Interface.SIMPLE,),
        observed_format_minimum=0.99,
        minimum_simple_cases=500,
        minimum_interface_cases=100,
    )
    averaged = baseline.model_copy(update={"candidate": Candidate.B, "parent": Candidate.A, "checkpoint": "B"})
    assert compare_candidates(baseline, baseline, averaged, policy, require_format=False)["decision"] == "accept"
    uncertain = policy.model_copy(update={"content_accuracy_decline": 0.001})
    assert (
        compare_candidates(baseline, baseline, averaged, uncertain, require_format=False)["decision"] == "inconclusive"
    )
    routed = averaged.model_copy(update={"routing": {"max_load": 1.3}})
    report = compare_candidates(baseline, baseline, routed, policy, require_format=False)
    assert report["decision"] == "diagnose" and "routing" in report["violations"]
    regressed = averaged.model_copy(
        update={"content": {serialization: {str(i): i > 100 for i in range(500)} for serialization in Serialization}}
    )
    report = compare_candidates(baseline, baseline, regressed, policy, require_format=False)
    assert report["decision"] == "diagnose" and report["content"]["legacy"]["losses"] == 101
    assert report["content"]["legacy"]["confidence_interval_95"][1] < -0.02
    formatted = _bound_report(
        {
            "panel_id": baseline.panel_id,
            "checkpoint": {"uri": "E", "metadata_digest": "e"},
            "backend": baseline.backend,
            "model_config_digest": "config-65k",
            "serialization": "marin",
            "tokenizer_revision": baseline.tokenizer_revision,
            "context_length": 16384,
        }
    )
    parent_d = baseline.model_copy(update={"candidate": Candidate.D, "parent": Candidate.C})
    extended = baseline.model_copy(
        update={
            "candidate": Candidate.E,
            "parent": Candidate.D,
            "checkpoint": "E",
            "checkpoint_metadata_digest": "e",
            "model_config_digest": "config-65k",
            "format_report": formatted,
            "serving_parity_report": "parity-65k.json",
            "serving_parity_passed": True,
        }
    )
    extension_policy = policy.model_copy(update={"long_context_panel_id": "fixed-65k-panel"})
    pending = compare_candidates(baseline, parent_d, extended, extension_policy, require_format=True)
    assert pending["decision"] == "inconclusive" and "actual_65k_context_panel" in pending["uncertain"]
    long = _bound_report(
        {
            **formatted,
            "context_length": 65536,
            "panel_id": "fixed-65k-panel",
            "tokenizer_revision": baseline.tokenizer_revision,
        }
    )
    extended = extended.model_copy(update={"long_context_report": long})
    assert (
        compare_candidates(baseline, parent_d, extended, extension_policy, require_format=True)["decision"] == "accept"
    )
    for change in ({"observed_rate": float("nan")}, {"successes": 0}, {"total": 1000}):
        invalid = {
            **formatted,
            "metrics": {**formatted["metrics"], "simple": {**formatted["metrics"]["simple"], **change}},
        }
        invalid["report_digest"] = digest({key: value for key, value in invalid.items() if key != "report_digest"})
        with pytest.raises(ValueError):
            compare_candidates(
                baseline,
                parent_d,
                extended.model_copy(update={"format_report": invalid}),
                extension_policy,
                require_format=True,
            )
    changed = {**formatted, "sample_id": "another-generation"}
    with pytest.raises(ValueError, match="digest changed"):
        compare_candidates(
            baseline,
            parent_d,
            extended.model_copy(update={"format_report": changed}),
            extension_policy,
            require_format=True,
        )
    stale_model = extended.model_copy(update={"model_config_digest": "wrong-rotary-config"})
    with pytest.raises(ValueError, match="does not belong"):
        compare_candidates(baseline, parent_d, stale_model, extension_policy, require_format=True)

    # Absolute formatting acceptance can grow while paired PT/content panels stay fixed.
    initial = baseline.model_copy(update={"panel_id": "initial-35-case-format-panel"})
    parent_c = initial.model_copy(update={"candidate": Candidate.C, "parent": Candidate.A, "checkpoint": "C"})
    final = _bound_report(
        {
            "panel_id": "final-503-case-format-panel",
            "checkpoint": {"uri": "D", "metadata_digest": "d"},
            "backend": baseline.backend,
            "model_config_digest": baseline.model_config_digest,
            "serialization": "marin",
            "tokenizer_revision": baseline.tokenizer_revision,
            "context_length": 16384,
        }
    )
    assert sum(metric["total"] for metric in final["metrics"].values()) == 503
    trained = initial.model_copy(
        update={
            "candidate": Candidate.D,
            "parent": Candidate.C,
            "checkpoint": "D",
            "checkpoint_metadata_digest": "d",
            "panel_id": final["panel_id"],
            "format_report": final,
            "serving_parity_passed": True,
            "serving_parity_report": "D-parity.json",
        }
    )
    staged_policy = policy.model_copy(update={"baseline_evaluation_id": initial.evaluation_id})
    assert compare_candidates(initial, parent_c, trained, staged_policy, require_format=True)["decision"] == "accept"
    pending_serving = trained.model_copy(update={"serving_parity_passed": False, "serving_parity_report": None})
    assert (
        compare_candidates(initial, parent_c, pending_serving, staged_policy, require_format=True)["decision"]
        == "inconclusive"
    )
    assert (
        compare_native_candidates(initial, parent_c, pending_serving, staged_policy, require_format=True)["decision"]
        == "accept"
    )
    with pytest.raises(ValueError, match="content_panel_id"):
        compare_candidates(
            initial,
            parent_c,
            trained.model_copy(update={"content_panel_id": "other-content-panel"}),
            staged_policy,
            require_format=True,
        )


def test_paired_interval_reports_exact_discordant_counts():
    parent = {"a": True, "b": False, "c": True, "d": False}
    child = {"a": False, "b": True, "c": True, "d": False}
    measured = paired_content_interval(parent, child)
    assert measured["wins"] == measured["losses"] == 1
    assert measured["observed_change"] == 0
    assert measured["confidence_interval_95"][0] < 0 < measured["confidence_interval_95"][1]


def smoke_optimizer_report() -> dict:
    return {
        "schema_version": 1,
        "execution_id": "real-execution",
        "checkpoint_metadata_digest": "donor-C",
        "measurement_identity_digest": "model-data-batch-mesh-resources",
        "optimizer_step": 0,
        "compact_optimizer_only": True,
        "errors": [],
        "trainable_parameter_bytes": 1024,
        "optimizer_state_bytes": 2064,
        "expected_optimizer_state_bytes": 2064,
    }


def smoke_record() -> SmokeRecord:
    return SmokeRecord(
        execution_id="real-execution",
        checkpoint_metadata_digest="donor-C",
        hardware="GB200-NVL72",
        gpu_count=64,
        configuration_digest="config",
        measurement_identity_digest="model-data-batch-mesh-resources",
        sft_config=SftConfig(
            rank=16,
            alpha=16,
            token_ids=(128009,),
            adapter_learning_rate=1e-4,
            vocab_learning_rate=1e-5,
            num_steps=4,
            warmup_steps=1,
            seed=0,
        ),
        terminal_checkpoint="measured-compact-checkpoint",
        terminal_checkpoint_metadata_digest="measured-compact-metadata",
        input_batches_digest="fixed-input",
        sequence_length=16384,
        global_batch_executed_tokens=65536,
        microbatch_size=1,
        gradient_accumulation=4,
        optimizer="Adam-adapters",
        trainable_parameter_bytes=1024,
        optimizer_state_bytes=2064,
        checkpoint_seconds=2,
        compile_seconds=30,
        warmup_updates=1,
        compiled_gradient_report="compiled-report.json",
        compact_optimizer_report="compact-optimizer-report.json",
        compact_optimizer_report_sha256=hashlib.sha256(json.dumps(smoke_optimizer_report()).encode()).hexdigest(),
        avoids_full_base_gradients=True,
        preserves_activation_gradients=True,
        updates=tuple(
            SmokeUpdate(
                update=i,
                elapsed_seconds=10 if i == 1 else 2,
                executed_tokens=65536,
                non_padding_tokens=50000,
                supervised_tokens=20000,
                peak_hbm_bytes=100000,
                optimizer_seconds=0.1,
                loss=2 - i / 100,
                gradient_norm=1,
                diagnostic_successes=80 + i,
                diagnostic_cases=100,
            )
            for i in range(1, 5)
        ),
    )


def test_pilot_uses_non_padding_measurements_and_rejects_too_few_updates():
    smoke = smoke_record()
    measured = summarize_smoke(smoke)
    assert measured["median_non_padding_tokens_per_second"] == 25000
    assert measured["non_padding_tokens"] == 200000
    assert measured["supervised_tokens"] == 80000
    assert measured["executed_tokens"] == 262144
    budget = PilotBudget(
        format_panel_id="heldout-format-acceptance",
        inspection_non_padding_tokens=1000000,
        cap_non_padding_tokens=4000000,
        minimum_learning_updates=20,
        early_inspection_update=10,
        warmup_updates=5,
        cooldown_updates=75,
        adapter_peak_lr=1e-4,
        vocab_peak_lr=1e-5,
        smoke_learning_review="learning-reviewed.json",
        reuse_smoke_optimizer=True,
    )
    plan = plan_pilot(smoke, budget)
    assert plan["expected_inspection_updates"] == 20
    assert plan["expected_cap_updates"] == 80
    assert plan["already_completed_updates"] == 4
    assert plan["expected_remaining_seconds"] == 152
    # Retained warmup batches need not contain the steady-state median tokens.
    variable = smoke.model_copy(
        update={
            "updates": (smoke.updates[0].model_copy(update={"non_padding_tokens": 25000}), *smoke.updates[1:]),
        }
    )
    retained = plan_pilot(variable, budget.model_copy(update={"cooldown_updates": 76}))
    assert retained["expected_cap_updates"] == 81
    assert retained["expected_inspection_updates"] == 21
    assert retained["already_consumed_non_padding_tokens"] == 175000
    with pytest.raises(ValueError, match="too few learning updates"):
        plan_pilot(smoke, budget.model_copy(update={"minimum_learning_updates": 21}))
    full = smoke.model_copy(
        update={
            "optimizer": "Muon-full",
            "updates": tuple(
                update.model_copy(update={"elapsed_seconds": update.elapsed_seconds * 2}) for update in smoke.updates
            ),
        }
    )
    assert matched_throughput(smoke, full)["non_padding_speedup"] == 2
    with pytest.raises(ValueError, match="input_batches_digest"):
        matched_throughput(smoke, full.model_copy(update={"input_batches_digest": "different-input"}))


def test_panel_report_keeps_failed_generation_and_exact_interval_counts():
    panel = Panel(
        release="test-panel",
        cases=tuple(
            PanelCase(
                id=f"qa-{i}",
                interface=Interface.SIMPLE,
                messages=({"role": "user", "content": "Hi"},),
                legacy_prompt="Hi",
                seed=i,
                source="local",
            )
            for i in range(2)
        ),
        tokenizer="prepared",
        tokenizer_revision="a" * 40,
        source_revision="b" * 40,
        backend="native-dropless",
        max_new_tokens=2,
        temperature=0,
    )
    request = SampleRequest(
        checkpoint=Checkpoint(uri="source-A", run_id="test", step=10, timestamp="2026-10-19", metadata_digest="meta"),
        spec=SamplingSpec(
            release=f"{panel.release}-marin-{panel.panel_id}",
            prompts=tuple(Prompt(id=case.id, text="Hi", seed=case.seed, source_url="local") for case in panel.cases),
            completions_per_prompt=1,
            tokenizer=panel.tokenizer,
            tokenizer_revision=panel.tokenizer_revision,
            temperature=0,
            max_new_tokens=2,
            context_length=16384,
        ),
        model={},
        source_revision=panel.source_revision,
        target_cluster="local-test",
    )
    result = SampleResult(
        request=request,
        completed_at="2026-10-19",
        eos_token_id=128009,
        completions=(
            PromptCompletions(
                prompt_id="qa-0",
                prompt_token_ids=(1,),
                samples=(
                    Completion(
                        sample_index=0, seed=0, token_ids=(5, 128009), text="Yes<|eot_id|>", stop_reason=StopReason.EOS
                    ),
                ),
            ),
            PromptCompletions(
                prompt_id="qa-1",
                prompt_token_ids=(1,),
                samples=(
                    Completion(
                        sample_index=0,
                        seed=1,
                        token_ids=(5, 6),
                        text="keeps going",
                        stop_reason=StopReason.MAX_NEW_TOKENS,
                    ),
                ),
            ),
        ),
    )
    report = grade_panel(panel, result)
    assert validated_format_metrics(report)["simple"].successes == 1
    assert report["metrics"]["simple"]["successes"] == report["metrics"]["simple"]["failures"] == 1
    assert report["metrics"]["simple"]["total"] == 2
    assert report["metrics"]["simple"]["confidence_interval_95"] == pytest.approx([0.0945312057, 0.9054687943])
    assert report["generations"][1]["generation"]["text"] == "keeps going"
    assert "termination" in report["generations"][1]["failures"]


def test_65k_panel_requires_actual_long_prompt_tokens(tmp_path):
    stage_gpt2_tokenizer(Path("lib/levanter/tests"), tmp_path)
    tokenizer = load_tokenizer(str(tmp_path))
    long = PanelCase(
        id="long",
        interface=Interface.CONTEXT_65K,
        messages=({"role": "user", "content": "x" * 65000},),
        legacy_prompt="x" * 65000,
        minimum_prompt_tokens=65000,
        seed=0,
        source="local",
    )
    panel = Panel(
        release="long-panel",
        cases=(long,),
        tokenizer="prepared",
        tokenizer_revision="a" * 40,
        source_revision="b" * 40,
        backend="native",
        max_new_tokens=32,
        context_length=65536,
        temperature=0,
    )
    rendered = render_panel(panel, tokenizer, Serialization.LEGACY)
    assert rendered.context_length == 65536
    assert len(tokenizer.encode(rendered.prompts[0].text)) >= 65000
    shortened = panel.model_copy(update={"cases": (long.model_copy(update={"legacy_prompt": "short"}),)})
    with pytest.raises(ValueError, match="actual context limits"):
        render_panel(shortened, tokenizer, Serialization.LEGACY)


def test_native_panel_sampler_retains_eot_and_distinguishes_truncation_and_legacy_stop(tmp_path):
    stage_gpt2_tokenizer(Path("lib/levanter/tests"), tmp_path)
    underlying = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    # Extend the checked-in tokenizer locally so the real generation boundary uses Hero's EOT ID.
    underlying.add_tokens([f"<unused-{i}>" for i in range(128009 - underlying.get_vocab_size())])
    underlying.add_special_tokens(["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"])
    underlying.save(str(tmp_path / "tokenizer.json"))
    tokenizer = load_tokenizer(str(tmp_path))
    answer_id = tokenizer.encode("x")[0]
    panel = Panel(
        release="native-test",
        cases=(
            PanelCase(
                id="qa",
                interface=Interface.SIMPLE,
                messages=({"role": "user", "content": "Hi"},),
                legacy_prompt="Hi",
                seed=0,
                source="local",
                expected="x",
            ),
        ),
        tokenizer="prepared",
        tokenizer_revision="a" * 40,
        source_revision="b" * 40,
        backend=NATIVE_BACKEND,
        max_new_tokens=2,
        temperature=0,
    )
    request = NativePanelRequest(
        panel=panel,
        checkpoint=Checkpoint(uri="source-A", run_id="test", step=10, timestamp="2026-10-19", metadata_digest="meta"),
        model={},
        serialization=Serialization.MARIN,
        target_cluster="local",
        expert_axis_size=1,
        replica_axis_size=1,
    )

    def logits(tokens, positions):
        scores = np.full((len(tokens), len(tokenizer)), -100.0)
        for row in range(len(tokens)):
            token = 128009 if tokens[row, positions[row]] == answer_id else answer_id
            scores[row, token] = 0
        return scores

    result, report = sample_panel(request, tokenizer, logits, batch_size=1)
    assert result.completions[0].samples[0].token_ids == (answer_id, 128009)
    assert result.completions[0].samples[0].text.endswith("<|eot_id|>")
    assert report["metrics"]["simple"]["successes"] == 1
    assert score_content_panel(panel, result, tokenizer)["outcomes"] == {"qa": True}

    def truncated_logits(tokens, positions):
        scores = np.full((len(tokens), len(tokenizer)), -100.0)
        scores[:, answer_id] = 0
        return scores

    truncated_result, truncated = sample_panel(request, tokenizer, truncated_logits, batch_size=1)
    assert truncated["metrics"]["simple"]["failures"] == 1
    assert "termination" in truncated["generations"][0]["failures"]
    assert score_content_panel(panel, truncated_result, tokenizer)["outcomes"] == {"qa": False}

    def legacy_logits(tokens, positions):
        scores = np.full((len(tokens), len(tokenizer)), -100.0)
        scores[:, tokenizer.eos_token_id] = 0
        return scores

    legacy, legacy_report = sample_panel(
        request.model_copy(update={"serialization": Serialization.LEGACY}), tokenizer, legacy_logits, batch_size=1
    )
    assert legacy.completions[0].samples[0].token_ids == (tokenizer.eos_token_id,)
    assert legacy.eos_token_id != 128009
    assert legacy_report["serialization"] == "legacy" and not legacy_report["metrics"]


def test_smoke_assembly_requires_persisted_runtime_measurements_and_compiled_proof(tmp_path):
    measured = smoke_record()
    metadata = {
        key: measured.model_dump(mode="json")[key]
        for key in (
            "execution_id",
            "checkpoint_metadata_digest",
            "configuration_digest",
            "measurement_identity_digest",
            "sft_config",
            "input_batches_digest",
            "hardware",
            "gpu_count",
            "sequence_length",
            "trainable_parameter_bytes",
            "optimizer_state_bytes",
        )
    }
    for update in measured.updates:
        raw = {
            **metadata,
            **update.model_dump(),
            "schema_version": 1,
            "kind": "hero_sft_raw_smoke_update",
            "compile_seconds": 30,
            "missing_smoke_measurements": [],
            "input_delta_gradient_norm": 0.01,
            "acquisition_diagnostic": {"panel_digest": "fixed-train-diagnostic"},
        }
        (tmp_path / f"update-{update.update:06d}.json").write_text(json.dumps(raw))
    compiled = tmp_path / "compiled-gradient-report.json"
    compiled.write_text(json.dumps({"avoids_full_base_gradients": True, "optimizer_frozen_parameter_inputs": 0}))
    optimizer = tmp_path / "compact-optimizer-report.json"
    optimizer.write_text(json.dumps(smoke_optimizer_report()))
    compact = tmp_path / "compact"
    compact.mkdir()
    completion_metadata = {
        "is_temporary": False,
        "step": 4,
        "hero_sft": {
            "base": {"metadata_digest": measured.checkpoint_metadata_digest},
            "sft": dataclasses.asdict(measured.sft_config),
        },
        "hero_sft_counters": {"sft_step": 4, "processed_tokens": 200000, "supervised_tokens": 80000},
    }
    (compact / "metadata.json").write_text(json.dumps(completion_metadata))
    (tmp_path / "checkpoint-000004.json").write_text(
        json.dumps(
            {
                "update": 4,
                "checkpoint_seconds": 2,
                "checkpoint": str(compact),
                "metadata_digest": digest(completion_metadata),
                "execution_id": measured.execution_id,
            }
        )
    )
    assembled = assemble_smoke(
        tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters"
    )
    assert summarize_smoke(assembled)["median_non_padding_tokens_per_second"] == 25000
    assert assembled.preserves_activation_gradients and assembled.avoids_full_base_gradients
    assert assembled.checkpoint_seconds == 2
    # Budget qualification rechecks the pinned structural report, not a copied
    # pass flag or the optimizer byte count alone.
    optimizer.write_text(json.dumps({**smoke_optimizer_report(), "optimizer_step": 1}))
    with pytest.raises(ValueError, match="optimizer report bytes changed"):
        validate_smoke_checkpoint(assembled)
    with pytest.raises(ValueError, match="fresh compact Adam"):
        assemble_smoke(tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters")
    for field in ("execution_id", "checkpoint_metadata_digest", "measurement_identity_digest"):
        optimizer.write_text(json.dumps({**smoke_optimizer_report(), field: "another-smoke"}))
        with pytest.raises(ValueError, match="fresh compact Adam"):
            assemble_smoke(
                tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters"
            )
    optimizer.write_text(json.dumps({**smoke_optimizer_report(), "optimizer_state_bytes": 2048}))
    with pytest.raises(ValueError, match="fresh compact Adam"):
        assemble_smoke(tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters")
    optimizer.unlink()
    with pytest.raises(FileNotFoundError):
        assemble_smoke(tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters")
    optimizer.write_text(json.dumps(smoke_optimizer_report()))
    compiled.write_text(json.dumps({"avoids_full_base_gradients": False, "optimizer_frozen_parameter_inputs": 0}))
    with pytest.raises(ValueError, match="Compiled evidence"):
        assemble_smoke(tmp_path, warmup_updates=1, microbatch_size=1, gradient_accumulation=4, optimizer="Adam-adapters")
