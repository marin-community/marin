# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from json import dumps
from pathlib import Path

import draccus
import pytest
import requests
from marin.inference.backend import ModelSpec
from marin.inference.vllm_release import VllmGpuWheelProvenance

from experiments.grug.moe_hero_ep.context_extension import ContextExtensionConfig, extended_model
from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Candidate, Serialization, grade_panel
from experiments.grug.moe_hero_ep.format_handoff import assemble_candidate
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, NativePanelRequest
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest, SuitePanel
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Completion,
    Prompt,
    PromptCompletions,
    SampleRequest,
    SampleResult,
    SamplingSpec,
    StopReason,
    TokenProbability,
    TokenScore,
    digest,
)
from experiments.grug.moe_hero_ep.serving_parity import (
    JsonPin,
    OwnedServer,
    ServingParityRequest,
    StopContract,
    Topology,
    Trajectory,
    collect_serving,
    prepare_serving,
    verify_serving_report,
)


def pin(path):
    return JsonPin(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pin(path)


def bound(value, key):
    return {**value, key: digest(value)}


def runtime(tmp_path, request, model, exported):
    wheel = VllmGpuWheelProvenance(
        "test-release",
        ("9.0",),
        "d" * 40,
        "test-version",
        "e" * 64,
        "https://wheel.invalid/download/test-release/test.whl",
    )
    wheel_record = {"filename": "test.whl", "url": wheel.wheel_url, "sha256": wheel.wheel_sha256}
    qualification = save(
        tmp_path / "release.json",
        {
            "release": {"status": "released", "tag": wheel.release_tag, "candidate_tag": "test-candidate"},
            "source": {"fork_commit": wheel.source_commit},
            "distribution": {"version": wheel.version},
            "platforms": [{"architecture": "x86_64", "wheel": {**wheel_record, "size_bytes": 1}}],
            "validation": {
                "status": "passed",
                "targets": [
                    {
                        "architecture": "x86_64",
                        "candidate_tag": "test-candidate",
                        "source_commit": wheel.source_commit,
                        "wheel": {**wheel_record, "url": wheel.wheel_url.replace("/test-release/", "/test-candidate/")},
                        "result": "passed",
                        "hardware": {"compute_capability": "9.0"},
                        "gates": {
                            name: {
                                "status": "passed",
                                **(
                                    {"metrics": {"completions": 2, "output_tokens_per_second": 1}}
                                    if name == "serve_smoke"
                                    else {}
                                ),
                            }
                            for name in (
                                "wheel_sha256",
                                "distribution_metadata",
                                "vllm._C_stable_libtorch",
                                "GrugMoeForCausalLM",
                                "cumem_allocator",
                                "torchaudio.resample",
                                "sparse_nccl_contract",
                                "serve_smoke",
                            )
                        },
                    }
                ],
            },
        },
    )
    startup = tmp_path / "startup.log"
    startup.write_text(
        "MARIN_VLLM_WHEEL_VERIFIED="
        + json.dumps({**dataclasses.asdict(wheel), "compute_capability": "9.0", "extension_path": "/owned/vllm/_C.so"})
        + "\n"
        + f"GrugMoE effective config: TP=0/1 PP=0/1 layers=[0,{model.num_layers}) DP=0/1 EP=0/1\n"
    )
    server = OwnedServer(
        execution_id="owned-fixture",
        server_url="http://owned.invalid/v1",
        model=ModelSpec(
            weights=str(tmp_path / "export"),
            api_model="hero",
            num_chips=1,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=model.max_seq_len,
            chat_template_content=None,
            tokenizer=request.tokenizer,
            tokenizer_revision=request.tokenizer_revision,
        ),
        export_manifest=exported,
        qualification_manifest=qualification,
        startup_log=pin(startup),
        wheel=wheel,
        architecture="x86_64",
        compute_capability="9.0",
        topology=Topology(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, expert_parallel_size=1
        ),
        source_revision="f" * 40,
        command=(
            "vllm",
            "serve",
            str(tmp_path / "export"),
            "--no-enable-prefix-caching",
            "--served-model-name",
            "hero",
            "--tokenizer",
            request.tokenizer,
            "--tokenizer-revision",
            request.tokenizer_revision,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(model.max_seq_len),
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size",
            "1",
            "--data-parallel-size",
            "1",
            "--attention-backend",
            "declared-test-backend",
        ),
        attention_backend="declared-test-backend",
        prefix_caching=False,
        speculative_decoding=False,
    )
    return save(tmp_path / "launch.json", server.model_dump(mode="json"))


@pytest.fixture
def parity_bundle(tmp_path, suite_inputs):
    request, _ = suite_inputs

    def build(candidate=Candidate.A, checkpoint=None):
        base = request if checkpoint is None else request.model_copy(update={"checkpoint": checkpoint})
        model = draccus.decode(GrugModelConfig, base.model)
        if candidate == Candidate.E:
            model = extended_model(
                ContextExtensionConfig(base.checkpoint.uri, base.checkpoint.metadata_digest, model, "b" * 40)
            )
        panels = []
        outputs = []
        selections = []
        for index, length in enumerate((8, 16000, 65000) if candidate == Candidate.E else (8, 16000)):
            panel = base.panels[0].panel.model_copy(
                update={"release": f"serving-{index}", "context_length": model.max_seq_len, "max_new_tokens": 3}
            )
            entry = SuitePanel(panel=panel, serialization=Serialization.MARIN, purpose=base.panels[0].purpose)
            panels.append(entry)
            tokens = (42, 43, EOT_ID)
            scores = tuple(
                TokenScore(
                    token_id=token,
                    text="x",
                    logprob=-0.1,
                    top_tokens=(
                        TokenProbability(token_id=token, text="x", logprob=-0.1),
                        TokenProbability(token_id=44, text="y", logprob=-3.0),
                    ),
                )
                for token in tokens
            )
            spec = SamplingSpec(
                release=f"{panel.release}-{Serialization.MARIN}-{panel.panel_id}",
                prompts=(Prompt(id="greeting", text="retained fixture", seed=0, source_url="fixture"),),
                completions_per_prompt=1,
                tokenizer=base.tokenizer,
                tokenizer_revision=base.tokenizer_revision,
                temperature=0,
                max_new_tokens=3,
                context_length=model.max_seq_len,
            )
            sample_request = SampleRequest(
                checkpoint=base.checkpoint,
                spec=spec,
                model=draccus.encode(model),
                source_revision=base.source_revision,
                target_cluster=base.target_cluster,
            )
            result = SampleResult(
                request=sample_request,
                completions=(
                    PromptCompletions(
                        prompt_id="greeting",
                        prompt_token_ids=(1,) * length,
                        samples=(
                            Completion(
                                sample_index=0,
                                seed=0,
                                token_ids=tokens,
                                text="xxx",
                                stop_reason=StopReason.EOS,
                                token_scores=scores,
                            ),
                        ),
                    ),
                ),
                completed_at="fixture",
                eos_token_id=EOT_ID,
            )
            path = tmp_path / "goldens" / entry.result_key / "samples.json"
            saved = save(path, result.model_dump(mode="json"))
            outputs.append(
                {
                    "panel_id": panel.panel_id,
                    "result_key": entry.result_key,
                    "sample_id": sample_request.sample_id,
                    "file_sha256": {"samples.json": saved.sha256},
                }
            )
            selections.append(
                Trajectory(
                    panel_id=panel.panel_id,
                    prompt_id="greeting",
                    sample_index=0,
                    minimum_decode_tokens=2,
                    stop=StopContract.EOT,
                )
            )
        suite = base.model_copy(update={"model": draccus.encode(model), "panels": tuple(panels)})
        manifest = bound(
            {
                "request": suite.model_dump(mode="json"),
                "suite_id": suite.suite_id,
                "model_digest": digest(draccus.encode(model)),
                "model": draccus.encode(model),
                "checkpoint": base.checkpoint.model_dump(),
                "backend": NATIVE_BACKEND,
                "compute_dtype": "bfloat16",
                "routing_control_dtype": "float32",
                "source_revision": base.source_revision,
                "panels": outputs,
            },
            "manifest_digest",
        )
        suite_pin = save(tmp_path / "goldens/manifest.json", manifest)
        hf = model.to_hf_config(model.vocab_size).to_dict()
        export_request = {
            "export_version": 3,
            "config": {
                "checkpoint": base.checkpoint.uri,
                "metadata_digest": base.checkpoint.metadata_digest,
                "model": draccus.encode(model),
                "destination": str(tmp_path / "export"),
            },
            "hf_config": hf,
        }
        save(tmp_path / "export/config.json", hf)
        save(tmp_path / "export/export-request.json", export_request)
        save(tmp_path / "export/model.safetensors.index.json", {"weight_map": {"fixture": "one.safetensors"}})
        exported = save(
            tmp_path / "export/export-manifest.json",
            {
                "export_id": digest(export_request),
                "request": export_request,
                "shards": [{"filename": "one.safetensors", "tensor_names": ["fixture"]}],
                "precision": {"learned_weights": "bfloat16", "effective_routing_bias": "float32"},
            },
        )
        extension = None
        if candidate == Candidate.E:
            extension = save(
                tmp_path / "extension.json",
                bound(
                    {
                        "candidate": "E",
                        "source_checkpoint": base.checkpoint.uri,
                        "source_metadata_digest": base.checkpoint.metadata_digest,
                        "model": draccus.encode(model),
                        "hf_config": hf,
                    },
                    "manifest_digest",
                ),
            )
        launch = runtime(tmp_path, base, model, exported)
        return ServingParityRequest(
            candidate=candidate,
            suite_manifest=suite_pin,
            export_manifest=exported,
            context_extension_manifest=extension,
            trajectories=tuple(selections),
            max_new_tokens=3,
            ranks=(0,),
            waves=2,
            max_probability_error=0.002,
            required_serving_source="d" * 40,
            source_revision="f" * 40,
            launch_manifest=launch,
        )

    return build


def responses(request, mutation=None):
    calls = []

    def post(url, headers, json, timeout):
        tokens = [42, 43, EOT_ID]
        choice = {
            "prompt_token_ids": json["prompt"],
            "token_ids": tokens,
            "logprobs": {"top_logprobs": [{f"token_id:{token}": -0.1, "token_id:44": -3.0} for token in tokens]},
            "finish_reason": "stop",
            "stop_reason": EOT_ID,
        }
        body = {"model": "hero", "choices": [choice]}
        if mutation is not None:
            mutation(body, len(calls))
        calls.append((url, headers, json, timeout))
        response = requests.Response()
        response.status_code = 200
        response._content = dumps(body).encode()
        return response

    return post, calls


def test_parsed_multitoken_http_and_retained_bytes_recompute_success(tmp_path, parity_bundle):
    request = parity_bundle(Candidate.E)
    post, calls = responses(request)
    report = collect_serving(request, str(tmp_path / "proof.json"), post=post)
    assert report["measurement"]["passed"]
    assert len(calls) == 6
    assert {row["prompt_tokens"] for row in report["measurement"]["trajectories"]} == {8, 16000, 65000}
    assert all(row["decode_positions"] == 2 for row in report["measurement"]["trajectories"])
    assert verify_serving_report(pin(tmp_path / "proof.json"))["measurement"] == report["measurement"]
    # Offline verification uses retained bytes, not mutable current source paths.
    Path(request.launch_manifest.path).unlink()
    assert verify_serving_report(pin(tmp_path / "proof.json"))["measurement"]["passed"]
    report["exchanges"][0]["response"]["choices"][0]["token_ids"][0] = 45
    report["report_digest"] = digest({key: value for key, value in report.items() if key != "report_digest"})
    altered = save(tmp_path / "altered.json", report)
    with pytest.raises(ValueError, match="recomputed"):
        verify_serving_report(altered)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda body, index: body.update(model="wrong"), "another served model"),
        (lambda body, index: body["choices"][0].update(prompt_token_ids=[1]), "raw prompt"),
        (lambda body, index: body["choices"][0]["token_ids"].__setitem__(0, 44), "maximum returned logprob"),
        (lambda body, index: body["choices"][0].update(finish_reason="length"), "EOT"),
        (
            lambda body, index: (
                body["choices"][0]["logprobs"]["top_logprobs"][1].update({"token_id:43": -0.11}) if index == 1 else None
            ),
            "same-rank",
        ),
    ],
)
def test_raw_http_wrong_identity_prefix_decode_stop_and_repeatability_reject(tmp_path, parity_bundle, mutation, reason):
    request = parity_bundle()
    post, _ = responses(request, mutation)
    report = collect_serving(request, str(tmp_path / "proof.json"), post=post)
    assert not report["measurement"]["passed"]
    assert reason.lower() in str(report["measurement"]["reasons"]).lower()
    assert not verify_serving_report(pin(tmp_path / "proof.json"))["measurement"]["passed"]


def test_pending_source_or_actual_context_coverage_makes_zero_http_calls(tmp_path, parity_bundle):
    request = parity_bundle(Candidate.E)
    launch = json.loads(Path(request.launch_manifest.path).read_text())
    launch["wheel"]["source_commit"] = "0" * 40
    changed = save(tmp_path / "wrong-launch.json", launch)
    request = request.model_copy(update={"launch_manifest": changed})
    post, calls = responses(request)
    report = collect_serving(request, str(tmp_path / "pending.json"), post=post)
    assert not calls and report["measurement"]["status"] == "pending"
    assert verify_serving_report(pin(tmp_path / "pending.json"))["measurement"]["status"] == "pending"
    no_long = request.model_copy(update={"trajectories": request.trajectories[:2]})
    assert any("65000" in reason for reason in prepare_serving(no_long).pending)


def test_handoff_uses_recomputed_report_and_rejects_cross_suite_proof(tmp_path, parity_bundle, completed_suite):
    request = parity_bundle()
    post, _ = responses(request)
    collect_serving(request, str(tmp_path / "proof.json"), post=post)
    handoff, _ = completed_suite
    attached = handoff.model_copy(
        update={
            "serving_parity_report": str(tmp_path / "proof.json"),
            "serving_parity_sha256": pin(tmp_path / "proof.json").sha256,
        }
    )
    with pytest.raises(ValueError, match="another candidate"):
        assemble_candidate(attached)


def test_divergence_retains_only_common_prefix_positions(tmp_path, parity_bundle):
    request = parity_bundle()

    def mutate(body, index):
        choice = body["choices"][0]
        choice["token_ids"][1] = 45
        choice["logprobs"]["top_logprobs"][1]["token_id:45"] = -0.09

    post, _ = responses(request, mutate)
    report = collect_serving(request, str(tmp_path / "divergence.json"), post=post)
    measured = report["measurement"]
    assert not measured["passed"]
    assert all(row["first_divergence"] == 1 and row["decode_positions"] == 1 for row in measured["trajectories"])
    assert {row["position"] for row in measured["comparisons"]} == {0}


def test_context_budget_rejected_before_http(tmp_path, parity_bundle):
    request = parity_bundle()
    manifest = json.loads(Path(request.suite_manifest.path).read_text())
    output = manifest["panels"][1]
    samples_path = Path(request.suite_manifest.path).parent / output["result_key"] / "samples.json"
    samples = json.loads(samples_path.read_text())
    samples["completions"][0]["prompt_token_ids"] = [1] * 16383
    output["file_sha256"]["samples.json"] = save(samples_path, samples).sha256
    manifest["manifest_digest"] = digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    request = request.model_copy(update={"suite_manifest": save(Path(request.suite_manifest.path), manifest)})
    post, calls = responses(request)
    with pytest.raises(ValueError, match="request limits"):
        collect_serving(request, str(tmp_path / "never.json"), post=post)
    assert not calls


def test_handoff_derives_pass_from_same_suite_retained_http(tmp_path, parity_bundle, completed_suite):
    handoff, manifest = completed_suite
    request = parity_bundle(checkpoint=handoff.checkpoint)
    extra = json.loads(Path(request.suite_manifest.path).read_text())
    suite = NativeSuiteRequest.model_validate(manifest["request"])
    added = NativeSuiteRequest.model_validate(extra["request"])
    root = Path(handoff.suite_root)
    for entry, output in zip(added.panels, extra["panels"], strict=True):
        samples = SampleResult.model_validate_json(
            (Path(request.suite_manifest.path).parent / entry.result_key / "samples.json").read_bytes()
        )
        panel_request = NativePanelRequest(
            panel=entry.panel,
            checkpoint=suite.checkpoint,
            model=suite.model,
            serialization=entry.serialization,
            target_cluster=suite.target_cluster,
            expert_axis_size=1,
            replica_axis_size=1,
        )
        report = grade_panel(entry.panel, samples, entry.serialization)
        output.update(
            serialization=entry.serialization.value,
            purpose=entry.purpose.value,
            format_report_digest=report["report_digest"],
        )
        output["file_sha256"] = {
            name: save(root / entry.result_key / name, value).sha256
            for name, value in (
                ("panel.json", entry.panel.model_dump(mode="json")),
                ("request.json", panel_request.model_dump(mode="json")),
                ("samples.json", samples.model_dump(mode="json")),
                ("report.json", report),
            )
        }
    suite = suite.model_copy(update={"panels": suite.panels + added.panels})
    manifest.update(
        request=suite.model_dump(mode="json"), suite_id=suite.suite_id, panels=manifest["panels"] + extra["panels"]
    )
    manifest["manifest_digest"] = digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    suite_pin = save(root / "manifest.json", manifest)
    save(root / "request.json", suite.model_dump(mode="json"))
    request = request.model_copy(update={"suite_manifest": suite_pin})
    post, _ = responses(request)
    collect_serving(request, str(tmp_path / "same-suite.json"), post=post)
    handoff = handoff.model_copy(
        update={
            "suite_manifest_digest": manifest["manifest_digest"],
            "serving_parity_report": str(tmp_path / "same-suite.json"),
            "serving_parity_sha256": pin(tmp_path / "same-suite.json").sha256,
        }
    )
    candidate, provenance = assemble_candidate(handoff)
    assert candidate.serving_parity_passed
    assert provenance["serving_parity"]["measurement"]["passed"]


def test_transport_error_is_retained_rejected_and_recomputable(tmp_path, parity_bundle):
    request = parity_bundle()
    calls = []

    def post(*args, **kwargs):
        calls.append(args)
        raise requests.ConnectionError("owned endpoint unavailable")

    report = collect_serving(request, str(tmp_path / "unavailable.json"), post=post)
    assert len(calls) == 1
    assert report["exchanges"][0]["response"]["transport_error"] == "ConnectionError"
    assert not verify_serving_report(pin(tmp_path / "unavailable.json"))["measurement"]["passed"]
