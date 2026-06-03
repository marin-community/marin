# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

import gzip
import importlib.util
import json
import numpy as np
import sys
import threading
from pathlib import Path

from test_utils import use_test_mesh


_BENCH_PATH = Path(__file__).parents[1] / "scripts" / "bench" / "bench_qwen3_tpu_inference_parity.py"
_SPEC = importlib.util.spec_from_file_location("bench_qwen3_tpu_inference_parity", _BENCH_PATH)
assert _SPEC is not None
bench = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def test_rl_decode_matrix_cases_are_decode_heavy():
    cases = bench.matrix_cases(bench.DEFAULT_MATRIX)

    assert [case.name for case in cases] == [
        "decode_b8_i1_o128_n1",
        "decode_b8_i1_o512_n1",
        "decode_b32_i1_o128_n1",
        "decode_b32_i1_o512_n1",
        "decode_b128_i1_o128_n1",
        "decode_b128_i1_o512_n1",
        "decode_b32_i1_o128_n4",
        "decode_b32_i1_o512_n4",
    ]
    assert all(case.input_tokens == 1 for case in cases)
    assert all(case.output_tokens >= 128 for case in cases)
    assert {case.active_sequences for case in cases if case.n == 1} == {8, 32, 128}
    assert {case.output_tokens for case in cases if case.n == 1} == {128, 512}
    assert any(case.n > 1 for case in cases)


def test_matrix_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown matrix"):
        bench.matrix_cases("prefill")


def test_main_rejects_non_positive_max_prefill_size():
    with pytest.raises(ValueError, match="max-prefill-size"):
        bench.main(["--backend", "levanter", "--measure-rounds", "1", "--max-prefill-size", "0"])


def test_main_rejects_non_positive_decode_round_knobs():
    with pytest.raises(ValueError, match="max-rounds"):
        bench.main(["--backend", "levanter", "--measure-rounds", "1", "--max-rounds", "0"])
    with pytest.raises(ValueError, match="max-tokens-per-round"):
        bench.main(["--backend", "levanter", "--measure-rounds", "1", "--max-tokens-per-round", "0"])
    with pytest.raises(ValueError, match="tensor-parallel-size"):
        bench.main(["--backend", "levanter", "--measure-rounds", "1", "--tensor-parallel-size", "0"])


def test_main_rejects_non_positive_rpa_kernel_knobs():
    with pytest.raises(ValueError, match="rpa-num-kv-pages-per-block"):
        bench.main(["--backend", "levanter", "--measure-rounds", "1", "--rpa-num-kv-pages-per-block", "0"])


def test_main_rejects_reference_logit_check_without_levanter():
    with pytest.raises(ValueError, match="requires the Levanter backend"):
        bench.main(["--backend", "vllm", "--check-levanter-reference-logits"])
    with pytest.raises(ValueError, match="reference-logit-max-prompts"):
        bench.main(
            ["--backend", "levanter", "--check-levanter-reference-logits", "--reference-logit-max-prompts", "0"]
        )
    with pytest.raises(ValueError, match="reference-logit-only"):
        bench.main(["--backend", "levanter", "--reference-logit-only"])


def test_parse_args_defaults_to_auto_tpu_paged_attention_backend():
    args = bench.parse_args(["--backend", "levanter"])

    assert args.levanter_tpu_paged_attention_backend == "auto"
    assert not args.levanter_allow_reference_fallback
    assert args.levanter_sampler_top_k_mode == "candidate"


def test_parse_args_defaults_to_two_warmup_rounds():
    args = bench.parse_args(["--backend", "levanter"])

    assert args.warmup_rounds == 2


def test_parse_args_defaults_to_streaming_greedy_lm_head_with_opt_out():
    default_args = bench.parse_args(["--backend", "levanter"])
    opt_out_args = bench.parse_args(["--backend", "levanter", "--no-levanter-streaming-greedy-lm-head"])

    assert default_args.levanter_streaming_greedy_lm_head
    assert not opt_out_args.levanter_streaming_greedy_lm_head


def test_start_backend_passes_levanter_tpu_paged_attention_config(monkeypatch, tmp_path):
    captured: dict = {}

    def start_levanter_server(**kwargs):
        captured.update(kwargs)
        return bench.ServerHandle(
            name=f"levanter:{kwargs['tpu_paged_attention_backend'].value}",
            base_url="http://127.0.0.1:8000/v1",
            model_id="levanter",
            close=lambda: None,
        )

    monkeypatch.setattr(bench, "start_levanter_server", start_levanter_server)
    args = bench.parse_args(
        [
            "--backend",
            "levanter",
            "--levanter-tpu-paged-attention-backend",
            "jax_rpa",
            "--levanter-allow-reference-fallback",
            "--levanter-compute-dtype",
            "float32",
            "--levanter-trainer-mp",
            "bf16",
            "--levanter-tpu-inference-out-dtype",
            "float32",
            "--levanter-preserve-attention-output-dtype",
            "--levanter-sampler-top-k-mode",
            "threshold_mask",
            "--check-levanter-reference-logits",
            "--reference-logit-max-prompts",
            "1",
            "--reference-logit-decode-backend",
            "tpu_inference",
            "--reference-logit-cache-dtype",
            "auto",
            "--reference-logit-cache-dtype",
            "bfloat16",
            "--reference-logit-only",
            "--tensor-parallel-size",
            "4",
        ]
    )

    handle = bench.start_backend(
        args,
        "levanter",
        output_dir=tmp_path,
        checkpoint="Qwen/Qwen3-8B",
        tokenizer_name="Qwen/Qwen3-8B",
        cases=[],
        prompts={},
    )

    assert handle.name == "levanter:jax_rpa"
    assert captured["tpu_paged_attention_backend"] == bench.TpuPagedAttentionBackend.JAX_RPA
    assert captured["allow_reference_fallback"]
    assert captured["compute_dtype"] == "float32"
    assert captured["trainer_mp_policy"] == "bf16"
    assert captured["tpu_inference_out_dtype"] == "float32"
    assert captured["preserve_attention_output_dtype"]
    assert captured["sampler_top_k_mode"] == bench.SamplerTopKMode.THRESHOLD_MASK
    assert captured["tensor_parallel_size"] == 4
    assert captured["reference_logit_check_dir"] == tmp_path
    assert captured["reference_logit_check_cases"] == []
    assert captured["reference_logit_check_prompts"] == {}
    assert captured["reference_logit_max_prompts"] == 1
    assert captured["reference_logit_decode_backends"] == [bench.TpuPagedAttentionBackend.TPU_INFERENCE]
    assert captured["reference_logit_cache_dtype_policies"] == ["auto", "bfloat16"]
    assert captured["reference_logit_only"]


def test_levanter_trainer_config_uses_model_axis_for_tensor_parallelism():
    trainer = bench.levanter_trainer_config(4, "bf16")

    assert trainer.mp.compute_dtype == bench.jnp.bfloat16
    assert trainer.mesh.axes["data"] == 1
    assert trainer.mesh.axes["replica"] == 1
    assert trainer.mesh.axes["model"] == 4
    assert trainer.compute_axis_mapping["heads"] == "model"
    assert trainer.compute_axis_mapping["kv_head"] == "model"
    assert trainer.compute_axis_mapping["mlp"] == "model"
    assert trainer.compute_axis_mapping["vocab"] == "model"
    assert trainer.parameter_axis_mapping["heads"] == "model"
    assert trainer.parameter_axis_mapping["kv_head"] == "model"
    assert trainer.parameter_axis_mapping["mlp"] == "model"
    assert trainer.parameter_axis_mapping["vocab"] == "model"


def test_write_levanter_kernel_artifacts_uses_trainer_mesh_and_compute_mapping(tmp_path):
    events: list[str] = []

    class Context:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")

        def __exit__(self, exc_type, exc, tb):
            events.append(f"exit:{self.name}")

    class Trainer:
        compute_axis_mapping = {"batch": "data"}

        def use_device_mesh(self):
            return Context("mesh")

    class Engine:
        def write_kernel_jaxprs(self, path, *, return_logprobs, use_streaming_greedy_lm_head, log_artifacts):
            events.append(f"write:{Path(path).name}:{return_logprobs}:{use_streaming_greedy_lm_head}:{log_artifacts}")

    class InferenceContext:
        engine = Engine()

    class Server:
        inference_context = InferenceContext()

    original_axis_mapping = bench.hax.axis_mapping

    def axis_mapping(mapping):
        assert mapping == {"batch": "data"}
        return Context("axis_mapping")

    bench.hax.axis_mapping = axis_mapping
    try:
        bench.write_levanter_kernel_artifacts(
            Trainer(),
            Server(),
            tmp_path,
            return_logprobs=True,
            use_streaming_greedy_lm_head=True,
        )
    finally:
        bench.hax.axis_mapping = original_axis_mapping

    assert events == [
        "enter:mesh",
        "enter:axis_mapping",
        f"write:{tmp_path.name}:True:True:False",
        "exit:axis_mapping",
        "exit:mesh",
    ]


def test_main_rejects_too_large_rpa_kv_block():
    with pytest.raises(ValueError, match="ceil"):
        bench.main(
            [
                "--backend",
                "levanter",
                "--measure-rounds",
                "1",
                "--max-model-len",
                "4096",
                "--rpa-num-kv-pages-per-block",
                "64",
            ]
        )


def test_request_count_accounts_for_n_generations():
    case = next(case for case in bench.matrix_cases(bench.DEFAULT_MATRIX) if case.name == "decode_b32_i1_o128_n4")

    assert case.request_count == 8


def test_write_prompt_corpus_records_exact_token_ids(tmp_path):
    class Tokenizer:
        def encode(self, prompt, *, add_special_tokens):
            assert not add_special_tokens
            if prompt == "x":
                return [7]
            raise AssertionError(f"unexpected prompt {prompt!r}")

    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    corpus = bench.write_prompt_corpus(tmp_path, [case], {case.name: ("x", 1)}, Tokenizer())

    assert corpus == {
        "prompts": [
            {
                "case_name": "decode_b1_i1_o2_n1",
                "active_sequences": 1,
                "input_tokens_target": 1,
                "output_tokens_target": 2,
                "n": 1,
                "request_count": 1,
                "prompt": "x",
                "prompt_token_ids": [7],
                "prompt_tokens": 1,
            }
        ]
    }
    assert json.loads((tmp_path / "prompt_corpus.json").read_text()) == corpus


def test_write_prompt_corpus_rejects_tokenization_drift(tmp_path):
    class Tokenizer:
        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "x"
            assert not add_special_tokens
            return [7, 8]

    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    with pytest.raises(ValueError, match="Prompt token count drift"):
        bench.write_prompt_corpus(tmp_path, [case], {case.name: ("x", 1)}, Tokenizer())


def test_deduplicated_prompt_token_groups_preserve_case_coverage():
    class Tokenizer:
        def encode(self, prompt, *, add_special_tokens):
            assert not add_special_tokens
            return {"x": [7], "y": [8, 9]}[prompt]

    cases = [
        bench.BenchmarkCase("a", active_sequences=1, input_tokens=1, output_tokens=2),
        bench.BenchmarkCase("b", active_sequences=1, input_tokens=1, output_tokens=2),
        bench.BenchmarkCase("c", active_sequences=1, input_tokens=2, output_tokens=2),
    ]

    groups = bench._deduplicated_prompt_token_groups(
        cases,
        {"a": ("x", 1), "b": ("x", 1), "c": ("y", 2)},
        Tokenizer(),
        max_prompts=None,
    )

    assert groups == [(["a", "b"], [7]), (["c"], [8, 9])]


def test_single_token_direct_attention_matches_full_attention_for_tiny_qwen3():
    Vocab = bench.hax.Axis("vocab", 32)
    Pos = bench.hax.Axis("position", 1)
    config = bench.Qwen3Config(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        scan_layers=True,
    )

    with use_test_mesh():
        model = bench.Qwen3LMHeadModel.init(Vocab, config, key=bench.jax.random.PRNGKey(0))
        input_ids = bench.hax.named(bench.jnp.asarray([7], dtype=bench.jnp.int32), Pos)
        pos_ids = bench.hax.arange(Pos, dtype=bench.jnp.int32)

        direct_hidden = bench._single_token_direct_attention_hidden(model, input_ids, pos_ids)
        unrolled_hidden = bench._unrolled_full_attention_hidden(
            model,
            input_ids,
            bench.AttentionMask.causal(),
            pos_ids,
        )
        scanned_hidden = model.activations(
            input_ids,
            attn_mask=bench.AttentionMask.causal(),
            pos_ids=pos_ids,
            key=None,
        )

    assert direct_hidden is not None
    np.testing.assert_allclose(direct_hidden.array, unrolled_hidden.array, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(scanned_hidden.array, unrolled_hidden.array, rtol=1e-4, atol=1e-4)


def test_write_reference_logit_check_outputs_records_pass_and_fail(tmp_path):
    passing = bench.ReferenceLogitCheckResult(
        prompt_index=0,
        case_names=["decode_b8_i1_o128_n1"],
        prompt_token_ids=[7],
        decode_backend="tpu_inference",
        kv_cache_dtype="bfloat16",
        kv_cache_dtype_policy="auto",
        reference_hidden_dtype="float32",
        decode_hidden_dtype="bfloat16",
        reference_logits_dtype="float32",
        decode_logits_dtype="bfloat16",
        tpu_inference_out_dtype="float32",
        preserve_attention_output_dtype=True,
        positions=1,
        vocab_size=32,
        hidden_max_abs_error=0.01,
        hidden_mean_abs_error=0.001,
        hidden_rms_abs_error=0.002,
        max_abs_error=0.1,
        max_abs_error_if_reference_rounded_to_decode_dtype=0.1,
        residual_max_abs_error_after_reference_rounding=0.0,
        max_rel_error=0.01,
        mean_abs_error=0.001,
        rms_abs_error=0.002,
        abs_error_p50=0.0001,
        abs_error_p90=0.001,
        abs_error_p99=0.01,
        abs_error_p999=0.05,
        top1_agreement=1.0,
        top_k_diagnostics=[
            {
                "k": 1,
                "max_abs_error": 0.01,
                "mean_abs_error": 0.01,
                "max_logprob_abs_error": 0.02,
                "mean_logprob_abs_error": 0.02,
                "overlap_fraction": 1.0,
            }
        ],
        abs_error_histogram=[{"lower": 0.0, "upper": 0.1, "count": 32, "fraction": 1.0}],
        passed=True,
    )
    failing = bench.ReferenceLogitCheckResult(
        prompt_index=1,
        case_names=["decode_b32_i1_o128_n4"],
        prompt_token_ids=[8],
        decode_backend="reference",
        kv_cache_dtype="float32",
        kv_cache_dtype_policy="default",
        reference_hidden_dtype="float32",
        decode_hidden_dtype="float32",
        reference_logits_dtype="float32",
        decode_logits_dtype="float32",
        tpu_inference_out_dtype=None,
        preserve_attention_output_dtype=False,
        positions=1,
        vocab_size=32,
        hidden_max_abs_error=0.2,
        hidden_mean_abs_error=0.1,
        hidden_rms_abs_error=0.15,
        max_abs_error=1.0,
        max_abs_error_if_reference_rounded_to_decode_dtype=0.0,
        residual_max_abs_error_after_reference_rounding=1.0,
        max_rel_error=0.5,
        mean_abs_error=0.1,
        rms_abs_error=0.2,
        abs_error_p50=0.1,
        abs_error_p90=0.5,
        abs_error_p99=0.9,
        abs_error_p999=1.0,
        top1_agreement=0.0,
        top_k_diagnostics=[
            {
                "k": 1,
                "max_abs_error": 1.0,
                "mean_abs_error": 1.0,
                "max_logprob_abs_error": 1.0,
                "mean_logprob_abs_error": 1.0,
                "overlap_fraction": 0.0,
            }
        ],
        abs_error_histogram=[{"lower": 0.0, "upper": 0.1, "count": 16, "fraction": 0.5}],
        passed=False,
    )

    bench.write_reference_logit_check_outputs(tmp_path, [passing], atol=0.5, rtol=0.05)
    payload = json.loads((tmp_path / "levanter_reference_logits.json").read_text())
    assert payload["passed"] is True
    assert payload["results"][0]["case_names"] == ["decode_b8_i1_o128_n1"]
    rendered = (tmp_path / "levanter_reference_logits.md").read_text()
    assert "decode_b8_i1_o128_n1" in rendered
    assert "reference-top-k diagnostics" in rendered
    assert "absolute-error histogram" in rendered
    assert "round residual max abs" in rendered
    assert "hidden max abs" in rendered
    assert "direct-decode hidden max abs" in rendered

    with pytest.raises(AssertionError, match="reference-logit check failed"):
        bench.write_reference_logit_check_outputs(tmp_path, [passing, failing], atol=0.5, rtol=0.05)
    payload = json.loads((tmp_path / "levanter_reference_logits.json").read_text())
    assert payload["passed"] is False

    bench.write_reference_logit_check_outputs(
        tmp_path,
        [passing, failing],
        atol=0.5,
        rtol=0.05,
        raise_on_failure=False,
    )
    payload = json.loads((tmp_path / "levanter_reference_logits.json").read_text())
    assert payload["passed"] is False


def test_reference_logit_error_metrics_reports_distribution_and_topk():
    reference = bench.jnp.asarray([[2.0, 1.0, 0.0, -1.0]], dtype=bench.jnp.float32)
    candidate = bench.jnp.asarray([[1.9, 1.2, 0.1, -1.0]], dtype=bench.jnp.float32)

    metrics = bench._reference_logit_error_metrics(reference, candidate)

    assert metrics["max_abs_error"] == pytest.approx(0.2)
    assert metrics["reference_logits_dtype"] == "float32"
    assert metrics["decode_logits_dtype"] == "float32"
    assert metrics["max_abs_error_if_reference_rounded_to_decode_dtype"] == pytest.approx(0.0)
    assert metrics["residual_max_abs_error_after_reference_rounding"] == pytest.approx(0.2)
    assert metrics["mean_abs_error"] == pytest.approx(0.1)
    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["top_k_diagnostics"][0]["k"] == 1
    assert metrics["top_k_diagnostics"][0]["max_abs_error"] == pytest.approx(0.1)
    assert metrics["top_k_diagnostics"][-1]["k"] == 4
    assert sum(bucket["count"] for bucket in metrics["abs_error_histogram"]) == 4


def test_reference_logit_cache_dtype_uses_bf16_for_tpu_inference_policy():
    assert (
        bench._reference_logit_cache_dtype(
            bench.TpuPagedAttentionConfig(backend=bench.TpuPagedAttentionBackend.AUTO),
            bench.jnp.float32,
        )
        == bench.jnp.bfloat16
    )
    assert (
        bench._reference_logit_cache_dtype(
            bench.TpuPagedAttentionConfig(backend=bench.TpuPagedAttentionBackend.TPU_INFERENCE),
            bench.jnp.float32,
        )
        == bench.jnp.bfloat16
    )
    assert (
        bench._reference_logit_cache_dtype(
            bench.TpuPagedAttentionConfig(backend=bench.TpuPagedAttentionBackend.REFERENCE),
            bench.jnp.float32,
        )
        == bench.jnp.float32
    )


def test_reference_logit_cache_dtype_policy_can_force_f32_for_tpu_inference():
    config = bench.TpuPagedAttentionConfig(backend=bench.TpuPagedAttentionBackend.TPU_INFERENCE)

    assert bench._reference_logit_cache_dtype_for_policy(config, bench.jnp.bfloat16, "auto") == bench.jnp.bfloat16
    assert bench._reference_logit_cache_dtype_for_policy(config, bench.jnp.bfloat16, "float32") == bench.jnp.float32


def test_percentile_interpolates():
    assert bench._percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert bench._percentile([0.0, 100.0], 0.9) == 90.0


def test_both_backends_run_sequentially():
    assert bench.selected_backends("both") == ["vllm", "levanter"]
    assert bench.selected_backends("vllm") == ["vllm"]
    assert bench.selected_backends("levanter") == ["levanter"]


def test_vllm_env_snapshot_does_not_probe_jax_devices(monkeypatch):
    def fail_if_called():
        raise AssertionError("vLLM env collection must not initialize JAX devices")

    monkeypatch.setattr(bench, "_jax_device_snapshot", fail_if_called)

    snapshot = bench._runtime_env_snapshot(include_jax_devices=False)

    assert snapshot["devices"] is None
    assert snapshot["device_kind"] is None
    assert "devices_skipped" in snapshot


def test_send_completion_includes_error_body(monkeypatch):
    class FailedResponse:
        text = '{"detail":"out of memory"}'

        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error", response=self)

    def post(url, json, timeout):
        assert url == "http://127.0.0.1:8000/v1/completions"
        assert json["model"] == "Qwen/Qwen3-8B"
        assert json["temperature"] == 0.0
        assert json["top_p"] == 1.0
        assert "logprobs" not in json
        assert timeout == 1.0
        return FailedResponse()

    monkeypatch.setattr(bench.requests, "post", post)

    with pytest.raises(requests.HTTPError, match="out of memory"):
        bench._send_completion(
            base_url="http://127.0.0.1:8000/v1",
            model_id="Qwen/Qwen3-8B",
            prompt="x",
            max_tokens=1,
            n=1,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            timeout=1.0,
            return_logprobs=False,
        )


def test_send_completion_requests_generated_logprobs(monkeypatch):
    captured: dict = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def post(_url, json, timeout):
        assert timeout == 1.0
        captured.update(json)
        return Response()

    monkeypatch.setattr(bench.requests, "post", post)

    bench._send_completion(
        base_url="http://127.0.0.1:8000/v1",
        model_id="Qwen/Qwen3-8B",
        prompt="x",
        max_tokens=1,
        n=1,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        timeout=1.0,
        return_logprobs=True,
    )

    assert captured["logprobs"] == 1


def test_send_completion_omits_seed_when_backend_does_not_support_it(monkeypatch):
    captured: dict = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def post(_url, json, timeout):
        assert timeout == 1.0
        captured.update(json)
        return Response()

    monkeypatch.setattr(bench.requests, "post", post)

    bench._send_completion(
        base_url="http://127.0.0.1:8000/v1",
        model_id="Qwen/Qwen3-8B",
        prompt="x",
        max_tokens=1,
        n=1,
        temperature=0.7,
        top_p=1.0,
        seed=None,
        timeout=1.0,
        return_logprobs=False,
    )

    assert "seed" not in captured


def _case_result(
    case_name: str,
    backend: str,
    decode_tokens_per_second: float,
    *,
    compile_including_seconds: float | None = None,
    hbm_used_bytes: int | None = None,
    compiled_shape_count: int | None = None,
) -> bench.CaseResult:
    return bench.CaseResult(
        case_name=case_name,
        backend=backend,
        request_count=1,
        active_sequences=1,
        n=1,
        input_tokens_target=1,
        output_tokens_target=128,
        prompt_tokens=1,
        completion_tokens=128,
        total_tokens=129,
        compile_including_seconds=compile_including_seconds,
        steady_state_seconds=1.0,
        request_latency_ms_p50=1.0,
        request_latency_ms_p90=1.0,
        ttft_ms_p50=None,
        decode_tokens_per_second=decode_tokens_per_second,
        total_tokens_per_second=decode_tokens_per_second + 1.0,
        hbm_used_bytes=hbm_used_bytes,
        compiled_shape_count=compiled_shape_count,
    )


def test_summary_markdown_includes_vllm_ratio_columns(tmp_path):
    bench.write_outputs(
        tmp_path,
        [
            _case_result("decode_b8_i1_o128_n1", "vllm-tpu", 100.0),
            _case_result(
                "decode_b8_i1_o128_n1",
                "levanter",
                90.0,
                compile_including_seconds=12.3456,
                hbm_used_bytes=123456,
                compiled_shape_count=2,
            ),
        ],
        {"backend": "both"},
    )

    summary = (tmp_path / "summary.md").read_text()

    assert "decode/vllm" in summary
    assert "total/vllm" in summary
    assert "target" in summary
    assert "compile incl s" in summary
    assert "ttft p50 ms" in summary
    assert "hbm bytes" in summary
    assert "shape buckets" in summary
    assert "0.900" in summary
    assert "pass" in summary
    assert "12.346" in summary
    assert "123456" in summary


def test_summary_json_includes_machine_readable_parity_comparisons(tmp_path):
    bench.write_outputs(
        tmp_path,
        [
            _case_result("decode_b8_i1_o128_n1", "vllm-tpu", 100.0),
            _case_result("decode_b8_i1_o128_n1", "levanter:auto", 90.0),
            _case_result("decode_b32_i1_o128_n1", "vllm-tpu", 100.0),
            _case_result("decode_b32_i1_o128_n1", "levanter:auto", 70.0),
        ],
        {"backend": "both"},
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["parity_decode_ratio_target"] == 0.85
    comparisons = {comparison["case_name"]: comparison for comparison in summary["comparisons"]}
    assert comparisons["decode_b8_i1_o128_n1"]["decode_ratio"] == 0.9
    assert comparisons["decode_b8_i1_o128_n1"]["meets_decode_ratio_target"] is True
    assert comparisons["decode_b32_i1_o128_n1"]["decode_ratio"] == 0.7
    assert comparisons["decode_b32_i1_o128_n1"]["meets_decode_ratio_target"] is False


def _stress_result() -> bench.StressResult:
    return bench.StressResult(
        case_name="decode_b32_i1_o128_n4",
        backend="levanter:auto",
        concurrent_requests=8,
        active_sequences=32,
        n=4,
        input_tokens_target=1,
        output_tokens_target=128,
        total_requests=10,
        successful_requests=9,
        failed_requests=1,
        timeout_requests=1,
        prompt_tokens=9,
        completion_tokens=1152,
        total_tokens=1161,
        load_seconds=60.0,
        wall_clock_seconds=62.0,
        steady_decode_tokens_per_second=19.2,
        wall_clock_decode_tokens_per_second=18.5806451613,
        wall_clock_total_tokens_per_second=18.7258064516,
        request_latency_ms_p50=1000.0,
        request_latency_ms_p90=1200.0,
        request_latency_ms_p99=1400.0,
        request_latency_ms_max=1500.0,
        hbm_used_bytes=1234,
        compiled_shape_count=2,
        max_request_queue_depth=3,
        max_batch_queue_depth=1,
        page_size=128,
        max_pages=1024,
        retry_errors=0,
        error_counts={"timeout": 1},
    )


def test_write_outputs_includes_stress_artifacts(tmp_path):
    bench.write_outputs(
        tmp_path,
        [_case_result("decode_b32_i1_o128_n4", "levanter:auto", 100.0)],
        {"backend": "levanter"},
        [_stress_result()],
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    stress_summary = (tmp_path / "stress_summary.md").read_text()
    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}

    assert summary["stress_results"][0]["successful_requests"] == 9
    assert "steady decode tok/s" in stress_summary
    assert "decode_b32_i1_o128_n4" in stress_summary
    assert "stress_summary.md" in artifact_paths


def test_write_outputs_records_artifact_manifest(tmp_path):
    hlo_dir = tmp_path / "levanter_hlo"
    hlo_dir.mkdir()
    (hlo_dir / "generate.txt").write_text("hlo")
    vllm_dir = tmp_path / "vllm_profiles"
    vllm_dir.mkdir()
    (vllm_dir / "stdout.log").write_text("server log")
    levanter_profile_dir = tmp_path / "levanter_profiles" / "levanter_auto" / "decode_b8_i1_o128_n1"
    levanter_profile_dir.mkdir(parents=True)
    (levanter_profile_dir / "trace.json.gz").write_text("profile")
    (tmp_path / "prompt_corpus.json").write_text("{}")
    (tmp_path / "levanter_reference_logits.json").write_text("{}")
    (tmp_path / "levanter_reference_logits.md").write_text("| prompt |\n")

    bench.write_outputs(
        tmp_path, [_case_result("decode_b8_i1_o128_n1", "levanter:auto", 90.0)], {"backend": "levanter"}
    )

    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}

    assert {
        "summary.json",
        "summary.md",
        "env.json",
        "prompt_corpus.json",
        "levanter_reference_logits.json",
        "levanter_reference_logits.md",
        "levanter_hlo",
        "levanter_profiles",
        "vllm_profiles",
    } <= set(artifacts)
    assert artifacts["levanter_hlo"]["kind"] == "directory"
    assert artifacts["levanter_hlo"]["file_count"] == 1
    assert artifacts["levanter_profiles"]["kind"] == "directory"
    assert artifacts["vllm_profiles"]["bytes"] == len("server log")


def test_write_outputs_logs_durable_hlo_summary(tmp_path):
    hlo_dir = tmp_path / "levanter_hlo"
    hlo_dir.mkdir()
    with gzip.open(hlo_dir / "gen_loop.hlo.txt.gz", "wt") as f:
        f.write(
            "\n".join(
                [
                    "ENTRY main {",
                    "  %rng = u32[2] parameter(0)",
                    '  %gather = "stablehlo.gather"(%x, %i) <{indices_are_sorted = false}>',
                    "  %sort = call @argsort(f32[8]{0} %logits)",
                    "  %all-gather = f32[8]{0} all-gather(f32[2]{0} %partial)",
                    '  %custom-call = f32[8]{0} custom-call(), custom_call_target="ragged_paged_attention"',
                    "}",
                ]
            )
        )

    bench.write_outputs(
        tmp_path, [_case_result("decode_b8_i1_o128_n1", "levanter:auto", 90.0)], {"backend": "levanter"}
    )

    hlo_summary = json.loads((tmp_path / "hlo_summary.json").read_text())
    file_summary = hlo_summary["files"][0]
    assert file_summary["path"] == "levanter_hlo/gen_loop.hlo.txt.gz"
    assert file_summary["pattern_counts"]["collective"] == 1
    assert file_summary["pattern_counts"]["rng_sampling"] == 1
    assert file_summary["pattern_counts"]["sort_or_topk"] == 1
    assert file_summary["pattern_counts"]["custom_call"] == 1

    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert {"hlo_summary.json", "hlo_summary.md"} <= artifact_paths


def test_start_backend_places_vllm_logs_under_output_dir(monkeypatch, tmp_path):
    captured: dict = {}

    def start_vllm_server(**kwargs):
        captured.update(kwargs)
        return bench.ServerHandle("vllm-tpu", "http://127.0.0.1:8000/v1", "Qwen/Qwen3-8B", close=lambda: None)

    monkeypatch.setattr(bench, "start_vllm_server", start_vllm_server)
    args = bench.parse_args(["--backend", "vllm"])

    bench.start_backend(
        args,
        "vllm",
        output_dir=tmp_path,
        checkpoint="Qwen/Qwen3-8B",
        tokenizer_name="Qwen/Qwen3-8B",
        cases=[],
        prompts={},
    )

    assert captured["log_dir"] == tmp_path / "vllm_profiles"


def test_main_rejects_vllm_greedy_multi_generation_cases():
    with pytest.raises(ValueError, match="n > 1 with greedy sampling"):
        bench.main(
            [
                "--backend",
                "vllm",
                "--case",
                "decode_b32_i1_o128_n4",
                "--measure-rounds",
                "1",
            ]
        )


def test_main_accepts_vllm_sampled_multi_generation_cases(monkeypatch, tmp_path):
    started: list[str] = []

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "Hello"
            assert not add_special_tokens
            return [1]

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1]
            assert not skip_special_tokens
            return "Hello"

    def start_backend(args, backend, *, output_dir, checkpoint, tokenizer_name, cases, prompts):
        del cases, prompts
        started.append(backend)
        return bench.ServerHandle(
            name="vllm-tpu",
            base_url="http://127.0.0.1:8000/v1",
            model_id="Qwen/Qwen3-8B",
            close=lambda: None,
        )

    def run_cases_for_backend(*, args, handle, cases, prompts, output_dir):
        assert args.temperature == 0.7
        assert output_dir == tmp_path
        assert [case.name for case in cases] == ["decode_b32_i1_o128_n4"]
        return [_case_result("decode_b32_i1_o128_n4", handle.name, 100.0)]

    monkeypatch.setattr(bench, "load_tokenizer", lambda tokenizer_name: Tokenizer())
    monkeypatch.setattr(bench, "start_backend", start_backend)
    monkeypatch.setattr(bench, "run_cases_for_backend", run_cases_for_backend)
    monkeypatch.setattr(
        bench,
        "_jax_device_snapshot",
        lambda: pytest.fail("vLLM benchmark finalization must not initialize JAX devices"),
    )

    bench.main(
        [
            "--backend",
            "vllm",
            "--case",
            "decode_b32_i1_o128_n4",
            "--temperature",
            "0.7",
            "--measure-rounds",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert started == ["vllm"]
    env = json.loads((tmp_path / "env.json").read_text())
    assert env["backend_envs"]["vllm-tpu"]["devices"] is None
    assert "devices_skipped" in env["backend_envs"]["vllm-tpu"]


def test_main_accepts_stress_only_without_measure_rounds(monkeypatch, tmp_path):
    started: list[str] = []

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "Hello"
            assert not add_special_tokens
            return [1]

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1]
            assert not skip_special_tokens
            return "Hello"

    def start_backend(args, backend, *, output_dir, checkpoint, tokenizer_name, cases, prompts):
        del args, output_dir, checkpoint, tokenizer_name, cases, prompts
        started.append(backend)
        return bench.ServerHandle("levanter:auto", "http://127.0.0.1:8000/v1", "levanter", close=lambda: None)

    def run_stress_cases_for_backend(*, args, handle, cases, prompts):
        assert args.measure_rounds == 0
        assert args.stress_max_requests == 2
        assert [case.name for case in cases] == ["decode_b32_i1_o128_n4"]
        assert prompts["decode_b32_i1_o128_n4"][1] == 1
        return [_stress_result()]

    monkeypatch.setattr(bench, "load_tokenizer", lambda tokenizer_name: Tokenizer())
    monkeypatch.setattr(bench, "start_backend", start_backend)
    monkeypatch.setattr(bench, "run_cases_for_backend", lambda **kwargs: pytest.fail("measure pass should be skipped"))
    monkeypatch.setattr(bench, "run_stress_cases_for_backend", run_stress_cases_for_backend)

    bench.main(
        [
            "--backend",
            "levanter",
            "--case",
            "decode_b32_i1_o128_n4",
            "--temperature",
            "0.7",
            "--measure-rounds",
            "0",
            "--stress-max-requests",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert started == ["levanter"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["results"] == []
    assert summary["stress_results"][0]["total_requests"] == 10


def test_run_case_propagates_backend_static_metrics(monkeypatch):
    def send_completion(**kwargs):
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "choices": []}, 0.1

    monkeypatch.setattr(bench, "_send_completion", send_completion)

    result = bench.run_case(
        handle=bench.ServerHandle(
            name="levanter",
            base_url="http://127.0.0.1:8000/v1",
            model_id="levanter",
            close=lambda: None,
            hbm_used_bytes=1234,
            compiled_shape_count=2,
        ),
        case=bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2),
        prompt="x",
        prompt_tokens=1,
        warmup=False,
        seed=0,
        temperature=0.0,
        top_p=1.0,
        request_timeout=1.0,
        return_logprobs=True,
    )

    assert result.hbm_used_bytes == 1234
    assert result.compiled_shape_count == 2


def test_run_stress_case_aggregates_success_failures_and_service_metrics(monkeypatch):
    calls = 0

    def send_completion(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise requests.exceptions.Timeout("timed out")
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "choices": []}, 0.01

    def metrics_snapshot():
        return {"request_queue_depth": 4, "batch_queue_depth": 2, "page_size": 128, "max_pages": 1024}

    monkeypatch.setattr(bench, "_send_completion", send_completion)

    result = bench.run_stress_case(
        handle=bench.ServerHandle(
            name="levanter:auto",
            base_url="http://127.0.0.1:8000/v1",
            model_id="levanter",
            close=lambda: None,
            hbm_used_bytes=1234,
            compiled_shape_count=2,
            metrics_snapshot=metrics_snapshot,
        ),
        case=bench.BenchmarkCase("decode_b4_i1_o2_n2", active_sequences=4, input_tokens=1, output_tokens=2, n=2),
        prompt="x",
        prompt_tokens=1,
        seed=0,
        temperature=0.7,
        top_p=1.0,
        top_k=4096,
        request_timeout=1.0,
        return_logprobs=True,
        duration_seconds=0.0,
        concurrent_requests=2,
        metrics_interval_seconds=0.001,
        max_requests=3,
    )

    assert result.total_requests == 3
    assert result.successful_requests == 2
    assert result.failed_requests == 1
    assert result.timeout_requests == 1
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 4
    assert result.error_counts == {"timeout": 1}
    assert result.request_latency_ms_p50 == 10.0
    assert result.max_request_queue_depth == 4
    assert result.max_batch_queue_depth == 2
    assert result.page_size == 128
    assert result.max_pages == 1024
    assert result.hbm_used_bytes == 1234
    assert result.compiled_shape_count == 2


def test_run_case_omits_seed_for_backends_without_seed_support(monkeypatch):
    seeds: list[int | None] = []

    def send_completion(**kwargs):
        seeds.append(kwargs["seed"])
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "choices": []}, 0.1

    monkeypatch.setattr(bench, "_send_completion", send_completion)

    bench.run_case(
        handle=bench.ServerHandle(
            name="vllm-tpu",
            base_url="http://127.0.0.1:8000/v1",
            model_id="Qwen/Qwen3-8B",
            close=lambda: None,
            supports_seed=False,
        ),
        case=bench.BenchmarkCase("decode_b4_i1_o2_n2", active_sequences=4, input_tokens=1, output_tokens=2, n=2),
        prompt="x",
        prompt_tokens=1,
        warmup=False,
        seed=123,
        temperature=0.7,
        top_p=1.0,
        request_timeout=1.0,
        return_logprobs=True,
    )

    assert seeds == [None, None]


def test_run_cases_reports_total_configured_warmup_time(monkeypatch):
    calls: list[bool] = []

    def run_case(**kwargs):
        calls.append(kwargs["warmup"])
        return bench.CaseResult(
            case_name=kwargs["case"].name,
            backend=kwargs["handle"].name,
            request_count=1,
            active_sequences=1,
            n=1,
            input_tokens_target=1,
            output_tokens_target=2,
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            compile_including_seconds=None,
            steady_state_seconds=10.0 if kwargs["warmup"] else 1.0,
            request_latency_ms_p50=1.0,
            request_latency_ms_p90=1.0,
            ttft_ms_p50=None,
            decode_tokens_per_second=2.0,
            total_tokens_per_second=3.0,
            hbm_used_bytes=None,
            compiled_shape_count=None,
        )

    monkeypatch.setattr(bench, "run_case", run_case)
    args = bench.parse_args(["--backend", "levanter", "--warmup-rounds", "2", "--measure-rounds", "1"])
    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    results = bench.run_cases_for_backend(
        args=args,
        handle=bench.ServerHandle("levanter", "http://127.0.0.1:8000/v1", "levanter", close=lambda: None),
        cases=[case],
        prompts={case.name: ("x", 1)},
    )

    assert calls == [True, True, False]
    assert results[0].compile_including_seconds == 20.0


def test_run_cases_profiles_measured_levanter_rounds(monkeypatch, tmp_path):
    calls: list[bool] = []
    profile_events: list[tuple[str, str]] = []

    class StepTraceAnnotation:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            profile_events.append(("step_enter", self.name))

        def __exit__(self, exc_type, exc, tb):
            profile_events.append(("step_exit", self.name))

    def start_trace(path, *, create_perfetto_trace):
        assert create_perfetto_trace is True
        profile_events.append(("start", path))

    def stop_trace():
        profile_events.append(("stop", ""))

    def run_case(**kwargs):
        calls.append(kwargs["warmup"])
        return _case_result(kwargs["case"].name, kwargs["handle"].name, 2.0)

    monkeypatch.setattr(bench, "run_case", run_case)
    monkeypatch.setattr(bench.jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(bench.jax.profiler, "stop_trace", stop_trace)
    monkeypatch.setattr(bench.jax.profiler, "StepTraceAnnotation", StepTraceAnnotation)

    args = bench.parse_args(
        [
            "--backend",
            "levanter",
            "--warmup-rounds",
            "1",
            "--measure-rounds",
            "1",
            "--profile-levanter",
        ]
    )
    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    bench.run_cases_for_backend(
        args=args,
        handle=bench.ServerHandle("levanter:auto", "http://127.0.0.1:8000/v1", "levanter", close=lambda: None),
        cases=[case],
        prompts={case.name: ("x", 1)},
        output_dir=tmp_path,
    )

    assert calls == [True, False]
    assert profile_events == [
        (
            "start",
            str(tmp_path / "levanter_profiles" / "levanter_auto" / "decode_b1_i1_o2_n1" / "measure_0"),
        ),
        ("step_enter", "levanter:auto:decode_b1_i1_o2_n1:measure_0"),
        ("step_exit", "levanter:auto:decode_b1_i1_o2_n1:measure_0"),
        ("stop", ""),
    ]


def test_run_cases_can_add_no_lm_head_diagnostic(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def run_case(**kwargs):
        calls.append((kwargs["handle"].name, kwargs["warmup"]))
        return _case_result(kwargs["case"].name, kwargs["handle"].name, 2.0)

    def diagnose_without_lm_head(case, prompt, prompt_tokens, warmup, seed, temperature, top_p, request_timeout):
        del prompt, prompt_tokens, seed, temperature, top_p, request_timeout
        calls.append(("levanter:auto:no_lm_head", warmup))
        return _case_result(case.name, "levanter:auto:no_lm_head", 20.0 if warmup else 10.0)

    monkeypatch.setattr(bench, "run_case", run_case)
    args = bench.parse_args(
        [
            "--backend",
            "levanter",
            "--warmup-rounds",
            "1",
            "--measure-rounds",
            "1",
            "--levanter-diagnose-without-lm-head",
        ]
    )
    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    results = bench.run_cases_for_backend(
        args=args,
        handle=bench.ServerHandle(
            "levanter:auto",
            "http://127.0.0.1:8000/v1",
            "levanter",
            close=lambda: None,
            diagnose_without_lm_head=diagnose_without_lm_head,
        ),
        cases=[case],
        prompts={case.name: ("x", 1)},
    )

    assert calls == [
        ("levanter:auto", True),
        ("levanter:auto", False),
        ("levanter:auto:no_lm_head", True),
        ("levanter:auto:no_lm_head", False),
    ]
    assert [result.backend for result in results] == ["levanter:auto", "levanter:auto:no_lm_head"]
    assert results[0].compile_including_seconds == 1.0
    assert results[1].compile_including_seconds == 1.0


def test_run_cases_can_add_lm_head_no_sampling_diagnostic(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def run_case(**kwargs):
        calls.append((kwargs["handle"].name, kwargs["warmup"]))
        return _case_result(kwargs["case"].name, kwargs["handle"].name, 2.0)

    def diagnose_with_lm_head_no_sampling(
        case, prompt, prompt_tokens, warmup, seed, temperature, top_p, request_timeout
    ):
        del prompt, prompt_tokens, seed, temperature, top_p, request_timeout
        calls.append(("levanter:auto:lm_head_no_sampling", warmup))
        return _case_result(case.name, "levanter:auto:lm_head_no_sampling", 20.0 if warmup else 10.0)

    monkeypatch.setattr(bench, "run_case", run_case)
    args = bench.parse_args(
        [
            "--backend",
            "levanter",
            "--warmup-rounds",
            "1",
            "--measure-rounds",
            "1",
            "--levanter-diagnose-lm-head-no-sampling",
        ]
    )
    case = bench.BenchmarkCase("decode_b1_i1_o2_n1", active_sequences=1, input_tokens=1, output_tokens=2)

    results = bench.run_cases_for_backend(
        args=args,
        handle=bench.ServerHandle(
            "levanter:auto",
            "http://127.0.0.1:8000/v1",
            "levanter",
            close=lambda: None,
            diagnose_with_lm_head_no_sampling=diagnose_with_lm_head_no_sampling,
        ),
        cases=[case],
        prompts={case.name: ("x", 1)},
    )

    assert calls == [
        ("levanter:auto", True),
        ("levanter:auto", False),
        ("levanter:auto:lm_head_no_sampling", True),
        ("levanter:auto:lm_head_no_sampling", False),
    ]
    assert [result.backend for result in results] == ["levanter:auto", "levanter:auto:lm_head_no_sampling"]
    assert results[0].compile_including_seconds == 1.0
    assert results[1].compile_including_seconds == 1.0


def test_run_levanter_without_lm_head_case_preserves_warmup_flag(monkeypatch):
    events: list[str] = []

    class Context:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")

        def __exit__(self, exc_type, exc, tb):
            events.append(f"exit:{self.name}")

    class Trainer:
        compute_axis_mapping = {"batch": "data"}

        def use_device_mesh(self):
            return Context("mesh")

    class Tokenizer:
        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "x"
            assert not add_special_tokens
            return [7]

    class Engine:
        def generate_without_lm_head(self, requests):
            assert len(requests) == 1
            assert requests[0].prompt_tokens == [7]
            assert requests[0].n_generations == 2

            class Result:
                total_generated = 2

            return Result()

        def generate_with_lm_head_no_sampling(self, requests):
            return self.generate_without_lm_head(requests)

    class InferenceContext:
        engine = Engine()

    class Server:
        inference_context = InferenceContext()

    original_axis_mapping = bench.hax.axis_mapping

    def axis_mapping(mapping):
        assert mapping == {"batch": "data"}
        return Context("axis_mapping")

    clock = iter([10.0, 12.5, 20.0, 21.0, 30.0, 31.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    bench.hax.axis_mapping = axis_mapping
    try:
        case = bench.BenchmarkCase("decode_b2_i1_o1_n2", active_sequences=2, input_tokens=1, output_tokens=1, n=2)
        warmup = bench.run_levanter_without_lm_head_case(
            trainer=Trainer(),
            server=Server(),
            tokenizer=Tokenizer(),
            backend_name="levanter:auto:no_lm_head",
            hbm_used_bytes=123,
            compiled_shape_count=2,
            case=case,
            prompt="x",
            prompt_tokens=1,
            warmup=True,
            seed=0,
            temperature=0.0,
            top_p=1.0,
        )
        measured = bench.run_levanter_without_lm_head_case(
            trainer=Trainer(),
            server=Server(),
            tokenizer=Tokenizer(),
            backend_name="levanter:auto:no_lm_head",
            hbm_used_bytes=123,
            compiled_shape_count=2,
            case=case,
            prompt="x",
            prompt_tokens=1,
            warmup=False,
            seed=0,
            temperature=0.0,
            top_p=1.0,
        )
        lm_head_no_sampling = bench.run_levanter_with_lm_head_no_sampling_case(
            trainer=Trainer(),
            server=Server(),
            tokenizer=Tokenizer(),
            backend_name="levanter:auto:lm_head_no_sampling",
            hbm_used_bytes=123,
            compiled_shape_count=2,
            case=case,
            prompt="x",
            prompt_tokens=1,
            warmup=False,
            seed=0,
            temperature=0.0,
            top_p=1.0,
        )
    finally:
        bench.hax.axis_mapping = original_axis_mapping

    assert warmup.compile_including_seconds == 2.5
    assert warmup.steady_state_seconds == 2.5
    assert measured.compile_including_seconds is None
    assert measured.steady_state_seconds == 1.0
    assert lm_head_no_sampling.backend == "levanter:auto:lm_head_no_sampling"
    assert measured.hbm_used_bytes == 123
    assert measured.compiled_shape_count == 2
    assert events == [
        "enter:mesh",
        "enter:axis_mapping",
        "exit:axis_mapping",
        "exit:mesh",
        "enter:mesh",
        "enter:axis_mapping",
        "exit:axis_mapping",
        "exit:mesh",
        "enter:mesh",
        "enter:axis_mapping",
        "exit:axis_mapping",
        "exit:mesh",
    ]


def test_send_completion_after_start_forwards_after_release(monkeypatch):
    calls: list[dict] = []

    def send_completion(**kwargs):
        calls.append(kwargs)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "choices": []}, 0.1

    monkeypatch.setattr(bench, "_send_completion", send_completion)
    start_event = threading.Event()

    start_event.set()
    payload, elapsed = bench._send_completion_after_start(
        start_event=start_event,
        base_url="http://127.0.0.1:8000/v1",
        model_id="levanter",
        prompt="x",
        max_tokens=1,
        n=1,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        timeout=1.0,
        return_logprobs=True,
    )

    assert payload["usage"]["completion_tokens"] == 1
    assert elapsed == 0.1
    assert calls[0]["model_id"] == "levanter"
    assert calls[0]["return_logprobs"] is True


def test_log_output_artifacts_emits_summary_and_env(tmp_path, caplog):
    bench.write_outputs(tmp_path, [_case_result("decode_b8_i1_o128_n1", "levanter", 90.0)], {"backend": "levanter"})

    with caplog.at_level("INFO", logger=bench.logger.name):
        bench.log_output_artifacts(tmp_path)

    assert "Benchmark summary.md:" in caplog.text
    assert "decode_b8_i1_o128_n1" in caplog.text
    assert "Benchmark env.json:" in caplog.text
    assert '"backend": "levanter"' in caplog.text
    assert "Benchmark artifacts.json:" in caplog.text
    assert "summary.md" in caplog.text
