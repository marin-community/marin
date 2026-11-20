# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete production InferenceEngine.generate calls at max_rounds 1/32.

CPU mode is an output-invariance control, not native performance evidence. TPU
mode requires four local TPUv4 devices and retains the accepted fixed-schedule
fixture, including its packed scheduling restrictions. No downloads or profiles
are performed. Rates describe completed finite bursts, not sustained serving.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import GenerationResult, InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.models.snowball import SnowballLMHeadModel
from snowball_zero_kl import MODEL_SEED, BenchmarkArgs, _fingerprint, _metadata, make_fixture, qualified_config

BASE_OID = "b490f651d8e7b32fb6b2c6a1ab47affdcafa3630"
SAMPLING_SEED = 44017
ROUNDS = (1, 32)
WARMUPS = 2
MIN_MEASURED = 5


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_tokens: tuple[int, ...]
    max_num_tokens: int
    stop_token: int | None = None


@dataclass(frozen=True)
class Workload:
    name: str
    requests: tuple[RequestSpec, ...]
    scope: str


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "benchmarks/snowball_zero_kl.py"]
    paths.extend(
        root / "src/levanter" / name
        for name in (
            "inference/engine.py",
            "inference/jit_scheduler.py",
            "inference/page_table.py",
            "layers/sampler.py",
            "models/snowball.py",
            "grug/attention/fixed_schedule.py",
            "kernels/pallas/fixed_schedule_linear.py",
        )
    )
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _engine_config(backend: Literal["cpu", "tpu"], max_rounds: int) -> InferenceEngineConfig:
    return InferenceEngineConfig(
        max_seq_len=256 if backend == "tpu" else 16,
        hbm_utilization=0.9,
        page_size=128 if backend == "tpu" else 4,
        max_rounds=max_rounds,
        max_stop_seqs=1,
        max_stop_tokens=1,
        seed=SAMPLING_SEED,
        max_seqs=4,
        max_pages=8 if backend == "tpu" else 16,
        compute_dtype=jnp.bfloat16,
        max_queued_tokens=4,
        max_seqs_in_prefill=4,
        max_prefill_size=508 if backend == "tpu" else 16,
        max_tokens_per_round=4,
    )


def _workloads(tokens: np.ndarray, backend: Literal["cpu", "tpu"]) -> tuple[Workload, ...]:
    prompt_length = 127 if backend == "tpu" else 3
    generated = 129 if backend == "tpu" else 6
    homogeneous = Workload(
        "homogeneous_burst",
        tuple(RequestSpec(i, tuple(map(int, tokens[i, :prompt_length])), prompt_length + generated) for i in range(4)),
        "Four requests admitted together; fixed prompt and generation lengths; LENGTH termination.",
    )
    lengths = (1, 3, 2, 4, 2, 1, 3) if backend == "cpu" else (127, 125, 126, 124, 127, 123, 122)
    additions = (3, 6, 4, 7, 5, 4, 6)
    heterogeneous = Workload(
        "heterogeneous_pending_reuse",
        tuple(
            RequestSpec(100 + i, tuple(map(int, tokens[i % 4, :length])), length + additions[i])
            for i, length in enumerate(lengths)
        ),
        "Seven requests for four slots; unequal prompts/length limits force pending admission and slot reuse.",
    )
    return homogeneous, heterogeneous


def _requests(workload: Workload) -> tuple[list[Request], list[dict[str, Any]]]:
    requests, identities = [], []
    for spec in workload.requests:
        key = jax.random.fold_in(jax.random.PRNGKey(SAMPLING_SEED), spec.request_id)
        stops = (
            None
            if spec.stop_token is None
            else hax.named(jnp.asarray([[spec.stop_token]], dtype=jnp.int32), ("stop_seq", "position"))
        )
        params = SeqDecodingParams(
            max_num_tokens=jnp.asarray(spec.max_num_tokens, dtype=jnp.int32),
            stop_tokens=stops,
            temperature=jnp.asarray(1.0, dtype=jnp.float32),
            top_p=jnp.asarray(1.0, dtype=jnp.float32),
            key=key,
        )
        requests.append(Request(list(spec.prompt_tokens), spec.request_id, params, n_generations=1))
        identities.append(
            {
                **dataclasses.asdict(spec),
                "n_generations": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "key": np.asarray(jax.device_get(key)).tolist(),
            }
        )
    jax.block_until_ready(eqx.filter(requests, eqx.is_array))
    return requests, identities


def _outputs(result: GenerationResult, requests: list[Request]) -> list[dict[str, Any]]:
    if result.logprobs is None:
        raise AssertionError("engine did not return sampled logprobs")
    rows = []
    for request, tokens, probs, reason in zip(
        requests, result.tokens, result.logprobs, result.finish_reasons, strict=True
    ):
        if reason not in (FinishReason.LENGTH, FinishReason.STOP):
            raise AssertionError(f"incomplete request {request.request_id}: {reason}")
        if not tokens or len(tokens) != len(probs) or not np.isfinite(probs).all():
            raise AssertionError(f"invalid generated tokens/logprobs for request {request.request_id}")
        rows.append(
            {
                "request_id": request.request_id,
                "tokens": tokens,
                "sampled_logprobs": probs,
                "sampled_logprobs_fp32_hex": np.asarray(probs, dtype=np.float32).tobytes().hex(),
                "finish_reason": reason.name,
            }
        )
    if result.total_generated != sum(len(row["tokens"]) for row in rows):
        raise AssertionError("engine generated count disagrees with returned output")
    return rows


def _comparison(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for a, b in zip(left, right, strict=True):
        comparisons.append(
            {
                "request_id": a["request_id"],
                "request_identity_equal": a["request_id"] == b["request_id"],
                "tokens_equal": a["tokens"] == b["tokens"],
                "finish_reason_equal": a["finish_reason"] == b["finish_reason"],
                "sampled_logprobs_bitwise_equal": a["sampled_logprobs_fp32_hex"] == b["sampled_logprobs_fp32_hex"],
            }
        )
    exact = all(all(value for key, value in row.items() if key != "request_id") for row in comparisons)
    return {"exact": exact, "requests": comparisons}


def _measure_call(engine: InferenceEngine, requests: list[Request], call: dict[str, Any]) -> None:
    started = time.perf_counter()
    try:
        result = engine.generate(requests)
        # generate() extracts host outputs itself; include all remaining device
        # work as well, including the final free-pages/reset bookkeeping.
        jax.block_until_ready(eqx.filter(engine.gen_state, eqx.is_array))
    except Exception:
        call["status"] = "failed"
        call["elapsed_until_failure"] = time.perf_counter() - started
        call["partial_outputs"] = [
            {
                "request_id": rid,
                "choice": choice,
                "tokens": value.token_list,
                "sampled_logprobs_fp32_hex": np.asarray(value.logprobs, dtype=np.float32).tobytes().hex(),
                "finish_reason": value.finish_reason.name,
            }
            for rid, choices in engine.results.items()
            for choice, value in choices.items()
        ]
        raise
    call["completed_batch_latency"] = time.perf_counter() - started
    # All validation, comparisons and hashes occur after the timed boundary.
    call["generated_tokens"] = result.total_generated
    try:
        call["outputs"] = _outputs(result, requests)
    except Exception:
        call["status"] = "output_validation_failed"
        call["returned_tokens"] = result.tokens
        call["returned_logprobs_fp32_hex"] = (
            None
            if result.logprobs is None
            else [np.asarray(row, dtype=np.float32).tobytes().hex() for row in result.logprobs]
        )
        call["returned_finish_reasons"] = [reason.name for reason in result.finish_reasons]
        raise
    call["input_tokens"] = sum(len(request.prompt_tokens) for request in requests)
    call["generated_tokens_per_second"] = result.total_generated / call["completed_batch_latency"]
    call["status"] = "complete"


def _benchmark_workload(
    report: dict[str, Any], workload: Workload, engines: dict[int, InferenceEngine], measured: int
) -> dict:
    requests, identities = _requests(workload)
    row: dict[str, Any] = {
        "name": workload.name,
        "scope": workload.scope,
        "requests": identities,
        "request_config_sha256": _json_hash(identities),
        "calls": [],
        "arms": {},
    }
    report["workloads"].append(row)
    for index in range(1 + WARMUPS + measured):
        phase = "first_call" if index == 0 else "warmup" if index <= WARMUPS else "measured"
        # AB / BA pairs prevent all reference calls preceding treatment calls.
        order = ROUNDS if index % 2 == 0 else tuple(reversed(ROUNDS))
        for rounds in order:
            call = {
                "order": len(row["calls"]),
                "phase": phase,
                "repetition": index,
                "max_rounds": rounds,
                "status": "started",
            }
            row["calls"].append(call)
            report["progress"] = {
                "workload": workload.name,
                **{key: call[key] for key in ("phase", "repetition", "max_rounds")},
            }
            _measure_call(engines[rounds], requests, call)
    reference = row["calls"][0]["outputs"]
    for call in row["calls"]:
        call["comparison_to_max_rounds1_first_call"] = _comparison(reference, call["outputs"])
    row["output_invariance_exact"] = all(
        call["comparison_to_max_rounds1_first_call"]["exact"] for call in row["calls"]
    )
    for rounds in ROUNDS:
        calls = [call for call in row["calls"] if call["max_rounds"] == rounds]
        steady = [call for call in calls if call["phase"] == "measured"]
        latencies = [call["completed_batch_latency"] for call in steady]
        row["arms"][str(rounds)] = {
            "first_call_latency": calls[0]["completed_batch_latency"],
            "first_call_compile_note": "Includes compilation needed at this point; earlier workloads and persistent caches may reuse compilations.",
            "warmups": WARMUPS,
            "measured": measured,
            "raw_measured_latencies": latencies,
            "mean_completed_batch_latency": float(np.mean(latencies)),
            "median_completed_batch_latency": float(np.median(latencies)),
            "aggregate_generated_tokens_per_second": sum(c["generated_tokens"] for c in steady) / sum(latencies),
            "generated_tokens_per_call": [c["generated_tokens"] for c in steady],
            "input_tokens_per_call": [c["input_tokens"] for c in steady],
        }
    if not row["output_invariance_exact"]:
        raise AssertionError(f"max_rounds changed actual outputs in {workload.name}")
    return row


def run(report: dict[str, Any], *, backend: Literal["cpu", "tpu"], measured: int, source_sha: str | None) -> None:
    """Run real engine bursts, updating the report after each actual call."""
    if measured < MIN_MEASURED:
        raise ValueError(f"at least {MIN_MEASURED} measured calls per arm are required")
    report.update(workloads=[], progress={"phase": "runtime_verification"}, requested_backend=backend)
    actual_backend = jax.default_backend()
    report["actual_backend"] = actual_backend
    if backend != actual_backend:
        raise RuntimeError(f"requested {backend}, actual backend {actual_backend}; fallback is forbidden")
    if backend == "tpu" and (
        jax.device_count() != 4
        or jax.local_device_count() != 4
        or jax.process_count() != 1
        or any("tpu v4" not in device.device_kind.lower() for device in jax.devices())
    ):
        raise RuntimeError("TPU mode requires exactly four local TPUv4 devices and one process")
    config = qualified_config(implementation="fixed_schedule" if backend == "tpu" else "native", backend=backend)
    if backend == "cpu":
        config = dataclasses.replace(
            config, hidden_dim=16, intermediate_dim=16, shared_expert_intermediate_dim=16, head_dim=8, max_seq_len=16
        )
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        metadata = _metadata(
            BenchmarkArgs(backend=backend, warmup=WARMUPS, measured=measured, source_sha=source_sha),
            backend=backend,
            mesh=mesh,
        )
        metadata.update(
            schema="snowball_production_engine_performance.v1",
            base_oid=BASE_OID,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fixture=(
                "accepted-four-slot256token-fixed-schedule" if backend == "tpu" else "CPU-reference-four-slot-H16-D8"
            ),
            timing="End-to-end generate, including reset, host extraction and synchronization; construction, hashing and parity excluded. Engine donation is unchanged between arms.",
            accuracy_note="Exact ordinary-engine sampled tokens/logprobs/finish reasons across max_rounds, not full-vocabulary training parity.",
            source_sha_origin="caller" if source_sha is not None else "git_head",
        )
        report["provenance"] = metadata
        report["provenance"].update(sampling_seed=SAMPLING_SEED, source_file_sha256=_source_hashes())
        report["model_config"] = dataclasses.asdict(config)
        report["engine_configs"] = {str(n): dataclasses.asdict(_engine_config(backend, n)) for n in ROUNDS}
        report["config_sha256"] = _json_hash({"model": report["model_config"], "engines": report["engine_configs"]})
        report["scope"] = {
            "metric": "finite completed batch latency and generated tokens per second; not sustained server throughput",
            "decode_iteration_metrics": "not collected; complete generate calls contain prefill and multiple host/device rounds",
            "distribution": "public Sampler, temperature=1, top_p=1, identical per-request keys in both arms; sampled logprobs compared bitwise",
            "eos": "Explicit per-request single-token STOP sequences; no tokenizer-specific EOS claim.",
            "native": "CPU is only an invariance control; TPU performance requires parent native review.",
            "packed_scheduling": "TPU retains fixed-schedule selectors and fails on unsupported packed layouts or nonfinite output; no fallback.",
        }
        report["progress"] = {"phase": "model_initialization"}
        fixture = make_fixture(
            sequence_length=256 if backend == "tpu" else 8, page_size=128 if backend == "tpu" else 4
        )
        model = SnowballLMHeadModel.init(hax.Axis("vocab", config.vocab_size), config, key=jax.random.key(MODEL_SEED))
        jax.block_until_ready(eqx.filter(model, eqx.is_array))
        report["provenance"]["model_weights_sha256"] = _fingerprint(model)
        report["provenance"]["fixture_tokens_sha256"] = _fingerprint(fixture.tokens)
        engines = {}
        report["engine_initialization_latency"] = {}
        for n in ROUNDS:
            started = time.perf_counter()
            engines[n] = InferenceEngine.from_model_with_config(model, None, _engine_config(backend, n))
            jax.block_until_ready(eqx.filter(engines[n].gen_state, eqx.is_array))
            report["engine_initialization_latency"][str(n)] = time.perf_counter() - started
        workloads = _workloads(np.asarray(fixture.tokens.array), backend)
        homogeneous = _benchmark_workload(report, workloads[0], engines, measured)
        _benchmark_workload(report, workloads[1], engines, measured)
        observed = homogeneous["calls"][0]["outputs"]
        stops = Workload(
            "seeded_first_token_stop",
            tuple(
                dataclasses.replace(spec, stop_token=output["tokens"][0])
                for spec, output in zip(workloads[0].requests, observed, strict=True)
            ),
            "Single-token STOP per request is its actual seeded first output in homogeneous_burst, derived outside timing; public sampler unchanged.",
        )
        stopped = _benchmark_workload(report, stops, engines, measured)
        if any(
            output["finish_reason"] != "STOP" or len(output["tokens"]) != 1
            for call in stopped["calls"]
            for output in call["outputs"]
        ):
            raise AssertionError("seeded STOP workload did not stop on its actual first sampled token")
    report["progress"] = {"phase": "complete"}
    report["status"] = "passed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--measured", type=int, default=MIN_MEASURED)
    parser.add_argument("--source-sha", help="Source commit for exported trees; otherwise read current Git HEAD.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"schema": "snowball_production_engine_performance.v1", "status": "started"}
    code = 0
    try:
        run(report, backend=args.backend, measured=args.measured, source_sha=args.source_sha)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        code = 1
    args.output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "progress": report.get("progress")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
