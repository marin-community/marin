# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B complete sampled generation against the immutable accepted host loop.

Supply the original API blob from BASE_OID as --reference-api. Both arms use
one model and the same arithmetic, inputs, sampling key, and physical page maps.
The production InferenceEngine already loops on device. This benchmark measures
full diagnostic route/logit retention, not production engine throughput. Its
fixed-slot RNG and capture contract differ from DecodeState and _DecodeOutputs.
CPU mode exercises instrumentation; only parent-run TPUv4 evidence qualifies
native parity or performance. Profiles are optional, local, and outside timing.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference import snowball_rollout_replay as candidate
from levanter.models.snowball import SnowballLMHeadModel
from snowball_rollout_replay import SAMPLING_SEED, _sampled_log_probs
from snowball_zero_kl import (
    MODEL_SEED,
    TOKEN_SEED,
    BenchmarkArgs,
    _array_metrics,
    _fingerprint,
    _metadata,
    _selector_metadata,
    make_fixture,
    qualified_config,
)

BASE_OID = "1b3d5e5cb02c9306657f8aae17b76979c7b4bbec"
REFERENCE_API_SHA256 = "2f10d17feaa01ae98090b31066814fe157ea6a5cdc6a80e2d073163720894f93"


def load_reference(path: Path, expected_sha256: str) -> ModuleType:
    """Load only the original accepted API, never a caller-selected substitute."""
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if expected_sha256 != REFERENCE_API_SHA256 or actual != REFERENCE_API_SHA256:
        raise ValueError(f"reference API is not the accepted {BASE_OID} blob: {actual}")
    name = "snowball_accepted_generation_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference API: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def evidence(generated: candidate.SnowballGeneratedRoutes, *, physical_layout: bool = True) -> dict:
    """Select the complete public payload, optionally excluding physical addresses."""
    captures = generated.serving_captures
    if not physical_layout:
        captures = tuple((c.route_captures, c.token_ids, c.position_ids, c.token_valid) for c in captures)
    return {
        "tokens": generated.tokens,
        "sampled_token_ids": generated.sampled_token_ids,
        "serving_logits": generated.serving_logits,
        "serving_captures": captures,
        "teacher_capture": generated.teacher_capture,
        "predictor_positions": generated.predictor_positions,
    }


def bit_comparison(left, right) -> dict:
    """Compare every array byte and the pytree's static layout, including axes."""
    a, a_tree = jax.tree.flatten(eqx.filter(left, eqx.is_array))
    b, b_tree = jax.tree.flatten(eqx.filter(right, eqx.is_array))
    if a_tree != b_tree:
        raise ValueError("evidence structures differ")
    different_bits = elements = 0
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("evidence shapes or dtypes differ")
        xor = np.bitwise_xor(np.frombuffer(x.tobytes(), np.uint8), np.frombuffer(y.tobytes(), np.uint8))
        different_bits += int(np.unpackbits(xor).sum())
        elements += x.size
    return {"bitwise": different_bits == 0, "different_bits": different_bits, "elements": elements, "leaves": len(a)}


def _complete(fn: Callable[[], candidate.SnowballGeneratedRoutes]) -> candidate.SnowballGeneratedRoutes:
    generated = fn()
    jax.block_until_ready(evidence(generated))
    return generated


def measure(
    fn: Callable[[], candidate.SnowballGeneratedRoutes], *, warmup: int, measured: int
) -> tuple[dict, candidate.SnowballGeneratedRoutes]:
    """Include cache initialization, sampling and completed public evidence."""
    started = time.perf_counter()
    first = _complete(fn)
    first_latency = time.perf_counter() - started
    results = [first]
    for _ in range(warmup):
        results.append(_complete(fn))
    samples = []
    for _ in range(measured):
        started = time.perf_counter()
        result = _complete(fn)
        samples.append(time.perf_counter() - started)
        results.append(result)
    # No diagnostic hashing or parity work inside any measured interval.
    digest = _fingerprint(evidence(first))
    if any(_fingerprint(evidence(result)) != digest for result in results[1:]):
        raise AssertionError("same-seed generation changed between timing samples")
    return {
        "first_call_including_compile": first_latency,
        "steady_samples": samples,
        "steady_mean": float(np.mean(samples)),
        "steady_median": float(np.median(samples)),
        "steady_min": min(samples),
        "steady_max": max(samples),
        "warmups": warmup,
        "measured": measured,
        "total_generations": 1 + warmup + measured,
    }, results[-1]


def compare_generated(reference, current, reference_teacher, current_teacher, *, physical_layout: bool) -> dict:
    reference_probs = np.asarray(
        _sampled_log_probs(jnp.asarray(reference.prediction_logits), jnp.asarray(reference.sampled_token_ids))
    )
    current_probs = np.asarray(
        _sampled_log_probs(jnp.asarray(current.prediction_logits), jnp.asarray(current.sampled_token_ids))
    )
    reference_teacher_probs = np.asarray(
        _sampled_log_probs(
            jnp.asarray(reference_teacher[:, reference.predictor_positions]), jnp.asarray(reference.sampled_token_ids)
        )
    )
    current_teacher_probs = np.asarray(
        _sampled_log_probs(
            jnp.asarray(current_teacher[:, current.predictor_positions]), jnp.asarray(current.sampled_token_ids)
        )
    )
    payload = evidence(reference, physical_layout=physical_layout)
    other = evidence(current, physical_layout=physical_layout)
    metrics = {name: bit_comparison(value, other[name]) for name, value in payload.items()}
    metrics.update(
        teacher_logits=bit_comparison(reference_teacher, current_teacher),
        sampled_log_probs=bit_comparison(reference_probs, current_probs),
        teacher_sampled_log_probs=bit_comparison(reference_teacher_probs, current_teacher_probs),
    )
    return {"bitwise": all(value["bitwise"] for value in metrics.values()), "features": metrics}


def profile_arm(fn: Callable, directory: Path, *, generations: int, max_bytes: int) -> dict:
    """Capture a separate bounded steady window; never upload or dump HLO."""
    if directory.exists():
        raise FileExistsError(f"profile directory must be new: {directory}")
    config = ProfilerConfig(
        enabled=True,
        start_step=0,
        num_steps=generations,
        profile_options=ProfileOptionsConfig(host_tracer_level=1, python_tracer_level=0, enable_hlo_proto=False),
        upload=XprofUploadConfig(enabled=False),
    )
    started = time.time()
    with jax.profiler.trace(
        str(directory),
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=config.build_jax_profile_options(),
    ):
        for step in range(generations):
            with jax.profiler.StepTraceAnnotation("generation", step_num=step):
                _complete(fn)
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": digest})
    total = sum(row["bytes"] for row in files)
    return {
        "status": "captured" if total <= max_bytes else "exceeds_parent_cap",
        "directory": str(directory),
        "started_unix": started,
        "finished_unix": time.time(),
        "generations": generations,
        "config": dataclasses.asdict(config),
        "files": files,
        "bytes": total,
        "max_bytes": max_bytes,
        "xplane_files": sum(row["path"].endswith(".xplane.pb") for row in files),
        "timed_comparison": False,
        "upload": False,
    }


def run(
    report: dict,
    *,
    backend: str,
    reference_api: Path,
    reference_sha256: str,
    source_sha: str | None,
    warmup: int = 2,
    measured: int = 8,
    profile_dir: Path | None = None,
    profile_generations: int = 1,
    profile_max_bytes: int | None = None,
) -> None:
    reference = load_reference(reference_api, reference_sha256)
    if warmup < 0 or measured <= 0:
        raise ValueError("warmup must be nonnegative and measured must be positive")
    if profile_generations not in (1, 2) or (
        profile_dir is not None and (profile_max_bytes is None or profile_max_bytes <= 0)
    ):
        raise ValueError("profiles require 1-2 generations per arm and an explicit positive parent byte cap")
    if backend != jax.default_backend():
        raise RuntimeError(f"requested {backend} but runtime is {jax.default_backend()}")
    if backend == "tpu" and (
        jax.device_count() != 4
        or jax.local_device_count() != 4
        or jax.process_count() != 1
        or any("tpu v4" not in d.device_kind.lower() for d in jax.devices())
    ):
        raise RuntimeError("native qualification requires exactly four local TPUv4 devices")
    total, prompt_length, page_size = (256, 127, 128) if backend == "tpu" else (8, 3, 4)
    config = qualified_config(implementation="fixed_schedule" if backend == "tpu" else "native", backend=backend)
    if backend == "cpu":
        config = dataclasses.replace(
            config,
            max_seq_len=total,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=2,
        )
    fixture = make_fixture(sequence_length=total, page_size=page_size)
    mesh = compact_grug_mesh(expert_axis_size=1)
    report.update(
        config=dataclasses.asdict(config),
        config_sha256=hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True).encode()).hexdigest(),
        selectors=_selector_metadata(config),
        page_maps=[m.tolist() for m in fixture.page_maps],
    )
    report["metadata"] = _metadata(
        BenchmarkArgs(source_sha=source_sha, warmup=warmup, measured=measured), backend=backend, mesh=mesh
    )
    report["metadata"].update(
        schema="snowball_rollout_performance.v1",
        base_oid=BASE_OID,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        api_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        reference_api_sha256=reference_sha256,
        reference_api_base=BASE_OID,
        fixture=(
            "B4/S256/P127/oneNoPE/E2/top1/H128/I32/Q2/KV1/D128/V23/page128"
            if backend == "tpu"
            else "CPU-reference-B4/S8/P3/two-layers/E4/top2/H16/I16/Q2/KV1/D8/V23/page4"
        ),
        prompt_length=prompt_length,
        generated_tokens=total - prompt_length,
        sampling_seed=SAMPLING_SEED,
        model_seed=MODEL_SEED,
        token_seed=TOKEN_SEED,
        sampling_rule="temperature-one categorical; fold_in(sampling_key, continuation_index)",
        timing="seconds; complete API + cache initialization + synchronized evidence; first call includes uncached compilation; no donation; no profiler",
        arm_order=["reference", "candidate"],
        scope="complete fixed-slot route/logit-retention diagnostic; not production InferenceEngine throughput",
    )
    results, teachers, functions = {}, {}, {}
    with (
        jax.set_mesh(mesh),
        hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}),
        jax.default_matmul_precision("highest"),
    ):
        report["metadata"].update(
            default_matmul_precision=str(jax.config.jax_default_matmul_precision),
            mesh_axis_types=[str(t) for t in mesh.axis_types],
        )
        model = SnowballLMHeadModel.init(hax.Axis("vocab", config.vocab_size), config, key=jax.random.key(MODEL_SEED))
        report["parameter_sha256_before"] = _fingerprint(model)
        prompts = hax.named(jnp.asarray(np.asarray(fixture.tokens.array)[:, :prompt_length]), ("batch", "position"))
        report["prompt_sha256"] = _fingerprint(prompts)
        jax.block_until_ready((model, prompts))
        for arm, api in (("reference", reference), ("candidate", candidate)):
            results[arm], teachers[arm] = [], []
            for index, page_map in enumerate(fixture.page_maps):

                def generation(api=api, page_map=page_map):
                    return api.generate_with_route_captures(
                        model,
                        prompts,
                        sampling_key=jax.random.key(SAMPLING_SEED),
                        generated_tokens=total - prompt_length,
                        page_map=page_map,
                        page_size=page_size,
                        cache_dtype=jnp.bfloat16,
                    )

                functions[arm] = generation
                row = {"arm": arm, "page_map_index": index, "status": "running"}
                report["runs"].append(row)
                timing, result = measure(generation, warmup=warmup, measured=measured)
                teacher = api.replay_generated_routes(model, result)
                if not np.isfinite(result.serving_logits).all() or not np.isfinite(teacher).all():
                    raise FloatingPointError("non-finite serving or teacher logits")
                row.update(
                    status="success",
                    timing=timing,
                    digests={name: _fingerprint(value) for name, value in evidence(result).items()},
                    teacher_sha256=_fingerprint(teacher),
                    capture_provenance={
                        "producer": "SnowballLMHeadModel.decode_with_route_captures",
                        "teacher_recapture": False,
                        "serving_calls": len(result.serving_captures),
                        "captured_token_rows_per_layer": 4 * total,
                        "layers": config.num_layers,
                        "top_k": config.num_experts_per_token,
                        "per_call_sha256": [_fingerprint(c) for c in result.serving_captures],
                    },
                    serving_to_teacher=_array_metrics(result.serving_logits, teacher),
                    aggregate_counts={
                        "sampled_tokens_per_generation": 4 * (total - prompt_length),
                        "serving_token_rows_per_generation": 4 * total,
                        "serving_calls_per_generation": total - prompt_length + 1,
                        "measured_sampled_tokens": measured * 4 * (total - prompt_length),
                        "measured_serving_token_rows": measured * 4 * total,
                    },
                )
                results[arm].append(result)
                teachers[arm].append(teacher)
        report["parity"] = {
            "reference_to_candidate": [
                compare_generated(
                    results["reference"][i],
                    results["candidate"][i],
                    teachers["reference"][i],
                    teachers["candidate"][i],
                    physical_layout=True,
                )
                for i in range(2)
            ],
            "cross_page_maps": {
                arm: compare_generated(
                    results[arm][0], results[arm][1], teachers[arm][0], teachers[arm][1], physical_layout=False
                )
                for arm in results
            },
        }
        exact = all(m["bitwise"] for m in report["parity"]["reference_to_candidate"]) and all(
            m["bitwise"] for m in report["parity"]["cross_page_maps"].values()
        )
        report["parity"]["bitwise"] = exact
        report["parameter_sha256_after"] = _fingerprint(model)
        if report["parameter_sha256_before"] != report["parameter_sha256_after"]:
            raise AssertionError("generation changed model parameters")
        if not exact:
            raise AssertionError("generation differs from accepted API or across physical page maps")
        if backend == "tpu" and not all(r["serving_to_teacher"]["bitwise_fp32"] for r in report["runs"]):
            raise AssertionError("native serving-to-teacher bits differ")
        report["comparison"] = [
            {
                "page_map_index": i,
                "reference_mean": report["runs"][i]["timing"]["steady_mean"],
                "candidate_mean": report["runs"][i + 2]["timing"]["steady_mean"],
                "reference_over_candidate": report["runs"][i]["timing"]["steady_mean"]
                / report["runs"][i + 2]["timing"]["steady_mean"],
            }
            for i in range(2)
        ]
        if profile_dir is not None:
            assert profile_max_bytes is not None
            report["profiles"] = {"page_map_index": 1, "arms": {}}
            for arm, fn in functions.items():
                report["profiles"]["arms"][arm] = profile_arm(
                    fn, profile_dir / arm, generations=profile_generations, max_bytes=profile_max_bytes
                )
        else:
            report["profiles"] = {
                "status": "not_requested",
                "host_device_attribution": "hypothesis; native profile pending parent",
            }
    report.update(
        status="success",
        native_qualification={
            "status": "observed" if backend == "tpu" else "unexecuted",
            "acceptance": "parent-owned",
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--reference-api", type=Path, required=True)
    parser.add_argument("--reference-sha256", default=REFERENCE_API_SHA256)
    parser.add_argument("--source-sha")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-generations", type=int, default=1)
    parser.add_argument("--profile-max-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"status": "running", "runs": [], "native_qualification": {"status": "unexecuted"}}
    try:
        run(
            report,
            backend=args.backend,
            reference_api=args.reference_api,
            reference_sha256=args.reference_sha256,
            source_sha=args.source_sha,
            warmup=args.warmup,
            measured=args.measured,
            profile_dir=args.profile_dir,
            profile_generations=args.profile_generations,
            profile_max_bytes=args.profile_max_bytes,
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
