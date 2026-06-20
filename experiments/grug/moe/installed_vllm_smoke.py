# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small installed-vLLM serving smoke for GrugMoE artifacts."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata as md
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.grug.moe.model import GRUG_MOE_ARTIFACT_SCHEMA_VERSION, GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY

_PROMPT_IDS = [1, 42, 128, 2048, 17, 3072, 5, 63]
_REFERENCE_JSON_NAME = "installed_vllm_reference.json"
_DEFAULT_CONFIG_NAME = "small-diagnostic"
_CONFIG_NAMES = ("small-diagnostic", "scaled", "canary")
_SMALL_MAX_SHARD_SIZE_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_SHARD_SIZE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_LOGPROB_DELTA = 5.0
_DEFAULT_VLLM_DTYPE = "bfloat16"
_VLLM_DTYPE_CHOICES = ("bfloat16", "float32")


def _direct_url(package: str) -> str:
    direct_url = md.distribution(package).read_text("direct_url.json")
    return direct_url.strip() if direct_url else ""


def _print_runtime_header() -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")

    print("remote_cwd=" + os.getcwd())
    try:
        marin_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError) as exc:  # pragma: no cover - diagnostic only
        marin_sha = f"unavailable:{exc!r}"
    print("marin_sha=" + marin_sha)
    for package in ("vllm", "tpu-inference"):
        print(f"{package}_direct_url=" + _direct_url(package))
    print("vllm_version=" + md.version("vllm"))
    print("tpu-inference_version=" + md.version("tpu-inference"))
    print("jax_version=" + md.version("jax"))
    print("libtpu_version=" + md.version("libtpu"))
    print("grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))


def _artifact_dir(output_dir: Path) -> Path:
    return output_dir / "grugmoe-inference"


def _reference_path(output_dir: Path) -> Path:
    return output_dir / _REFERENCE_JSON_NAME


def _default_max_shard_size(config_name: str) -> int:
    if config_name == "small-diagnostic":
        return _SMALL_MAX_SHARD_SIZE_BYTES
    return _DEFAULT_MAX_SHARD_SIZE_BYTES


def _artifact_config(artifact_dir: Path) -> dict[str, Any]:
    with (artifact_dir / "config.json").open() as f:
        config = json.load(f)
    if int(config.get("vocab_size", 0)) <= 0:
        raise AssertionError(f"invalid artifact vocab_size={config.get('vocab_size')!r}")
    if config.get(GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY) != GRUG_MOE_ARTIFACT_SCHEMA_VERSION:
        raise AssertionError(
            "invalid artifact "
            f"{GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY}={config.get(GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY)!r}"
        )
    return config


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _shard_count(path: Path) -> int:
    return len(list(path.glob("*.safetensors")))


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["VLLM_TARGET_DEVICE"] = "tpu"
    env["MODEL_IMPL_TYPE"] = "auto"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _configure_vllm_env() -> None:
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    # Marin's generic vLLM helper defaults MODEL_IMPL_TYPE to "vllm".
    # GrugMoE is exposed by tpu-inference's JAX registry, so force auto.
    os.environ["MODEL_IMPL_TYPE"] = "auto"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _server_base_url(server_url: str) -> str:
    return server_url.removesuffix("/v1")


def _parse_token_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _reference_payload(reference_path: Path | None) -> dict[str, Any] | None:
    if reference_path is None or not reference_path.exists():
        return None
    with reference_path.open() as f:
        return json.load(f)


def _expected_token_ids(reference_path: Path | None, explicit_ids: list[int] | None) -> list[int] | None:
    if explicit_ids is not None:
        return explicit_ids
    reference = _reference_payload(reference_path)
    if reference is None:
        return None
    continuation_ids = reference.get("continuation_ids")
    if continuation_ids is None:
        return None
    return [int(token_id) for token_id in continuation_ids]


def _reference_logprobs(reference_path: Path | None) -> np.ndarray | None:
    reference = _reference_payload(reference_path)
    if reference is None:
        return None
    logprobs = reference.get("levanter_continuation_logprobs")
    if logprobs is None:
        return None
    return np.asarray(logprobs, dtype=np.float64)


def export_artifact(
    output_dir: Path,
    *,
    config_name: str,
    max_shard_size: int,
    generation_tokens: int,
) -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")

    import tpu_inference.models.jax.grugmoe as tpu_grugmoe  # noqa: PLC0415

    from experiments.grug.moe import vllm_tpu_parity as parity  # noqa: PLC0415

    print("artifact_generation_process=started")
    print("artifact_config_name=" + config_name)
    parity.check_realistic_training_state_roundtrip(
        tpu_grugmoe,
        config_name=config_name,
        output_dir=output_dir,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        reference_output_path=_reference_path(output_dir),
    )
    print("artifact_generation_process=completed")


def _decode_routed_experts(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    with io.BytesIO(raw) as buf:
        return np.load(buf, allow_pickle=False)


def _assert_routed_experts(choice: dict[str, Any], artifact_config: dict[str, Any]) -> list[int]:
    encoded = choice.get("routed_experts")
    if encoded is None:
        raise AssertionError("installed vLLM did not return routed_experts with the feature enabled")
    routed_experts = _decode_routed_experts(encoded)
    if routed_experts.ndim != 3:
        raise AssertionError(f"routed_experts ndim {routed_experts.ndim} != 3")
    if 0 in routed_experts.shape:
        raise AssertionError(f"routed_experts has an empty dimension: {routed_experts.shape}")
    expected_layers = int(artifact_config["num_layers"])
    expected_topk = int(artifact_config["num_experts_per_token"])
    if routed_experts.shape[1] != expected_layers:
        raise AssertionError(f"routed_experts layers {routed_experts.shape[1]} != {expected_layers}")
    if routed_experts.shape[2] != expected_topk:
        raise AssertionError(f"routed_experts top-k {routed_experts.shape[2]} != {expected_topk}")
    num_experts = int(artifact_config["num_experts"])
    if not np.all((routed_experts >= 0) & (routed_experts < num_experts)):
        raise AssertionError(f"routed_experts contained IDs outside [0, {num_experts})")
    return list(routed_experts.shape)


def _assert_no_routed_experts(choice: dict[str, Any]) -> None:
    if choice.get("routed_experts") is not None:
        raise AssertionError("installed vLLM returned routed_experts while the feature was disabled")


def serve_artifact(
    artifact_dir: Path,
    *,
    reference_path: Path | None,
    prompt_ids: list[int],
    expected_generated_token_ids: list[int] | None,
    check_reference_continuation: bool,
    check_reference_logprobs: bool,
    max_logprob_delta: float,
    return_routed_experts: bool,
    vllm_dtype: str,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
) -> None:
    if os.environ.get("PYTHONPATH"):
        raise SystemExit(f"PYTHONPATH unexpectedly set: {os.environ['PYTHONPATH']}")
    _configure_vllm_env()

    import requests  # noqa: PLC0415
    from marin.evaluation.evaluators.evaluator import ModelConfig  # noqa: PLC0415
    from marin.inference.vllm_server import VllmEnvironment  # noqa: PLC0415

    artifact_config = _artifact_config(artifact_dir)
    max_model_len = max(16, len(prompt_ids) + generation_tokens)
    print("serve_artifact_dir=" + str(artifact_dir))
    print("serve_vllm_dtype=" + vllm_dtype)
    print("serve_return_routed_experts=" + str(return_routed_experts))
    for package in ("vllm", "tpu-inference"):
        print(f"serve_{package}_direct_url=" + _direct_url(package))
    print("serve_grugmoe_spec=" + repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe")))

    model = ModelConfig(
        name="grugmoe-installed-vllm-smoke",
        path=str(artifact_dir),
        engine_kwargs={
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
        },
    )
    extra_args = [
        "--runner",
        "generate",
        "--dtype",
        vllm_dtype,
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--skip-tokenizer-init",
        "--tokens-only",
    ]
    if return_routed_experts:
        extra_args.append("--enable-return-routed-experts")

    with VllmEnvironment(
        model=model,
        port=server_port,
        timeout_seconds=server_timeout_seconds,
        extra_args=extra_args,
    ) as env:
        print("vllm_server_initialized=True")
        print("vllm_server_log_dir=" + (env.vllm_server.log_dir if env.vllm_server else ""))
        generate_url = _server_base_url(env.server_url) + "/inference/v1/generate"
        response = requests.post(
            generate_url,
            json={
                "model": env.model_id,
                "token_ids": prompt_ids,
                "sampling_params": {
                    "max_tokens": generation_tokens,
                    "temperature": 0.0,
                    **({"logprobs": 1} if check_reference_logprobs else {}),
                },
                "stream": False,
            },
            timeout=300,
        )
        print("vllm_generate_status_code=" + str(response.status_code))
        if not response.ok:
            print("vllm_generate_response_text=" + response.text[:4000])
            print("vllm_server_logs_tail_begin")
            print(env.logs_tail(max_lines=400))
            print("vllm_server_logs_tail_end")
            response.raise_for_status()
        payload = response.json()
        print("vllm_server_logs_tail_begin")
        print(env.logs_tail(max_lines=120))
        print("vllm_server_logs_tail_end")

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"expected exactly one generate choice, got {payload!r}")
    choice = choices[0]
    generated_ids = [int(token_id) for token_id in choice.get("token_ids", [])]
    if len(generated_ids) != generation_tokens:
        raise AssertionError(f"generated {len(generated_ids)} tokens, expected {generation_tokens}: {generated_ids}")
    vocab_size = int(artifact_config["vocab_size"])
    if any(token_id < 0 or token_id >= vocab_size for token_id in generated_ids):
        raise AssertionError(f"generated token IDs outside [0, {vocab_size}): {generated_ids}")

    expected_ids = _expected_token_ids(
        reference_path if check_reference_continuation or check_reference_logprobs else None,
        expected_generated_token_ids,
    )
    if (check_reference_continuation or check_reference_logprobs) and expected_ids is None:
        raise AssertionError(f"reference continuation was requested but unavailable: {reference_path}")
    if expected_ids is not None and generated_ids != expected_ids:
        raise AssertionError(f"generated token IDs {generated_ids} != expected {expected_ids}")

    logprob_summary = None
    if check_reference_logprobs:
        expected_logprobs = _reference_logprobs(reference_path)
        if expected_logprobs is None:
            raise AssertionError(f"reference logprobs were requested but unavailable: {reference_path}")
        content = ((choice.get("logprobs") or {}).get("content")) or []
        actual_logprobs = np.asarray([float(item["logprob"]) for item in content], dtype=np.float64)
        if actual_logprobs.shape != expected_logprobs.shape:
            raise AssertionError(
                f"generated logprobs shape {actual_logprobs.shape} != reference {expected_logprobs.shape}"
            )
        deltas = np.abs(actual_logprobs - expected_logprobs)
        logprob_summary = {
            "count": int(deltas.size),
            "max_abs_delta": float(np.max(deltas)) if deltas.size else 0.0,
            "mean_abs_delta": float(np.mean(deltas)) if deltas.size else 0.0,
            "max_allowed_abs_delta": float(max_logprob_delta),
        }
        if logprob_summary["max_abs_delta"] > max_logprob_delta:
            raise AssertionError(f"selected-token logprob delta exceeded tolerance: {logprob_summary}")

    routed_shape = None
    if return_routed_experts:
        routed_shape = _assert_routed_experts(choice, artifact_config)
    else:
        _assert_no_routed_experts(choice)

    summary = {
        "artifact_dir": str(artifact_dir),
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "expected_generated_ids": expected_ids,
        "logprob_summary": logprob_summary,
        "return_routed_experts": return_routed_experts,
        "routed_experts_shape": routed_shape,
    }
    print("installed_vllm_smoke=" + json.dumps(summary, sort_keys=True))


def _run_phase_subprocess(
    phase: str,
    *,
    output_dir: Path,
    config_name: str,
    artifact_dir: Path | None,
    reference_path: Path | None,
    max_shard_size: int,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
    vllm_dtype: str,
    return_routed_experts: bool,
    check_reference_continuation: bool,
    check_reference_logprobs: bool,
    max_logprob_delta: float,
    expected_generated_token_ids: list[int] | None,
) -> None:
    args = [
        sys.executable,
        "-m",
        "experiments.grug.moe.installed_vllm_smoke",
        "--phase",
        phase,
        "--output-dir",
        str(output_dir),
        "--config-name",
        config_name,
        "--max-shard-size",
        str(max_shard_size),
        "--generation-tokens",
        str(generation_tokens),
        "--server-port",
        str(server_port),
        "--server-timeout-seconds",
        str(server_timeout_seconds),
        "--vllm-dtype",
        vllm_dtype,
    ]
    if artifact_dir is not None:
        args.extend(["--artifact-dir", str(artifact_dir)])
    if reference_path is not None:
        args.extend(["--reference-path", str(reference_path)])
    if return_routed_experts:
        args.append("--return-routed-experts")
    if check_reference_continuation:
        args.append("--check-reference-continuation")
    if check_reference_logprobs:
        args.extend(["--check-reference-logprobs", "--max-logprob-delta", str(max_logprob_delta)])
    if expected_generated_token_ids is not None:
        args.extend(
            ["--expected-generated-token-ids", ",".join(str(token_id) for token_id in expected_generated_token_ids)]
        )
    subprocess.run(args, check=True, env=_subprocess_env())


def run(
    output_dir: Path,
    *,
    config_name: str,
    artifact_dir: Path | None,
    reference_path: Path | None,
    max_shard_size: int,
    generation_tokens: int,
    server_port: int,
    server_timeout_seconds: int,
    vllm_dtype: str,
    return_routed_experts: bool,
    check_reference_continuation: bool,
    check_reference_logprobs: bool,
    max_logprob_delta: float,
    expected_generated_token_ids: list[int] | None,
) -> None:
    _print_runtime_header()
    if artifact_dir is None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _run_phase_subprocess(
            "export",
            output_dir=output_dir,
            config_name=config_name,
            artifact_dir=None,
            reference_path=None,
            max_shard_size=max_shard_size,
            generation_tokens=generation_tokens,
            server_port=server_port,
            server_timeout_seconds=server_timeout_seconds,
            vllm_dtype=vllm_dtype,
            return_routed_experts=return_routed_experts,
            check_reference_continuation=check_reference_continuation,
            check_reference_logprobs=check_reference_logprobs,
            max_logprob_delta=max_logprob_delta,
            expected_generated_token_ids=expected_generated_token_ids,
        )
        artifact_dir = _artifact_dir(output_dir)
        reference_path = _reference_path(output_dir)
        print("installed_artifact_dir=" + str(artifact_dir))
        print("installed_reference_json=" + str(reference_path))
        print("installed_artifact_bytes=" + str(_directory_size_bytes(artifact_dir)))
        print("installed_safetensors_file_count=" + str(_shard_count(artifact_dir)))

    _run_phase_subprocess(
        "serve",
        output_dir=output_dir,
        config_name=config_name,
        artifact_dir=artifact_dir,
        reference_path=reference_path,
        max_shard_size=max_shard_size,
        generation_tokens=generation_tokens,
        server_port=server_port,
        server_timeout_seconds=server_timeout_seconds,
        vllm_dtype=vllm_dtype,
        return_routed_experts=return_routed_experts,
        check_reference_continuation=check_reference_continuation,
        check_reference_logprobs=check_reference_logprobs,
        max_logprob_delta=max_logprob_delta,
        expected_generated_token_ids=expected_generated_token_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "export", "serve"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/grugmoe-installed-vllm-smoke"))
    parser.add_argument(
        "--config-name",
        choices=_CONFIG_NAMES,
        default=_DEFAULT_CONFIG_NAME,
        help="Artifact profile to export when --artifact-dir is not provided.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Existing HF/safetensors artifact directory for --phase serve or --phase all.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help=f"Optional reference JSON, defaults to OUTPUT_DIR/{_REFERENCE_JSON_NAME} after export.",
    )
    parser.add_argument("--max-shard-size", type=int, default=None)
    parser.add_argument("--generation-tokens", type=int, default=1)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--vllm-dtype",
        choices=_VLLM_DTYPE_CHOICES,
        default=_DEFAULT_VLLM_DTYPE,
        help="Installed vLLM dtype for the serve path.",
    )
    parser.add_argument(
        "--return-routed-experts",
        action="store_true",
        help="Start vLLM with enable_return_routed_experts and assert returned routed-expert shape/range.",
    )
    parser.add_argument(
        "--check-reference-continuation",
        action="store_true",
        help="Assert generated IDs match continuation_ids in the reference JSON.",
    )
    parser.add_argument(
        "--check-reference-logprobs",
        action="store_true",
        help="Assert serving selected-token logprobs are close to the reference JSON.",
    )
    parser.add_argument("--max-logprob-delta", type=float, default=_DEFAULT_MAX_LOGPROB_DELTA)
    parser.add_argument(
        "--expected-generated-token-ids",
        default=None,
        help="Comma-separated deterministic continuation to assert instead of reading reference JSON.",
    )
    args = parser.parse_args()

    max_shard_size = args.max_shard_size
    if max_shard_size is None:
        max_shard_size = _default_max_shard_size(args.config_name)
    expected_generated_token_ids = _parse_token_ids(args.expected_generated_token_ids)

    if args.phase == "export":
        _print_runtime_header()
        export_artifact(
            args.output_dir,
            config_name=args.config_name,
            max_shard_size=max_shard_size,
            generation_tokens=args.generation_tokens,
        )
    elif args.phase == "serve":
        _print_runtime_header()
        serve_artifact(
            args.artifact_dir or _artifact_dir(args.output_dir),
            reference_path=args.reference_path or _reference_path(args.output_dir),
            prompt_ids=_PROMPT_IDS,
            expected_generated_token_ids=expected_generated_token_ids,
            check_reference_continuation=args.check_reference_continuation,
            check_reference_logprobs=args.check_reference_logprobs,
            max_logprob_delta=args.max_logprob_delta,
            return_routed_experts=args.return_routed_experts,
            vllm_dtype=args.vllm_dtype,
            generation_tokens=args.generation_tokens,
            server_port=args.server_port,
            server_timeout_seconds=args.server_timeout_seconds,
        )
    else:
        run(
            args.output_dir,
            config_name=args.config_name,
            artifact_dir=args.artifact_dir,
            reference_path=args.reference_path,
            max_shard_size=max_shard_size,
            generation_tokens=args.generation_tokens,
            server_port=args.server_port,
            server_timeout_seconds=args.server_timeout_seconds,
            vllm_dtype=args.vllm_dtype,
            return_routed_experts=args.return_routed_experts,
            check_reference_continuation=args.check_reference_continuation,
            check_reference_logprobs=args.check_reference_logprobs,
            max_logprob_delta=args.max_logprob_delta,
            expected_generated_token_ids=expected_generated_token_ids,
        )


if __name__ == "__main__":
    main()
