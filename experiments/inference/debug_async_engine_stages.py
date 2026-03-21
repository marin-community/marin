# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DEBUG SCRIPT — diagnose async-native vLLM engine hang on TPU.

Runs each stage of the async pipeline in isolation with stderr logging
at every step. NOT for production use — delete once the issue is resolved.

Usage (via Iris):
    uv run iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --memory 24GB --region us-central1 \
        --extra tpu --extra vllm --job-name debug-async-engine \
        --no-wait \
        -- python experiments/inference/debug_async_engine_stages.py \
        --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback


def _log(msg: str) -> None:
    """Print to stderr — always captured by Iris regardless of stdout redirects."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[DEBUG {ts}] {msg}", file=sys.stderr, flush=True)


def _log_stage(stage: int, name: str) -> None:
    _log(f"===== STAGE {stage}: {name} =====")


def stage_1_configure_environment() -> None:
    _log_stage(1, "CONFIGURE ENVIRONMENT")
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("TPU_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "3")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    _log("Env vars set")


def stage_2_bootstrap_metadata(model_path: str) -> str:
    _log_stage(2, "BOOTSTRAP METADATA")
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_inprocess import _resolve_bootstrap_model_source_for_start

    model = ModelConfig(
        name="debug-model",
        path=model_path,
        engine_kwargs={},
    )
    t0 = time.time()
    bootstrap_source, bootstrap_dir = _resolve_bootstrap_model_source_for_start(model)
    _log(f"Bootstrap source: {bootstrap_source}")
    _log(f"Bootstrap dir: {bootstrap_dir}")
    _log(f"Bootstrap took {time.time() - t0:.1f}s")

    config_path = os.path.join(bootstrap_source, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        _log(f"config.json keys: {sorted(cfg.keys())}")
        _log(f"  num_attention_heads={cfg.get('num_attention_heads')}")
        _log(f"  hidden_size={cfg.get('hidden_size')}")
        _log(f"  num_hidden_layers={cfg.get('num_hidden_layers')}")
    else:
        _log(f"WARNING: config.json not found at {config_path}")

    return bootstrap_source


def stage_3_import_vllm_symbols() -> tuple:
    _log_stage(3, "IMPORT VLLM SYMBOLS")
    t0 = time.time()
    from vllm import AsyncEngineArgs

    _log(f"  AsyncEngineArgs imported ({time.time() - t0:.1f}s)")

    import uvicorn

    _log(f"  uvicorn imported ({time.time() - t0:.1f}s)")

    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client_from_engine_args,
        init_app_state,
    )

    _log(f"  api_server symbols imported ({time.time() - t0:.1f}s)")

    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    _log(f"  cli_args imported ({time.time() - t0:.1f}s)")

    _log(f"All vLLM symbols imported in {time.time() - t0:.1f}s")
    return (
        AsyncEngineArgs,
        build_app,
        build_async_engine_client_from_engine_args,
        FlexibleArgumentParser,
        init_app_state,
        make_arg_parser,
        uvicorn,
        validate_parsed_serve_args,
    )


def stage_4_parse_engine_args(symbols: tuple, bootstrap_source: str, max_model_len: int) -> object:
    _log_stage(4, "PARSE ENGINE ARGS")
    (
        async_engine_args_cls,
        _build_app,
        _build_engine,
        flexible_parser_cls,
        _init_app,
        make_arg_parser,
        _uvicorn,
        validate_parsed_serve_args,
    ) = symbols

    worker_ext = "marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension"
    cli_args = [
        "--model",
        bootstrap_source,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "debug-model",
        "--trust-remote-code",
        "--disable-frontend-multiprocessing",
        "--load-format",
        "dummy",
        "--worker-extension-cls",
        worker_ext,
        "--max-model-len",
        str(max_model_len),
        "--enforce-eager",
    ]
    _log(f"CLI args: {cli_args}")

    parser = flexible_parser_cls(description="debug")
    parser = make_arg_parser(parser)
    args = parser.parse_args(cli_args)
    validate_parsed_serve_args(args)
    engine_args = async_engine_args_cls.from_cli_args(args)
    _log(f"Engine args created: model={engine_args.model}, load_format={engine_args.load_format}")
    _log(f"  tensor_parallel_size={engine_args.tensor_parallel_size}")
    _log(f"  enforce_eager={engine_args.enforce_eager}")
    _log(f"  worker_extension_cls={getattr(engine_args, 'worker_extension_cls', 'N/A')}")
    return engine_args, args


async def stage_5_create_engine(symbols: tuple, engine_args: object) -> object:
    _log_stage(5, "CREATE ASYNC ENGINE (this is where the hang likely is)")
    build_async_engine_client_from_engine_args = symbols[2]

    _log("Calling build_async_engine_client_from_engine_args...")
    _log("  disable_frontend_multiprocessing=True")

    t0 = time.time()

    # Use asyncio.wait_for to detect hangs
    async def _build():
        async with build_async_engine_client_from_engine_args(
            engine_args,
            disable_frontend_multiprocessing=True,
        ) as engine_client:
            elapsed = time.time() - t0
            _log(f"Engine client created in {elapsed:.1f}s")
            _log(f"Engine client type: {type(engine_client).__name__}")
            _log(f"Engine client attributes: {[a for a in dir(engine_client) if not a.startswith('_')]}")

            # Check if collective_rpc is available
            has_rpc = hasattr(engine_client, "collective_rpc")
            _log(f"Has collective_rpc: {has_rpc}")

            if has_rpc:
                await stage_6_test_collective_rpc(engine_client)
            else:
                _log("SKIP stage 6: no collective_rpc method")

            await stage_7_test_single_shard(engine_client)

            _log("All stages complete inside engine context — exiting context now")
            return engine_client

    try:
        await asyncio.wait_for(_build(), timeout=300)
        _log("Engine context exited cleanly")
    except asyncio.TimeoutError:
        _log(f"TIMEOUT: Engine creation hung for 300s. Last check at {time.time() - t0:.1f}s")
        raise


async def stage_6_test_collective_rpc(engine_client) -> None:
    _log_stage(6, "TEST COLLECTIVE_RPC (tiny tensor)")
    import numpy as np
    from marin.rl.environments.inference_ctx.async_vllm import serialize_state_dict_for_rpc

    # Create a tiny fake state dict with one tensor
    fake_dict = {"debug.test_tensor": np.zeros((2, 2), dtype=np.bfloat16)}
    serialized = serialize_state_dict_for_rpc(fake_dict)
    _log(f"Serialized test tensor: {list(serialized.keys())}")

    t0 = time.time()
    try:
        # This will likely fail (key won't match any model param) but should
        # NOT hang. We want to see if collective_rpc itself works.
        await engine_client.collective_rpc(
            "update_weight",
            args=(serialized, "meta-llama/Llama-3.1-8B-Instruct"),
        )
        _log(f"collective_rpc returned in {time.time() - t0:.1f}s (no error)")
    except Exception as exc:
        _log(f"collective_rpc raised after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        # This is expected — the test tensor won't match real weights.
        # The important thing is it didn't hang.


async def stage_7_test_single_shard(engine_client) -> None:
    _log_stage(7, "TEST SINGLE REAL SHARD")
    import jax
    from iris.marin_fs import url_to_fs
    from levanter.compat.fsspec_safetensor import read_safetensors_fsspec
    from marin.inference.vllm_inprocess import _discover_safetensor_shards, _reshape_attention_tensors
    from marin.rl.environments.inference_ctx.async_vllm import serialize_state_dict_for_rpc

    model_path = os.environ.get("DEBUG_MODEL_PATH")
    if not model_path:
        _log("SKIP: DEBUG_MODEL_PATH not set")
        return

    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)
    _log(f"Found {len(shard_files)} shards")

    if not shard_files:
        _log("No shards found — skipping")
        return

    # Just load the first shard
    shard_file = shard_files[0]
    shard_path = os.path.join(remote_path, shard_file)
    _log(f"Loading shard 1/{len(shard_files)}: {shard_file}")

    t0 = time.time()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        shard_dict = await read_safetensors_fsspec(shard_path, fs=fs, sharding_fn=None)
    t_download = time.time() - t0
    shard_bytes = sum(v.nbytes for v in shard_dict.values())
    _log(f"Downloaded {len(shard_dict)} tensors, {shard_bytes / (1024**3):.2f} GiB in {t_download:.1f}s")

    # Read config for reshape
    config_path = os.path.join(os.environ.get("DEBUG_BOOTSTRAP_SOURCE", "/tmp"), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        num_heads = cfg["num_attention_heads"]
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        head_dim = cfg["hidden_size"] // num_heads
        _log(f"Reshaping attention: heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
        _reshape_attention_tensors(shard_dict, num_heads, num_kv_heads, head_dim)
    else:
        _log(f"No config.json at {config_path} — skipping reshape")

    serialized = serialize_state_dict_for_rpc(shard_dict)
    _log(f"Serialized shard: {len(serialized)} keys")

    t0 = time.time()
    try:
        await engine_client.collective_rpc(
            "update_weight",
            args=(serialized, "meta-llama/Llama-3.1-8B-Instruct"),
        )
        _log(f"Shard 1 injected via collective_rpc in {time.time() - t0:.1f}s")
    except Exception as exc:
        _log(f"Shard 1 injection failed after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())

    del shard_dict, serialized


async def run_all_stages(model_path: str, max_model_len: int) -> None:
    t_total = time.time()
    _log(f"DEBUG SCRIPT START — model={model_path}, max_model_len={max_model_len}")
    _log(f"Python: {sys.version}")
    _log(f"PID: {os.getpid()}")

    stage_1_configure_environment()
    bootstrap_source = stage_2_bootstrap_metadata(model_path)

    # Stash for stage 7
    os.environ["DEBUG_MODEL_PATH"] = model_path
    os.environ["DEBUG_BOOTSTRAP_SOURCE"] = bootstrap_source

    symbols = stage_3_import_vllm_symbols()
    engine_args, _args = stage_4_parse_engine_args(symbols, bootstrap_source, max_model_len)
    await stage_5_create_engine(symbols, engine_args)

    _log(f"ALL STAGES COMPLETE in {time.time() - t_total:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="DEBUG: async vLLM engine stage-by-stage diagnosis")
    parser.add_argument("--model", required=True, help="GCS model path")
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    _log("=" * 60)
    _log("DEBUG ASYNC ENGINE STAGES — NOT FOR PRODUCTION")
    _log("=" * 60)

    try:
        asyncio.run(run_all_stages(args.model, args.max_model_len))
    except Exception:
        _log(f"FATAL: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
