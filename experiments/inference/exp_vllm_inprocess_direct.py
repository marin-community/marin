# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct in-process vLLM test — no HTTP server.

Validates fast weight loading via Levanter fsspec + shard-streaming sync_weights,
then runs inference directly with LLM.generate().

Usage (8B on v5p-8):
    uv run iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --memory 24GB --region us-central1 \
        --extra tpu --extra vllm \
        -- python experiments/inference/exp_vllm_inprocess_direct.py \
        --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f

Usage (70B on v5p-8, needs TP=4):
    uv run iris --config=lib/iris/examples/marin.yaml job run \
        --tpu v5p-8 --memory 24GB --region us-central1 \
        --extra tpu --extra vllm \
        -- python experiments/inference/exp_vllm_inprocess_direct.py \
        --model gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b \
        --tp 4
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

# Must be set before LLM() is called.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Duplicate raw stdout fd for Iris-visible logging — immune to JAX/vLLM
# redirecting sys.stdout.
_IRIS_LOG_FD = os.dup(1)


def _iris_log(message: str) -> None:
    """Write directly to container stdout fd so Iris dashboard captures it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    line = f"I{ts} 0 vllm.inprocess.direct {message}\n"
    os.write(_IRIS_LOG_FD, line.encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GCS path to model")
    parser.add_argument("--prompt", default="Write a short haiku about TPUs.", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_inprocess import (
        _load_and_inject_streaming,
        _resolve_bootstrap_model_source_for_start,
        _resolve_mapping_model_name,
        _resolve_sync_weights_callable,
    )

    model = ModelConfig(
        name="direct-test",
        path=args.model,
        engine_kwargs={"max_model_len": args.max_model_len},
    )

    # Step 1: Resolve bootstrap source (local tmpdir with config.json + tokenizer)
    t0 = time.time()
    bootstrap_source, bootstrap_local_dir = _resolve_bootstrap_model_source_for_start(model)
    t_bootstrap = time.time() - t0
    _iris_log(f"[1/4] Bootstrap resolved in {t_bootstrap:.1f}s")

    # Step 2: Create LLM with dummy weights
    _iris_log("[2/4] Creating LLM skeleton (load_format=dummy, enforce_eager=True)...")
    t0 = time.time()
    llm = LLM(
        model=bootstrap_source,
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
    )
    t_skeleton = time.time() - t0
    _iris_log(f"[2/4] LLM skeleton created in {t_skeleton:.1f}s")

    # Step 3: Stream weights from GCS — one shard at a time (low memory)
    mapping_name = _resolve_mapping_model_name(model, args.model)
    _iris_log(f"[3/4] Streaming weights from {args.model} (mapping={mapping_name})...")

    sync_weights_fn = _resolve_sync_weights_callable(llm)
    events: list[str] = []
    t0 = time.time()
    _load_and_inject_streaming(
        model_path=args.model,
        sync_weights_fn=sync_weights_fn,
        mapping_model_name=mapping_name,
        bootstrap_model_source=bootstrap_source,
        events=events,
    )
    t_pipeline = time.time() - t0
    _iris_log(f"[3/4] Weight pipeline (streaming): {t_pipeline:.1f}s")

    # Step 4: Generate
    _iris_log(f"[4/4] Generating with prompt: {args.prompt[:80]}")
    t0 = time.time()
    outputs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=0.7))
    t_gen = time.time() - t0

    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    tps = tokens / t_gen if t_gen > 0 else 0

    _iris_log(f"[4/4] Generated {tokens} tokens in {t_gen:.1f}s ({tps:.1f} tok/s)")
    _iris_log(f"Output: {text[:200]}")

    total = t_bootstrap + t_skeleton + t_pipeline + t_gen
    _iris_log(
        f"TOTAL: {total:.1f}s "
        f"(bootstrap={t_bootstrap:.1f} skeleton={t_skeleton:.1f} "
        f"weights={t_pipeline:.1f} generate={t_gen:.1f})"
    )
    _iris_log("SUCCESS")

    # Cleanup
    if bootstrap_local_dir:
        import shutil

        shutil.rmtree(bootstrap_local_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
