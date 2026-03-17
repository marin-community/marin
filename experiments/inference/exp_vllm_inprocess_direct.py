# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct in-process vLLM test — no HTTP server.

Validates fast weight loading via Levanter fsspec + sync_weights,
then runs inference directly with LLM.generate().
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Save real stdout — vLLM may wrap sys.stdout with rank-prefixed streams.
_REAL_STDOUT = sys.stdout

# Must be set before LLM() is called.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _iris_log(message: str) -> None:
    """Write directly to real stdout so Iris dashboard captures it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    _REAL_STDOUT.write(f"I{ts} 0 vllm.inprocess.direct {message}\n")
    _REAL_STDOUT.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GCS path to model")
    parser.add_argument("--prompt", default="Write a short haiku about TPUs.", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    from marin.inference.vllm_inprocess import (
        _resolve_bootstrap_model_source_for_start,
        _resolve_mapping_model_name,
        load_safetensors_from_remote,
    )
    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
    from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu

    model = ModelConfig(
        name="direct-test",
        path=args.model,
        engine_kwargs={"max_model_len": args.max_model_len},
    )

    # Step 1: Resolve bootstrap source (local tmpdir with config.json + tokenizer)
    t0 = time.time()
    bootstrap_source, _ = _resolve_bootstrap_model_source_for_start(model)
    t_bootstrap = time.time() - t0
    logger.info(f"Bootstrap source resolved in {t_bootstrap:.1f}s: {bootstrap_source}")
    _iris_log(f"Bootstrap resolved in {t_bootstrap:.1f}s")

    # Step 2: Create LLM with dummy weights
    _iris_log("Creating LLM skeleton with load_format=dummy...")
    t0 = time.time()
    llm = LLM(
        model=bootstrap_source,
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=True,  # Avoid XLA compilation for faster testing
    )
    t_skeleton = time.time() - t0
    logger.info(f"LLM skeleton created in {t_skeleton:.1f}s")
    _iris_log(f"LLM skeleton created in {t_skeleton:.1f}s")

    # Step 3: Load weights from GCS via Levanter fsspec
    _iris_log(f"Loading weights from {args.model} via Levanter fsspec...")
    t0 = time.time()
    state_dict = load_safetensors_from_remote(args.model)
    t_load = time.time() - t0
    logger.info(f"Loaded {len(state_dict)} tensors from GCS in {t_load:.1f}s")
    _iris_log(f"Loaded {len(state_dict)} tensors from GCS in {t_load:.1f}s")

    # Step 4: Reshape HF attention projections to Levanter 3D format
    # HF stores q/k/v/o_proj as 2D (heads*dim, hidden), but MODEL_MAPPINGS
    # expect Levanter's 3D format (heads, dim, hidden).
    import numpy as np

    config_path = os.path.join(bootstrap_source, "config.json")
    with open(config_path) as f:
        model_config = __import__("json").load(f)
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    head_dim = model_config["hidden_size"] // num_heads

    reshaped = 0
    for key in list(state_dict.keys()):
        if "q_proj" in key and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(num_heads, head_dim, -1)
            reshaped += 1
        elif ("k_proj" in key or "v_proj" in key) and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(num_kv_heads, head_dim, -1)
            reshaped += 1
        elif "o_proj" in key and "bias" not in key:
            w = np.array(state_dict[key])
            state_dict[key] = w.reshape(-1, num_heads, head_dim)
            reshaped += 1
    logger.info(f"Reshaped {reshaped} attention projections to Levanter 3D format")
    _iris_log(f"Reshaped {reshaped} attention projections to Levanter 3D format")

    # Step 5: Inject weights via sync_weights
    mapping_name = _resolve_mapping_model_name(model, args.model)
    logger.info(f"Using mapping: {mapping_name}")
    _iris_log(f"Injecting weights via sync_weights (mapping={mapping_name})...")

    t0 = time.time()
    nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)
    t_convert = time.time() - t0
    logger.info(f"Converted to NNX state in {t_convert:.1f}s")

    t0 = time.time()
    llm.llm_engine.model_executor.driver_worker.sync_weights(
        nnx_state,
        mappings=MODEL_MAPPINGS[mapping_name],
        transpose_keys=MODEL_TRANSPOSE_KEYS[mapping_name],
        reshard_fn=None,
    )
    t_inject = time.time() - t0
    logger.info(f"Weights injected in {t_inject:.1f}s")

    t_total = t_load + t_convert + t_inject
    logger.info(f"=== TOTAL WEIGHT PIPELINE: {t_total:.1f}s ===")
    _iris_log(
        f"WEIGHT PIPELINE COMPLETE: {t_total:.1f}s " f"(load={t_load:.1f} convert={t_convert:.1f} inject={t_inject:.1f})"
    )

    # Step 5: Generate!
    logger.info(f"Generating with prompt: {args.prompt}")
    t0 = time.time()
    outputs = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=0.7))
    t_gen = time.time() - t0

    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    logger.info(f"Generated {tokens} tokens in {t_gen:.1f}s ({tokens/t_gen:.1f} tok/s)")
    _iris_log(f"Generated {tokens} tokens in {t_gen:.1f}s ({tokens/t_gen:.1f} tok/s)")
    _iris_log(f"Output: {text[:200]}")
    _iris_log("SUCCESS")
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {text}")
    print(f"{'='*60}")
    print(f"Weight loading: {t_total:.1f}s")
    print(f"Generation: {t_gen:.1f}s ({tokens} tokens)")
    print("SUCCESS")


if __name__ == "__main__":
    main()
