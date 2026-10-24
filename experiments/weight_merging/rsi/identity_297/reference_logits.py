# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare identical short token sequences with the eager FP32 Grug reference."""

import argparse
import hashlib
import io
import json
import sys
import types
from importlib import import_module
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret
from transformers import AutoTokenizer

MODEL = "open-athena/Marin-Snowball-67B-A2B-v0.1.1"
REVISION = "59852d6f07261722bbec1edb6ab148317156b58e"
PROMPTS = (
    "What is your name and who created you?",
    "Write the opening sentence of a horror story about a fairy.",
    "Janet has 3 apples and buys 5 more. How many apples does she have?",
    "Solve for x: 2*x + 7 = 19.",
    "The capital of France is",
    "Remember the opening codeword: lavender.\n"
    + "This is a neutral passage about gardens, weather, books, and travel.\n" * 250
    + "What was the opening codeword?",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-probe", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "vendor/skyrl/skyrl-train"))
    # Import the unchanged optional attention bindings without the unrelated
    # SkyRL training utility __init__ (which imports Ray and rollout drivers).
    utils = types.ModuleType("skyrl_train.utils")
    utils.__path__ = [str(root / "vendor/skyrl/skyrl-train/skyrl_train/utils")]
    sys.modules["skyrl_train.utils"] = utils
    grug = import_module("skyrl_train.models.grug_moe")

    fs, prefix = filesystem_for(args.output.rstrip("/"))
    if fs.exists(prefix + "/done.json"):
        print("Reference probe already complete", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in PROMPTS
    ]
    assert all(isinstance(ids, list) and all(isinstance(token, int) for token in ids) for ids in prompt_ids)
    assert len(prompt_ids[5]) > 3000, f"Window probe too short: {len(prompt_ids[5])}"
    print(f"Validated prompt token lengths: {[len(ids) for ids in prompt_ids]}", flush=True)
    model, loading_info = grug.GrugMoeForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
        device_map="auto",
        max_memory={index: "65GiB" for index in range(torch.cuda.device_count())},
        output_loading_info=True,
    )
    model.eval()
    assert all(not value for value in loading_info.values()), f"Incomplete checkpoint load: {loading_info}"
    loading_info = {key: sorted(value) if isinstance(value, set) else value for key, value in loading_info.items()}
    index_path = hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)
    index = json.loads(Path(index_path).read_text())
    state = model.state_dict()
    assert set(state) == set(index["weight_map"]), "Loaded state keys differ from checkpoint"
    state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    assert state_bytes == index["metadata"]["total_size"], "Loaded tensor bytes differ from checkpoint"
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    provenance = {
        "model": MODEL,
        "revision": REVISION,
        "loading_info": loading_info,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensor_count": len(state),
        "state_elements": sum(tensor.numel() for tensor in state.values()),
        "state_bytes": state_bytes,
        "reference": "SkyRL Grug HF-compatible eager FP32, TF32 disabled",
        "implementation_sha256": (
            hashlib.sha256((root / "vendor/skyrl/skyrl-train/skyrl_train/models/grug_moe.py").read_bytes()).hexdigest()
        ),
        "windows": [None if long else model.config.sliding_window for long in grug.grug_long_layer_flags(26)],
        "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
        "limitation": "Five short prompts and one >3000-token prompt. API values are log probabilities.",
    }
    fs.pipe_file(prefix + "/provenance.json", json.dumps(provenance, indent=2).encode())
    print(json.dumps(provenance), flush=True)
    if args.decode_probe:
        prompts = (
            "Write a detailed 600-word horror story about a fairy tending a deserted garden.",
            "Explain how to solve linear equations, with five worked examples and detailed reasoning for each step.",
            "Describe the history and culture of Paris in a detailed 600-word essay.",
        )
        for index, prompt in enumerate(prompts):
            done_path = f"{prefix}/decode-{index}.json"
            if fs.exists(done_path):
                continue
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            payload = {
                "model": MODEL,
                "prompt": ids,
                "max_tokens": 200,
                "temperature": 0,
                "seed": 0,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            }
            base = require_secret("FP32_ENDPOINT").rstrip("/")
            response = requests.post(base + "/v1/completions", json=payload, timeout=180)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            tokens = choice["logprobs"]["tokens"]
            assert tokens and all(token.startswith("token_id:") for token in tokens)
            generated = [int(token.split(":", 1)[1]) for token in tokens]
            inputs = torch.tensor([ids + generated[:-1]], device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                logits = model(input_ids=inputs, use_cache=False, logits_to_keep=len(generated)).logits[0].float().cpu()
            log_probs = logits.log_softmax(-1)
            comparisons = []
            for step, token_id in enumerate(generated):
                server = choice["logprobs"]["token_logprobs"][step]
                reference = float(log_probs[step, token_id])
                comparisons.append(
                    {
                        "step": step,
                        "token_id": token_id,
                        "server": server,
                        "reference": reference,
                        "error": server - reference,
                        "reference_top_id": int(logits[step].argmax()),
                    }
                )
            result = {"prompt": prompt, "input_ids": ids, "choice": choice, "comparisons": comparisons}
            fs.pipe_file(done_path, json.dumps(result, indent=2).encode())
            print(f"Completed decode prompt {index}: {len(generated)} tokens", flush=True)
        fs.pipe_file(prefix + "/done.json", json.dumps({"decode_prompts": 3, "provenance": provenance}).encode())
        return
    for index, prompt in enumerate(PROMPTS):
        done_path = f"{prefix}/prompt-{index}.json"
        if fs.exists(done_path):
            continue
        ids = prompt_ids[index]
        inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
        log_probs = logits.log_softmax(-1)
        buffer = io.BytesIO()
        np.save(buffer, logits.numpy())
        fs.pipe_file(f"{prefix}/prompt-{index}-logits.npy", buffer.getvalue())
        result = {"prompt": prompt, "token_ids": ids, "reference_top_id": int(logits.argmax()), "servers": {}}
        for arm in ("BF16", "FP32"):
            base = require_secret(f"{arm}_ENDPOINT").rstrip("/")
            payload = {
                "model": MODEL,
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
                "logprobs": 20,
                "return_tokens_as_token_ids": True,
            }
            response = requests.post(base + "/v1/completions", json=payload, timeout=180)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            comparisons = []
            for token, value in choice["logprobs"]["top_logprobs"][0].items():
                if not token.startswith("token_id:"):
                    raise ValueError(f"Server did not return unambiguous token IDs: {token!r}")
                token_id = int(token.split(":", 1)[1])
                reference = float(log_probs[token_id])
                comparisons.append(
                    {"token_id": token_id, "server": value, "reference": reference, "error": value - reference}
                )
            result["servers"][arm] = {"choice": choice, "comparisons": comparisons}
        fs.pipe_file(done_path, json.dumps(result, indent=2).encode())
        print(f"Completed prompt {index}", flush=True)
    fs.pipe_file(prefix + "/done.json", json.dumps({"prompts": len(PROMPTS), "provenance": provenance}).encode())


if __name__ == "__main__":
    main()
