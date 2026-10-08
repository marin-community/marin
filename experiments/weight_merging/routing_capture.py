# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture logical expert selections from a routing-enabled vLLM endpoint."""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests


def summarize_routing(payload: dict, *, num_layers: int, num_experts: int, top_k: int) -> dict:
    """Validate full-request routing and count selections by input-token phase."""
    (choice,) = payload["choices"]
    prompt_tokens = payload["prompt_token_ids"]
    generated_tokens = choice["token_ids"]
    routes = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"], validate=True)), allow_pickle=False)
    # The last sampled token has not been forwarded through the model.
    expected_tokens = len(prompt_tokens) + len(generated_tokens) - 1
    if not prompt_tokens or not generated_tokens or routes.shape != (expected_tokens, num_layers, top_k):
        raise ValueError(f"Incomplete routing capture: {routes.shape}; expected {(expected_tokens, num_layers, top_k)}")
    if not np.issubdtype(routes.dtype, np.integer) or np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("Routing capture contains invalid expert IDs")
    if np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
        raise ValueError("Routing capture selects the same expert twice for one token")
    phases = {}
    for phase, values in (("prompt", routes[: len(prompt_tokens)]), ("generated", routes[len(prompt_tokens) :])):
        counts = np.stack(
            [np.bincount(values[:, layer, :].ravel(), minlength=num_experts) for layer in range(num_layers)]
        )
        phases[phase] = {"forwarded_tokens": len(values), "selection_counts": counts.tolist()}
    return {
        "request_id": payload["id"],
        "model": payload["model"],
        "prompt_tokens": len(prompt_tokens),
        "sampled_tokens": len(generated_tokens),
        "forwarded_tokens": expected_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "finish_reason": choice["finish_reason"],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI API URL ending in /v1")
    parser.add_argument("--request", required=True, type=Path, help="One non-streaming chat-completion request JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--num-experts", required=True, type=int)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    request.update(stream=False, return_token_ids=True, routed_experts_prompt_start=0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    response = requests.post(f"{args.base_url.rstrip('/')}/chat/completions", json=request, timeout=args.timeout)
    response.raise_for_status()
    payload = response.json()
    (args.output / "response.json").write_text(json.dumps(payload) + "\n")
    summary = summarize_routing(payload, num_layers=args.num_layers, num_experts=args.num_experts, top_k=args.top_k)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
