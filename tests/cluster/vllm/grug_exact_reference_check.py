# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen July 27 Grug fixture against the Marin vLLM fork.

The fixture contains observations produced by the exact training-reference
commit. Router and shared-expert checks run directly through the vLLM model
classes. Logit and prefix-reuse checks run through a live OpenAI-compatible
server loaded from the same checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    from tests.cluster.vllm.backend_parity import TokenScore, parity_from_logprob_map
except ModuleNotFoundError:
    # When this file is executed by path in the isolated vLLM environment,
    # its own directory is importable but the repository's namespace package
    # may lose to an installed package named ``tests``.
    from backend_parity import TokenScore, parity_from_logprob_map

MAX_PROBABILITY_ERROR = 0.075
HTTP_TIMEOUT_SECONDS = 600


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fixture(fixture_dir: Path) -> dict[str, Any]:
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    if manifest["training_reference_commit"] != "fd3e9bc5b428633027f944be7fdf1136567db028":
        raise AssertionError("fixture is not pinned to the frozen training reference")
    for name, expected in manifest["files"].items():
        actual = _sha256(fixture_dir / name)
        if actual != expected:
            raise AssertionError(f"{name}: fixture hash {actual} != {expected}")
    return manifest


def run_tensor_parity(fixture_dir: Path) -> dict[str, Any]:
    """Use the vLLM router and dense MLP formula on frozen Levanter tensors."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as torch_f  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from vllm.model_executor.models.grugmoe import GrugMoeRouter  # noqa: PLC0415

    manifest = verify_fixture(fixture_dir)
    observations = np.load(fixture_dir / "observations.npz", allow_pickle=False)
    state = load_file(fixture_dir / "model.safetensors", device="cpu")
    layer_count = int(manifest["semantics"]["layers"])
    top_k = int(manifest["semantics"]["qb_top_k"])
    num_experts = int(state["model.layers.0.mlp.router.bias"].numel())
    max_weight_error = 0.0
    selected_hash = hashlib.sha256()
    for layer_index in range(layer_count):
        prefix = f"model.layers.{layer_index}"
        hidden = torch.from_numpy(observations[f"mlp_input.layer.{layer_index}"])
        router_weight = state[f"{prefix}.mlp.router.weight"]
        router_bias = state[f"{prefix}.mlp.router.bias"]
        router_logits = torch_f.linear(hidden.float(), router_weight.float())
        router = GrugMoeRouter(
            top_k=top_k,
            global_num_experts=num_experts,
            bias=router_bias,
        )
        actual_weights, actual_ids = router.select_experts(
            hidden_states=hidden,
            router_logits=router_logits,
        )
        expected_ids = observations[f"selected_experts.layer.{layer_index}"]
        expected_weights = observations[f"normalized_weights.layer.{layer_index}"]
        np.testing.assert_array_equal(actual_ids.numpy(), expected_ids)
        np.testing.assert_allclose(
            actual_weights.numpy(),
            expected_weights,
            rtol=1e-5,
            atol=1e-6,
        )
        max_weight_error = max(
            max_weight_error,
            float(np.max(np.abs(actual_weights.numpy() - expected_weights))),
        )
        selected_hash.update(actual_ids.numpy().tobytes())

    # Concatenating gate/up rows and down columns is algebraically identical to
    # summing two half-width shared experts. Check every layer on its real input.
    max_shared_error = 0.0
    shared_weight_hashes: list[str] = []
    for layer_index in range(layer_count):
        prefix = f"model.layers.{layer_index}.shared_experts"
        hidden = torch.from_numpy(observations[f"mlp_input.layer.{layer_index}"]).float()
        experts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for shared_index in range(2):
            expert = (
                state[f"{prefix}.{shared_index}.gate_proj.weight"].float(),
                state[f"{prefix}.{shared_index}.up_proj.weight"].float(),
                state[f"{prefix}.{shared_index}.down_proj.weight"].float(),
            )
            if not any(torch.count_nonzero(weight).item() for weight in expert):
                raise AssertionError(f"layer {layer_index} shared expert {shared_index} is zero")
            experts.append(expert)
            shared_weight_hashes.append(
                hashlib.sha256(b"".join(weight.numpy().tobytes() for weight in expert)).hexdigest()
            )
        if all(torch.equal(left, right) for left, right in zip(experts[0], experts[1], strict=True)):
            raise AssertionError(f"layer {layer_index} shared experts are identical")

        separate = sum(
            torch_f.linear(
                torch_f.silu(torch_f.linear(hidden, gate)) * torch_f.linear(hidden, up),
                down,
            )
            for gate, up, down in experts
        )
        fused_gate = torch.cat([expert[0] for expert in experts], dim=0)
        fused_up = torch.cat([expert[1] for expert in experts], dim=0)
        fused_down = torch.cat([expert[2] for expert in experts], dim=1)
        fused = torch_f.linear(
            torch_f.silu(torch_f.linear(hidden, fused_gate)) * torch_f.linear(hidden, fused_up),
            fused_down,
        )
        np.testing.assert_allclose(fused.numpy(), separate.numpy(), rtol=1e-5, atol=2e-5)
        max_shared_error = max(max_shared_error, float(torch.max(torch.abs(fused - separate)).item()))

    return {
        "passed": True,
        "training_reference_commit": manifest["training_reference_commit"],
        "layers_checked": layer_count,
        "tokens_checked_per_layer": int(observations["semantic_input_ids"].size),
        "selected_experts_sha256": selected_hash.hexdigest(),
        "max_normalized_weight_absolute_error": max_weight_error,
        "shared_experts": {
            "count": 2,
            "width_each": int(manifest["semantics"]["shared_expert_width_each"]),
            "all_nonzero_and_pairwise_distinct": len(set(shared_weight_hashes)) == len(shared_weight_hashes),
            "fused_equivalence_max_absolute_error": max_shared_error,
        },
    }


def _completion(
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    *,
    max_tokens: int = 1,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt_token_ids,
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "logprobs": 64,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if len(payload.get("choices", ())) != 1:
        raise AssertionError(f"expected one completion choice: {payload!r}")
    return payload


def _routes(payload: dict[str, Any]) -> np.ndarray:
    encoded = payload["choices"][0].get("routed_experts")
    if not encoded:
        raise AssertionError("completion omitted routed experts")
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def _top_logprobs(payload: dict[str, Any]) -> dict[int, float]:
    (row,) = payload["choices"][0]["logprobs"]["top_logprobs"]
    return {int(token.removeprefix("token_id:")): float(value) for token, value in row.items()}


def _score_logprobs(expected: np.ndarray, payload: dict[str, Any]) -> dict[str, Any]:
    actual = _top_logprobs(payload)
    if set(actual) != set(range(expected.size)):
        raise AssertionError(f"server returned {len(actual)}/{expected.size} vocabulary logprobs")
    expected_greedy = int(expected.argmax())
    actual_greedy = int(payload["choices"][0]["token_ids"][0])
    parity = parity_from_logprob_map(
        "grug-exact-reference",
        tuple(TokenScore(logprob=float(logprob), token_id=token_id) for token_id, logprob in enumerate(expected)),
        actual_greedy,
        actual,
        backend_rank=0,
    )
    parity.assert_matches(max_probability_error=MAX_PROBABILITY_ERROR)
    return {
        "greedy_token_id": actual_greedy,
        "expected_greedy_token_id": expected_greedy,
        "greedy_token_agrees": actual_greedy == expected_greedy,
        "golden_probability_gap_to_greedy": parity.golden_probability_gap_to_greedy,
        "max_probability_error": parity.max_probability_error,
        "probability_l1_error": parity.top_probability_l1_error,
        "tolerance": MAX_PROBABILITY_ERROR,
    }


def run_server_parity(base_url: str, model: str, fixture_dir: Path) -> dict[str, Any]:
    """Check full-prefill parity, then exact 512-token prefix reuse."""
    verify_fixture(fixture_dir)
    observations = np.load(fixture_dir / "observations.npz", allow_pickle=False)

    semantic_ids = observations["semantic_input_ids"].astype(int).tolist()
    semantic = _completion(base_url, model, semantic_ids)
    expected_routes = np.stack(
        [observations[f"selected_experts.layer.{index}"] for index in range(7)],
        axis=1,
    )
    actual_routes = _routes(semantic)
    if actual_routes.shape[0] < len(semantic_ids):
        raise AssertionError(
            f"semantic response returned {actual_routes.shape[0]} routed positions for {len(semantic_ids)} prompt tokens"
        )
    np.testing.assert_array_equal(actual_routes[: len(semantic_ids)], expected_routes)
    semantic_score = _score_logprobs(observations["semantic_logprobs"][-1], semantic)

    boundary_ids = observations["boundary_input_ids"].astype(int).tolist()
    cold = _completion(base_url, model, boundary_ids)
    reused = _completion(base_url, model, boundary_ids)
    cold_choice = cold["choices"][0]
    reused_choice = reused["choices"][0]
    if cold_choice["token_ids"] != reused_choice["token_ids"]:
        raise AssertionError("cold and reused boundary requests generated different tokens")
    if _top_logprobs(cold) != _top_logprobs(reused):
        raise AssertionError("cold and reused boundary requests returned different logprobs")
    np.testing.assert_array_equal(_routes(cold), _routes(reused))
    cached_tokens = int(reused["usage"]["prompt_tokens_details"]["cached_tokens"])
    if cached_tokens != 512:
        raise AssertionError(f"expected 512 cached prompt tokens, got {cached_tokens}")
    boundary_cold_score = _score_logprobs(observations["boundary_logprobs"], cold)
    boundary_reused_score = _score_logprobs(observations["boundary_logprobs"], reused)

    return {
        "passed": True,
        "semantic": {
            **semantic_score,
            "prompt_tokens": len(semantic_ids),
            "selected_experts_sha256": hashlib.sha256(actual_routes[: len(semantic_ids)].tobytes()).hexdigest(),
        },
        "boundary": {
            "prompt_tokens": len(boundary_ids),
            "cache_block_size": 16,
            "sliding_window": 512,
            "reused_prompt_tokens": cached_tokens,
            "cold": boundary_cold_score,
            "reused": boundary_reused_score,
            "cold_reuse_identical": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"tensor": run_tensor_parity(args.fixture)}
    if args.base_url or args.model:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model must be given together")
        result["server"] = run_server_parity(args.base_url, args.model, args.fixture)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
