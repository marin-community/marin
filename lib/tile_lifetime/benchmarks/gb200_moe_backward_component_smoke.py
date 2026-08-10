# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness-only GPU smoke for generated segmented-reverse scalar stages."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch

PACKAGE_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from tile_lifetime.cuda_expert_parallel_training_codegen import (  # noqa: E402
    expert_training_scalar_program_from_pair_map,
)
from tile_lifetime.cuda_map_fold_codegen import default_map_fold_semantics  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-output", type=Path, required=True)
    return parser


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_mok_gmm_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
    }


def _assert_repeated(function, output: torch.Tensor) -> str:
    function()
    torch.cuda.synchronize(output.device)
    first = output.clone()
    function()
    torch.cuda.synchronize(output.device)
    if not torch.equal(first, output):
        raise AssertionError("generated component is not deterministic across repeated execution")
    return _tensor_sha256(output)


def _weighted_pack_case(module: ModuleType, device: torch.device) -> dict[str, object]:
    received = (torch.arange(32, device=device).reshape(4, 8) / 17).to(torch.bfloat16)
    route_rows = torch.tensor(((0, 2), (4, -1), (1, 6), (3, 5)), dtype=torch.int64, device=device)
    weights = torch.tensor(((0.25, 0.75), (0.6, 0.0), (0.4, 0.6), (0.2, 0.8)), device=device)
    output = torch.empty((8, 8), dtype=torch.bfloat16, device=device)

    def run() -> None:
        module.route_weighted_padded_pack_bf16_out(received, route_rows, weights, output)

    output_hash = _assert_repeated(run, output)
    expected = torch.zeros_like(output)
    for source in range(route_rows.size(0)):
        for slot in range(route_rows.size(1)):
            row = int(route_rows[source, slot].item())
            if row >= 0:
                expected[row] = (received[source].float() * weights[source, slot]).to(torch.bfloat16)
    return {"error": _error(output, expected), "deterministic_sha256": output_hash}


def _pair_vjp_case(module: ModuleType, device: torch.device) -> dict[str, object]:
    pairs = (torch.arange(64, device=device).reshape(8, 8) / 31 - 1).to(torch.bfloat16)
    cotangent = (torch.arange(32, device=device).reshape(8, 4) / 19 - 0.5).to(torch.bfloat16)
    output = torch.empty_like(pairs)

    def run() -> None:
        module.row_halves_pair_map_vjp_bf16_out(pairs, cotangent, output)

    output_hash = _assert_repeated(run, output)
    left = pairs[:, :4].float().requires_grad_()
    right = pairs[:, 4:].float().requires_grad_()
    mapped = torch.nn.functional.silu(left) * right
    expected_left, expected_right = torch.autograd.grad(mapped, (left, right), cotangent.float())
    expected = torch.cat((expected_left, expected_right), dim=1).to(torch.bfloat16)
    return {"error": _error(output, expected), "deterministic_sha256": output_hash}


def _route_fold_case(module: ModuleType, device: torch.device) -> dict[str, object]:
    edge_output = (torch.arange(64, device=device).reshape(8, 8) / 23 - 1).to(torch.bfloat16)
    received_cotangent = (torch.arange(32, device=device).reshape(4, 8) / 29 - 0.5).to(torch.bfloat16)
    route_rows = torch.tensor(((0, 2), (4, -1), (1, 6), (3, 5)), dtype=torch.int64, device=device)
    output = torch.empty((4, 2), dtype=torch.float32, device=device)

    def run() -> None:
        module.route_weight_feature_fold_out(edge_output, received_cotangent, route_rows, output)

    output_hash = _assert_repeated(run, output)
    expected = torch.zeros_like(output)
    for source in range(route_rows.size(0)):
        for slot in range(route_rows.size(1)):
            row = int(route_rows[source, slot].item())
            if row >= 0:
                state = torch.tensor(0.0, device=device)
                for feature in range(edge_output.size(1)):
                    state = torch.add(
                        state,
                        torch.mul(edge_output[row, feature].float(), received_cotangent[source, feature].float()),
                    )
                expected[source, slot] = state
    return {"error": _error(output, expected), "deterministic_sha256": output_hash}


def _source_fold_case(module: ModuleType, device: torch.device) -> dict[str, object]:
    values = (torch.arange(48, device=device).reshape(6, 8) / 13 - 1).to(torch.bfloat16)
    row_indices = torch.tensor(((0, 4), (2, -1), (5, 1)), dtype=torch.int64, device=device)
    output = torch.empty((3, 8), dtype=torch.bfloat16, device=device)

    def run() -> None:
        module.indexed_ordered_source_fold_bf16_out(values, row_indices, output)

    output_hash = _assert_repeated(run, output)
    expected = torch.zeros_like(output)
    for source in range(row_indices.size(0)):
        state = torch.zeros(output.size(1), dtype=torch.float32, device=device)
        for slot in range(row_indices.size(1)):
            row = int(row_indices[source, slot].item())
            if row >= 0:
                state = torch.add(state, values[row].float())
        expected[source] = state.to(torch.bfloat16)
    return {"error": _error(output, expected), "deterministic_sha256": output_hash}


def main() -> None:
    args = _parser().parse_args()
    if not args.probe_extension.is_file():
        raise FileNotFoundError(args.probe_extension)
    if not torch.cuda.is_available():
        raise RuntimeError("the generated reverse component smoke requires CUDA")
    device = torch.device(args.device)
    module = _load_extension(args.probe_extension)
    program = expert_training_scalar_program_from_pair_map(default_map_fold_semantics().pair_map)
    observed_fingerprint = str(module.generated_expert_training_program_sha256())
    if observed_fingerprint != program.fingerprint:
        raise RuntimeError(
            f"extension reverse program {observed_fingerprint} does not match selected {program.fingerprint}"
        )

    cases = {
        "route_weighted_padded_pack": _weighted_pack_case(module, device),
        "pair_map_vjp": _pair_vjp_case(module, device),
        "route_weight_feature_fold": _route_fold_case(module, device),
        "source_input_ordered_fold": _source_fold_case(module, device),
    }
    maximum_error = max(case["error"]["maximum_absolute_error"] for case in cases.values())
    result = {
        "status": "accepted" if maximum_error == 0.0 else "rejected",
        "purpose": "correctness_only_no_timing",
        "generated_program_sha256": program.fingerprint,
        "extension": str(args.probe_extension.resolve()),
        "device": torch.cuda.get_device_name(device),
        "cases": cases,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "accepted":
        raise AssertionError(f"generated reverse component smoke failed: maximum error {maximum_error}")


if __name__ == "__main__":
    main()
