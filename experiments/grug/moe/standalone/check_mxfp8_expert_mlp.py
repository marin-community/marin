# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the fused MXFP8 expert MLP against an independent f32 reference."""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.mxfp8 import mxfp8_expert_mlp

CUTLASS_DISTRIBUTIONS = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)


def _distribution_file_matches(distribution: importlib.metadata.Distribution, path: Path) -> bool:
    for package_path in distribution.files or ():
        installed_path = Path(distribution.locate_file(package_path)).resolve()
        if installed_path != path:
            continue
        file_hash = package_path.hash
        if file_hash is None or file_hash.mode != "sha256":
            return False
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded_digest = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return encoded_digest == file_hash.value
    return False


def cutlass_environment_sentinel() -> dict:
    """Describe the exact CUTLASS DSL payload and NVVM path selected by this process."""
    distributions = {}
    installed_distributions = {}
    for name in CUTLASS_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
            continue
        installed_distributions[name] = distribution
        distributions[name] = {
            "version": distribution.version,
            "dist_info": str(getattr(distribution, "_path", "")),
        }

    cutlass_module_path = None
    cutlass_extension_path = None
    cutlass_import_error = None
    try:
        cutlass = importlib.import_module("cutlass")
        cutlass_module_path = str(Path(cutlass.__file__).resolve())
        importlib.import_module("cutlass.cute")
        importlib.import_module("cutlass._mlir._mlir_libs")
    except Exception as error:
        cutlass_import_error = f"{type(error).__name__}: {error}"
    for module_name, module in sys.modules.items():
        if module_name.endswith("._cutlass_ir") and getattr(module, "__file__", None):
            cutlass_extension_path = str(Path(module.__file__).resolve())
            break

    cutlass_payload = None
    payload_owners = []
    if cutlass_extension_path is not None:
        extension_path = Path(cutlass_extension_path)
        if ".cu13." in extension_path.name:
            cutlass_payload = "nvidia-cutlass-dsl-libs-cu13"
        elif ".cu12." in extension_path.name:
            cutlass_payload = "nvidia-cutlass-dsl-libs-cu12"
        for name, distribution in installed_distributions.items():
            if _distribution_file_matches(distribution, extension_path):
                payload_owners.append(name)
        if cutlass_payload is None and len(payload_owners) == 1:
            cutlass_payload = payload_owners[0]

    libnvvm_path = None
    libnvvm_error = None
    try:
        pathfinder = importlib.import_module("cuda.pathfinder")
        libnvvm_path = str(pathfinder.load_nvidia_dynamic_lib("nvvm").abs_path)
    except Exception as error:
        libnvvm_error = f"{type(error).__name__}: {error}"

    return {
        "cuda_toolkit_path": os.environ.get("CUDA_TOOLKIT_PATH"),
        "cutlass_extension_path": cutlass_extension_path,
        "cutlass_import_error": cutlass_import_error,
        "cutlass_module_path": cutlass_module_path,
        "cutlass_payload": cutlass_payload,
        "cutlass_payload_record_owners": payload_owners,
        "distributions": distributions,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "libnvvm_error": libnvvm_error,
        "libnvvm_path": libnvvm_path,
        "nvidia_cutlass_dsl_module_path": cutlass_module_path,
    }


def _relative_frobenius(actual: jax.Array, expected: jax.Array) -> float:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    return float(jnp.linalg.norm(actual_f32 - expected_f32) / jnp.linalg.norm(expected_f32))


def _group_sizes(tokens: int, experts: int, seed: int) -> list[int]:
    if experts < 4:
        raise ValueError(f"experts must be at least 4, got {experts}")
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(experts, 0.5))
    groups = np.floor(weights * tokens).astype(np.int64)
    groups[1] = 0
    groups[-1] = 0
    groups[0] += tokens - int(groups.sum())
    assert int(groups.sum()) == tokens
    return [int(group) for group in groups]


def _reference_mlp(groups: list[int]):
    def reference(x: jax.Array, w13: jax.Array, w2: jax.Array) -> jax.Array:
        outputs = []
        start = 0
        for expert, size in enumerate(groups):
            x_expert = x[start : start + size].astype(jnp.float32)
            hidden = x_expert @ w13[expert].astype(jnp.float32)
            gate, up = jnp.split(hidden, 2, axis=-1)
            outputs.append((jax.nn.silu(gate) * up) @ w2[expert].astype(jnp.float32))
            start += size
        return jnp.concatenate(outputs)

    return reference


def _inputs(
    *,
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts: int,
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, list[int], jax.Array]:
    key = jax.random.PRNGKey(seed)
    key_x, key_w13, key_w2, key_cotangent = jax.random.split(key, 4)
    x = jax.random.normal(key_x, (tokens, hidden_dim), jnp.bfloat16)
    w13 = (
        jax.random.normal(key_w13, (experts, hidden_dim, 2 * intermediate_dim), jnp.bfloat16) / hidden_dim**0.5
    ).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(key_w2, (experts, intermediate_dim, hidden_dim), jnp.bfloat16) / intermediate_dim**0.5
    ).astype(jnp.bfloat16)
    cotangent = jax.random.normal(key_cotangent, (tokens, hidden_dim), jnp.bfloat16)
    groups = _group_sizes(tokens, experts, seed)
    return x, w13, w2, cotangent, groups, jnp.asarray(groups, jnp.int32)


def run(args: argparse.Namespace) -> dict:
    if args.hidden_dim % 128 or args.intermediate_dim % 128:
        raise ValueError("hidden-dim and intermediate-dim must be divisible by 128")
    device = jax.devices()[0]
    inputs = _inputs(
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        experts=args.experts,
        seed=args.seed,
    )
    x, w13, w2, cotangent, groups, group_sizes = inputs

    @jax.jit
    def treatment(x_value, w13_value, w2_value):
        output, pullback = jax.vjp(
            lambda a, b, c: mxfp8_expert_mlp(a, b, c, group_sizes),
            x_value,
            w13_value,
            w2_value,
        )
        return output, pullback(cotangent)

    reference_mlp = _reference_mlp(groups)

    @jax.jit
    def control(x_value, w13_value, w2_value):
        output, pullback = jax.vjp(reference_mlp, x_value, w13_value, w2_value)
        return output, pullback(cotangent.astype(jnp.float32))

    output, (dx, dw13, dw2) = treatment(x, w13, w2)
    reference_output, (reference_dx, reference_dw13, reference_dw2) = control(x, w13, w2)
    jax.block_until_ready((output, dx, dw13, dw2))

    errors = {
        "output": _relative_frobenius(output, reference_output),
        "dx": _relative_frobenius(dx, reference_dx),
        "dw13": _relative_frobenius(dw13, reference_dw13),
        "dw2": _relative_frobenius(dw2, reference_dw2),
    }
    for name, error in errors.items():
        if error >= args.error_gate:
            raise AssertionError(f"{name} relative-Frobenius error {error:.6f} >= {args.error_gate}")

    zero_expert_max = {}
    for expert, size in enumerate(groups):
        if size == 0:
            dw13_max = float(jnp.abs(dw13[expert]).max())
            dw2_max = float(jnp.abs(dw2[expert]).max())
            if dw13_max != 0.0 or dw2_max != 0.0:
                raise AssertionError(f"zero-token expert {expert} has nonzero gradients: dw13={dw13_max}, dw2={dw2_max}")
            zero_expert_max[str(expert)] = {"dw13": dw13_max, "dw2": dw2_max}

    return {
        "device": str(device.device_kind),
        "compute_capability": str(getattr(device, "compute_capability", None)),
        "jax": jax.__version__,
        "shape": {
            "tokens": args.tokens,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "experts": args.experts,
        },
        "group_sizes": groups,
        "errors": errors,
        "zero_expert_gradient_max": zero_expert_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65_536)
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-gate", type=float, default=0.1)
    parser.add_argument("--out", default="/tmp/ep25d2-mxfp8-numerics.json")
    parser.add_argument("--sentinel-only", action="store_true")
    args = parser.parse_args()

    sentinel = cutlass_environment_sentinel()
    print("CUTLASS_ENV_SENTINEL " + json.dumps(sentinel, sort_keys=True), flush=True)
    if args.sentinel_only:
        return

    result = run(args)
    with open(args.out, "w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
    print("RESULTS " + json.dumps(result, sort_keys=True), flush=True)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
