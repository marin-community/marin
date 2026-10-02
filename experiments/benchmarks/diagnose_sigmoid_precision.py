# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exhaustive finite-BF16 sigmoid outputs with an independent FP64 reference."""

import argparse
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np

BF16_BIT_PATTERNS = 1 << 16
EXAMPLE_COUNT = 5
CAPTURED_GATE_BOUND = 0.04


def sigmoid_reference(inputs: np.ndarray) -> np.ndarray:
    """Evaluate a stable sigmoid in NumPy FP64, without accelerator arithmetic."""
    values = inputs.astype(np.float64)
    exponential = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


def finite_bf16_inputs() -> np.ndarray:
    values = np.arange(BF16_BIT_PATTERNS, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    return values[np.isfinite(values)]


def error_summary(inputs: np.ndarray, actual: np.ndarray, reference: np.ndarray) -> dict:
    rounded = reference.astype(ml_dtypes.bfloat16).astype(np.float64)
    actual = actual.astype(np.float64)
    if not np.all(np.isfinite(actual)) or not np.all((actual >= 0) & (actual <= 1)):
        raise ValueError("Sigmoid returned nonfinite values or values outside [0, 1]")
    actual_bits = actual.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int32)
    reference_bits = rounded.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int32)
    distance = np.abs(actual_bits - reference_bits)
    min_normal = float(ml_dtypes.finfo(ml_dtypes.bfloat16).tiny)
    regions = {
        "all_finite_inputs": np.ones(inputs.shape, dtype=bool),
        "captured_gate_range_abs_le_0_04": np.abs(inputs) <= CAPTURED_GATE_BOUND,
        "reference_rounds_to_zero": rounded == 0,
        "reference_rounds_to_one": rounded == 1,
        "reference_subnormal": (rounded > 0) & (rounded < min_normal),
        "reference_normal_interior": (rounded >= min_normal) & (rounded < 1),
    }
    summaries = {}
    for name, mask in regions.items():
        error = np.abs(actual[mask] - reference[mask])
        summaries[name] = {
            "count": int(np.count_nonzero(mask)),
            "rounded_reference_mismatches": int(np.count_nonzero(distance[mask])),
            "max_bf16_steps_from_rounded_reference": int(distance[mask].max(initial=0)),
            "max_absolute_error_fp64": float(error.max(initial=0)),
            "rms_error_fp64": float(np.sqrt(np.mean(np.square(error)))) if error.size else None,
        }
    indices = np.flatnonzero(distance)
    indices = indices[np.argsort(-distance[indices], kind="stable")][:EXAMPLE_COUNT]
    examples = [
        {
            "input": float(inputs[i]),
            "actual": float(actual[i]),
            "fp64_reference": float(reference[i]),
            "rounded_bf16_reference": float(rounded[i]),
            "bf16_steps": int(distance[i]),
        }
        for i in indices
    ]
    return {"regions": summaries, "largest_bf16_step_examples": examples}


def jax_outputs(inputs: np.ndarray, platform: str) -> tuple[dict, dict]:
    # Optional runtime: the Torch invocation must not initialize a JAX backend.
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    device = jax.devices(platform)[0]
    values = jax.device_put(inputs.astype(ml_dtypes.bfloat16), device)
    outputs = {
        "default_bf16": jax.jit(jax.nn.sigmoid)(values),
        "fp32_intermediates": jax.jit(lambda x: jax.nn.sigmoid(x.astype(jnp.float32)).astype(x.dtype))(values),
    }
    provenance = {
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "device": str(device),
        "device_kind": device.device_kind,
        "execution": "standalone_jit_elementwise",
    }
    return {name: np.asarray(value.astype(jnp.float32)) for name, value in outputs.items()}, provenance


def torch_outputs(inputs: np.ndarray, platform: str) -> tuple[dict, dict]:
    # Optional runtime: import the Torch supplied by the selected vLLM environment.
    import torch  # noqa: PLC0415

    device = "cuda:0" if platform == "gpu" else "cpu"
    values = torch.tensor(inputs, dtype=torch.bfloat16, device=device)
    outputs = {
        "default_bf16": torch.sigmoid(values),
        "fp32_intermediates": torch.sigmoid(values.float()).to(values.dtype),
    }
    provenance = {
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device": device,
        "device_kind": torch.cuda.get_device_name(0) if platform == "gpu" else "cpu",
        "execution": "standalone_eager_elementwise",
    }
    return {name: value.float().cpu().numpy() for name, value in outputs.items()}, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("jax", "torch"), required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = finite_bf16_inputs()
    reference = sigmoid_reference(inputs)
    run = jax_outputs if args.runtime == "jax" else torch_outputs
    outputs, provenance = run(inputs, args.platform)
    report = {
        "boundary": "untimed_standalone_sigmoid_exhaustive_finite_bf16",
        "runtime": args.runtime,
        "platform": args.platform,
        "provenance": provenance,
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(inputs.astype("<f4").tobytes()).hexdigest(),
        "reference": "NumPy stable FP64 sigmoid, rounded once to BF16 for mismatch counts",
        "reference_versions": {"numpy": np.__version__, "ml_dtypes": ml_dtypes.__version__},
        "model_math_changed": False,
        "results": {name: error_summary(inputs, output, reference) for name, output in outputs.items()},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
