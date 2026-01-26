"""Test runners for comparing HuggingFace and Grug implementations."""

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .bridges import jax_to_torch, torch_to_jax
from .compare import ComparisonResult, compare

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


def run_hf(
    module: "nn.Module",
    *inputs: Any,
    output_idx: int | None = None,
    device: str = "cpu",
) -> jnp.ndarray:
    """Run a PyTorch module on inputs, return output as JAX array.

    Handles:
    - Converting JAX inputs to torch tensors
    - Running the module in eval mode with no_grad
    - Converting outputs back to JAX arrays

    Args:
        module: PyTorch module to run
        *inputs: Input arrays (JAX arrays, numpy arrays, or torch tensors)
        output_idx: If module returns a tuple, extract this index
        device: Device to run on ("cpu", "cuda", etc.)

    Returns:
        JAX array with the module output
    """
    if torch is None:
        raise ImportError("torch is required for run_hf")

    # Convert inputs to torch tensors
    torch_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            torch_inputs.append(inp.to(device))
        elif hasattr(inp, "__jax_array__") or isinstance(inp, jnp.ndarray):
            np_inp = np.array(inp)
            if np_inp.dtype.kind in ("i", "u"):
                torch_inputs.append(torch.from_numpy(np_inp.astype(np.int32, copy=True)).to(device))
            elif np_inp.dtype.kind == "b":
                torch_inputs.append(torch.from_numpy(np_inp.astype(np.bool_, copy=True)).to(device))
            else:
                torch_inputs.append(jax_to_torch(inp, device))
        elif isinstance(inp, np.ndarray):
            # Convert float64 to float32 for torch compatibility
            if inp.dtype == np.float64:
                inp = inp.astype(np.float32)
            torch_inputs.append(torch.from_numpy(inp.copy()).to(device))
        else:
            # Assume it's already suitable (e.g., Python int/float)
            torch_inputs.append(inp)

    # Run module in eval mode with no_grad
    module.eval()
    module.to(device)
    with torch.no_grad():
        output = module(*torch_inputs)

    # Handle tuple outputs
    if output_idx is not None:
        if isinstance(output, tuple):
            output = output[output_idx]
        else:
            raise ValueError(f"output_idx={output_idx} but output is not a tuple")

    # Convert output to JAX
    if isinstance(output, tuple):
        # If still a tuple after indexing, take first element
        output = output[0]

    return torch_to_jax(output)


@dataclass
class SuiteResult:
    """Result of running a comparison test suite."""

    results: list[ComparisonResult] = field(default_factory=list)
    all_passed: bool = True
    first_failure: ComparisonResult | None = None
    num_passed: int = 0
    num_failed: int = 0

    def __str__(self) -> str:
        lines = [f"Suite: {self.num_passed}/{self.num_passed + self.num_failed} tests passed"]
        for r in self.results:
            lines.append(f"  {r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


def run_comparison_suite(
    tests: list[tuple[str, Callable, Callable, dict[str, Any]]],
    stop_on_failure: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> SuiteResult:
    """Run a suite of comparison tests.

    Args:
        tests: List of (name, hf_fn, grug_fn, inputs_dict) tuples.
            hf_fn and grug_fn are callables that take **inputs_dict and return arrays.
        stop_on_failure: Whether to stop at first failure
        atol: Absolute tolerance for comparisons
        rtol: Relative tolerance for comparisons

    Returns:
        SuiteResult with all results and first failure info

    Example:
        suite = run_comparison_suite([
            ("embedding", lambda t: run_hf(hf_embed, t), lambda t: grug_embed[t], {"t": tokens}),
            ("layer0_norm", lambda h: run_hf(hf_norm, h), lambda h: rms_norm(h, w, eps), {"h": hidden}),
        ])
        print(suite)
    """
    result = SuiteResult()

    for name, hf_fn, grug_fn, inputs in tests:
        # Run both implementations
        try:
            hf_out = hf_fn(**inputs)
        except Exception as e:
            # Create a failure result for HF error
            comp = ComparisonResult(
                name=name,
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                diff_locations=(),
                expected_shape=(),
                actual_shape=(),
                atol=atol,
                rtol=rtol,
                failure_summary=f"HF function raised: {type(e).__name__}: {e}",
            )
            result.results.append(comp)
            result.num_failed += 1
            result.all_passed = False
            if result.first_failure is None:
                result.first_failure = comp
            if stop_on_failure:
                break
            continue

        try:
            grug_out = grug_fn(**inputs)
        except Exception as e:
            # Create a failure result for Grug error
            comp = ComparisonResult(
                name=name,
                passed=False,
                max_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                diff_locations=(),
                expected_shape=(),
                actual_shape=(),
                atol=atol,
                rtol=rtol,
                failure_summary=f"Grug function raised: {type(e).__name__}: {e}",
            )
            result.results.append(comp)
            result.num_failed += 1
            result.all_passed = False
            if result.first_failure is None:
                result.first_failure = comp
            if stop_on_failure:
                break
            continue

        # Compare outputs
        comp = compare(hf_out, grug_out, name=name, atol=atol, rtol=rtol)
        result.results.append(comp)

        if comp.passed:
            result.num_passed += 1
        else:
            result.num_failed += 1
            result.all_passed = False
            if result.first_failure is None:
                result.first_failure = comp
            if stop_on_failure:
                break

    return result
