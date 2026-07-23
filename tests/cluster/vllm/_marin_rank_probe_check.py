#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU dry-run for _marin_rank_probe.py (plan G0): import safety, collision-safe
naming, checksum round-trip, and fixed-order reference behavior — everything that
can be validated without a GPU or vLLM.

    uv run python tests/cluster/vllm/_marin_rank_probe_check.py
"""

import sys

import _marin_rank_probe as probe
import torch


def main() -> int:
    # Import must succeed without vllm installed (lazy imports) and without the
    # NCCL overlay env var (hook inert).
    assert "vllm" not in sys.modules, "probe module must not import vllm at module scope"

    # Worker-extension collision assert in vLLM checks every non-dunder attr.
    public = [name for name in dir(probe.MarinRankProbe) if not name.startswith("__")]
    assert public and all(name.startswith(("marin_probe_", "_marin_")) for name in public), public
    assert all(callable(getattr(probe.MarinRankProbe, name)) for name in public), public

    # Every probe method must report failure instead of raising: an exception escaping
    # collective_rpc kills the worker process and the whole serve with it.
    instance = probe.MarinRankProbe()
    for name in (name for name in public if name.startswith("marin_probe_")):
        result = getattr(instance, name)(out_dir="/nonexistent/probe/path")
        assert isinstance(result, dict) and result.get("ok") is False, (name, result)
        assert "error" in result, (name, result)

    # Checksums: bf16 and fp32 tensors, deterministic, sensitive to one-bit changes.
    tensor = torch.randn(16, 8, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    assert probe._checksum(tensor) == probe._checksum(tensor.clone())
    flipped = tensor.clone()
    flipped[0, 0] = flipped[0, 0] + 2.0
    assert probe._checksum(tensor) != probe._checksum(flipped)
    assert probe._checksum(tensor.float()) == probe._checksum(tensor.clone().float())

    # Fixed-order reference: exact-representable partials sum identically in any
    # order; a catastrophic bf16 mix must expose order sensitivity in a *bf16
    # sequential* sum while the probe's fp32 accumulate stays order-invariant here.
    exact = [torch.full((4, 4), float(2**i), dtype=torch.bfloat16) for i in range(8)]
    orders = [tuple(range(8)), tuple(reversed(range(8)))]
    a, b = (probe._rank_order_reference(exact, order) for order in orders)
    assert torch.equal(a, b)

    mixed = [torch.full((2, 2), value, dtype=torch.bfloat16) for value in (256.0, 1.0, 1.0, -256.0, 1.0, 1.0, 1.0, 1.0)]
    fp32_fwd, fp32_rev = (probe._rank_order_reference(mixed, order) for order in orders)
    assert torch.equal(fp32_fwd, fp32_rev), "fp32 accumulation should absorb this magnitude gap"

    def bf16_sequential(parts, order):
        total = parts[order[0]].clone()
        for index in order[1:]:
            total = total + parts[index]
        return total

    bf16_fwd, bf16_rev = (bf16_sequential(mixed, order) for order in orders)
    assert not torch.equal(bf16_fwd, bf16_rev), "expected bf16 order sensitivity in the control"

    # Sizes-log trim arithmetic used by the harness.
    lines = list(range(1000))
    trimmed = lines[:200] + lines[-50:]
    assert len(trimmed) == 250 and trimmed[199] == 199 and trimmed[200] == 950

    print("probe check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
