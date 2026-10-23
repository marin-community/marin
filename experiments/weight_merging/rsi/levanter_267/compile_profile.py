# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive compiler memory evidence for the first production prefill kernels."""

import functools
import json
import time

import jax
from levanter.inference import snowball_prefill
from rigging.filesystem.storage_path import StoragePath


def instrument_prefill(output):
    for name in ("_cache_prefix", "_finish_prompt"):
        original = getattr(snowball_prefill, name)
        setattr(snowball_prefill, name, _profile_once(original, output, name))


def _profile_once(function, output, name):
    measured = False

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        nonlocal measured
        if measured:
            return function(*args, **kwargs)
        root = StoragePath(output)
        stem = f"compile-{name}-rank{jax.process_index()}"
        start = time.monotonic()
        lowered = None
        report = dict(kernel=name, rank=jax.process_index())
        try:
            lowered = function.lower(*args, **kwargs)
            report["cost_analysis"] = lowered.cost_analysis()
            (root / f"{stem}.json").write_text(json.dumps(report, indent=2))
            compiled = lowered.compile()
        except Exception as error:
            # The optimized buffer assignment may be unavailable after an OOM;
            # retain the lowered program rather than guessing at its buffers.
            if lowered is not None:
                (root / f"{stem}.hlo.txt").write_text(lowered.compiler_ir("hlo").as_hlo_text())
            report.update(error=str(error), compile_seconds=time.monotonic() - start)
            (root / f"{stem}.json").write_text(json.dumps(report, indent=2))
            raise
        stats = compiled.memory_analysis()
        fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
        memory = None if stats is None else {key: getattr(stats, key) for key in fields}
        report.update(memory=memory, compile_seconds=time.monotonic() - start)
        (root / f"{stem}.json").write_text(json.dumps(report, indent=2))
        print("COMPILE_MEMORY", json.dumps(report), flush=True)
        measured = True
        # Preserve JIT's static-argument handling and donated-call contract.
        return function(*args, **kwargs)

    return wrapped
