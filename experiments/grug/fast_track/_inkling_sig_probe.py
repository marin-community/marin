# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-rank dump of flash-attn version + preprocess __call__ param names (detect per-node mismatch)."""
import importlib
import inspect
import os


def main():
    rank = os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "?")
    try:
        fa = importlib.import_module("flash_attn")
        ver = getattr(fa, "__version__", "?")
    except Exception as e:
        ver = f"err:{e}"
    pre = importlib.import_module("flash_attn.cute.flash_bwd_preprocess").FlashAttentionBackwardPreprocess
    # param names of __call__ from the source (cute.jit-safe: read the def line region)
    src = inspect.getsource(pre.__call__).splitlines()
    params = [
        ln.strip().split(":")[0].strip()
        for ln in src[1:20]
        if (":" in ln and "cute.Tensor" in ln) or "Optional" in ln or "CUstream" in ln
    ]
    hasscale = any("mScaleP" in ln or "softmax_scale" in ln for ln in src[:20])
    path = getattr(pre, "__module__", "?")
    print(
        f"RANK{rank} fa={ver} module={path} n_params={len(params)} mScaleP_or_scale_in_sig={hasscale} params={params}",
        flush=True,
    )
    print("INKLING_SIG7_DONE", flush=True)


if __name__ == "__main__":
    main()
