# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Dump flash_attn.cute sm90 backward score_mod / mask_mod / preprocess conventions (GPU)."""
import importlib
import inspect
import re


def _dump(modname, clsname):
    m = importlib.import_module(modname)
    c = getattr(m, clsname)
    src = c.__init__ and inspect.getsource(c)
    # print windows around score_mod_fn = partial(  and  score_mod_fn(  and mask_mod
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if re.search(
            r"partial\(\s*$|= partial\(|score_mod_fn\(|score_mod_bwd_fn\(|mask_mod\(|apply_mask|aux_tensors\b", ln
        ):
            lo = max(0, i - 1)
            hi = min(len(lines), i + 8)
            print(f"--- {clsname} @L{i} ---", flush=True)
            for j in range(lo, hi):
                print("   " + lines[j][:150], flush=True)


def main():
    try:
        pre = importlib.import_module("flash_attn.cute.flash_bwd_preprocess").FlashAttentionBackwardPreprocess
        print("PREPROCESS __call__:", inspect.signature(pre.__call__), flush=True)
    except Exception as e:
        print("preprocess sig err:", e, flush=True)
    for mod, cls in [
        ("flash_attn.cute.flash_bwd_sm90", "FlashAttentionBackwardSm90"),
    ]:
        try:
            _dump(mod, cls)
        except Exception as e:
            print(f"{cls} dump err: {type(e).__name__}: {e}", flush=True)
    print("INKLING_SIG2_DONE", flush=True)


if __name__ == "__main__":
    main()
