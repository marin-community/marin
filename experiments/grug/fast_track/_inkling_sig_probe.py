# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Dump b28 preprocess + postprocess __call__ signatures (training env)."""
import importlib
import importlib.metadata as md
import inspect


def dump(modname, clsname, n):
    c = getattr(importlib.import_module(modname), clsname)
    src = inspect.getsource(c.__call__).splitlines()
    print(f"{clsname} params:", flush=True)
    for ln in src[:n]:
        s = ln.strip()
        if s and (":" in s or s.startswith("def") or s.startswith(")")):
            print(f"   {s[:120]}", flush=True)


def main():
    try:
        print("flash-attn-4 version:", md.version("flash-attn-4"), flush=True)
    except Exception as e:
        print("ver err:", e, flush=True)
    dump("flash_attn.cute.flash_bwd_preprocess", "FlashAttentionBackwardPreprocess", 16)
    dump("flash_attn.cute.flash_bwd_postprocess", "FlashAttentionBackwardPostprocess", 12)
    print("INKLING_B28_DONE", flush=True)


if __name__ == "__main__":
    main()
