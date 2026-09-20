# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Dump sm90 score_mod partial binding via repr (GPU)."""
import importlib
import inspect


def main():
    try:
        pre = importlib.import_module("flash_attn.cute.flash_bwd_preprocess").FlashAttentionBackwardPreprocess
        print("PREPROCESS_SIG:", repr(str(inspect.signature(pre.__call__))), flush=True)
    except Exception as e:
        print("preprocess err:", repr(str(e)), flush=True)
    m = importlib.import_module("flash_attn.cute.flash_bwd_sm90")
    src = inspect.getsource(m.FlashAttentionBackwardSm90).splitlines()
    for i, ln in enumerate(src):
        if "score_mod_fn_cur = partial(" in ln:
            block = "\\n".join(x.strip() for x in src[i : i + 10])
            print("SCOREMOD_PARTIAL:", repr(block), flush=True)
        if "score_mod_bwd_fn_cur = partial(" in ln:
            block = "\\n".join(x.strip() for x in src[i : i + 10])
            print("SCOREMODBWD_PARTIAL:", repr(block), flush=True)
    print("INKLING_SIG4_DONE", flush=True)


if __name__ == "__main__":
    main()
