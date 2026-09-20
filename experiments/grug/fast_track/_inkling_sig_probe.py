# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Dump FlashAttentionBackwardPreprocess.__call__ SOURCE (definitive signature) (GPU)."""
import importlib
import inspect


def main():
    c = importlib.import_module("flash_attn.cute.flash_bwd_preprocess").FlashAttentionBackwardPreprocess
    src = inspect.getsource(c.__call__)
    lines = src.splitlines()
    # print the def signature block (up to the closing of the param list) + any use_padded_offsets / mScaleP refs
    for i, ln in enumerate(lines):
        if (
            i < 40
            or ("use_padded_offsets" in ln)
            or ("mScaleP" in ln)
            or ("softmax_scale" in ln)
            or ("self." in ln and "offset" in ln.lower())
        ):
            print(f"C{i}: {ln.rstrip()[:150]}", flush=True)
    print("INKLING_SIG6_DONE", flush=True)


if __name__ == "__main__":
    main()
