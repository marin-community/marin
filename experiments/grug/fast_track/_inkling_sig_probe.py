# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Dump flash_attn.cute backward postprocess signature (GPU)."""
import importlib
import inspect


def main():
    p = importlib.import_module("flash_attn.cute.flash_bwd_postprocess").FlashAttentionBackwardPostprocess
    print("POSTPROCESS_SIG:", repr(str(inspect.signature(p.__call__))), flush=True)
    print("INKLING_SIG5_DONE", flush=True)


if __name__ == "__main__":
    main()
