# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper so `iris job run` can launch the vLLM smoke test."""

from marin.inference.vllm_smoke_test import main

if __name__ == "__main__":
    raise SystemExit(main())
