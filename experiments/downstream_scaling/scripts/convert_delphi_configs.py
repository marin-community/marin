# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Set ``architectures`` to ``Qwen3ForCausalLM`` in every Delphi checkpoint config.

The levanter HF export labels Delphi checkpoints ``architectures: ["LlamaForCausalLM"]``
even though the safetensors carry Qwen3-style ``q_norm``/``k_norm`` weights. vLLM/tpu-inference
dispatches on ``architectures``, so the misnamed field routes loads to a Llama3 class that
doesn't know those keys. This script rewrites the field in-place under a single GCS region.

Usage:
    uv run python experiments/downstream_scaling/scripts/convert_delphi_configs.py \
        --region eu-west4
    uv run python experiments/downstream_scaling/scripts/convert_delphi_configs.py \
        --region eu-west4 --apply
"""

from __future__ import annotations

import argparse
import json

import fsspec

from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS

TARGET_ARCH = "Qwen3ForCausalLM"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="GCS region suffix (e.g. eu-west4, us-east5)")
    parser.add_argument("--apply", action="store_true", help="Apply writes (default: dry-run)")
    args = parser.parse_args()

    bucket = f"marin-{args.region}"
    fs = fsspec.filesystem("gs")

    for slug, rel_path in DELPHI_CHECKPOINTS.items():
        base = f"{bucket}/{rel_path}"
        configs = sorted(p for p in fs.find(base) if p.endswith("/config.json"))
        if not configs:
            print(f"NONE   {slug}: no config.json under gs://{base}")
            continue
        for cp in configs:
            url = f"gs://{cp}"
            with fs.open(url, "r") as f:
                cfg = json.load(f)
            current = cfg.get("architectures")
            if current == [TARGET_ARCH]:
                print(f"OK     {url}")
                continue
            print(f"WRITE  {url}: {current} -> [{TARGET_ARCH!r}]")
            if not args.apply:
                continue
            cfg["architectures"] = [TARGET_ARCH]
            with fs.open(url, "w") as f:
                json.dump(cfg, f)


if __name__ == "__main__":
    main()
