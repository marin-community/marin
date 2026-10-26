# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared calibration inputs, also serialized across the isolated runtime boundary."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

HERE = Path("experiments/evaluation/serving_parity")
POLICY = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs")
MODEL = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
REVISION = "50fe2f488aee7730bbccc721024b35229075602e"
TASKS = ("gsm8k-0shot", "boolq")


class HardwareKind(StrEnum):
    H100 = "h100"
    V6E = "v6e-8"
    V5P = "v5p-8"
    V4 = "v4-32"


@dataclass(frozen=True)
class ProbeSettings:
    model: str = MODEL
    revision: str = REVISION
    max_model_len: int = 8192
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 16
    num_hidden_layers: int = 26
    full_attention_interval: int = 4


PROBE = ProbeSettings()


def frozen_items(task: str) -> list[dict]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    return manifest["gsm8k" if task == "gsm8k-0shot" else task]["items"]
