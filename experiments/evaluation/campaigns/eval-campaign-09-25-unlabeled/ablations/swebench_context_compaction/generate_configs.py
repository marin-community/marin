# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the canonical model and Harbor configs for the SWE-bench context grid."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent
MODEL_CONFIGS = CAMPAIGN / "model-configs"
BASE_HARBOR_CONFIG = CAMPAIGN / "harbor-configs" / "swebench-verified.yaml"
HARBOR_COMMIT = "39267af200a50ad06f5318535aa034f40f5ef1e5"


@dataclass(frozen=True)
class Model:
    key: str
    source: str


@dataclass(frozen=True)
class Arm:
    key: str
    context_tokens: int
    output_tokens: int
    summarize: bool


MODELS = (
    Model("qwen36", "Qwen-Qwen3.6-35B-A3B.yaml"),
    Model("gemma4", "google-gemma-4-26B-A4B-it.yaml"),
    Model("nemotron-lightning", "nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.yaml"),
    Model("snowball-v0.1", "open-athena-Snowball-Smartie-RL-2026.10.15.32-step14.yaml"),
)
ARMS = (
    Arm("65k-16k-no-summary", 65_536, 16_384, False),
    Arm("131k-16k-no-summary", 131_072, 16_384, False),
    Arm("32k-8k-summary", 32_768, 8_192, True),
    Arm("65k-16k-summary", 65_536, 16_384, True),
)
SUMMARIZATION_THRESHOLD_CHARACTERS = 8_192


def _write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def generate() -> None:
    """Write all exact configs from the pinned campaign sources and declared grid."""
    base_harbor = yaml.safe_load(BASE_HARBOR_CONFIG.read_text())
    for arm in ARMS:
        harbor = dict(base_harbor)
        harbor["agents"] = [dict(agent) for agent in base_harbor["agents"]]
        agent = harbor["agents"][0]
        agent["kwargs"] = dict(agent["kwargs"])
        agent["kwargs"]["model_info"] = {
            "max_input_tokens": arm.context_tokens,
            "max_output_tokens": arm.output_tokens,
        }
        if arm.summarize:
            agent["kwargs"]["summarization_threshold_chars"] = SUMMARIZATION_THRESHOLD_CHARACTERS
        else:
            agent["kwargs"].pop("summarization_threshold_chars", None)
        harbor["datasets"] = [dict(dataset) for dataset in base_harbor["datasets"]]
        harbor["datasets"][0][
            "registry_url"
        ] = f"https://raw.githubusercontent.com/marin-community/harbor/{HARBOR_COMMIT}/registry.json"
        _write_yaml(ROOT / "configs" / "harbor" / f"{arm.key}.yaml", harbor)

        for model in MODELS:
            document = yaml.safe_load((MODEL_CONFIGS / model.source).read_text())
            document["name"] = f"{document['name']}-swebench-{arm.key}"
            document.setdefault("serve", {})["max_model_len"] = arm.context_tokens
            document.setdefault("generation", {})["max_gen_toks"] = arm.output_tokens
            _write_yaml(ROOT / "configs" / "models" / model.key / f"{arm.key}.yaml", document)


if __name__ == "__main__":
    generate()
