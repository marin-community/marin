# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a separately addressable 65K YaRN candidate from format-SFT weights."""

import argparse
import dataclasses
import json
from pathlib import Path

import draccus
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig
from rigging.filesystem.buckets import filesystem_for

from experiments.grug.moe_hero_ep.averaging import read_prepared_provenance
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.yarn import yarn_frequencies

SOURCE_CONTEXT = 16384
TARGET_CONTEXT = 65536


@dataclasses.dataclass(frozen=True)
class ContextExtensionConfig:
    source_checkpoint: str
    metadata_digest: str
    model: GrugModelConfig
    source_revision: str
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0


def extended_model(config: ContextExtensionConfig) -> GrugModelConfig:
    """Extend the 16K format candidate by four while retaining Hero attention semantics."""
    if config.model.max_seq_len != SOURCE_CONTEXT or config.model.yarn is not None:
        raise ValueError("Context extension requires the original 16K format-SFT model configuration")
    if config.model.rope.scaling_factor is not None:
        raise ValueError("Do not combine YaRN with an existing rotary scaling intervention")
    yarn = YarnRotaryEmbeddingsConfig(
        theta=config.model.rope.theta,
        factor=TARGET_CONTEXT / SOURCE_CONTEXT,
        original_max_position_embeddings=SOURCE_CONTEXT,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        mscale=config.mscale,
    )
    # Validate the configuration and keep the rotary half, global rope-free
    # layers, window width and QK multiplier identical to the source candidate.
    result = dataclasses.replace(config.model, max_seq_len=TARGET_CONTEXT, yarn=yarn)
    yarn_frequencies(result.inferred_head_dim // 2, yarn)
    result.to_hf_config(result.vocab_size)
    return result


def prepare_extension(config: ContextExtensionConfig, destination: Path) -> dict:
    """Pin E to D's immutable arrays and emit its native and serving configuration.

    YaRN changes forward configuration without changing parameter shapes. No
    weight arrays are copied or read here. E remains unvalidated until native
    short/16K/65K evaluation and serving parity succeed.
    """
    model = extended_model(config)
    fs, source = filesystem_for(config.source_checkpoint)
    with fs.open(f"{source.rstrip('/')}/metadata.json") as stream:
        metadata = json.load(stream)
    if digest(metadata) != config.metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Context extension source metadata changed or is not permanent")
    sft = metadata.get("hero_sft", {})
    if sft.get("candidate") != "D" or metadata.get("merge_base") != "BF16-rounded candidate C":
        raise ValueError("Context extension requires an ordinary merged candidate-D export")
    base = sft["base"]
    prepared = read_prepared_provenance(base["checkpoint"], base["metadata_digest"])
    if prepared["candidate"] != "C":
        raise ValueError("The format checkpoint does not reference a prepared candidate-C base")
    source_model = draccus.encode(draccus.decode(GrugModelConfig, prepared["model"]))
    if digest(source_model) != digest(draccus.encode(config.model)):
        raise ValueError("Context-extension forward settings differ from the format checkpoint's provenanced model")
    if len(config.source_revision) != 40 or any(char not in "0123456789abcdef" for char in config.source_revision):
        raise ValueError("Context extension requires an immutable implementation revision")
    payload = {
        "candidate": "E",
        "parent_candidate": "D",
        "source_checkpoint": config.source_checkpoint,
        "source_metadata_digest": config.metadata_digest,
        "source_revision": config.source_revision,
        "source_model_digest": digest(draccus.encode(config.model)),
        "model": draccus.encode(model),
        "hf_config": model.to_hf_config(model.vocab_size).to_dict(),
        "weights": "immutable reference to D; no weight or optimizer changes",
        "validation_status": "pending",
        "required_evaluations": [
            "D versus E on identical short and actual 16K prompts, loss, content and routing gates",
            "E on actual prompts of at least 65000 tokens within the 65536-token context",
            "native-to-serving logits and generation parity with the same partial/interleaved YaRN",
        ],
        "training": "No context-extension training; evaluate configuration-only extension first",
    }
    payload["manifest_digest"] = digest(payload)
    with destination.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="ContextExtensionConfig JSON with the merged D checkpoint pin")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_extension(draccus.decode(ContextExtensionConfig, json.loads(args.config.read_text())), args.destination)


if __name__ == "__main__":
    main()
