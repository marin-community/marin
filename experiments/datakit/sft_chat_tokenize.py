# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare rendered SFT text, then tokenize it with an immutable Marin tokenizer.

Run the text stage while the tokenizer is being finalized. Each token step waits
for its own normalized text, so completed sources can tokenize while other text
steps are still running. Tokenization requires an immutable object-storage
artifact with the expected chat template and reserved token IDs. The artifact
path is part of the step identity, so publish changed tokenizer bytes to a new
versioned path.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from levanter.data.text.datasets import PackedTokenDataset
from levanter.store.cache import TreeCache
from levanter.tokenizers import TokenizerBackend, load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step
from marin.processing.tokenize.store_builder import LevanterStoreData, build_levanter_store_step
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers.utils.chat_template_utils import render_jinja_template

EXPECTED_VOCAB_SIZE = 128256
CONTEXT_LENGTH = 262144
STORE_MAX_WORKERS = 256
STORE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="32g", disk="10g")
MIX_PATH = Path(__file__).parents[1] / "grug_sft" / "special_token_1000_mix.json"
EXPECTED_TOKEN_IDS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_think|>": 128002,
    "<|end_think|>": 128003,
    "<tool_call>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009,
    "</tool_call>": 128011,
}


def sft_text_steps() -> list[StepSpec]:
    """Return rendered and normalized text steps for every registered SFT source."""
    return [source.normalized for source in all_sft_sources().values()]


def sft_token_steps(tokenizer: str) -> list[StepSpec]:
    """Return tokenizer-specific attribute steps over the normalized SFT text."""
    return [
        tokenize_attributes_step(
            name=f"datakit/tokenize/sft/{source.name}",
            train_normalize=source.normalized,
            tokenizer=tokenizer,
            tokenizer_backend=TokenizerBackend.HF,
        )
        for source in all_sft_sources().values()
    ]


def selected_source_names() -> tuple[str, ...]:
    """Return the sources retained by the fixed 48-hour SFT allocation."""
    plan = json.loads(MIX_PATH.read_text())
    names = tuple(sorted(plan["allocations_tokens"]))
    missing = sorted(set(names) - set(all_sft_sources()))
    if missing:
        raise ValueError(f"SFT allocation contains unregistered sources: {missing}")
    return names


def sft_store_steps(tokenizer: str) -> dict[str, StepSpec]:
    """Build one Levanter store from each selected source's token attributes."""
    token_steps = {
        source.name: step for source, step in zip(all_sft_sources().values(), sft_token_steps(tokenizer), strict=True)
    }
    return {
        name: build_levanter_store_step(
            name=f"datakit/store/sft/{name}",
            tokenize_steps=[token_steps[name]],
            max_workers=STORE_MAX_WORKERS,
            worker_resources=STORE_WORKER_RESOURCES,
        )
        for name in selected_source_names()
    }


def _write_store_manifest(output_path: str, stores: dict[str, StepSpec], tokenizer: str) -> dict[str, int | str]:
    manifest = {}
    total_tokens = 0
    total_packed_sequences = 0
    for name, step in stores.items():
        store = read_artifact(step.output_path, LevanterStoreData)
        if store.tokenizer != tokenizer:
            raise ValueError(f"{name} store uses tokenizer {store.tokenizer}, expected {tokenizer}")
        stats = store.splits.get("train")
        if stats is None or stats.total_elements <= 0 or stats.total_tokens <= 0:
            raise ValueError(f"{name} has no nonempty train store")
        cache = TreeCache.load(stats.path, {"input_ids": np.zeros(0, dtype=np.int32)})
        packed_sequences = len(
            PackedTokenDataset(
                cache,
                Axis("position", CONTEXT_LENGTH),
                max_segments_per_example=CONTEXT_LENGTH,
                slice_strategy="left",
            ).as_sync_dataset()
        )
        if packed_sequences <= 0:
            raise ValueError(f"{name} has no packed training sequences")
        total_tokens += stats.total_tokens
        total_packed_sequences += packed_sequences
        manifest[name] = {
            "path": step.output_path,
            "cache_path": store.cache_path,
            "tokenizer": tokenizer,
            "max_length": CONTEXT_LENGTH,
            "seed": 0,
            "sources": {
                name: {
                    "conversations": stats.total_elements,
                    "tokens": stats.total_tokens,
                    "overlength_conversations": 0,
                    "overlength_tokens": 0,
                }
            },
            "packed_sequences": packed_sequences,
        }

    stores_path = prefix_join(output_path, "stores.json")
    StoragePath(stores_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "stores_path": stores_path,
        "source_count": len(manifest),
        "tokens": total_tokens,
        "packed_sequences": total_packed_sequences,
    }


def sft_store_manifest_step(tokenizer: str, output_path: str) -> StepSpec:
    """Return the selected source stores and their training manifest."""
    stores = sft_store_steps(tokenizer)
    return StepSpec(
        name="datakit/store/sft-1000-step-manifest",
        deps=list(stores.values()),
        fn=lambda output_path: _write_store_manifest(output_path, stores, tokenizer),
        hash_attrs={"context_length": CONTEXT_LENGTH, "tokenizer": tokenizer, "version": "2026.09.19"},
        override_output_path=output_path,
    )


def verify_tokenizer(tokenizer: str) -> None:
    """Check the immutable tokenizer's vocabulary, reserved IDs, and chat template."""
    if "://" not in tokenizer:
        raise ValueError("The SFT tokenizer must be an immutable object-storage URI")
    loaded = load_tokenizer(tokenizer)
    if len(loaded) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"{tokenizer} has vocabulary size {len(loaded)}, expected {EXPECTED_VOCAB_SIZE}")
    for token, expected_id in EXPECTED_TOKEN_IDS.items():
        actual_ids = loaded.encode(token, add_special_tokens=False)
        if actual_ids != [expected_id]:
            raise ValueError(f"{token} must encode as [{expected_id}], got {actual_ids}")
    if loaded.bos_token_id != EXPECTED_TOKEN_IDS["<|begin_of_text|>"]:
        raise ValueError(f"Unexpected BOS token ID: {loaded.bos_token_id}")
    if loaded.eos_token_id != EXPECTED_TOKEN_IDS["<|end_of_text|>"]:
        raise ValueError(f"Unexpected EOS token ID: {loaded.eos_token_id}")

    conversation = [{"role": "user", "content": "template probe"}]
    actual = loaded.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    expected, _ = render_jinja_template(
        conversations=[conversation],
        chat_template=MARIN_CHAT_TEMPLATE,
        bos_token=loaded.bos_token or "",
        eos_token=loaded.eos_token or "",
        add_generation_prompt=False,
    )
    if actual != expected[0]:
        raise ValueError(f"{tokenizer} has an unexpected chat template")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("text", "tokens", "stores"))
    parser.add_argument("--tokenizer", help="Immutable object-storage tokenizer URI for token or store stages")
    parser.add_argument("--manifest-output", help="Object-storage directory for stores.json")
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--run", action="store_true", help="Execute the steps; otherwise print the plan")
    args = parser.parse_args()

    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")
    if args.stage in ("tokens", "stores") and not args.tokenizer:
        parser.error(f"the {args.stage} stage requires --tokenizer")
    if args.stage == "stores" and not args.manifest_output:
        parser.error("the stores stage requires --manifest-output")

    text_steps = sft_text_steps()
    print(f"{args.stage}: {len(text_steps)} SFT sources", flush=True)
    if args.stage == "text":
        if args.run:
            StepRunner().run(text_steps, max_concurrent=args.max_concurrent)
        return

    verify_tokenizer(args.tokenizer)
    steps = (
        sft_token_steps(args.tokenizer)
        if args.stage == "tokens"
        else [sft_store_manifest_step(args.tokenizer, args.manifest_output)]
    )
    print(f"{args.stage}: {len(steps)} terminal step(s)", flush=True)
    if args.run:
        StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
