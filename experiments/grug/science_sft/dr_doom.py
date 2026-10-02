# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train two packed passes over a frozen science SFT snapshot from Dr Doom."""

import argparse
import hashlib
import json
import logging

from levanter.tokenizers import load_tokenizer
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datasets.science_forward_converted import SOURCE_NAME
from experiments.grug.science_sft.launch import CONTEXT, ScienceSftRecipe, build_run_config
from experiments.grug.science_sft.train import run_grug

MODEL_REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
MODEL_REVISION = "5b5705e8345bb111de03b3f0f9a00c8041e57db4"
TOKENIZER = f"{MODEL_REPO}@{MODEL_REVISION}"
MODEL_PATH = (
    "s3://marin-us-east-02a/marin/users/benfeuer/checkpoints/"
    "antidoom-ftpo-dapo-mixed-long-smoke/2026.10.02.2/exports/global_step_2/policy"
)
CHAT_TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
RECIPE = ScienceSftRecipe(
    tokenizer=TOKENIZER,
    model_path=MODEL_PATH,
    output_root="s3://marin-us-east-02a/marin/users/benfeuer/grug-science-converted-sft-runs",
    identity_prefix="science-forward-dr-doom-two-epochs",
    epochs=2,
    qk_mult=1.75,
    tags=("science-forward", "converted-chat", "dr-doom", "two-epochs", "router-bias:per-step-qb"),
)


def pinned_chat_template() -> str:
    """Read and verify the model's own chat template before tokenizing or training."""
    template = load_tokenizer(TOKENIZER).chat_template
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != CHAT_TEMPLATE_SHA256:
        raise ValueError("Pinned Dr Doom chat template differs from the audited template")
    if "<|start_think|>" not in template or '"/think"' not in template:
        raise ValueError("Pinned Dr Doom template lacks expected thinking-token formatting")
    return template


def prepare(snapshot_root: str, output_path: str, num_shards: int, max_workers: int) -> SftTokenStore:
    """Tokenize a sealed snapshot with the model template and preserve masked turns."""
    configure_coreweave_s3()
    root = StoragePath(snapshot_root)
    if not (root / "_READY").exists():
        raise ValueError(f"Science snapshot is not verified: {snapshot_root}")
    manifest = json.loads((root / "snapshot-manifest.json").read_text())
    if manifest["destination"] != snapshot_root or manifest["conversations"] < 1:
        raise ValueError("Science snapshot manifest is inconsistent")
    store = build_sft_store(
        [SftInput(SOURCE_NAME, str(root / "outputs/main"))],
        output_path=output_path,
        tokenizer=TOKENIZER,
        chat_template=pinned_chat_template(),
        max_length=CONTEXT,
        seed=0,
        num_shards=num_shards,
        max_workers=max_workers,
    )
    counts = store.sources[SOURCE_NAME]
    if counts.conversations + counts.overlength_conversations != manifest["conversations"]:
        raise ValueError("Tokenized conversation count differs from the sealed snapshot")
    if counts.overlength_conversations or counts.assistant_tokens == 0:
        raise ValueError("SFT snapshot contains overlength conversations or no assistant targets")
    logging.info("Prepared %d conversations, %d assistant tokens, %d packed sequences", counts.conversations,
                 counts.assistant_tokens, store.packed_sequences)
    return store


def launch(store_path: str, version: str) -> None:
    configure_coreweave_s3()
    store = SftTokenStore.raw_load(store_path)
    template = pinned_chat_template()
    if store.chat_template != template:
        raise ValueError("SFT store chat template differs from the pinned Dr Doom model")
    if StoragePath(prefix_join(MODEL_PATH, "chat_template.jinja")).read_text() != template:
        raise ValueError("Regional Dr Doom export chat template differs from the pinned model")
    model_config = json.loads(StoragePath(prefix_join(MODEL_PATH, "config.json")).read_text())
    if model_config["qk_mult"] != RECIPE.qk_mult or model_config["vocab_size"] != 128_256:
        raise ValueError("Regional Dr Doom export architecture differs from the training recipe")
    if not StoragePath(prefix_join(MODEL_PATH, "model.safetensors.index.json")).exists():
        raise FileNotFoundError(f"Missing regional model export: {MODEL_PATH}")
    run_grug(build_run_config(store, version, RECIPE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--snapshot-root", required=True)
    prep.add_argument("--output-path", required=True)
    prep.add_argument("--num-shards", type=int, required=True)
    prep.add_argument("--max-workers", type=int, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--store-path", required=True)
    train.add_argument("--version", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "prepare":
        prepare(args.snapshot_root, args.output_path, args.num_shards, args.max_workers)
    else:
        launch(args.store_path, args.version)


if __name__ == "__main__":
    main()
