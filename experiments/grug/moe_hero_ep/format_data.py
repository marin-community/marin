# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned, deduplicated format-SFT conversations through the shared renderer.

Input sources are already materialized canonical OpenAI-message JSONL files. No
source is downloaded or trusted implicitly. The resulting JSONL is directly
usable by ``ArtifactDatasetSpec`` in the shared SFT launcher.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Self

import datasets
import numpy as np
from huggingface_hub import HfApi
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.transform.conversation.transform_conversation import TransformSFTDatasetConfig, transform_row
from pydantic import Field, model_validator

from experiments.datasets.instruction import INSTRUCTION_DATASET_NAME_TO_CONFIG
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Record, digest

EOT_TEXT = "<|eot_id|>"
logger = logging.getLogger(__name__)
CATALOG_SOURCES = ("HuggingFaceH4/no_robots", "allenai/tulu-v2-sft-mixture", "teknium/OpenHermes-2.5")
HEADER_TEXTS = ("<|start_header_id|>", "<|end_header_id|>")


class SourceSpec(Record):
    slug: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    path: str
    revision: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trust_basis: str = Field(min_length=1)
    in_pretraining: bool
    weight: float = Field(gt=0, allow_inf_nan=False)
    max_examples: int = Field(gt=0)


class DataRecipe(Record):
    sources: tuple[SourceSpec, ...] = Field(min_length=1)
    tokenizer_path: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_snapshot_digest: str | None = None
    heldout_fraction: float = Field(gt=0, lt=1)
    split_seed: int
    max_tokens: int = Field(gt=1, le=16384)

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if len({source.slug for source in self.sources}) != len(self.sources):
            raise ValueError("Source slugs must be unique")
        return self


def shifted_chat_targets(processed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal inputs, next-token targets and assistant target weights."""
    ids = np.asarray(processed["input_ids"], dtype=np.int32)
    mask = np.asarray(processed["assistant_masks"], dtype=np.int32)
    if ids.shape != mask.shape or ids.ndim != 1:
        raise ValueError("Chat tokens and assistant masks must be aligned one-dimensional arrays")
    return ids[:-1], ids[1:], mask[1:]


def validate_chat_mask(processed: dict, tokenizer: MarinTokenizer) -> None:
    """Check atomic EOT supervision and exclusion of all role-header tokens."""
    ids = np.asarray(processed["input_ids"])
    mask = np.asarray(processed["assistant_masks"])
    eot = tokenizer.encode(EOT_TEXT)
    headers = [tokenizer.encode(text) for text in HEADER_TEXTS]
    if len(eot) != 1 or any(len(header) != 1 for header in headers):
        raise ValueError("The prepared tokenizer must have atomic Marin EOT and header tokens")
    if len(ids) < 2 or not mask[1:].any() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Conversation has no supervised causal targets or has an invalid mask")
    if not np.any((ids == eot[0]) & (mask == 1)):
        raise ValueError("Assistant EOT is not supervised")
    if any(np.any((ids == header[0]) & (mask == 1)) for header in headers):
        raise ValueError("Role headers must not be supervised")
    if ids[-1] != eot[0] or mask[-1] != 1:
        raise ValueError("Training conversations must finish with a supervised assistant EOT")


def valid_turn_sequence(messages: Sequence[dict]) -> bool:
    """Require complete user/assistant turns, allowing leading instructions and tool handoffs."""
    transitions = {
        None: {"system", "developer", "user"},
        "system": {"system", "developer", "user"},
        "developer": {"system", "developer", "user"},
        "user": {"assistant"},
        "assistant": {"user", "tool"},
        "tool": {"tool", "assistant"},
    }
    previous = None
    for message in messages:
        role = message["role"]
        if role not in transitions[previous]:
            return False
        previous = role
    return previous == "assistant"


def prepare_dataset(recipe: DataRecipe, output_directory: Path, tokenizer: MarinTokenizer | None = None) -> dict:
    """Write canonical train/heldout JSONL and exact source/token provenance.

    Deduplication and splitting use conversation content rather than source
    IDs or template options, so the same chat cannot leak across reasoning modes.
    Overlength conversations are excluded whole;
    assistant answers, reasoning and tool trajectories are never truncated.
    """
    tokenizer_path = Path(recipe.tokenizer_path)
    snapshot_digest = None
    if tokenizer_path.is_dir():
        snapshot_digest = digest(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(tokenizer_path.iterdir())
                if path.is_file()
            }
        )
        if recipe.tokenizer_snapshot_digest != snapshot_digest:
            raise ValueError("Local tokenizer content must match tokenizer_snapshot_digest in the recipe")
        tokenizer_reference = recipe.tokenizer_path
    else:
        if "@" in recipe.tokenizer_path:
            raise ValueError("Declare tokenizer revision separately from its Hub name")
        tokenizer_reference = f"{recipe.tokenizer_path}@{recipe.tokenizer_revision}"
    tokenizer = tokenizer or load_tokenizer(tokenizer_reference)
    processor = ChatProcessor(tokenizer, chat_template=MARIN_CHAT_TEMPLATE, mask_user_turns=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    destinations = [output_directory / name for name in ("train.jsonl", "heldout.jsonl", "manifest.json")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Dataset outputs already exist; choose a new immutable artifact directory")
    seen: set[str] = set()
    counts: dict[str, Counter] = {}
    rows: dict[str, list[dict]] = {"train": [], "heldout": []}
    token_totals: dict[str, Counter] = {"train": Counter(), "heldout": Counter()}
    for source in recipe.sources:
        raw = Path(source.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError(f"Source bytes do not match the pinned digest: {source.slug}")
        counts[source.slug] = counter = Counter()
        # Content-hash order makes selection independent of input shard order.
        candidates = []
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            content = {key: row[key] for key in ("messages", "system", "chat_template_kwargs") if key in row}
            if "messages" not in content:
                raise ValueError(f"Source {source.slug} is missing canonical messages")
            kwargs = content.get("chat_template_kwargs") or {}
            if any(
                key in kwargs for key in ("chat_template", "tokenize", "return_dict", "return_assistant_tokens_mask")
            ):
                raise ValueError("Dataset rows may not override the shared template or mask computation")
            if not content["messages"] or content["messages"][-1]["role"] != "assistant":
                raise ValueError("Dataset rows must end with an assistant response")
            identity = digest({key: value for key, value in content.items() if key != "chat_template_kwargs"})
            candidates.append((identity, content))
        for identity, content in sorted(candidates, key=lambda pair: pair[0]):
            counter["input"] += 1
            if identity in seen:
                counter["duplicates"] += 1
                continue
            if not valid_turn_sequence(content["messages"]):
                counter["excluded_invalid_turns"] += 1
                continue
            processed = processor([content])[0]
            validate_chat_mask(processed, tokenizer)
            if len(processed["input_ids"]) > recipe.max_tokens:
                counter["excluded_long"] += 1
                continue
            if counter["selected"] >= source.max_examples:
                counter["excluded_quota"] += 1
                continue
            seen.add(identity)
            split_value = int(digest({"seed": recipe.split_seed, "conversation": identity}), 16) / 2**256
            split = "heldout" if split_value < recipe.heldout_fraction else "train"
            row = {"id": identity, "source": source.slug, **content}
            rows[split].append(row)
            counter["selected"] += 1
            counter[split] += 1
            token_totals[split]["non_padding"] += len(processed["input_ids"])
            token_totals[split]["assistant_supervised"] += int(processed["assistant_masks"][1:].sum())
    if not rows["train"] or not rows["heldout"]:
        raise ValueError("Selection produced an empty train or heldout split; enlarge the trusted source sample")
    source_outputs: dict[str, dict[str, object]] = {}
    for source in recipe.sources:
        source_outputs[source.slug] = {"weight": source.weight}
        for split in rows:
            source_rows = [row for row in rows[split] if row["source"] == source.slug]
            if not source_rows:
                continue
            source_directory = output_directory / split
            source_directory.mkdir(exist_ok=True)
            destination = source_directory / f"{source.slug}.jsonl"
            with destination.open("x") as stream:
                stream.writelines(json.dumps(row, sort_keys=True) + "\n" for row in source_rows)
            source_outputs[source.slug][split] = {
                "path": str(destination),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "examples": len(source_rows),
            }
    for split, destination in zip(rows, destinations[:2], strict=True):
        destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[split]))
    manifest = {
        "recipe": recipe.model_dump(),
        "template_sha256": hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest(),
        "tokenizer_vocabulary_sha256": digest(tokenizer.get_vocab()),
        "tokenizer_revision": recipe.tokenizer_revision,
        "tokenizer_snapshot_digest": snapshot_digest,
        "tokenizer_reference": tokenizer_reference,
        "sources": {slug: dict(count) for slug, count in counts.items()},
        "source_outputs": source_outputs,
        "tokens": {split: dict(count) for split, count in token_totals.items()},
        "outputs": {
            split: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "examples": len(rows[split]),
            }
            for split, path in zip(rows, destinations[:2], strict=True)
        },
        "split_policy": "content-hash with seeded heldout assignment; whole-conversation exclusion",
        "token_count_policy": "non-padding includes prompts; supervised counts causal assistant targets",
        "packing": "Use SFTSpec(pack=True) conversation isolation; do not concatenate raw token arrays",
    }
    manifest["manifest_digest"] = digest(manifest)
    destinations[2].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class CatalogSelection(Record):
    name: str
    max_scanned_rows: int = Field(gt=0, le=10000)
    weight: float = Field(gt=0, allow_inf_nan=False)
    trust_basis: str = Field(min_length=1)
    in_pretraining: bool


def materialize_catalog_sources(
    selections: Sequence[CatalogSelection], output_directory: Path
) -> tuple[SourceSpec, ...]:
    """Stream bounded prefixes of explicitly selected existing SFT catalog sources.

    Only the listed simple-chat sources are eligible here. Their repository
    adapters are reused with text replacements disabled to retain content. Tool
    and reasoning trajectories should use reviewed canonical-message artifacts
    to preserve fields the older adapters do not represent. This performs network
    I/O only when this explicit materialization command is invoked.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    sources = []
    for selection in selections:
        if selection.name not in CATALOG_SOURCES:
            raise ValueError(f"Source is outside the reviewed simple-chat catalog selection: {selection.name}")
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[selection.name]
        revision = HfApi().dataset_info(config.hf_dataset_id, revision=config.revision).sha
        if not revision:
            raise ValueError("Hub did not return an immutable source commit")
        slug = selection.name.replace("/", "--")
        destination = output_directory / f"{slug}.jsonl"
        adapter = config.adapter.copy()
        adapter.replacements = {}
        transform_config = TransformSFTDatasetConfig(
            source=config.hf_dataset_id,
            revision=revision,
            output_path=str(output_directory),
            metadata_columns=config.metadata_columns,
            adapter=adapter,
        )
        stream = datasets.load_dataset(config.hf_dataset_id, revision=revision, split="train", streaming=True)
        accepted = 0
        with destination.open("x") as output:
            for row in islice(stream, selection.max_scanned_rows):
                transformed = transform_row(row, transform_config, adapter)
                if transformed is None:
                    continue
                record = transformed.model_dump()
                canonical = {key: record[key] for key in ("messages", "system", "chat_template_kwargs") if key in record}
                if not canonical["messages"] or canonical["messages"][-1]["role"] != "assistant":
                    continue
                output.write(json.dumps(canonical, sort_keys=True) + "\n")
                accepted += 1
        if not accepted:
            raise ValueError(f"No complete assistant conversations in the bounded source sample: {selection.name}")
        sources.append(
            SourceSpec(
                slug=slug,
                path=str(destination),
                revision=revision,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                trust_basis=selection.trust_basis,
                in_pretraining=selection.in_pretraining,
                weight=selection.weight,
                max_examples=accepted,
            )
        )
        logger.info("Materialized %s conversations from %s at %s", accepted, selection.name, revision)
    with (output_directory / "sources.json").open("x") as output:
        output.write(json.dumps([source.model_dump() for source in sources], indent=2) + "\n")
    return tuple(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe", type=Path, help="Pinned DataRecipe JSON, or CatalogSelection list with --materialize-catalog"
    )
    parser.add_argument(
        "--materialize-catalog", action="store_true", help="Explicit bounded network source materialization"
    )
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    if args.materialize_catalog:
        materialize_catalog_sources(
            [CatalogSelection.model_validate(row) for row in json.loads(args.recipe.read_text())], args.output_directory
        )
        return
    manifest = prepare_dataset(DataRecipe.model_validate_json(args.recipe.read_text()), args.output_directory)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
