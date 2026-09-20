# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Wikipedia-as-QA (wiki_to_rcqa) finite pool and its held-out cache for the TPP-10 sweep.

The Dolmino pool stores this component as thousands of small zstd shards, so the pool is packed
from four groups of 350 pinned files: each group is one packed part with a quarter of the
190,316,544-token quota, read file after file until the quota is met. Parent permutation, matched
subset, tokenizer and geometry are the frozen ones. Evaluations reuse the instruction sweep's five
QA-format caches and add a held-out rcQA stream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from levanter.store.cache import CacheMetadata, consolidate_shard_cache_ledgers
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as original
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as previous
from experiments.domain_phase_mix import prepare_tpp10_instruction_sweep as instruction
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps, persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("tpp10_rcqa_sweep_assets")
VERSION = "2026.09.19"
DOMAIN = "wiki_to_rcqa"
HELDOUT = f"{DOMAIN}/heldout-tpp10"
HELDOUT_TOKENS = instruction.HELDOUT_TOKENS
QA_EVALS = instruction.QA_EVALS
ON_TARGET_METRICS = (*(f"eval/qa/{name}-tpp10/bpb" for name in QA_EVALS), f"eval/{HELDOUT}/bpb")
CPU = instruction.CPU


@dataclass(frozen=True)
class GroupedZstdRecipe:
    """A raw recipe whose sources are read in consecutive groups, one packed part per group."""

    raw: original.RawRecipe
    group_size: int
    implementation_sha256: str


def grouped_documents(sources: tuple[original.Source, ...]) -> Iterator[dict]:
    """Stream the pinned zstd files of one group in order; the token quota stops the stream."""
    for source in sources:
        documents = instruction.zstd_documents(source, None)
        try:
            yield from documents
        finally:
            documents.close()


def prepare_grouped_zstd(recipe: GroupedZstdRecipe) -> None:
    experiment.require_central1()
    if recipe.implementation_sha256 != file_sha256(Path(__file__)):
        raise ValueError("Grouped zstd preparation code differs from its recipe")
    raw = recipe.raw
    if raw.implementation_sha256 != file_sha256(Path(original.__file__)):
        raise ValueError("Frozen tokenization code differs from its recipe")
    if raw.design_sha256 != experiment.load_design()["design_sha256"] or raw.tokens is None:
        raise ValueError("Finite grouped pool requires the frozen geometry and an exact quota")
    if recipe.group_size <= 0 or len(raw.sources) % recipe.group_size:
        raise ValueError("Sources must form whole groups")
    groups = [raw.sources[i : i + recipe.group_size] for i in range(0, len(raw.sources), recipe.group_size)]
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(raw.tokenizer)).metadata)
    root = raw.cache_path + "/" + raw.split
    paths, receipts = [], []
    for i, group in enumerate(groups):
        quota = raw.tokens // len(groups) + (i < raw.tokens % len(groups))
        path = f"{root}/parts/{i:03d}"
        docs = grouped_documents(group)
        try:
            receipt = original.write_part(
                original.token_records(docs, tokenizer_path=raw.tokenizer, quota=quota),
                path,
                metadata=metadata,
                identity={
                    "recipe_sha256": canonical_sha256(asdict(recipe)),
                    "sources": [asdict(source) for source in group],
                    "quota": quota,
                },
                expected_tokens=quota,
            )
        finally:
            docs.close()
        paths.append(path)
        receipts.append(receipt)
        logger.info("Prepared %s group %d/%d (%d files)", raw.component, i + 1, len(groups), len(group))
    consolidate_shard_cache_ledgers(paths, root, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata)
    tokens = original.cache_token_count(root, metadata)
    if tokens != raw.tokens:
        raise ValueError("Grouped packed pool differs from the prescribed quota")
    persist_submission_plan(
        {
            "design_sha256": raw.design_sha256,
            "recipe_sha256": canonical_sha256(asdict(recipe)),
            "tokens": tokens,
            "part_receipts": receipts,
        },
        raw.cache_path + "/receipt.json",
    )


def _inventory() -> dict:
    return json.loads((ASSETS / "sources.json").read_text())


def _sources(key: str) -> tuple[original.Source, ...]:
    return tuple(original.Source(**row) for row in _inventory()[key]["selected"])


def data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Raw pool (four grouped parts), permuted parent and matched subset, with the frozen index draws."""
    inventory = _inventory()
    original_hash = file_sha256(Path(original.__file__))
    grouped_hash = file_sha256(Path(__file__))
    sources = _sources(DOMAIN)
    group_size = int(inventory[DOMAIN]["group_size"])
    environment = {"MARIN_PREFIX": experiment.PREFIX}

    def raw_config(ctx: StepContext) -> GroupedZstdRecipe:
        recipe = original.RawRecipe(
            DOMAIN,
            sources,
            experiment.PARENT_SEQUENCES * experiment.SEQ_LEN,
            "train",
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            original_hash,
        )
        return GroupedZstdRecipe(recipe, group_size, grouped_hash)

    raw = ArtifactStep(
        name=f"tpp10_domain_sweeps/{DOMAIN}/raw",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(prepare_grouped_zstd, resources=CPU, env_vars=environment),
        build_config=raw_config,
    )
    raw = replace(raw, expected_fingerprint=raw.fingerprint())

    def parent_config(ctx: StepContext) -> original.ParentRecipe:
        return original.ParentRecipe(
            ctx.artifact_path(raw), ctx.output_path, experiment.TOKENIZER, design["design_sha256"], original_hash
        )

    parent = ArtifactStep(
        name=f"tpp10_domain_sweeps/{DOMAIN}/parent",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(original.prepare_parent, resources=CPU, env_vars=environment),
        build_config=parent_config,
        deps=(raw,),
    )
    parent = replace(parent, expected_fingerprint=parent.fingerprint())

    def subset_config(ctx: StepContext) -> original.SubsetRecipe:
        return original.SubsetRecipe(
            ctx.artifact_path(parent),
            experiment.SUBSET_SEEDS[0],
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            original_hash,
        )

    subset = ArtifactStep(
        name=f"tpp10_domain_sweeps/{DOMAIN}/matched",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(original.prepare_subset, resources=CPU, env_vars=environment),
        build_config=subset_config,
        deps=(parent,),
    )
    return {
        f"{DOMAIN}/raw": raw,
        f"{DOMAIN}/parent": parent,
        f"{DOMAIN}/matched": replace(subset, expected_fingerprint=subset.fingerprint()),
    }


def evaluation_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """The instruction sweep's five QA caches plus a held-out rcQA stream (one grouped part)."""
    steps = {name: step for name, step in instruction.evaluation_steps(design).items() if name.startswith("qa/")}
    sources = _sources(f"{DOMAIN}_heldout")
    original_hash = file_sha256(Path(original.__file__))
    grouped_hash = file_sha256(Path(__file__))

    def heldout_config(ctx: StepContext) -> GroupedZstdRecipe:
        recipe = original.RawRecipe(
            HELDOUT,
            sources,
            HELDOUT_TOKENS,
            "validation",
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            original_hash,
        )
        return GroupedZstdRecipe(recipe, len(sources), grouped_hash)

    heldout = ArtifactStep(
        name=HELDOUT,
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(prepare_grouped_zstd, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
        build_config=heldout_config,
    )
    steps[HELDOUT] = replace(heldout, expected_fingerprint=heldout.fingerprint())
    return steps


def evaluation_paths(design: dict) -> dict[str, str]:
    """Held-out evaluation caches: the seven Uncheatable components, five QA sets and the rcQA stream."""
    return {
        **previous.evaluation_paths(),
        **{name: step.path(experiment.PREFIX) for name, step in evaluation_steps(design).items()},
    }


def verify_evaluation_caches(design: dict, steps: dict[str, ArtifactStep[TokenizedCache]]) -> dict:
    """Check every evaluation cache is complete, non-empty and built from its pinned recipe."""
    pending = pending_training_steps(tuple(steps.values()), marin_prefix=experiment.PREFIX)
    if pending:
        raise ValueError(f"Evaluation caches are incomplete: {[step.name for step in pending]}")
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(experiment.TOKENIZER)).metadata)
    audit = {}
    for name, step in steps.items():
        path = step.path(experiment.PREFIX)
        receipt = uncheatable.read_json(path + "/receipt.json")
        if receipt["recipe_sha256"] != canonical_sha256(asdict(materialized_config(step, experiment.PREFIX))):
            raise ValueError(f"Evaluation cache recipe changed: {name}")
        if receipt["design_sha256"] != design["design_sha256"]:
            raise ValueError(f"Evaluation cache geometry changed: {name}")
        tokens = original.cache_token_count(path + "/validation", metadata)
        expected = HELDOUT_TOKENS if name == HELDOUT else receipt["tokens"]
        if tokens <= 0 or tokens != expected or tokens != receipt["tokens"]:
            raise ValueError(f"Evaluation cache length differs from its receipt: {name}")
        audit[name] = {
            "path": path,
            "fingerprint": step.fingerprint(),
            "tokens": tokens,
            "receipt_sha256": canonical_sha256(receipt),
        }
    return audit
