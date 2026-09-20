# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare the arXiv-papers finite pool and the S2ORC evaluation cache for the TPP-10 sweep.

The pool follows the Wikipedia recipe exactly: two pinned gzip JSON shards are tokenized with the
frozen TinyLlama tokenizer into the 190,316,544-token parent, permuted with the frozen parent
order, and one 10,485,760-token matched subset is drawn with the frozen subset seed. The S2ORC
validation set is tokenized like the frozen Paloma programming-languages set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from levanter.store.cache import CacheMetadata
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as original
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as previous
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

ASSETS = Path(__file__).with_name("tpp10_arxiv_sweep_assets")
VERSION = "2026.09.18"
DOMAIN = "arxiv_papers"
S2ORC = "paloma/m2d2_s2orc_unsplit-tpp10"
S2ORC_METRIC = f"eval/{S2ORC}/bpb"
ON_TARGET_METRICS = (
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    S2ORC_METRIC,
)
CPU = previous.CPU


def data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Raw pool, permuted parent and matched subset for arXiv, with the frozen index draws."""
    inventory = json.loads((ASSETS / "sources.json").read_text())
    original_hash = file_sha256(Path(original.__file__))
    sources = tuple(original.Source(**row) for row in inventory[DOMAIN]["selected"])
    environment = {"MARIN_PREFIX": experiment.PREFIX}

    def raw_config(ctx: StepContext) -> original.RawRecipe:
        return original.RawRecipe(
            DOMAIN,
            sources,
            experiment.PARENT_SEQUENCES * experiment.SEQ_LEN,
            "train",
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            original_hash,
        )

    raw = ArtifactStep(
        name=f"tpp10_domain_sweeps/{DOMAIN}/raw",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(original.prepare_raw, resources=CPU, env_vars=environment),
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


def evaluation_step(design: dict) -> ArtifactStep[TokenizedCache]:
    """The complete Paloma S2ORC validation split, tokenized with the frozen preparer."""
    inventory = json.loads((ASSETS / "sources.json").read_text())
    sources = tuple(original.Source(**row) for row in inventory["s2orc_evaluation"]["selected"])
    implementation = file_sha256(Path(original.__file__))

    def config(ctx: StepContext) -> original.RawRecipe:
        return original.RawRecipe(
            "s2orc_evaluation",
            sources,
            None,
            "validation",
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            implementation,
        )

    step = ArtifactStep(
        name=S2ORC,
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(original.prepare_raw, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
        build_config=config,
    )
    return replace(step, expected_fingerprint=step.fingerprint())


def evaluation_paths(design: dict) -> dict[str, str]:
    """Held-out evaluation caches: the seven Uncheatable components plus S2ORC."""
    return {**previous.evaluation_paths(), S2ORC: evaluation_step(design).path(experiment.PREFIX)}


def verify_evaluation_cache(design: dict, step: ArtifactStep[TokenizedCache]) -> dict:
    """Check the S2ORC cache is complete, non-empty and built from the pinned recipe."""
    if pending_training_steps((step,), marin_prefix=experiment.PREFIX):
        raise ValueError("S2ORC evaluation cache is incomplete")
    path = step.path(experiment.PREFIX)
    receipt = uncheatable.read_json(path + "/receipt.json")
    if receipt["recipe_sha256"] != canonical_sha256(asdict(materialized_config(step, experiment.PREFIX))):
        raise ValueError("S2ORC cache recipe changed")
    if receipt["design_sha256"] != design["design_sha256"]:
        raise ValueError("S2ORC cache geometry changed")
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(experiment.TOKENIZER)).metadata)
    tokens = original.cache_token_count(path + "/validation", metadata)
    if tokens <= 0 or tokens != receipt["tokens"]:
        raise ValueError("S2ORC cache length differs from its receipt")
    return {
        "path": path,
        "fingerprint": step.fingerprint(),
        "tokens": tokens,
        "receipt_sha256": canonical_sha256(receipt),
    }
