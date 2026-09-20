# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Dolmino FLAN finite pool and QA-format evaluation caches for the TPP-10 instruction sweep.

The pool follows the Wikipedia and arXiv recipes with a zstd reader: four pinned Dolmino FLAN shards
(the swarm's ``dolmino_synth_instruction`` bucket) are tokenized with the frozen TinyLlama tokenizer
into the 190,316,544-token parent, permuted with the frozen parent order, and one 10,485,760-token
matched subset is drawn with the frozen subset seed. Evaluations add five QA validation sets rendered
as question-answer documents and a bounded held-out FLAN shard.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import BinaryIO

import numpy as np
import zstandard
from fray.types import ResourceConfig
from google.cloud import storage
from levanter.store.cache import CacheMetadata, consolidate_shard_cache_ledgers
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as original
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as previous
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps, persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("tpp10_instruction_sweep_assets")
VERSION = "2026.09.19"
DOMAIN = "dolmino_flan"
QA_EVALS = ("arc_easy", "arc_challenge", "openbookqa", "qasc", "sciq")
HELDOUT = f"{DOMAIN}/heldout-tpp10"
HELDOUT_TOKENS = 2048 * experiment.SEQ_LEN
ON_TARGET_METRICS = (*(f"eval/qa/{name}-tpp10/bpb" for name in QA_EVALS), f"eval/{HELDOUT}/bpb")
# The 8 GiB on-demand class queued for hours on 18-19 September; the arXiv pool tokenized under 4 GiB.
CPU = ResourceConfig(cpu=2, ram="4g", disk="32g", preemptible=False, regions=(experiment.REGION,), zone=experiment.ZONE)
RECORD_BOUND = 64 * original.MIB


def zstd_lines(raw: BinaryIO, limit: int, uri: str) -> Iterator[dict]:
    """Decode bounded zstd JSON lines from ``raw``; every record must carry a text field."""
    reader = zstandard.ZstdDecompressor().stream_reader(original.BudgetedReader(raw, limit))
    stream = io.BufferedReader(reader)
    while line := stream.readline(RECORD_BOUND + 1):
        if len(line) > RECORD_BOUND:
            raise ValueError(f"Raw JSON record exceeds the 64 MiB bound: {uri}")
        row = json.loads(line)
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Missing text field: {uri}")
        yield row


def zstd_documents(source: original.Source, quota: int | None) -> Iterator[dict]:
    """Read one pinned regional zstd object; no fallback to a new generation or another region."""
    if not source.uri.startswith(experiment.PREFIX + "/"):
        raise ValueError(f"Nonregional source: {source.uri}")
    name = source.uri.removeprefix(experiment.PREFIX + "/")
    blob = storage.Client().bucket("marin-us-central1").blob(name, generation=int(source.generation))
    blob.reload()
    if (int(blob.size), blob.crc32c) != (source.size_bytes, source.crc32c):
        raise ValueError(f"Source metadata changed: {source.uri}")
    limit = min(source.size_bytes, max(8 * original.MIB, 4 * quota)) if quota is not None else source.size_bytes
    with blob.open("rb", chunk_size=8 * original.MIB) as raw:
        yield from zstd_lines(raw, limit, source.uri)


@dataclass(frozen=True)
class ZstdRecipe:
    raw: original.RawRecipe
    implementation_sha256: str


def prepare_zstd(recipe: ZstdRecipe) -> None:
    experiment.require_central1()
    if recipe.implementation_sha256 != file_sha256(Path(__file__)):
        raise ValueError("Zstd preparation code differs from its recipe")
    raw = recipe.raw
    if raw.implementation_sha256 != file_sha256(Path(original.__file__)):
        raise ValueError("Frozen tokenization code differs from its recipe")
    if raw.design_sha256 != experiment.load_design()["design_sha256"] or raw.tokens is None:
        raise ValueError("Finite zstd pool requires the frozen geometry and an exact quota")
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(raw.tokenizer)).metadata)
    root = raw.cache_path + "/" + raw.split
    paths, receipts = [], []
    for i, source in enumerate(raw.sources):
        quota = raw.tokens // len(raw.sources) + (i < raw.tokens % len(raw.sources))
        path = f"{root}/parts/{i:03d}"
        docs = zstd_documents(source, quota)
        try:
            receipt = original.write_part(
                original.token_records(docs, tokenizer_path=raw.tokenizer, quota=quota),
                path,
                metadata=metadata,
                identity={"recipe_sha256": canonical_sha256(asdict(recipe)), "source": asdict(source), "quota": quota},
                expected_tokens=quota,
            )
        finally:
            docs.close()
        paths.append(path)
        receipts.append(receipt)
        logger.info("Prepared %s shard %d/%d", raw.component, i + 1, len(raw.sources))
    consolidate_shard_cache_ledgers(paths, root, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata)
    tokens = original.cache_token_count(root, metadata)
    if tokens != raw.tokens:
        raise ValueError("Zstd packed pool differs from the prescribed quota")
    persist_submission_plan(
        {
            "design_sha256": raw.design_sha256,
            "recipe_sha256": canonical_sha256(asdict(recipe)),
            "tokens": tokens,
            "part_receipts": receipts,
        },
        raw.cache_path + "/receipt.json",
    )


def _sources(key: str) -> tuple[original.Source, ...]:
    inventory = json.loads((ASSETS / "sources.json").read_text())
    return tuple(original.Source(**row) for row in inventory[key]["selected"])


def data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Raw pool, permuted parent and matched subset for FLAN, with the frozen index draws."""
    original_hash = file_sha256(Path(original.__file__))
    zstd_hash = file_sha256(Path(__file__))
    sources = _sources(DOMAIN)
    environment = {"MARIN_PREFIX": experiment.PREFIX}

    def raw_config(ctx: StepContext) -> ZstdRecipe:
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
        return ZstdRecipe(recipe, zstd_hash)

    raw = ArtifactStep(
        name=f"tpp10_domain_sweeps/{DOMAIN}/raw",
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(prepare_zstd, resources=CPU, env_vars=environment),
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
    """QA validation documents (gzip, frozen preparer) and a bounded held-out FLAN shard (zstd)."""
    inventory = json.loads((ASSETS / "sources.json").read_text())
    original_hash = file_sha256(Path(original.__file__))
    environment = {"MARIN_PREFIX": experiment.PREFIX}
    steps: dict[str, ArtifactStep[TokenizedCache]] = {}
    for qa_name in QA_EVALS:
        row = inventory["qa_evaluations"]["sets"][qa_name]
        source = original.Source(**{key: row[key] for key in ("uri", "generation", "size_bytes", "crc32c")})
        name = f"qa/{qa_name}-tpp10"

        def config(ctx: StepContext, name=name, source=source) -> original.RawRecipe:
            return original.RawRecipe(
                name,
                (source,),
                None,
                "validation",
                ctx.output_path,
                experiment.TOKENIZER,
                design["design_sha256"],
                original_hash,
            )

        step = ArtifactStep(
            name=name,
            version=VERSION,
            artifact_type=TokenizedCache,
            run=remote(original.prepare_raw, resources=CPU, env_vars=environment),
            build_config=config,
        )
        steps[name] = replace(step, expected_fingerprint=step.fingerprint())
    heldout_sources = _sources(f"{DOMAIN}_heldout")
    zstd_hash = file_sha256(Path(__file__))

    def heldout_config(ctx: StepContext) -> ZstdRecipe:
        recipe = original.RawRecipe(
            HELDOUT,
            heldout_sources,
            HELDOUT_TOKENS,
            "validation",
            ctx.output_path,
            experiment.TOKENIZER,
            design["design_sha256"],
            original_hash,
        )
        return ZstdRecipe(recipe, zstd_hash)

    heldout = ArtifactStep(
        name=HELDOUT,
        version=VERSION,
        artifact_type=TokenizedCache,
        run=remote(prepare_zstd, resources=CPU, env_vars=environment),
        build_config=heldout_config,
    )
    steps[HELDOUT] = replace(heldout, expected_fingerprint=heldout.fingerprint())
    return steps


def evaluation_paths(design: dict) -> dict[str, str]:
    """Held-out evaluation caches: the seven Uncheatable components, five QA sets and the FLAN shard."""
    return {
        **previous.evaluation_paths(),
        **{name: step.path(experiment.PREFIX) for name, step in evaluation_steps(design).items()},
    }


def verify_evaluation_caches(design: dict, steps: dict[str, ArtifactStep[TokenizedCache]]) -> dict:
    """Check every new evaluation cache is complete, non-empty and built from its pinned recipe."""
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
