# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, resumable regional caches for the TPP10 experiment; dry-run by default."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import fsspec
import numpy as np
from fray.types import ResourceConfig
from google.cloud import storage
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.datasets import TokenSeqDataset
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, CacheMetadata, SerialCacheWriter, consolidate_shard_cache_ledgers
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config, run
from marin.execution.remote import remote
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps, persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
MIB = 1024**2
FORMAT = TextLmDatasetFormat()
CPU = ResourceConfig(cpu=8, ram="32g", disk="32g", regions=(experiment.REGION,), zone=experiment.ZONE)


@dataclass(frozen=True)
class Source:
    uri: str
    generation: str
    size_bytes: int
    crc32c: str


@dataclass(frozen=True)
class RawRecipe:
    component: str
    sources: tuple[Source, ...]
    tokens: int | None
    split: str
    cache_path: str
    tokenizer: str
    design_sha256: str
    implementation_sha256: str
    format: TextLmDatasetFormat = FORMAT


@dataclass(frozen=True)
class SubsetRecipe:
    parent_path: str
    seed: int
    cache_path: str
    tokenizer: str
    design_sha256: str
    implementation_sha256: str
    format: TextLmDatasetFormat = FORMAT


@dataclass(frozen=True)
class ParentRecipe:
    parent_path: str
    cache_path: str
    tokenizer: str
    design_sha256: str
    implementation_sha256: str
    format: TextLmDatasetFormat = FORMAT


class BudgetedReader:
    """Bound compressed bytes consumed by a gzip reader, with an extra EOF probe."""

    def __init__(self, stream: BinaryIO, limit: int):
        self.stream = stream
        self.limit = limit
        self.consumed = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit - self.consumed
        data = self.stream.read(min(size, remaining + 1) if size >= 0 else remaining + 1)
        self.consumed += len(data)
        if self.consumed > self.limit:
            raise ValueError("Compressed source-read budget exhausted before its token quota")
        return data


def remote_documents(source: Source, quota: int | None) -> Iterator[dict]:
    """Read one pinned regional object; no fallback to a new generation or another region."""
    if not source.uri.startswith(experiment.PREFIX + "/"):
        raise ValueError(f"Nonregional source: {source.uri}")
    name = source.uri.removeprefix(experiment.PREFIX + "/")
    blob = storage.Client().bucket("marin-us-central1").blob(name, generation=int(source.generation))
    blob.reload()
    if (int(blob.size), blob.crc32c) != (source.size_bytes, source.crc32c):
        raise ValueError(f"Source metadata changed: {source.uri}")
    # GCS may prefetch one 8 MiB chunk beyond bytes requested by gzip.
    limit = min(source.size_bytes, max(8 * MIB, 4 * quota)) if quota is not None else source.size_bytes
    with blob.open("rb", chunk_size=8 * MIB) as raw:
        with gzip.GzipFile(fileobj=BudgetedReader(raw, limit)) as stream:
            while line := stream.readline(64 * MIB + 1):
                if len(line) > 64 * MIB:
                    raise ValueError(f"Raw JSON record exceeds the 64 MiB bound: {source.uri}")
                row = json.loads(line)
                if not isinstance(row.get("text"), str):
                    raise ValueError(f"Missing text field: {source.uri}")
                yield row


def token_records(documents: Iterable[dict], *, tokenizer_path: str, quota: int | None) -> Iterator[dict]:
    """Tokenize with the runtime preprocessor and stop at an exact token budget."""
    processor = FORMAT.build_preprocessor(load_tokenizer(tokenizer_path))
    remaining = quota
    for document in documents:
        for row in processor([document]):
            ids = np.asarray(row["input_ids"], dtype=np.int32)
            if remaining is not None:
                ids = ids[:remaining]
                remaining -= len(ids)
            if len(ids):
                yield {"input_ids": ids}
            if remaining == 0:
                return
    if remaining is not None and remaining > 0:
        raise ValueError(f"Pinned source exhausted with {remaining} tokens still required")


def cache_token_count(path: str, metadata: CacheMetadata) -> int:
    ledger = CacheLedger.load(path, metadata)
    if not ledger.is_finished or ledger.metadata.compare_to(metadata):
        raise ValueError(f"Unfinished cache or tokenizer metadata mismatch: {path}")
    return ledger.field_counts["input_ids"]


def write_part(
    records: Iterable[dict], path: str, *, metadata: CacheMetadata, identity: dict, expected_tokens: int | None
) -> dict:
    """Reuse finished matching parts; restart only this part after an incomplete attempt."""
    receipt_path = path + "/receipt.json"
    fs, _ = fsspec.core.url_to_fs(receipt_path)
    if fs.exists(receipt_path):
        with fsspec.open(receipt_path, "rt") as handle:
            receipt = json.load(handle)
        if receipt["identity"] != identity:
            raise ValueError(f"Completed part has a different recipe: {path}")
        if cache_token_count(path, metadata) != receipt["tokens"] or (
            expected_tokens is not None and receipt["tokens"] != expected_tokens
        ):
            raise ValueError(f"Completed part has an unexpected token count: {path}")
        return receipt
    digest = hashlib.sha256()
    tokens = 0
    buffer = []
    buffered_tokens = 0
    with SerialCacheWriter(path, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata) as writer:
        for row in records:
            ids = np.asarray(row["input_ids"], dtype="<i4")
            digest.update(ids.tobytes())
            tokens += len(ids)
            buffer.append({"input_ids": ids})
            buffered_tokens += len(ids)
            if buffered_tokens >= 262144:
                writer.write_batch(buffer)
                buffer.clear()
                buffered_tokens = 0
        if buffer:
            writer.write_batch(buffer)
        if not tokens or (expected_tokens is not None and tokens != expected_tokens):
            raise ValueError(f"Wrong token count for {path}: {tokens}, expected {expected_tokens}")
    receipt = {"identity": identity, "tokens": tokens, "token_stream_sha256": digest.hexdigest()}
    persist_submission_plan(receipt, receipt_path)
    return receipt


def prepare_raw(recipe: RawRecipe) -> None:
    experiment.require_central1()
    if experiment.load_design()["design_sha256"] != recipe.design_sha256:
        raise ValueError("Preparation recipe belongs to another design")
    if recipe.implementation_sha256 != file_sha256(Path(__file__)):
        raise ValueError("Preparation code differs from its recipe")
    processor = FORMAT.build_preprocessor(load_tokenizer(recipe.tokenizer))
    metadata = CacheMetadata(processor.metadata)
    root = recipe.cache_path + "/" + recipe.split
    paths, receipts = [], []
    for i, source in enumerate(recipe.sources):
        quota = (
            None
            if recipe.tokens is None
            else recipe.tokens // len(recipe.sources) + (i < recipe.tokens % len(recipe.sources))
        )
        path = f"{root}/parts/{i:03d}"
        identity = {
            "design_sha256": recipe.design_sha256,
            "implementation_sha256": recipe.implementation_sha256,
            "source": asdict(source),
            "quota": quota,
        }
        docs = remote_documents(source, quota)
        try:
            receipt = write_part(
                token_records(docs, tokenizer_path=recipe.tokenizer, quota=quota),
                path,
                metadata=metadata,
                identity=identity,
                expected_tokens=quota,
            )
        finally:
            docs.close()
        paths.append(path)
        receipts.append(receipt)
        logger.info("Prepared %s shard %d/%d", recipe.component, i + 1, len(recipe.sources))
    consolidate_shard_cache_ledgers(paths, root, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata)
    total = cache_token_count(root, metadata)
    if recipe.tokens is not None and total != recipe.tokens:
        raise ValueError("Consolidated token count differs from finite-pool recipe")
    persist_submission_plan(
        {
            "design_sha256": recipe.design_sha256,
            "recipe_sha256": canonical_sha256(asdict(recipe)),
            "tokens": total,
            "part_receipts": receipts,
        },
        recipe.cache_path + "/receipt.json",
    )


def prepare_subset(recipe: SubsetRecipe) -> None:
    experiment.require_central1()
    design = experiment.load_design()
    if design["design_sha256"] != recipe.design_sha256:
        raise ValueError("Subset recipe belongs to another design")
    indices = experiment.subset_indices(recipe.seed)
    digest = canonical_sha256({"indices": indices.tolist()})
    if digest != design["subset_indices_sha256"][str(recipe.seed)]:
        raise ValueError("Subset index draw differs from frozen manifest")
    materialize_indices(recipe, indices, digest, parent_sequences=experiment.PARENT_SEQUENCES)


def prepare_parent(recipe: ParentRecipe) -> None:
    experiment.require_central1()
    design = experiment.load_design()
    if design["design_sha256"] != recipe.design_sha256:
        raise ValueError("Parent recipe belongs to another design")
    indices = experiment.parent_permutation()
    digest = canonical_sha256({"indices": indices.tolist()})
    if digest != design["parent_permutation_sha256"]:
        raise ValueError("Parent permutation differs from frozen manifest")
    materialize_indices(recipe, indices, digest, parent_sequences=experiment.PARENT_SEQUENCES)


def materialize_indices(
    recipe: ParentRecipe | SubsetRecipe, indices: np.ndarray, digest: str, *, parent_sequences: int
) -> None:
    """Write a verified view of finite parent sequences, using bounded reads."""
    if recipe.implementation_sha256 != file_sha256(Path(__file__)):
        raise ValueError("Preparation code differs from its recipe")
    tok = load_tokenizer(recipe.tokenizer)
    parent = load_lm_dataset_cache(recipe.parent_path + "/train", FORMAT, tok)
    sequences = TokenSeqDataset(parent, experiment.SEQ_LEN)
    if len(sequences.as_sync_dataset()) != parent_sequences:
        raise ValueError("Parent cache does not have the prespecified sequence count")

    def records() -> Iterator[dict]:
        for start in range(0, len(indices), 128):
            for row in asyncio.run(sequences.get_batch(indices[start : start + 128].tolist())):
                yield {"input_ids": row["input_ids"]}

    metadata = CacheMetadata(FORMAT.build_preprocessor(tok).metadata)
    receipt = write_part(
        records(),
        recipe.cache_path + "/train",
        metadata=metadata,
        identity=asdict(recipe),
        expected_tokens=len(indices) * experiment.SEQ_LEN,
    )
    persist_submission_plan(
        {
            "design_sha256": recipe.design_sha256,
            "recipe_sha256": canonical_sha256(asdict(recipe)),
            "tokens": receipt["tokens"],
            "indices_sha256": digest,
            "token_stream_sha256": receipt["token_stream_sha256"],
        },
        recipe.cache_path + "/receipt.json",
    )


def data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Build bounded preparation identities without touching remote data."""
    inventory = json.loads((experiment.ASSETS / "sources.json").read_text())
    implementation = file_sha256(Path(__file__))
    steps = {}
    for name in (*experiment.WEB_COUNTS, "starcoder", "evaluation"):
        sources = tuple(Source(**row) for row in inventory[name]["selected"])
        tokens = (
            None
            if name == "evaluation"
            else (experiment.PARENT_SEQUENCES if name == "starcoder" else design["web_sequences"][name])
            * experiment.SEQ_LEN
        )

        def config(ctx: StepContext, name=name, sources=sources, tokens=tokens) -> RawRecipe:
            return RawRecipe(
                name,
                sources,
                tokens,
                "validation" if name == "evaluation" else "train",
                ctx.output_path,
                experiment.TOKENIZER,
                design["design_sha256"],
                implementation,
            )

        steps[name] = ArtifactStep(
            name=("paloma/dolma_100_programing_languages-tpp10" if name == "evaluation" else f"starcoder_tpp10/{name}"),
            version=experiment.VERSION,
            artifact_type=TokenizedCache,
            run=remote(prepare_raw, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
            build_config=config,
        )
    raw_parent = steps.pop("starcoder")
    steps["starcoder_raw"] = raw_parent

    def parent_config(ctx: StepContext) -> ParentRecipe:
        return ParentRecipe(
            ctx.artifact_path(raw_parent), ctx.output_path, experiment.TOKENIZER, design["design_sha256"], implementation
        )

    parent = ArtifactStep(
        name="starcoder_tpp10/parent",
        version=experiment.VERSION,
        artifact_type=TokenizedCache,
        run=remote(prepare_parent, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
        build_config=parent_config,
        deps=(raw_parent,),
    )
    steps["starcoder"] = parent
    for seed in experiment.SUBSET_SEEDS:

        def subset_config(ctx: StepContext, seed=seed) -> SubsetRecipe:
            return SubsetRecipe(
                ctx.artifact_path(parent),
                seed,
                ctx.output_path,
                experiment.TOKENIZER,
                design["design_sha256"],
                implementation,
            )

        steps[f"subset_{seed}"] = ArtifactStep(
            name=f"starcoder_tpp10/subset_{seed}",
            version=experiment.VERSION,
            artifact_type=TokenizedCache,
            run=remote(prepare_subset, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
            build_config=subset_config,
            deps=(parent,),
        )
    return steps


def verify_caches(design: dict, steps: dict[str, ArtifactStep[TokenizedCache]], prefix: str) -> dict:
    """Require completed fingerprinted caches with matching tokenizer metadata and lengths."""
    if pending_training_steps(tuple(steps.values()), marin_prefix=prefix):
        raise ValueError("Data preparation is incomplete")
    metadata = CacheMetadata(FORMAT.build_preprocessor(load_tokenizer(experiment.TOKENIZER)).metadata)
    caches = {}
    for name, step in steps.items():
        path = step.path(prefix)
        with fsspec.open(path + "/receipt.json", "rt") as handle:
            receipt = json.load(handle)
        if receipt["design_sha256"] != design["design_sha256"]:
            raise ValueError(f"Data receipt belongs to another design: {name}")
        if receipt["recipe_sha256"] != canonical_sha256(asdict(materialized_config(step, prefix))):
            raise ValueError(f"Data receipt belongs to another recipe: {name}")
        if name == "starcoder" and receipt["indices_sha256"] != design["parent_permutation_sha256"]:
            raise ValueError("Parent permutation receipt differs from the frozen design")
        if name.startswith("subset_"):
            seed = name.removeprefix("subset_")
            if receipt["indices_sha256"] != design["subset_indices_sha256"][seed]:
                raise ValueError(f"Subset membership receipt differs from the frozen design: {name}")
        split = "validation" if name == "evaluation" else "train"
        count = cache_token_count(path + "/" + split, metadata)
        expected = (
            receipt["tokens"]
            if name == "evaluation"
            else (
                experiment.MATCHED_SEQUENCES
                if name.startswith("subset_")
                else (
                    experiment.PARENT_SEQUENCES
                    if name in ("starcoder", "starcoder_raw")
                    else design["web_sequences"][name]
                )
            )
            * experiment.SEQ_LEN
        )
        if count != expected or count != receipt["tokens"]:
            raise ValueError(f"Wrong finite cache length: {name}")
        caches[name] = {
            "path": path,
            "fingerprint": step.fingerprint(),
            "tokens": count,
            "receipt_sha256": canonical_sha256(receipt),
        }
    return {"status": "passed", "design_sha256": design["design_sha256"], "caches": caches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    logging.basicConfig(level=logging.INFO)
    design = experiment.load_design()
    steps = data_steps(design)
    if args.run or args.audit:
        experiment.require_central1()
    if args.run:
        pending = pending_training_steps(tuple(steps.values()), marin_prefix=experiment.PREFIX)
        if pending:
            run(*pending, max_concurrent=args.max_concurrent, force_run_failed=True)
    result = (
        verify_caches(design, steps, experiment.PREFIX)
        if args.audit or args.run
        else {
            "design_sha256": design["design_sha256"],
            "caches": {k: {"path": s.path(experiment.PREFIX), "fingerprint": s.fingerprint()} for k, s in steps.items()},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.run or args.audit:
        digest = canonical_sha256(result)
        uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/data_audits/{digest}/cache_audit.json"
        persist_submission_plan(result, uri)
        print(json.dumps({"cache_audit_sha256": digest, "cache_audit_uri": uri}))


if __name__ == "__main__":
    main()
