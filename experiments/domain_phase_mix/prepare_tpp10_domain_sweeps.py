# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare the two finite regional pools while preserving frozen StarCoder inputs."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from google.cloud import storage
from levanter.store.cache import CacheMetadata, consolidate_shard_cache_ledgers
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as original
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps, persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("tpp10_domain_sweeps_assets")
VERSION = "2026.09.11"
DOMAINS = ("wikipedia", "finemath_3plus")
CPU = ResourceConfig(cpu=2, ram="8g", disk="32g", preemptible=False, regions=(experiment.REGION,), zone=experiment.ZONE)
MAX_PARQUET_ROW_GROUP_BYTES = 512 * original.MIB


class BoundedParquetReader(io.RawIOBase):
    """Bound bytes returned through a seekable parquet input, including repeated reads."""

    def __init__(self, stream, limit: int):
        self.stream = stream
        self.limit = limit
        self.consumed = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit - self.consumed
        data = self.stream.read(min(size, remaining + 1) if size >= 0 else remaining + 1)
        self.consumed += len(data)
        if self.consumed > self.limit:
            raise ValueError("Parquet source-read budget exhausted before its token quota")
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def check_parquet_memory(parquet: pq.ParquetFile, maximum_bytes: int) -> None:
    """Reject row groups whose uncompressed size exceeds the preparation budget."""
    for i in range(parquet.metadata.num_row_groups):
        size = parquet.metadata.row_group(i).total_byte_size
        if size > maximum_bytes:
            raise ValueError(f"Parquet row group {i} needs {size} bytes; limit is {maximum_bytes}")


def parquet_documents(source: original.Source) -> Iterator[dict]:
    """Read only text batches from one immutable regional parquet object."""
    if not source.uri.startswith(experiment.PREFIX + "/raw/finemath-7090a5/finemath-3plus/"):
        raise ValueError(f"Unexpected FineMath source: {source.uri}")
    blob = (
        storage.Client()
        .bucket("marin-us-central1")
        .blob(source.uri.removeprefix(experiment.PREFIX + "/"), generation=int(source.generation))
    )
    blob.reload()
    if (int(blob.size), blob.crc32c) != (source.size_bytes, source.crc32c):
        raise ValueError(f"Source metadata changed: {source.uri}")
    with blob.open("rb", chunk_size=8 * original.MIB) as raw:
        # At most one object plus footer/range-read overhead is delivered to Arrow.
        # GCS can additionally prefetch one 8 MiB buffer per noncontiguous seek.
        reader = BoundedParquetReader(raw, source.size_bytes + 32 * original.MIB)
        try:
            with pq.ParquetFile(reader, pre_buffer=False) as parquet:
                if "text" not in parquet.schema_arrow.names:
                    raise ValueError(f"Missing parquet text column: {source.uri}")
                check_parquet_memory(parquet, MAX_PARQUET_ROW_GROUP_BYTES)
                for batch in parquet.iter_batches(batch_size=128, columns=["text"], use_threads=False):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str):
                            raise ValueError(f"Non-string parquet text: {source.uri}")
                        yield {"text": text}
        finally:
            logger.info("Parquet bytes delivered: %d for %s", reader.consumed, source.uri)


@dataclass(frozen=True)
class ParquetRecipe:
    raw: original.RawRecipe
    implementation_sha256: str


def prepare_parquet(recipe: ParquetRecipe) -> None:
    experiment.require_central1()
    if recipe.implementation_sha256 != file_sha256(Path(__file__)):
        raise ValueError("Parquet preparation code differs from its recipe")
    raw = recipe.raw
    if raw.implementation_sha256 != file_sha256(Path(original.__file__)):
        raise ValueError("Frozen tokenization code differs from its recipe")
    if raw.design_sha256 != experiment.load_design()["design_sha256"] or raw.tokens is None:
        raise ValueError("Finite parquet pool requires the frozen geometry and an exact quota")
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(raw.tokenizer)).metadata)
    root = raw.cache_path + "/train"
    paths, receipts = [], []
    for i, source in enumerate(raw.sources):
        quota = raw.tokens // len(raw.sources) + (i < raw.tokens % len(raw.sources))
        path = f"{root}/parts/{i:03d}"
        docs = parquet_documents(source)
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
        logger.info("Prepared FineMath shard %d/%d", i + 1, len(raw.sources))
    consolidate_shard_cache_ledgers(paths, root, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata)
    tokens = original.cache_token_count(root, metadata)
    if tokens != raw.tokens:
        raise ValueError("FineMath packed pool differs from the prescribed quota")
    persist_submission_plan(
        {
            "design_sha256": raw.design_sha256,
            "recipe_sha256": canonical_sha256(asdict(recipe)),
            "tokens": tokens,
            "part_receipts": receipts,
        },
        raw.cache_path + "/receipt.json",
    )


def data_steps(design: dict) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Build distinct domain cache identities with the original index materializer."""
    inventory = json.loads((ASSETS / "sources.json").read_text())
    original_hash = file_sha256(Path(original.__file__))
    parquet_hash = file_sha256(Path(__file__))
    steps = {}
    for domain in DOMAINS:
        sources = tuple(original.Source(**row) for row in inventory[domain])

        def raw_config(ctx: StepContext, domain=domain, sources=sources):
            recipe = original.RawRecipe(
                domain,
                sources,
                experiment.PARENT_SEQUENCES * experiment.SEQ_LEN,
                "train",
                ctx.output_path,
                experiment.TOKENIZER,
                design["design_sha256"],
                original_hash,
            )
            return ParquetRecipe(recipe, parquet_hash) if domain == "finemath_3plus" else recipe

        raw = ArtifactStep(
            name=f"tpp10_domain_sweeps/{domain}/raw",
            version=VERSION,
            artifact_type=TokenizedCache,
            run=remote(
                prepare_parquet if domain == "finemath_3plus" else original.prepare_raw,
                resources=CPU,
                env_vars={"MARIN_PREFIX": experiment.PREFIX},
            ),
            build_config=raw_config,
        )
        raw = replace(raw, expected_fingerprint=raw.fingerprint())

        def parent_config(ctx: StepContext, raw=raw) -> original.ParentRecipe:
            return original.ParentRecipe(
                ctx.artifact_path(raw), ctx.output_path, experiment.TOKENIZER, design["design_sha256"], original_hash
            )

        parent = ArtifactStep(
            name=f"tpp10_domain_sweeps/{domain}/parent",
            version=VERSION,
            artifact_type=TokenizedCache,
            run=remote(original.prepare_parent, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
            build_config=parent_config,
            deps=(raw,),
        )
        parent = replace(parent, expected_fingerprint=parent.fingerprint())

        def subset_config(ctx: StepContext, parent=parent) -> original.SubsetRecipe:
            return original.SubsetRecipe(
                ctx.artifact_path(parent),
                experiment.SUBSET_SEEDS[0],
                ctx.output_path,
                experiment.TOKENIZER,
                design["design_sha256"],
                original_hash,
            )

        subset = ArtifactStep(
            name=f"tpp10_domain_sweeps/{domain}/matched",
            version=VERSION,
            artifact_type=TokenizedCache,
            run=remote(original.prepare_subset, resources=CPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}),
            build_config=subset_config,
            deps=(parent,),
        )
        steps.update(
            {
                f"{domain}/raw": raw,
                f"{domain}/parent": parent,
                f"{domain}/matched": replace(subset, expected_fingerprint=subset.fingerprint()),
            }
        )
    return steps


def evaluation_paths() -> dict[str, str]:
    """Resolve the existing immutable Uncheatable cache identities without I/O."""
    spec = json.loads((ASSETS / "evaluation.json").read_text())
    paths = {}
    for name in uncheatable.COMPONENTS:
        identity = {
            "source": spec["sources"][name],
            "tokenizer": spec["tokenizer_pins"],
            "preparer": file_sha256(Path(original.__file__)),
            "layout": "one-source-flat-validation-v1",
        }
        paths[f"uncheatable_eval/{name}"] = (
            f"{experiment.PREFIX}/tokenized/uncheatable_eval/{name}-tpp10/{uncheatable.VERSION}/"
            + canonical_sha256(identity)
        )
    return paths


def verify_caches(design: dict, steps: dict[str, ArtifactStep[TokenizedCache]]) -> dict:
    """Check exact quotas, index draws, tokenizer and recipe receipts before training."""
    if pending_training_steps(tuple(steps.values()), marin_prefix=experiment.PREFIX):
        raise ValueError("Domain data preparation is incomplete")
    metadata = CacheMetadata(original.FORMAT.build_preprocessor(load_tokenizer(experiment.TOKENIZER)).metadata)
    caches = {}
    for name, step in steps.items():
        path = step.path(experiment.PREFIX)
        receipt = uncheatable.read_json(path + "/receipt.json")
        config = materialized_config(step, experiment.PREFIX)
        if receipt["recipe_sha256"] != canonical_sha256(asdict(config)):
            raise ValueError(f"Cache recipe changed: {name}")
        if receipt["design_sha256"] != design["design_sha256"]:
            raise ValueError(f"Cache geometry changed: {name}")
        subset = name.endswith("/matched")
        expected = (experiment.MATCHED_SEQUENCES if subset else experiment.PARENT_SEQUENCES) * experiment.SEQ_LEN
        if original.cache_token_count(path + "/train", metadata) != expected or receipt["tokens"] != expected:
            raise ValueError(f"Finite cache length changed: {name}")
        if not name.endswith("/raw"):
            digest = (
                design["subset_indices_sha256"][str(experiment.SUBSET_SEEDS[0])]
                if subset
                else design["parent_permutation_sha256"]
            )
            if receipt["indices_sha256"] != digest:
                raise ValueError(f"Finite cache membership changed: {name}")
        caches[name] = {
            "path": path,
            "fingerprint": step.fingerprint(),
            "tokens": expected,
            "receipt_sha256": canonical_sha256(receipt),
        }
    return {"status": "passed", "caches": caches}
