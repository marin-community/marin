# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read each catalog branch and the assembled train/holdout datasets before launch."""

import asyncio
import json
import logging
from dataclasses import dataclass

import draccus
import jax
import numpy as np
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.data.text.cache import load_lm_dataset_cache_from_ledger
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig, dataset_for_component
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.models.lm_model import LmExample
from levanter.store.cache import CacheCatalog
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreflightConfig:
    data: LmDataConfig
    sequence_length: int
    vocab_size: int
    expected_train_sequences: dict[str, int]
    output_path: str


async def inspect_dataset(dataset: AsyncDataset, config: PreflightConfig) -> dict:
    count = await dataset.async_len()
    if count <= 0:
        raise ValueError("Empty dataset")
    samples = []
    for index in sorted({0, count - 1}):
        example = (await dataset.get_batch([index]))[0]
        if isinstance(example, LmExample):
            example = grug_lm_example_from_named(example)
        if not isinstance(example, GrugLmExample):
            raise TypeError(f"Unexpected example type: {type(example)}")
        tokens, weights = np.asarray(example.tokens), np.asarray(example.loss_weight)
        if tokens.shape != (config.sequence_length,) or weights.shape != tokens.shape:
            raise ValueError(f"Invalid tokens/weights shapes: {tokens.shape}/{weights.shape}")
        if not np.issubdtype(tokens.dtype, np.integer) or tokens.min() < 0 or tokens.max() >= config.vocab_size:
            raise ValueError(f"Invalid token range or dtype: {tokens.min()}..{tokens.max()} ({tokens.dtype})")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid loss weights")
        samples.append(
            dict(
                index=index,
                min_token=int(tokens.min()),
                max_token=int(tokens.max()),
                loss_positions=float(weights.sum()),
            )
        )
    return dict(sequences=count, samples=samples)


async def preflight(config: PreflightConfig) -> None:
    if config.data.cache_catalog is None:
        raise ValueError("Preflight requires a complete cache catalog")
    catalog = CacheCatalog.load(config.data.cache_catalog)
    position = Axis("position", config.sequence_length)
    tokenizer = config.data.the_tokenizer
    results = []
    errors = []
    for split in ("train", "validation"):
        for name, component in config.data.components.items():
            children = component.children if isinstance(component, ConcatDatasetComponent) else {None: component}
            for child_name, child in children.items():
                key = name if child_name is None else f"{name}/{child_name}"
                entry = catalog.entry(split, key)
                if entry is None and split == "validation":
                    continue
                try:
                    if entry is None:
                        raise ValueError("Missing training cache catalog entry")
                    if not isinstance(child, DatasetComponent):
                        raise TypeError(f"Expected cache-backed component: {type(child)}")
                    cache = load_lm_dataset_cache_from_ledger(
                        entry.cache_dir, entry.ledger, child.format, tokenizer, config.data.enforce_eos
                    )
                    dataset = dataset_for_component(
                        child,
                        position,
                        cache,
                        eos_id=tokenizer.eos_token_id,
                        block_cross_document_attention=config.data.block_cross_document_attention,
                    )
                    result = dict(stage="branch", split=split, component=key, **await inspect_dataset(dataset, config))
                    results.append(result)
                    logger.info("Checked %s", result)
                except Exception as exc:
                    # Report every failed branch in one audit instead of retrying one at a time.
                    errors.append(dict(stage="branch", split=split, component=key, error=f"{type(exc).__name__}: {exc}"))
                    logger.exception("Preflight failed for %s/%s", split, key)

    if not errors:
        for split in ("train", "validation"):
            try:
                datasets = (
                    config.data.train_sets(position, key=jax.random.key(0))
                    if split == "train"
                    else config.data.validation_sets(position)
                )
            except Exception as exc:
                errors.append(dict(stage="assembly", split=split, error=f"{type(exc).__name__}: {exc}"))
                continue
            for name, dataset in datasets.items():
                try:
                    result = dict(
                        stage="assembly", split=split, component=name, **await inspect_dataset(dataset, config)
                    )
                    expected = (
                        config.expected_train_sequences.get(name)
                        if split == "train"
                        else (config.data.num_validation_sequences or {}).get(name)
                    )
                    if expected is not None and result["sequences"] != expected:
                        raise ValueError(f"Expected {expected} sequences, found {result['sequences']}")
                    results.append(result)
                    logger.info("Checked %s", result)
                except Exception as exc:
                    errors.append(
                        dict(stage="assembly", split=split, component=name, error=f"{type(exc).__name__}: {exc}")
                    )
                    logger.exception("Preflight failed for assembled %s/%s", split, name)

    StoragePath(config.output_path).write_text(
        json.dumps(dict(config=draccus.encode(config), checked=results, errors=errors), indent=2) + "\n"
    )
    if errors:
        raise ValueError(f"Preflight found {len(errors)} failures; see {config.output_path}")


@draccus.wrap()
def main(config: PreflightConfig) -> None:
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(preflight(config))


if __name__ == "__main__":
    main()
