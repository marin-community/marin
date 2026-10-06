# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the training cache and prove its global batch dose on one CPU device."""

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import draccus
import jax
import numpy as np
from haliax import Axis
from jax.sharding import Mesh
from levanter.data.dataset import BlockShufflingDataset, PermutationDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import ChatDataset, DatasetComponent, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat, preprocessor_for_format
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.store.cache import write_levanter_cache
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_collection import StudentRow

ROWS = 16
PASSES = 2
BATCH_SIZE = 8
UPDATES = 4
CONTEXT_TOKENS = 16384


@dataclass(frozen=True)
class CoverageLoaderProofConfig:
    train_config: TrainLmConfig
    input_pins: dict[str, str]
    collection_path: str
    output_path: str


def run_coverage_loader_proof(config: CoverageLoaderProofConfig) -> dict:
    """Prepare a missing cache, then certify the global CPU dose against saved collection witnesses."""
    directory = StoragePath(config.collection_path)
    dataset_bytes = (directory / "dataset.json").read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != config.input_pins["dataset_sha256"]:
        raise ValueError("Completed dataset hash differs from its input pin")
    dataset_record = json.loads(dataset_bytes)
    expected_dose = {
        "rows": ROWS,
        "passes": PASSES,
        "batch_size": BATCH_SIZE,
        "optimizer_updates": UPDATES,
        "example_exposures": ROWS * PASSES,
    }
    if dataset_record["sha256"] != config.input_pins["jsonl_sha256"] or any(
        dataset_record[key] != value for key, value in expected_dose.items()
    ):
        raise ValueError("Completed dataset does not declare the pinned sixteen-row dose")
    if str(directory / "train.jsonl") != config.input_pins["jsonl_uri"]:
        raise ValueError("Completed collection differs from the pinned training source")
    collection_bytes = (directory / "collection.json").read_bytes()
    if hashlib.sha256(collection_bytes).hexdigest() != config.input_pins["collection_sha256"]:
        raise ValueError("Collection hash differs from its input pin")
    collection = json.loads(collection_bytes)
    if dataset_record["collection_sha256"] != compact_json_sha256(collection) or collection["status"] != "passed":
        raise ValueError("Completed dataset differs from the accepted collection")
    for entry in collection["accepted"]:
        if entry["row"] != entry["witness"]["example"] or entry["row_sha256"] != compact_json_sha256(entry["row"]):
            raise ValueError("Accepted collection row differs from its saved witness")
    rows = tuple(
        StudentRow(
            entry["witness"]["example"],
            tuple(entry["witness"]["input_ids"]),
            tuple(entry["witness"]["assistant_mask"]),
        )
        for entry in collection["accepted"]
    )
    proof = teacher_coverage_loader_proof(config.train_config, rows, config.input_pins)
    write_once(StoragePath(config.output_path), proof)
    return proof


def teacher_coverage_loader_proof(
    config: TrainLmConfig, rows: tuple[StudentRow, ...], input_pins: Mapping[str, str]
) -> dict:
    """Prepare the cache and validate four global CPU batches against independent saved witnesses."""
    if len(rows) != ROWS or config.trainer.num_train_steps != UPDATES or config.trainer.train_batch_size != BATCH_SIZE:
        raise ValueError("Coverage proof requires sixteen rows and four batches of eight")
    if config.train_seq_len != CONTEXT_TOKENS or config.data.mixture_block_size != BATCH_SIZE:
        raise ValueError("Coverage proof requires the fixed context and mixture block size")
    if len(config.data.components) != 1:
        raise ValueError("Coverage proof requires one canonical teacher source")
    if config.data.cache_catalog is not None:
        raise ValueError("Coverage cache preparation does not support a cache catalog")
    component = next(iter(config.data.components.values()))
    if not isinstance(component, DatasetComponent) or not isinstance(component.format, ChatLmDatasetFormat):
        raise ValueError("Coverage proof requires the resolved chat component")
    if not isinstance(component.source, UrlDatasetSourceConfig):
        raise ValueError("Coverage proof requires a canonical JSONL source")
    if component.source.train_urls != [input_pins["jsonl_uri"]]:
        raise ValueError("Resolved training source differs from the pinned JSONL")
    if component.format.pack is not False or component.pack not in (None, False) or not component.format.mask_user_turns:
        raise ValueError("Coverage proof requires one conversation per example and assistant-only loss")

    source_bytes = StoragePath(input_pins["jsonl_uri"]).read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != input_pins["jsonl_sha256"]:
        raise ValueError("Canonical JSONL hash differs from its input pin")
    source_rows = [json.loads(line) for line in source_bytes.splitlines()]
    if source_rows != [row.example for row in rows]:
        raise ValueError("Canonical JSONL differs from the row witnesses")
    row_hashes = [compact_json_sha256(row.example) for row in rows]
    if len(set(row_hashes)) != ROWS:
        raise ValueError("Coverage proof requires sixteen distinct canonical rows")

    processor = preprocessor_for_format(
        component.format, config.data.the_tokenizer, enforce_bos=True, enforce_eos=config.data.enforce_eos
    )
    processed = processor(source_rows)
    for row, tokens in zip(rows, processed, strict=True):
        if not np.array_equal(tokens["input_ids"], row.input_ids) or not np.array_equal(
            tokens["assistant_masks"], row.assistant_mask
        ):
            raise ValueError("Resolved preprocessing differs from the saved full-token witness")
        if not 0 < len(row.input_ids) <= config.train_seq_len or not sum(row.assistant_mask[1:]):
            raise ValueError("Canonical row has no full-context supervised targets")
    by_tokens = {row.input_ids: (row_hash, row) for row_hash, row in zip(row_hashes, rows, strict=True)}
    if len(by_tokens) != ROWS:
        raise ValueError("Canonical rows do not have distinct full-token witnesses")

    cache_root = component.cache_dir
    if cache_root is None:
        if config.data.cache_dir is None:
            raise ValueError("Coverage proof requires a configured cache path")
        cache_root = str(StoragePath(config.data.cache_dir) / next(iter(config.data.components)))
    cache_path = StoragePath(cache_root) if component.flat_cache else StoragePath(cache_root) / "train"
    created_cache = not cache_path.exists()
    if created_cache:
        write_levanter_cache(processed, str(cache_path), metadata=processor.metadata)

    data_key, seed_binding = training_data_key(config)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("data",))
    batch_rows = []
    batch_doses = []
    with mesh:
        dataset = config.data.train_set(
            Axis("position", config.train_seq_len), config.trainer.batch_schedule, key=data_key
        )
        mixture = dataset.dataset
        assert isinstance(mixture, MixtureDataset)
        chat_dataset = next(iter(mixture.datasets.values()))
        if isinstance(chat_dataset, (BlockShufflingDataset, PermutationDataset)):
            chat_dataset = chat_dataset.dataset
        assert isinstance(chat_dataset, ChatDataset)
        effective_slice_strategy = chat_dataset.packed.slice_strategy
        loader = DataLoader(
            dataset,
            config.trainer.train_batch_size,
            mesh=mesh,
            axis_resources={"batch": "data"},
            batch_axis_name=config.trainer.batch_axis_name,
            max_buffered_batches=0,
            fetch_batch_size=1,
        )
        batches = loader.iter_from_step(0)
        for _ in range(config.trainer.num_train_steps):
            batch = next(batches)
            observed_rows = []
            supervised_tokens = 0
            segment_ids = batch.attn_mask.segment_ids
            assert segment_ids is not None, "Coverage proof requires conversation and padding segment IDs"
            for tokens, weights, segments in zip(
                np.asarray(batch.tokens.array),
                np.asarray(batch.loss_weight.array),
                np.asarray(segment_ids[0].array),
                strict=True,
            ):
                length = int(np.count_nonzero(segments >= 0))
                witness = by_tokens.get(tuple(int(token) for token in tokens[:length]))
                if witness is None:
                    raise ValueError("Loaded example differs from every full-token witness")
                row_hash, row = witness
                np.testing.assert_array_equal(segments[:length], np.full(length, segments[0], dtype=segments.dtype))
                np.testing.assert_array_equal(segments[length:], -np.ones(len(segments) - length, dtype=segments.dtype))
                expected_weights = np.zeros(config.train_seq_len, dtype=np.float32)
                expected_weights[: length - 1] = row.assistant_mask[1:]
                np.testing.assert_array_equal(weights, expected_weights)
                observed_rows.append(row_hash)
                supervised_tokens += int(weights.sum())
            batch_rows.append(observed_rows)
            batch_doses.append(supervised_tokens)

    counts = Counter(row_hash for batch in batch_rows for row_hash in batch)
    if counts != Counter({row_hash: PASSES for row_hash in row_hashes}):
        raise ValueError("Four loaded batches do not expose each canonical row exactly twice")
    batches_per_pass = ROWS // BATCH_SIZE
    for start in range(0, UPDATES, batches_per_pass):
        if Counter(row_hash for batch in batch_rows[start : start + batches_per_pass] for row_hash in batch) != Counter(
            row_hashes
        ):
            raise ValueError("A loaded pass does not contain each canonical row exactly once")
    return {
        "protocol": "teacher-sixteen-family-loader-proof-v1",
        "status": "passed",
        "input_pins": dict(input_pins),
        "jsonl_sha256": source_sha256,
        "resolved_train_config_sha256": compact_json_sha256(draccus.encode(config)),
        "cache_recipe": {"format": draccus.encode(component.format), "preprocessor": processor.metadata},
        "cache_preparation": {"path": str(cache_path), "method": "serial_write_if_missing"},
        "data_seed_binding": seed_binding,
        "scope": "single_cpu_global_batches",
        "distributed_sharding_verified": False,
        "canonical_row_sha256": row_hashes,
        "batch_row_sha256": batch_rows,
        "row_exposure_counts": dict(counts),
        "batch_supervised_tokens": batch_doses,
        "supervised_tokens": sum(batch_doses),
        "example_exposures": sum(counts.values()),
        "effective_slice_strategy": effective_slice_strategy,
    }


def training_data_key(config: TrainLmConfig) -> tuple[jax.Array, dict]:
    """Bind the CPU proof key to the current training source assignments."""
    source_bytes = Path(train_lm.__file__).read_bytes()
    source = ast.parse(source_bytes)
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"seed", "data_key"} for target in ast.walk(node.targets[0])
        )
    ]
    expected = ast.parse(
        "seed = config.trainer.seed\n"
        "data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)\n"
        "data_key = jrandom.PRNGKey(config.data_seed)\n"
    ).body
    if sorted(ast.dump(node) for node in assignments) != sorted(ast.dump(node) for node in expected):
        raise ValueError("Training source changed its data-key derivation")
    override = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Assign) and ast.dump(child) == ast.dump(expected[2]) for child in node.body)
    )
    if ast.dump(override.test) != ast.dump(ast.parse("config.data_seed is not None", mode="eval").body):
        raise ValueError("Training source changed its data-seed override")
    key = jax.random.split(jax.random.PRNGKey(config.trainer.seed), 4)[0]
    if config.data_seed is not None:
        key = jax.random.PRNGKey(config.data_seed)
    return key, {
        "train_lm_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "trainer_seed": config.trainer.seed,
        "data_seed": config.data_seed,
        "data_key": np.asarray(key).tolist(),
    }
