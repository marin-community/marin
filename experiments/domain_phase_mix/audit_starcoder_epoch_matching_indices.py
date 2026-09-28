# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit source-index identities without reading StarCoder token payloads."""

import argparse
import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import jax
import numpy as np
from levanter.data.dataset import AsyncDataset, BlockShufflingDataset
from levanter.utils.jax_utils import key_iterator

from experiments.domain_phase_mix import starcoder_epoch_matching as experiment


class SourceIndices(AsyncDataset[int]):
    """A finite source returning sequence indices instead of corpus tokens."""

    def __init__(self, count: int):
        self.count = count

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        return self.count

    async def get_batch(self, indices: Sequence[int]) -> Sequence[int]:
        if any(index < 0 or index >= self.count for index in indices):
            raise IndexError("Source index outside the packed corpus")
        return list(indices)


async def audit_indices(
    packed_sequences: int, output_dir: Path, *, design: experiment.ExperimentDesign | None = None
) -> dict:
    """Compare the legacy interior, explicit parent, nested subset and old endpoint."""
    if design is None:
        design = experiment.load_design()
    parent_count = experiment.PARENT_BATCHES * experiment.BATCH_SIZE
    matched_count = experiment.MATCHED_BATCHES * experiment.BATCH_SIZE
    if packed_sequences < parent_count:
        raise ValueError("Packed source cannot cover the frozen parent")
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(experiment.REFERENCE_SEED))
    legacy_keys = key_iterator(shuffle_key)
    keys = {name: next(legacy_keys) for name in design.component_order}
    source = SourceIndices(packed_sequences)

    async def mapped(key, count):
        dataset = BlockShufflingDataset(source, 256, window_blocks=512, key=key, perm_type="feistel")
        return np.asarray(await dataset.get_batch(range(count)), dtype="<i8")

    legacy = await mapped(keys["dolma/starcoder"], parent_count)
    explicit_key = np.asarray(design.component_shuffle_keys["dolma/starcoder"], dtype=np.uint32)
    parent = await mapped(explicit_key, parent_count)
    matched = await mapped(explicit_key, matched_count)
    endpoint = await mapped(next(key_iterator(shuffle_key)), parent_count)
    if not np.array_equal(legacy, parent) or not np.array_equal(parent[:matched_count], matched):
        raise ValueError("Legacy-parent or nested-subset source identity differs")
    if len(np.unique(parent)) != parent_count:
        raise ValueError("Parent contains duplicate source sequence indices")
    endpoint_differences = int(np.count_nonzero(endpoint != parent))
    if endpoint_differences == 0:
        raise ValueError("Expected historical endpoint mismatch was not reproduced")
    observed_environment = {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    result = {
        "design_sha256": design.design_sha256,
        "packed_starcoder_sequence_count": packed_sequences,
        "parent_sequence_count": parent_count,
        "matched_sequence_count": matched_count,
        "legacy_parent_match": True,
        "nested_subset_match": True,
        "historical_endpoint_different_positions": endpoint_differences,
        "parent_indices_sha256": hashlib.sha256(parent.tobytes()).hexdigest(),
        "matched_indices_sha256": hashlib.sha256(matched.tobytes()).hexdigest(),
        "observed_environment": observed_environment,
        "matches_training_environment": observed_environment == design.training_environment,
        "scope": "Index mapping under the recorded audit runtime; no token reads or historical binary replay.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parent_indices.i64le").write_bytes(parent.tobytes())
    (output_dir / "matched_indices.i64le").write_bytes(matched.tobytes())
    (output_dir / "index_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-sequences", required=True, type=int, help="Actual cache length verified from metadata")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(audit_indices(args.packed_sequences, args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
