# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote a checkpoint's fp32 pinned-host master into its params, offline and once per lineage.

A checkpoint written with ``MasterParamMode.FP32_PINNED_HOST`` stores the weights twice: the
authoritative fp32 master under ``master_params/`` and the bf16 compute copy under ``params/``.
The hero now trains without a master, and its restore refuses a master-bearing checkpoint rather
than silently keeping the bf16 copy (see ``refuse_master_layout_mismatch``). This writes a new
checkpoint whose ``params/`` are the stored fp32 master arrays and whose ``master_params/`` are
gone, leaving the source untouched. The copy is a raw OCDBT key rename, so no fleet, mesh, or
device memory is involved and dtypes are preserved exactly.

The new checkpoint's metadata carries a fresh timestamp, so at the same step it sorts newer than
its source and a resume under the same search root picks it first.

    uv run python -m experiments.grug.moe_hero_ep.convert_master_checkpoint \
        <source-checkpoint> --output <new-checkpoint>
"""

import asyncio
import datetime
import json
import logging

import click
import tensorstore as ts
from levanter.checkpoint_manifest import CheckpointArray, CheckpointManifest, read_manifest, write_manifest
from levanter.tensorstore_serialization import KVSTORE_DRIVER, build_kvstore_spec
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.checkpointing import LEGACY_STATE_KEY, MASTER_PARAMS_KEY

logger = logging.getLogger(__name__)

MASTER_FIELD = MASTER_PARAMS_KEY
PARAMS_FIELD = "params"
METADATA_FILENAME = "metadata.json"
COPY_CONCURRENCY = 16


def promoted_array_paths(manifest: CheckpointManifest) -> dict[str, str | None]:
    """Map each source array path to its destination path, or None for a dropped array.

    Raises unless the master tree mirrors the params tree leaf for leaf, so a partially written
    or structurally surprising checkpoint is refused rather than half-converted.
    """
    prefix = LEGACY_STATE_KEY + "/" if any(p.startswith(LEGACY_STATE_KEY + "/") for p in manifest.array_paths) else ""

    def field_and_leaf(path: str) -> tuple[str, str]:
        field, _, leaf = path.removeprefix(prefix).partition("/")
        return field, leaf

    by_field: dict[str, set[str]] = {}
    for path in manifest.array_paths:
        field, leaf = field_and_leaf(path)
        by_field.setdefault(field, set()).add(leaf)
    if MASTER_FIELD not in by_field:
        raise ValueError("checkpoint stores no master_params arrays; nothing to convert")
    master_leaves, params_leaves = by_field[MASTER_FIELD], by_field.get(PARAMS_FIELD, set())
    if master_leaves != params_leaves:
        raise ValueError(
            f"master_params does not mirror params: only in master {sorted(master_leaves - params_leaves)[:5]}, "
            f"only in params {sorted(params_leaves - master_leaves)[:5]}"
        )

    mapping: dict[str, str | None] = {}
    for path in manifest.array_paths:
        field, leaf = field_and_leaf(path)
        if field == PARAMS_FIELD:
            mapping[path] = None
        elif field == MASTER_FIELD:
            mapping[path] = prefix + PARAMS_FIELD + ("/" + leaf if leaf else "")
        else:
            mapping[path] = path
    return mapping


def _promoted_manifest(manifest: CheckpointManifest, mapping: dict[str, str | None]) -> CheckpointManifest:
    arrays: list[CheckpointArray] = []
    for array in manifest.arrays:
        destination = mapping[array.path]
        if destination is not None:
            arrays.append(array.model_copy(update={"path": destination}))
    return manifest.model_copy(update={"arrays": tuple(arrays)})


async def _copy_kvstore(source_root: str, output_root: str, mapping: dict[str, str | None]) -> int:
    """Copy every kept OCDBT key from source to output, renaming per ``mapping``. Returns key count."""
    source = await ts.KvStore.open({"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(source_root)})
    output = await ts.KvStore.open({"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(output_root)})
    # Keys are "<array path>/<zarr key>"; map on the longest array-path prefix.
    array_paths = sorted(mapping, key=len, reverse=True)

    def key_destination(key: str) -> str | None:
        for path in array_paths:
            if key == path or key.startswith(path + "/"):
                destination = mapping[path]
                return None if destination is None else destination + key.removeprefix(path)
        raise ValueError(f"kvstore key {key!r} belongs to no array in the manifest")

    async def copy_one(key: str, destination: str) -> None:
        value = await source.read(key)
        await output.write(destination, value.value)

    keys = [key.decode() for key in await source.list()]
    copied = 0
    for start in range(0, len(keys), COPY_CONCURRENCY):
        batch = [(key, key_destination(key)) for key in keys[start : start + COPY_CONCURRENCY]]
        await asyncio.gather(*(copy_one(key, destination) for key, destination in batch if destination is not None))
        copied += sum(1 for _, destination in batch if destination is not None)
    return copied


@click.command()
@click.argument("source")
@click.option("--output", required=True, help="Path for the converted checkpoint; must not exist.")
def main(source: str, output: str) -> None:
    logging.basicConfig(level=logging.INFO)
    manifest = read_manifest(source)
    if manifest is None:
        raise click.ClickException(f"{source} has no manifest.json; only manifest-bearing checkpoints convert")
    output_metadata = StoragePath(prefix_join(output, METADATA_FILENAME))
    if output_metadata.exists():
        raise click.ClickException(f"{output} already holds a checkpoint")

    mapping = promoted_array_paths(manifest)
    copied = asyncio.run(_copy_kvstore(source, output, mapping))
    logger.info("Copied %d kvstore keys from %s", copied, source)

    write_manifest(output, _promoted_manifest(manifest, mapping))
    metadata = json.loads(StoragePath(prefix_join(source, METADATA_FILENAME)).read_text())
    # A fresh timestamp makes the converted checkpoint the newest candidate at its step.
    metadata["timestamp"] = datetime.datetime.now().isoformat()
    output_metadata.write_text(json.dumps(metadata))
    logger.info("Wrote converted checkpoint (step %s) to %s", metadata.get("step"), output)


if __name__ == "__main__":
    main()
