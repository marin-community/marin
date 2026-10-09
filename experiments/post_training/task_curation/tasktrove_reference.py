# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated frozen-converter worker; emits archive snapshots, never executes graders."""

import argparse
import gzip
import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.dataset as ds
import pyarrow.parquet as pq

REFERENCE_COMPRESSION_LEVEL = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--snapshots-module", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--archive-compression", type=int, choices=(1, 9), required=True)
    parser.add_argument("--path")
    parser.add_argument("--payload-root", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    # Load only the dependency-free wire encoder from current code. Importing current
    # TaskCompendium or Verifyit here would contaminate the frozen reference process.
    spec = importlib.util.spec_from_file_location("parity_snapshots", args.snapshots_module)
    assert spec is not None and spec.loader is not None
    snapshots = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = snapshots
    spec.loader.exec_module(snapshots)
    sys.path[:0] = [str(checkout), str(checkout / "lib/taskcompendium/src"), str(checkout / "lib/verifyit/src")]
    # experiments is a namespace package in the checkout; exclude any editable host roots.
    for package, relative in (
        ("experiments", "experiments"),
        ("experiments.post_training", "experiments/post_training"),
    ):
        module = ModuleType(package)
        module.__path__ = [str(checkout / relative)]
        sys.modules[package] = module
    convert = importlib.import_module("experiments.post_training.tasktrove.convert")
    registry = importlib.import_module("experiments.post_training.tasktrove.converters.registry")
    dataset = importlib.import_module("experiments.post_training.tasktrove.dataset")
    verify = importlib.import_module("experiments.post_training.tasktrove.verify")
    taskbinary = importlib.import_module("experiments.post_training.tasktrove.taskbinary")
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("taskcompendium.", "verifyit.", "experiments.post_training.tasktrove")):
            filename = module.__file__
            if filename is not None and not Path(filename).resolve().is_relative_to(checkout):
                raise RuntimeError(f"Frozen reference imported host code: {name} from {filename}")
    # Keep the frozen file writer and its tar headers unchanged. Compression is outside
    # the snapshot contract, and level 9 wastes CPU on archives discarded immediately.
    taskbinary.gzip = SimpleNamespace(
        compress=partial(gzip.compress, compresslevel=args.archive_compression),
        decompress=gzip.decompress,
    )
    index, info = registry.converter_index(), dataset.load_source_verdicts()[args.source]
    batches = (
        pq.ParquetFile(args.input).iter_batches(batch_size=16)
        if args.path is None
        else ds.dataset(args.input, format="parquet")
        .scanner(filter=ds.field("path") == args.path, batch_size=16)
        .to_batches()
    )
    for batch in batches:
        for row in batch.to_pylist():
            result = convert.convert_one(info, row["path"], row["task_binary"], index, args.revision)
            if args.payload_root is not None:
                for name, blob in (("task", result.task_binary), ("oracle", result.solution_binary)):
                    if blob is not None:
                        (args.payload_root / name).write_bytes(blob)
            snapshot = snapshots.task_snapshot(
                args.source,
                row["path"],
                result.status,
                task_binary=result.task_binary,
                solution_binary=result.solution_binary,
                detail=result.error,
                metadata=(
                    {
                        key: getattr(result, key)
                        for key in (
                            "family",
                            "template_id",
                            "converter",
                            "mode",
                            "dockerfile_id",
                            "language",
                            "tags",
                            "has_solution",
                        )
                    }
                    if result.task_binary is not None
                    else {}
                ),
            )
            if result.task_binary is not None:
                task = taskbinary.read_task_binary(result.task_binary)
                parsed, rejection = verify.check_spec(task)
                if rejection is None and parsed is not None:
                    rejection = (
                        verify.check_dockerfile(task)
                        or verify.check_gold_leak(task, parsed)
                        or verify.check_shape(task, parsed)
                    )
                if rejection is not None:
                    snapshot = replace(
                        snapshot, legacy_static_rejection=str(rejection.check), legacy_static_detail=rejection.detail
                    )
            print(json.dumps(snapshots.snapshot_dict(snapshot), default=snapshots.json_temporal), flush=True)


if __name__ == "__main__":
    main()
