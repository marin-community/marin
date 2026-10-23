# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count changed entries and donor deltas rounded back to S in both averages."""

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)


def count_changes(
    anchor: CheckpointReader, outputs: dict[str, CheckpointReader], donors: dict[str, CheckpointReader]
) -> dict:
    totals = {"changed": 0, "numel": 0, "donor_delta": 0, "erased_delta": 0}
    reports = {name: {**totals, "by_dtype": {}, "per_key": {}} for name in outputs}
    for reader in [*outputs.values(), *donors.values()]:
        assert reader.weight_map.keys() == anchor.weight_map.keys()
    ordered = sorted(anchor.weight_map, key=lambda key: (anchor.weight_map[key], key))
    for index, key in enumerate(ordered, 1):
        base = anchor.tensor(key)
        for name, reader in outputs.items():
            tensor = reader.tensor(key)
            donor = donors[name].tensor(key)
            assert tensor.shape == base.shape and tensor.dtype == base.dtype, key
            assert donor.shape == base.shape and donor.dtype == base.dtype, key
            changed = donor_delta = erased_delta = 0
            for left, right, source in zip(
                base.reshape(-1).split(1_048_576),
                tensor.reshape(-1).split(1_048_576),
                donor.reshape(-1).split(1_048_576),
                strict=True,
            ):
                real_delta = source != left
                retained = right != left
                changed += torch.count_nonzero(retained).item()
                donor_delta += torch.count_nonzero(real_delta).item()
                erased_delta += torch.count_nonzero(real_delta & ~retained).item()
            count = base.numel()
            dtype = str(base.dtype)
            row = {"changed": changed, "numel": count, "donor_delta": donor_delta, "erased_delta": erased_delta}
            report = reports[name]
            group = report["by_dtype"].setdefault(dtype, dict(totals))
            for field, value in row.items():
                report[field] += value
                group[field] += value
            report["per_key"][key] = {
                **row,
                "fraction_changed": changed / count,
                "fraction_donor_delta_erased": erased_delta / donor_delta if donor_delta else None,
                "shape": list(base.shape),
                "dtype": dtype,
            }
            del tensor, donor
        del base
        logger.info("Compared %d/%d: %s", index, len(ordered), key)
    for report in reports.values():
        for group in [report, *report["by_dtype"].values()]:
            group["fraction_changed"] = group["changed"] / group["numel"]
            group["fraction_donor_delta_erased"] = (
                group["erased_delta"] / group["donor_delta"] if group["donor_delta"] else None
            )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    recipes = {name: json.loads(Path(__file__).with_name(name + ".json").read_text()) for name in ("avg-SC", "avg-SM")}
    anchor = CheckpointReader(CheckpointSource(**recipes["avg-SC"]["anchor"]))
    assert recipes["avg-SC"]["anchor"] == recipes["avg-SM"]["anchor"]
    outputs = {
        name: CheckpointReader(CheckpointSource(recipe["output"], args.code_revision))
        for name, recipe in recipes.items()
    }
    for reader in outputs.values():
        assert json.loads(reader.fs.cat_file(prefix_join(reader.path, "merge_manifest.json")))["verified"]
    donors = {name: CheckpointReader(CheckpointSource(**recipe["donors"][0])) for name, recipe in recipes.items()}
    reports = count_changes(anchor, outputs, donors)
    for name, report in reports.items():
        report.update(
            anchor=recipes[name]["anchor"],
            donor=recipes[name]["donors"][0],
            output=recipes[name]["output"],
            code_revision=args.code_revision,
            wall_seconds=time.monotonic() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        reader = outputs[name]
        payload = json.dumps(report, indent=2, allow_nan=False).encode()
        path = prefix_join(reader.path, "change_from_S.json")
        reader.fs.pipe_file(path, payload)
        assert reader.fs.cat_file(path) == payload
        logger.info("%s changed %d/%d = %.8f", name, report["changed"], report["numel"], report["fraction_changed"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
