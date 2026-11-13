# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare disjoint autosomal VEP panels from pinned development parquet files."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from experiments.evals.vep.protocol import AUTOSOMES, SEQUENCE_LENGTH, VARIANT_INDEX

INPUTS = {
    "variants-train.parquet": "919abf532debd9f5e339b94cd4eef9868cbfa92dbe4fc57a0b1398da481f469f",
    "sequences-train.parquet": "3e8a6fa99f94ced9e86972a56857b60ddaf7e0cea3f4deddbee555aef9832506",
}


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def variant_id(row: dict) -> str:
    return f"{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}"


def rank(seed: int, namespace: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{identity}".encode()).hexdigest()


def read_rows(path: Path, columns: list[str]):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns, use_threads=False):
        yield from batch.to_pylist()


def eligible_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        # Filter before constructing any selection pool, including demonstrations.
        if row["chrom"] in AUTOSOMES and row["subset"] == "missense_variant":
            groups[str(row["match_group"])].append(row)
    for group in groups.values():
        group.sort(key=variant_id)
        assert len(group) == 10 and sum(row["label"] for row in group) == 1
        assert len({row["chrom"] for row in group}) == 1
        assert len({variant_id(row) for row in group}) == 10
    return dict(groups)


def select_panels(groups: dict[str, list[dict]], seed: int, validation_groups: int) -> dict[str, list[dict]]:
    demonstrations = [row for group in groups.values() for row in group if row["chrom"] == "1"]
    chosen = []
    for label in (True, False):
        candidates = sorted(
            (row for row in demonstrations if row["label"] == label),
            key=lambda row: rank(seed, "demonstration", variant_id(row)),
        )
        chosen.append(
            next(
                row
                for row in candidates
                if all(
                    row["match_group"] != other["match_group"] and abs(row["pos"] - other["pos"]) > SEQUENCE_LENGTH - 1
                    for other in chosen
                )
            )
        )
    chosen.sort(key=lambda row: rank(seed, "demonstration-order", variant_id(row)))
    query_groups = sorted(
        (key for key, rows in groups.items() if rows[0]["chrom"] != "1"),
        key=lambda key: rank(seed, "query-group", key),
    )
    assert len(query_groups) >= 10 + validation_groups
    pilot = [row for key in query_groups[:10] for row in groups[key]]
    # Preserve whole groups and reject any group overlapping a pilot window.
    validation = []
    for key in query_groups[10:]:
        group = groups[key]
        if any(
            a["chrom"] == b["chrom"] and abs(a["pos"] - b["pos"]) <= SEQUENCE_LENGTH - 1 for a in group for b in pilot
        ):
            continue
        validation.extend(group)
        if len(validation) == validation_groups * 10:
            break
    assert len(validation) == validation_groups * 10
    return {"demonstrations": chosen, "pilot": pilot, "validation": validation}


def attach_sequences(panels: dict[str, list[dict]], path: Path) -> None:
    wanted = {variant_id(row): row for panel in panels.values() for row in panel}
    assert len(wanted) == sum(len(panel) for panel in panels.values())
    sequences = defaultdict(dict)
    columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "target",
        "subset",
        "match_group",
        "strand",
        "context",
        "ref_completion",
        "alt_completion",
    ]
    for row in read_rows(path, columns):
        identity = variant_id(row)
        if identity not in wanted:
            continue
        source = wanted[identity]
        assert row["target"] == source["label"]
        assert row["match_group"] == source["match_group"] and row["subset"] == source["subset"]
        strand = row["strand"]
        assert strand in {"+", "-"} and strand not in sequences[identity]
        ref = row["context"] + row["ref_completion"]
        alt = row["context"] + row["alt_completion"]
        assert len(row["context"]) == VARIANT_INDEX and len(ref) == len(alt) == SEQUENCE_LENGTH
        assert set(ref + alt) <= set("ACGT")
        assert [i for i, (a, b) in enumerate(zip(ref, alt, strict=True)) if a != b] == [VARIANT_INDEX]
        alleles = (source["ref"], source["alt"])
        if strand == "-":
            alleles = tuple(reverse_complement(base) for base in alleles)
        assert (ref[VARIANT_INDEX], alt[VARIANT_INDEX]) == alleles
        sequences[identity][strand] = {"reference": ref, "alternate": alt}
    for identity, row in wanted.items():
        pair = sequences[identity]
        assert set(pair) == {"+", "-"}
        for allele in ("reference", "alternate"):
            assert reverse_complement(pair["+"][allele]) == pair["-"][allele]
        row["id"] = identity
        row["sequences"] = pair
        row["sequence_sha256"] = {
            strand: {allele: hashlib.sha256(seq.encode()).hexdigest() for allele, seq in values.items()}
            for strand, values in pair.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-groups", type=int, required=True)
    args = parser.parse_args()
    assert args.validation_groups > 10
    for name, expected in INPUTS.items():
        with (args.inputs / name).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    columns = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    groups = eligible_groups(list(read_rows(args.inputs / "variants-train.parquet", columns)))
    panels = select_panels(groups, args.seed, args.validation_groups)
    attach_sequences(panels, args.inputs / "sequences-train.parquet")
    payload = {
        "seed": args.seed,
        "genome": "GRCh38",
        "split": "train",
        "inputs_sha256": INPUTS,
        "source_revision": "4aed58e50c5dea0b878a665007af2ef9e5108e9f",
        "sequence_revision": "7b92f047f9a36f90e9ac47886afa2a99264ee35c",
        "demonstration_chromosome": "1",
        "eligible_groups": len(groups),
        "reference_validation": "materialized harness center and strand checks; independent genome validation pending",
        "panels": panels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
