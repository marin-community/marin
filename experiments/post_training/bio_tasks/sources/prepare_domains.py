# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search a complete observed proteome and independently measure domain artifacts."""

import argparse
import csv
import gzip
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from Bio import SearchIO, SeqIO


def prepare(inputs: Path, output: Path) -> None:
    """Run HMMER and derive complete reference tables using Biopython coordinates."""
    output.mkdir(parents=True, exist_ok=False)
    query = json.loads((inputs / "query.json").read_text())
    proteins = SeqIO.to_dict(SeqIO.parse(inputs / "proteins.fa", "fasta"))
    with (inputs / "proteins.tsv").open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    metadata = {row["Entry"]: row for row in rows}
    if len(metadata) != len(rows) or set(metadata) != {name.split("|")[1] for name in proteins}:
        raise ValueError("Proteome and version metadata identities disagree")
    command = [
        "hmmsearch",
        "--cpu",
        "1",
        "--seed",
        str(query["seed"]),
        "--cut_ga",
        "--noali",
        "--domtblout",
        str(output / "domains.domtblout"),
        str(inputs / "models.hmm"),
        str(inputs / "proteins.fa"),
    ]
    started = time.monotonic()
    with (output / "hmmsearch.stdout").open("w") as stdout, (output / "hmmsearch.stderr").open("w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=900)
    execution = {"argv": command, "seconds": time.monotonic() - started}
    domains = {}
    model_proteins = {accession: set() for accession in query["model_accessions"]}
    model_domains = dict.fromkeys(model_proteins, 0)
    model_coverage = {accession: {} for accession in model_proteins}
    protein_domains = dict.fromkeys(metadata, 0)
    protein_models = {accession: set() for accession in metadata}
    protein_coverage = {accession: set() for accession in metadata}
    for result in SearchIO.parse(output / "domains.domtblout", "hmmsearch3-domtab"):
        model = result.accession
        if model not in model_proteins:
            raise ValueError(f"Unexpected model accession: {model}")
        for hit in result:
            accession = hit.id.split("|")[1]
            sequence = str(proteins[hit.id].seq)
            if len(sequence) != hit.seq_len or len(sequence) != int(metadata[accession]["Length"]):
                raise ValueError(f"Protein length differs between native search and input: {accession}")
            for hsp in hit:
                if not 0 <= hsp.hit_start < hsp.hit_end <= len(sequence):
                    raise ValueError("Domain coordinates exceed the observed protein")
                identity = f"{model}:{accession}:{hsp.domain_index}"
                if identity in domains:
                    raise ValueError(f"Repeated domain identity: {identity}")
                domains[identity] = {
                    "protein": accession,
                    "model": model,
                    "domain_index": hsp.domain_index,
                    "alignment_start": hsp.hit_start + 1,
                    "alignment_end": hsp.hit_end,
                    "envelope_start": hsp.env_start + 1,
                    "envelope_end": hsp.env_end,
                    "model_start": hsp.query_start + 1,
                    "model_end": hsp.query_end,
                    "protein_score": hit.bitscore,
                    "domain_score": hsp.bitscore,
                    "independent_evalue": hsp.evalue,
                    "conditional_evalue": hsp.evalue_cond,
                    "sequence": sequence[hsp.hit_start : hsp.hit_end],
                }
                model_domains[model] += 1
                model_proteins[model].add(accession)
                model_coverage[model].setdefault(accession, set()).update(range(hsp.hit_start, hsp.hit_end))
                protein_domains[accession] += 1
                protein_models[accession].add(model)
                protein_coverage[accession].update(range(hsp.hit_start, hsp.hit_end))
    if not domains:
        raise ValueError("This candidate produced no domains; review the task before admitting it")
    protein_rows = {
        accession: {
            "length": int(metadata[accession]["Length"]),
            "sequence_version": int(metadata[accession]["Sequence version"]),
            "domains": protein_domains[accession],
            "distinct_models": len(protein_models[accession]),
            "covered_residues": len(protein_coverage[accession]),
        }
        for accession in sorted(metadata)
    }
    summaries = {
        model: {
            "searched_proteins": len(proteins),
            "matched_proteins": len(model_proteins[model]),
            "domains": model_domains[model],
            "covered_residues": sum(map(len, model_coverage[model].values())),
        }
        for model in model_proteins
    }
    for filename, records in [("domains.tsv", domains), ("proteins.tsv", protein_rows)]:
        with (output / filename).open("w") as destination:
            writer = csv.DictWriter(destination, fieldnames=["id", *next(iter(records.values()))], delimiter="\t")
            writer.writeheader()
            writer.writerows({"id": key, **records[key]} for key in sorted(records))
    (output / "answer.json").write_text(json.dumps([{"id": key, **value} for key, value in summaries.items()]) + "\n")
    reference = {
        "query": query,
        "summaries": summaries,
        "domains": domains,
        "proteins": protein_rows,
        "execution": execution,
        "biopython_version": version("biopython"),
        "coordinate_policy": "1-based closed intervals; sequences use alignment boundaries, not envelopes.",
        "coverage_policy": "Union of aligned residue positions per protein; overlapping domains do not double count.",
    }
    (output / "reference.json.gz").write_bytes(gzip.compress((json.dumps(reference, indent=2) + "\n").encode(), mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs, args.output)


if __name__ == "__main__":
    main()
