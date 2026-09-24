# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additional sequence operations grounded in the inspected repository inventory."""

import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Instance, Recipe, csv_text


def generate_repo_sequences(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "dna-unique-mapping":
        # Distinct random flanks and long exact queries make accidental matches negligible;
        # explicitly reject ambiguity before accepting a generated fixture.
        starts = [rng.randint(10, 50), rng.randint(110, 150), rng.randint(240, 280)]
        read_length = rng.randint(35, 45)
        while True:
            genome = "".join(rng.choice("ACGT") for _ in range(360))
            complement = str.maketrans("ACGT", "TGCA")
            fragments = [genome[start : start + read_length] for start in starts]
            queries = [fragments[0], fragments[1].translate(complement)[::-1], fragments[2], "N" * read_length]
            if all(
                genome.count(word) == 1 and genome.count(word.translate(complement)[::-1]) == 0 for word in fragments
            ):
                break
        inputs = {
            "reference.fa": ">chr1\n" + genome + "\n",
            "reads.fa": "".join(f">q{i}\n{query}\n" for i, query in enumerate(queries)),
        }
        for i, (start, strand) in enumerate([(starts[0], "+"), (starts[1], "-"), (starts[2], "+"), (None, ".")]):
            expected[f"q{i}"] = {
                "reference": "chr1" if start is not None else None,
                "start": start,
                "end": start + read_length if start is not None else None,
                "strand": strand,
            }
        columns = {
            "reference": Column(
                kind="text", unit="sequence ID", description="unique exact-match reference or null", nullable=True
            ),
            "start": Column(
                kind="integer", unit="0-based bases", description="inclusive reference start or null", nullable=True
            ),
            "end": Column(
                kind="integer", unit="0-based bases", description="exclusive reference end or null", nullable=True
            ),
            "strand": Column(kind="text", unit="orientation", description="+ or -; . when unmapped"),
        }
        prompt = (
            "Map reads.fa to reference.fa, requiring a full-length exact A/C/G/T match on either "
            "strand. All mappable reads have exactly one placement. Report reference ID, 0-based "
            "half-open span, and strand for every query. Queries containing N have no exact A/C/G/T "
            "placement and must be unmapped with null reference/start/end and strand=.. Do not discard "
            "unmapped IDs."
        )
        wrong = [{"id": k, **v, "strand": "+" if v["strand"] == "-" else v["strand"]} for k, v in expected.items()]
        reason = "ignored_reverse_strand_mapping"
    elif operation == "protein-local-search":
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        core = "".join(rng.choice(alphabet) for _ in range(35))
        prefix = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 20)))
        suffix = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 20)))
        database = {"target": prefix + core + suffix, "short": core[:12], "decoy": "W" * 60}
        inputs = {
            "proteins.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in database.items()),
            "query.fa": ">query\n" + core + "\n",
        }
        expected = {
            "target": {"start": len(prefix), "end": len(prefix) + len(core), "identity": 1.0, "query_coverage": 1.0}
        }
        columns = {
            "start": Column(kind="integer", unit="0-based residues", description="inclusive target start"),
            "end": Column(kind="integer", unit="0-based residues", description="exclusive target end"),
            "identity": Column(
                kind="number", unit="fraction", description="exact aligned residue identity", atol=1e-10, rtol=1e-8
            ),
            "query_coverage": Column(
                kind="number",
                unit="fraction",
                description="matched query residues divided by full query length",
                atol=1e-10,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Find proteins.fa targets containing the entire query.fa protein as an exact contiguous "
            "match. Require 100% query coverage and identity; a short perfect local match does not "
            "qualify. Return each qualifying target ID with its 0-based half-open matching span, "
            "identity fraction, and query-coverage fraction. This bounded search does not ask for "
            "E-values or infer homology from an arbitrary score."
        )
        wrong = [
            {"id": k, **v, "query_coverage": len(core) / (len(prefix) + len(core) + len(suffix))}
            for k, v in expected.items()
        ]
        reason = "used_target_length_for_query_coverage"
    elif operation == "alignment-sum-of-pairs":
        columns = ["AA-", "AC-", "GGG", "T-T", "-CA", "NNN"]
        rng.shuffle(columns)
        sequences = {f"s{i}": "".join(column[i] for column in columns) for i in range(3)}
        inputs = {"alignment.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items())}
        known = {
            "AA-": (1, 0, 2),
            "AC-": (0, 1, 2),
            "GGG": (3, 0, 0),
            "T-T": (1, 0, 2),
            "-CA": (0, 1, 2),
            "NNN": (0, 0, 0),
        }
        for i, column in enumerate(columns):
            match, mismatch, gap = known[column]
            expected[str(i)] = {
                "matches": match,
                "mismatches": mismatch,
                "gap_pairs": gap,
                "score": 2 * match - mismatch - 2 * gap,
            }
        columns_schema = {
            name: Column(
                kind="integer",
                unit="score units" if name == "score" else "unordered sequence pairs",
                description=description,
            )
            for name, description in [
                ("matches", "equal A/C/G/T pairs"),
                ("mismatches", "different A/C/G/T pairs"),
                ("gap_pairs", "one gap and one called base"),
                ("score", "2*matches - mismatches - 2*gap_pairs"),
            ]
        }
        columns = columns_schema
        prompt = (
            "Score every column of alignment.fa by all unordered sequence pairs. A/C/G/T match=+2, "
            "mismatch=-1, exactly one gap paired with A/C/G/T=-2, gap-gap=0. Any pair involving N "
            "contributes zero to both score and counts. Report match, mismatch and penalized-gap pair "
            "counts plus score per 0-based column. This is a linear-gap sum-of-pairs audit of a "
            "supplied alignment, not a claim of global optimality."
        )
        wrong = [{"id": k, **v, "score": v["score"] + v["gap_pairs"]} for k, v in expected.items()]
        reason = "used_wrong_gap_penalty"
    elif operation == "hmmer-domain-extraction":
        sequence = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(65))
        lines = ["# HMMER domtblout profile; target is protein, query is domain HMM"]
        for index, (start, end) in enumerate([(8, 20), (35, 48)], 1):
            lines.append(
                f"protein - 65 DomainX PF00001 20 1e-20 80.0 0.0 {index} 2 1e-10 1e-10 40.0 0.0 "
                f"1 {end-start+1} {start} {end} {start-2} {end+3} 0.99 example domain"
            )
            expected[f"protein:DomainX:{index}"] = {
                "sequence": sequence[start - 1 : end],
                "start": start - 1,
                "end": end,
            }
        inputs = {"proteins.fa": ">protein\n" + sequence + "\n", "domains.domtblout": "\n".join(lines) + "\n"}
        columns = {
            "sequence": Column(kind="text", unit="amino acids", description="aligned target domain sequence"),
            "start": Column(kind="integer", unit="0-based residues", description="inclusive aligned-domain start"),
            "end": Column(kind="integer", unit="0-based residues", description="exclusive aligned-domain end"),
        }
        prompt = (
            "Extract each aligned protein domain from proteins.fa using HMMER domains.domtblout. Use "
            "target name (field 1), query/domain name (4), domain number (10), and ali_from/ali_to "
            "(18/19), which are 1-based inclusive target coordinates. Do not use HMM coordinates or "
            "wider envelope coordinates. Return 0-based half-open spans and sequence, "
            "id=target:query:domain_number. Ignore # comments and free-text descriptions."
        )
        wrong = [{"id": k, **v, "start": v["start"] - 2} for k, v in expected.items()]
        reason = "used_envelope_instead_of_alignment_coordinates"
    elif operation == "sequence-identity-clusters":
        length = rng.randint(12, 18)
        connected_c = rng.choice([True, False])
        connected_e = rng.choice([True, False])
        sequences = {
            "a": "A" * length,
            "b": "C" + "A" * (length - 1),
            "c": "C" * (2 if connected_c else 3) + "A" * (length - (2 if connected_c else 3)),
            "d": "G" * length,
            "e": "G" * (length - (1 if connected_e else 3)) + "T" * (1 if connected_e else 3),
            "f": "W" * length,
        }
        inputs = {
            "proteins.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items()),
            "max_mismatches.txt": "1\n",
        }
        groups = [("a", "b", "c")] if connected_c else [("a", "b"), ("c",)]
        groups += ([("d", "e")] if connected_e else [("d",), ("e",)]) + [("f",)]
        for group in groups:
            for name in group:
                expected[name] = {"representative": min(group), "size": len(group)}
        columns = {
            "representative": Column(
                kind="text", unit="protein ID", description="lexically smallest member of single-linkage component"
            ),
            "size": Column(kind="integer", unit="proteins", description="component size"),
        }
        prompt = (
            "Cluster equal-length proteins.fa sequences by single linkage: connect two proteins if "
            "full-length ungapped Hamming mismatches <= max_mismatches.txt, then take connected "
            "components. Coverage is 100%; report every protein with the lexically smallest member as "
            "representative and component size. A component need not satisfy the pairwise cutoff for "
            "every member pair. This exact miniature contract specifies the clustering mode."
        )
        wrong = [{"id": k, **v, "representative": "f" if k == "a" else v["representative"]} for k, v in expected.items()]
        reason = "merged_unrelated_sequence_families"
    else:
        assert operation == "fasta-indexed-regions"
        records = {f"chr{i}": "".join(rng.choice("ACGT") for _ in range(75 + i * 5)) for i in range(2)}
        lines = []
        index = []
        offset = 0
        for name, sequence in records.items():
            header = f">{name} reference contig\n"
            lines.append(header)
            offset += len(header)
            index.append(f"{name}\t{len(sequence)}\t{offset}\t11\t12\n")
            wrapped = "".join(sequence[i : i + 11] + "\n" for i in range(0, len(sequence), 11))
            lines.append(wrapped)
            offset += len(wrapped)
        regions = []
        for i, (name, start, end) in enumerate([("chr0", 8, 25), ("chr1", 1, 1), ("chr1", 68, 80)]):
            regions.append({"region": f"r{i}", "contig": name, "start": start, "end": end})
            expected[f"r{i}"] = {"sequence": records[name][start - 1 : end]}
        inputs = {"reference.fa": "".join(lines), "reference.fa.fai": "".join(index), "regions.csv": csv_text(regions)}
        columns = {"sequence": Column(kind="text", unit="bases", description="requested 1-based closed region sequence")}
        prompt = (
            "Retrieve regions.csv intervals from reference.fa using its supplied .fai index. Region "
            "start/end are 1-based inclusive, unlike BED. The FASTA is line-wrapped and has "
            "descriptions after contig names; account for line bytes versus bases and cross-line spans."
            " Return exact sequence using region id."
        )
        wrong = [{"id": k, "sequence": v["sequence"][1:]} for k, v in expected.items()]
        reason = "treated_region_start_as_zero_based"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "dna-unique-mapping": ("reference-mapping", "orientation", "unmapped-queries"),
    "protein-local-search": ("query-coverage", "local-search", "identity"),
    "alignment-sum-of-pairs": ("alignment-scoring", "gap-policy", "residue-correspondence"),
    "hmmer-domain-extraction": ("domtblout", "alignment-versus-envelope", "domain-identities"),
    "sequence-identity-clusters": ("sequence-clustering", "transitive-membership", "coverage-policy"),
    "fasta-indexed-regions": ("fasta-index", "line-wrapping", "coordinate-conversion"),
}
FORMATS = {
    "dna-unique-mapping": ("fasta",),
    "protein-local-search": ("fasta",),
    "alignment-sum-of-pairs": ("aligned-fasta",),
    "hmmer-domain-extraction": ("fasta", "hmmer-domtblout"),
    "sequence-identity-clusters": ("fasta",),
    "fasta-indexed-regions": ("fasta", "fai", "csv-header"),
}
RECIPES = tuple(
    Recipe(name, "1", skills, FORMATS[name], (), partial(generate_repo_sequences, operation=name))
    for name, skills in SKILLS.items()
)
