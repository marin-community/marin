# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tasks on observed paired sequencing reads, including native FASTQ output checks."""

import hashlib
import json
import random
from functools import partial

import numpy as np

from experiments.post_training.bio_tasks.contract import Column, Contract, FastqContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Difficulty, Instance, Recipe


def generate_real_reads(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    block = rng.randrange(1, 4)
    inputs = {
        f"reads_R{mate}.fastq": source_text("ENA:ERR266411", f"err266411-block{block}-r{mate}.fastq.gz")
        for mate in (1, 2)
    }
    mates = []
    for text in inputs.values():
        lines = text.splitlines()
        records = np.array([lines[index : index + 4] for index in range(0, len(lines), 4)])
        names = [header.split()[0][1:] for header in records[:, 0]]
        qualities = np.array([[ord(base) - 33 for base in quality] for quality in records[:, 3]], dtype=np.int16)
        mates.append((records, names, qualities))
    if mates[0][1] != mates[1][1]:
        raise ValueError("Paired source IDs disagree")
    query = {"operation": operation, "source_block": block}
    prompt = (
        "Analyze the observed paired-end PhiX sequencing reads from ERR266411 in /app/inputs. "
        "The input contains 2000 intact pairs from one declared consecutive archive block; these blocks are "
        "technical subsets, not biological replicates. Use Phred+33 qualities and the first whitespace-delimited "
        "header token as the read ID. Corresponding R1/R2 records have the same ID. Read parameters from query.json. "
    )
    expected = {}
    wrong = {}
    artifacts = {}
    if operation == "real-fastq-pair-filter":
        query.update(
            qualified_phred=rng.choice((20, 25, 30)), maximum_unqualified_percent=40, maximum_ns=2, minimum_length=40
        )
        passed = []
        for records, _names, qualities in mates:
            unqualified = (qualities < query["qualified_phred"]).sum(axis=1)
            ns = np.array([sequence.count("N") for sequence in records[:, 1]])
            lengths = np.array(list(map(len, records[:, 1])))
            passed.append(
                (unqualified * 100 <= lengths * query["maximum_unqualified_percent"])
                & (ns <= query["maximum_ns"])
                & (lengths >= query["minimum_length"])
            )
        keep = passed[0] & passed[1]
        expected = {name: {"keep": int(value)} for name, value in zip(mates[0][1], keep, strict=True)}
        wrong = {name: {"keep": int(value)} for name, value in zip(mates[0][1], passed[0] | passed[1], strict=True)}
        for mate, (records, names, _qualities) in enumerate(mates, 1):
            digest = hashlib.sha256()
            for index in np.flatnonzero(keep):
                digest.update((names[index] + "\n" + records[index, 1] + "\n" + records[index, 3] + "\n").encode())
            artifacts[f"filtered_R{mate}.fastq"] = FastqContract(
                records=int(keep.sum()),
                max_bytes=len(inputs[f"reads_R{mate}.fastq"].encode()) + 1024,
                sha256=digest.hexdigest(),
            )
        columns = {
            "keep": Column(
                kind="integer", description="1 only when both mates pass every declared filter; else 0", unit="indicator"
            )
        }
        prompt += (
            "A base is unqualified when its Phred score is strictly below qualified_phred. A read passes when "
            "its unqualified percentage is <= maximum_unqualified_percent, N count <= maximum_ns, and length "
            ">= minimum_length. Keep a pair only when both mates pass. Do not trim, correct, merge or deduplicate "
            "reads. Report keep for every input pair and write filtered_R1.fastq and filtered_R2.fastq."
        )
        mutation = "retained_pair_when_only_one_mate_passed"
    elif operation == "real-fastq-fixed-trim":
        query.update(trim_front=rng.choice((3, 5, 8)), trim_tail=rng.choice((2, 4, 7)))
        for mate, (records, names, _qualities) in enumerate(mates, 1):
            digest = hashlib.sha256()
            bases = 0
            for index in range(len(records)):
                sequence = records[index, 1][query["trim_front"] : -query["trim_tail"]]
                quality = records[index, 3][query["trim_front"] : -query["trim_tail"]]
                digest.update((names[index] + "\n" + sequence + "\n" + quality + "\n").encode())
                bases += len(sequence)
            expected[f"R{mate}"] = {"reads": len(records), "bases": bases}
            wrong[f"R{mate}"] = {"reads": len(records), "bases": sum(map(len, records[:, 1]))}
            artifacts[f"trimmed_R{mate}.fastq"] = FastqContract(
                records=len(records),
                max_bytes=len(inputs[f"reads_R{mate}.fastq"].encode()) + 1024,
                sha256=digest.hexdigest(),
            )
        columns = {
            "reads": Column(kind="integer", description="retained reads in this mate file", unit="reads"),
            "bases": Column(kind="integer", description="total retained sequence length", unit="bases"),
        }
        prompt += (
            "Remove trim_front bases from the start and trim_tail bases from the end of every read, changing "
            "the quality string at the identical positions. Preserve every pair; perform no other trimming, "
            "filtering, correction or merging. Write trimmed_R1.fastq and trimmed_R2.fastq and report read/base "
            "counts for IDs R1 and R2."
        )
        mutation = "reported_untrimmed_yield"
    elif operation == "real-fastq-cycle-quality":
        for mate, (_records, _names, qualities) in enumerate(mates, 1):
            for cycle in range(qualities.shape[1]):
                column = qualities[:, cycle]
                key = f"R{mate}:{cycle + 1}"
                expected[key] = {
                    "mean_phred": float(column.mean()),
                    "q20": int((column >= 20).sum()),
                    "q30": int((column >= 30).sum()),
                    "bases": len(column),
                }
                wrong[key] = {**expected[key], "mean_phred": float(column.mean()) - 31}
        columns = {
            "mean_phred": Column(
                kind="number", description="arithmetic mean of Phred scores at this cycle", unit="Phred", atol=1e-10
            ),
            "q20": Column(kind="integer", description="scores >=20", unit="bases"),
            "q30": Column(kind="integer", description="scores >=30", unit="bases"),
            "bases": Column(kind="integer", description="reads contributing to this cycle", unit="bases"),
        }
        prompt += (
            "Report per-cycle mean Phred, Q20/Q30 base counts and denominator for each mate; "
            "use IDs R1:1, R1:2, ... and R2:1, R2:2, ... . Cycles are 1-based."
        )
        mutation = "decoded_phred64_instead_of_phred33"
    elif operation == "real-fastq-expected-errors":
        errors = [np.power(10.0, -qualities.astype(float) / 10).sum(axis=1) for _records, _names, qualities in mates]
        wrong_errors = [
            qualities.shape[1] * np.power(10.0, -qualities.mean(axis=1) / 10) for _records, _names, qualities in mates
        ]
        for index, name in enumerate(mates[0][1]):
            expected[name] = {"errors": float(errors[0][index] + errors[1][index])}
            wrong[name] = {"errors": float(wrong_errors[0][index] + wrong_errors[1][index])}
        columns = {
            "errors": Column(
                kind="number",
                description="sum of 10**(-Q/10) across both mates",
                unit="expected base errors",
                atol=1e-10,
                rtol=1e-10,
            )
        }
        prompt += (
            "For each pair ID, report the sum of per-base error probabilities 10**(-Q/10) across both mates. "
            "Sum probabilities, not Phred scores, and do not transform the mean quality. "
            "This is a quality-derived expectation, not an observed mismatch count."
        )
        mutation = "transformed_mean_quality_instead_of_summing_probabilities"
    elif operation == "real-fastq-quality-yield":
        qualities = np.concatenate([mate[2].reshape(-1) for mate in mates])
        expected = {
            "library": {
                "reads": sum(len(mate[0]) for mate in mates),
                "bases": int(qualities.size),
                "q20": int((qualities >= 20).sum()),
                "q30": int((qualities >= 30).sum()),
            }
        }
        wrong = {"library": {**expected["library"], "reads": len(mates[0][0])}}
        columns = {
            name: Column(kind="integer", description=description, unit=unit)
            for name, description, unit in (
                ("reads", "number of individual reads across both mates", "reads"),
                ("bases", "all sequenced bases", "bases"),
                ("q20", "sequenced bases with Phred >=20", "bases"),
                ("q30", "sequenced bases with Phred >=30", "bases"),
            )
        }
        prompt += (
            "Report total individual reads, bases, Q20 bases and Q30 bases across both mate files, with ID library. "
            "Count each mate as a read; apply no filtering, trimming or duplicate removal."
        )
        mutation = "counted_pairs_as_individual_reads"
    else:
        raise ValueError(operation)
    inputs["query.json"] = json.dumps(query) + "\n"
    return Instance(
        prompt,
        inputs,
        Contract(columns=columns, expected=expected, fastq=artifacts),
        {mutation: [{"id": key, **row} for key, row in wrong.items()]},
        data_origin=DataOrigin.REAL,
        source_ids=("ENA:ERR266411",),
        derivation=f"Unmodified observed FASTQ pairs {(block - 1) * 2000 + 1}-{block * 2000} from ERR266411. "
        "Original IDs, bases and qualities retained; three disjoint technical subsets, not biological replicates.",
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        difficulty,
        skills,
        ("FASTQ",),
        ("https://www.ebi.ac.uk/ena/browser/view/ERR266411",),
        partial(generate_real_reads, operation=name),
    )
    for name, difficulty, skills in (
        ("real-fastq-pair-filter", Difficulty.MEDIUM, ("paired-read-qc", "threshold-boundaries", "native-fastq-output")),
        (
            "real-fastq-fixed-trim",
            Difficulty.EASY,
            ("paired-reads", "sequence-quality-synchronization", "native-fastq-output"),
        ),
        ("real-fastq-cycle-quality", Difficulty.EASY, ("phred-encoding", "sequencing-cycles", "quality-denominators")),
        (
            "real-fastq-expected-errors",
            Difficulty.MEDIUM,
            ("phred-probabilities", "paired-read-qc", "nonlinear-aggregation"),
        ),
        ("real-fastq-quality-yield", Difficulty.EASY, ("phred-encoding", "read-pair-denominators", "quality-yield")),
    )
)
