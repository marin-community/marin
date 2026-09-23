# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native BLAST, BWA, Bowtie 2, minimap2, and DIAMOND mapping oracles."""

from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fasta, tab_rows


def unmapped_rows(inputs: Path) -> dict[str, dict]:
    return {
        name: {"id": name, "reference": None, "start": None, "end": None, "strand": "."}
        for name in fasta(inputs / "reads.fa")
    }


def solve_blast(inputs: Path, work: Path) -> list[dict]:
    output = execute(
        [
            "blastn",
            "-task",
            "blastn-short",
            "-query",
            str(inputs / "reads.fa"),
            "-subject",
            str(inputs / "reference.fa"),
            "-dust",
            "no",
            "-evalue",
            "1000",
            "-num_threads",
            "1",
            "-outfmt",
            "6 qseqid sseqid qlen qstart qend sstart send nident length sstrand",
        ],
        work,
        "blast.tsv",
    )
    answer = unmapped_rows(inputs)
    for query, target, length, qstart, qend, start, end, identity, aligned, strand in tab_rows(output):
        if int(qstart) != 1 or int(qend) != int(length) or int(identity) != int(length) or int(aligned) != int(length):
            continue
        assert answer[query]["reference"] is None, "Native BLAST reported multiple full exact placements"
        answer[query] = {
            "id": query,
            "reference": target,
            "start": min(int(start), int(end)) - 1,
            "end": max(int(start), int(end)),
            "strand": "+" if strand == "plus" else "-",
        }
    return list(answer.values())


def sam_placements(path: Path, inputs: Path) -> list[dict]:
    lengths = {name: len(sequence) for name, sequence in fasta(inputs / "reads.fa").items()}
    answer = unmapped_rows(inputs)
    for row in tab_rows(path):
        if row[0].startswith("@"):
            continue
        name, flag, reference, position = row[:4]
        if int(flag) & (4 | 256 | 2048):
            continue
        if row[5] != f"{lengths[name]}M" or "NM:i:0" not in row[11:]:
            continue
        assert answer[name]["reference"] is None, "Multiple primary exact placements"
        start = int(position) - 1
        answer[name] = {
            "id": name,
            "reference": reference,
            "start": start,
            "end": start + lengths[name],
            "strand": "-" if int(flag) & 16 else "+",
        }
    return list(answer.values())


def solve_bwa(inputs: Path, work: Path) -> list[dict]:
    index = str(work / "bwa-index")
    execute(["bwa", "index", "-p", index, str(inputs / "reference.fa")], work, "bwa-index.log")
    output = execute(["bwa", "mem", "-t", "1", "-k", "11", "-T", "0", index, str(inputs / "reads.fa")], work, "bwa.sam")
    return sam_placements(output, inputs)


def solve_bowtie(inputs: Path, work: Path) -> list[dict]:
    index = str(work / "bowtie-index")
    execute(["bowtie2-build", "--threads", "1", str(inputs / "reference.fa"), index], work, "bowtie-index.log")
    output = execute(
        [
            "bowtie2",
            "--threads",
            "1",
            "--end-to-end",
            "--very-sensitive",
            "--score-min",
            "L,0,0",
            "--n-ceil",
            "L,0,0",
            "-f",
            "-x",
            index,
            "-U",
            str(inputs / "reads.fa"),
        ],
        work,
        "bowtie.sam",
    )
    return sam_placements(output, inputs)


def solve_minimap(inputs: Path, work: Path) -> list[dict]:
    output = execute(
        [
            "minimap2",
            "-t",
            "1",
            "-k",
            "7",
            "-w",
            "1",
            "-m",
            "10",
            "-s",
            "10",
            "-n",
            "2",
            "--secondary=no",
            "-c",
            str(inputs / "reference.fa"),
            str(inputs / "reads.fa"),
        ],
        work,
        "minimap.paf",
    )
    answer = unmapped_rows(inputs)
    for row in tab_rows(output):
        query, length, start, end, strand, target, _, target_start, target_end, matches, block, _ = row[:12]
        if int(start) != 0 or int(end) != int(length) or int(matches) != int(length) or int(block) != int(length):
            continue
        assert answer[query]["reference"] is None, "Multiple exact primary query placements"
        answer[query] = {
            "id": query,
            "reference": target,
            "start": int(target_start),
            "end": int(target_end),
            "strand": strand,
        }
    return list(answer.values())


def solve_diamond(inputs: Path, work: Path) -> list[dict]:
    database = str(work / "proteins")
    execute(
        ["diamond", "makedb", "--in", str(inputs / "proteins.fa"), "--db", database, "--threads", "1"],
        work,
        "diamond-index.log",
    )
    output = execute(
        [
            "diamond",
            "blastp",
            "--query",
            str(inputs / "query.fa"),
            "--db",
            database,
            "--threads",
            "1",
            "--masking",
            "0",
            "--evalue",
            "10000",
            "--id",
            "100",
            "--query-cover",
            "100",
            "--max-target-seqs",
            "0",
            "--outfmt",
            "6",
            "sseqid",
            "sstart",
            "send",
            "pident",
            "qcovhsp",
        ],
        work,
        "diamond.tsv",
    )
    return [
        {
            "id": name,
            "start": int(start) - 1,
            "end": int(end),
            "identity": float(identity) / 100,
            "query_coverage": float(coverage) / 100,
        }
        for name, start, end, identity, coverage in tab_rows(output)
    ]


def solve_star(inputs: Path, work: Path) -> list[dict]:
    index = work / "index"
    index.mkdir()
    execute(
        [
            "STAR",
            "--runMode",
            "genomeGenerate",
            "--runThreadN",
            "1",
            "--genomeDir",
            str(index),
            "--genomeFastaFiles",
            str(inputs / "reference.fa"),
            "--genomeSAindexNbases",
            "2",
            "--genomeChrBinNbits",
            "9",
        ],
        work,
        "star-index.log",
    )
    prefix = str(work / "mapped.")
    execute(
        [
            "STAR",
            "--runThreadN",
            "1",
            "--genomeDir",
            str(index),
            "--readFilesIn",
            str(inputs / "reads.fa"),
            "--outFileNamePrefix",
            prefix,
            "--outSAMtype",
            "SAM",
            "--outSAMattributes",
            "NH",
            "HI",
            "AS",
            "nM",
            "NM",
            "--alignIntronMax",
            "1",
            "--outFilterMismatchNmax",
            "0",
            "--outFilterMultimapNmax",
            "1",
            "--outFilterScoreMinOverLread",
            "0",
            "--outFilterMatchNminOverLread",
            "1",
        ],
        work,
        "star-map.log",
    )
    return sam_placements(work / "mapped.Aligned.out.sam", inputs)
