# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute sample-keyed lane aggregation through actual workflow engines."""

import json
import sys
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute

LANE_COUNTER = """import argparse
import csv
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("lanes", nargs="+", type=Path)
args = parser.parse_args()
counts = Counter()
for lane in args.lanes:
    with lane.open() as handle:
        for row in csv.DictReader(handle):
            counts[row["gene"]] += int(row["count"])
args.output.write_text(json.dumps([
    {"id": args.sample + ":" + gene, "count": count, "lanes": len(args.lanes)}
    for gene, count in sorted(counts.items())
]) + "\\n")
"""

SNAKEFILE = """import csv
from collections import defaultdict
from pathlib import Path

lanes = defaultdict(list)
with open(config["manifest"]) as handle:
    for row in csv.DictReader(handle):
        if row["include"] == "1":
            lanes[row["sample"]].append(str(Path(config["inputs"]) / row["counts"]))

rule all:
    input:
        expand("results/{sample}.json", sample=sorted(lanes))

rule merge_lanes:
    input:
        lambda wildcards: lanes[wildcards.sample]
    output:
        "results/{sample}.json"
    params:
        python=config["python"], counter=config["counter"]
    threads: 1
    shell:
        "{params.python:q} {params.counter:q} --sample {wildcards.sample:q} --output {output:q} {input:q}"
"""

NEXTFLOW = '''nextflow.enable.dsl=2

process MERGE_LANES {
    cpus 1
    maxForks 1
    publishDir params.results, mode: 'copy'

    input:
    tuple val(sample), path(counts)
    path counter

    output:
    path "${sample}.json"

    script:
    def lane_args = counts.collect { "'${it}'" }.join(' ')
    """
    python '${counter}' --sample '${sample}' --output '${sample}.json' ${lane_args}
    """
}

workflow {
    lanes = Channel.fromPath(params.manifest)
        .splitCsv(header: true)
        .filter { row -> row.include == '1' }
        .map { row -> tuple(row.sample, file("${params.inputs}/${row.counts}")) }
        .groupTuple()
    MERGE_LANES(lanes, file(params.counter))
}
'''


def workflow_parameters(inputs: Path, work: Path) -> Path:
    counter = work / "merge_lanes.py"
    counter.write_text(LANE_COUNTER)
    parameters = work / "parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "manifest": str(inputs / "samples.csv"),
                "inputs": str(inputs),
                "counter": str(counter),
                "results": str(work / "results"),
                "python": sys.executable,
            }
        )
        + "\n"
    )
    return parameters


def workflow_answers(work: Path) -> list[dict]:
    paths = sorted((work / "results").glob("*.json"))
    assert paths, "Workflow produced no biological sample counts"
    return [row for path in paths for row in json.loads(path.read_text())]


def solve_snakemake(inputs: Path, work: Path) -> list[dict]:
    parameters = workflow_parameters(inputs, work)
    snakefile = work / "Snakefile"
    snakefile.write_text(SNAKEFILE)
    execute(
        [
            "snakemake",
            "--snakefile",
            str(snakefile),
            "--cores",
            "1",
            "--scheduler",
            "greedy",
            "--configfile",
            str(parameters),
        ],
        work,
        "snakemake.log",
    )
    return workflow_answers(work)


def solve_nextflow(inputs: Path, work: Path) -> list[dict]:
    parameters = workflow_parameters(inputs, work)
    pipeline = work / "main.nf"
    pipeline.write_text(NEXTFLOW)
    configuration = work / "nextflow.config"
    configuration.write_text("process.executor = 'local'\nexecutor.queueSize = 1\n")
    execute(
        [
            "nextflow",
            "-c",
            str(configuration),
            "run",
            str(pipeline),
            "-offline",
            "-params-file",
            str(parameters),
            "-with-trace",
            str(work / "trace.tsv"),
        ],
        work,
        "nextflow.log",
    )
    return workflow_answers(work)
