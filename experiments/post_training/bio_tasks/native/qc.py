# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile sample accounting through FastQC and MultiQC data operations."""

from pathlib import Path

from experiments.post_training.bio_tasks.native.api import python_api
from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import table


def fastqc_read_count(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.startswith("Total Sequences\t"):
            return int(line.split("\t")[1])
    raise ValueError(f"FastQC Basic Statistics count missing: {path}")


def solve_fastqc(inputs: Path, work: Path) -> list[dict]:
    samples = [row for row in table(inputs / "samples.csv") if row["include"] == "1"]
    execute(
        [
            "fastqc",
            "--threads",
            "1",
            "--extract",
            "--outdir",
            str(work),
            *[str(inputs / row["fastq"]) for row in samples],
        ],
        work,
        "fastqc.log",
    )
    answer = []
    for sample in samples:
        report = work / (Path(sample["fastq"]).stem + "_fastqc") / "fastqc_data.txt"
        actual = fastqc_read_count(report)
        reported = fastqc_read_count(inputs / sample["report"])
        answer.append(
            {
                "id": sample["sample"],
                "actual_reads": actual,
                "reported_reads": reported,
                "difference": actual - reported,
                "matches": int(actual == reported),
            }
        )
    return answer


def solve_multiqc(inputs: Path, work: Path) -> list[dict]:
    return python_api(
        """import csv
import json
import shutil
import sys
from pathlib import Path
import multiqc

inputs, output = map(Path, sys.argv[1:])
with (inputs / "samples.csv").open() as handle:
    samples = [row for row in csv.DictReader(handle) if row["include"] == "1"]
reports = output.parent / "reports"
for sample in samples:
    directory = reports / (sample["sample"] + "_fastqc")
    directory.mkdir(parents=True)
    shutil.copyfile(inputs / sample["report"], directory / "fastqc_data.txt")
multiqc.parse_logs(str(reports), run_modules=["fastqc"], no_version_check=True, quiet=True)
parsed = multiqc.get_module_data(module="fastqc")
(output.parent / "parsed_reports.json").write_text(json.dumps(parsed, indent=2) + "\\n")
answer = []
for sample in samples:
    with (inputs / sample["fastq"]).open() as handle:
        lines = sum(1 for _ in handle)
    assert lines % 4 == 0
    actual = lines // 4
    reported = int(parsed[sample["sample"]]["Total Sequences"])
    answer.append({"id": sample["sample"], "actual_reads": actual, "reported_reads": reported,
                   "difference": actual - reported, "matches": int(actual == reported)})
output.write_text(json.dumps(answer) + "\\n")
""",
        inputs,
        work,
    )
