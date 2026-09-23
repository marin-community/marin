# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a repository's native data operation without loading its private reference."""

import argparse
import hashlib
import json
import signal
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from experiments.post_training.bio_tasks.native.api import (
    solve_biopython,
    solve_biostrings,
    solve_deseq,
    solve_edger,
    solve_genomicranges,
    solve_limma,
    solve_pybedtools,
    solve_pysam,
    solve_scanpy,
    solve_seurat,
)
from experiments.post_training.bio_tasks.native.htslib import solve_htslib
from experiments.post_training.bio_tasks.native.intervals import solve_bedtools, solve_kent, solve_macs
from experiments.post_training.bio_tasks.native.mapping import (
    solve_blast,
    solve_bowtie,
    solve_bwa,
    solve_diamond,
    solve_minimap,
    solve_star,
)
from experiments.post_training.bio_tasks.native.qc import solve_fastqc, solve_multiqc
from experiments.post_training.bio_tasks.native.reads import solve_cutadapt, solve_plink, solve_samtools, solve_vcftools
from experiments.post_training.bio_tasks.native.variants import solve_bcftools, solve_gatk
from experiments.post_training.bio_tasks.native.workflow import solve_nextflow, solve_snakemake


@dataclass(frozen=True)
class NativeOperation:
    recipe: str
    solve: Callable[[Path, Path], list[dict]]
    packages: tuple[str, ...] = ()


OPERATIONS = {
    1: NativeOperation("dna-unique-mapping", solve_blast),
    2: NativeOperation("sam-cigar-coverage", solve_samtools),
    3: NativeOperation("dna-unique-mapping", solve_bwa),
    4: NativeOperation("dna-unique-mapping", solve_bowtie),
    5: NativeOperation("bulk-size-factors", solve_deseq, ("r-jsonlite",)),
    6: NativeOperation("dna-unique-mapping", solve_star),
    7: NativeOperation("bed12-exons", solve_bedtools),
    8: NativeOperation("vcf-site-filtering", solve_gatk),
    9: NativeOperation("sam-cigar-coverage", solve_pysam),
    12: NativeOperation("matrixmarket-log-normalization", solve_seurat, ("r-jsonlite",)),
    13: NativeOperation("dna-unique-mapping", solve_minimap),
    14: NativeOperation("vcf-allelic-depth", solve_bcftools),
    16: NativeOperation("sample-sheet-lanes", solve_snakemake),
    17: NativeOperation("fasta-indexed-regions", solve_htslib, ("c-compiler", "pkg-config")),
    18: NativeOperation("fastqc-report-reconciliation", solve_fastqc),
    19: NativeOperation("gtf-splicing", solve_biopython),
    20: NativeOperation("sample-sheet-lanes", solve_nextflow),
    21: NativeOperation("protein-local-search", solve_diamond),
    22: NativeOperation("vcf-sample-qc", solve_plink),
    23: NativeOperation("matrixmarket-cell-qc", solve_scanpy),
    24: NativeOperation("fastqc-report-reconciliation", solve_multiqc),
    25: NativeOperation("bulk-cpm-filter", solve_edger, ("r-jsonlite",)),
    26: NativeOperation("adjusted-linear-effect", solve_limma, ("r-jsonlite",)),
    31: NativeOperation("fastq-adapter-trimming", solve_cutadapt),
    38: NativeOperation("vcf-sample-qc", solve_vcftools),
    41: NativeOperation("bedgraph-threshold-peaks", solve_macs),
    46: NativeOperation("bedgraph-weighted-signal", solve_kent),
    47: NativeOperation("interval-overlap", solve_genomicranges, ("r-jsonlite",)),
    48: NativeOperation("fasta-six-frame-translation", solve_biostrings, ("r-jsonlite",)),
    49: NativeOperation("strand-extraction", solve_pybedtools, ("bedtools=2.31.1",)),
}


def run_operation(repository: int, inputs: Path, output: Path) -> None:
    """Retain operation evidence; a separate process must grade answer.json."""
    operation = OPERATIONS[repository]
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(path.relative_to(inputs)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs.rglob("*"))
        if path.is_file()
    }
    status = {
        "repository_index": repository,
        "recipe": operation.recipe,
        "input_sha256": hashes,
        "execution": "running",
        "verification": "pending",
    }
    try:
        answer = operation.solve(inputs, output)
        (output / "answer.json").write_text(json.dumps(answer, allow_nan=False, indent=2) + "\n")
        status["execution"] = "completed"
    except Exception:
        # Retain evidence, then fail the operation. No generated answer or reward on a tool failure.
        status["execution"] = "failed"
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (output / "execution.json").write_text(json.dumps(status, indent=2) + "\n")


def interrupt_operation(signum: int, frame: FrameType | None) -> None:
    """Unwind command scopes so worker timeouts also reap the tool subprocesses."""
    raise TimeoutError(f"Native check interrupted by signal {signum}")


def main() -> None:
    signal.signal(signal.SIGTERM, interrupt_operation)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=int, choices=sorted(OPERATIONS), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_operation(args.repository, args.inputs, args.output)


if __name__ == "__main__":
    main()
