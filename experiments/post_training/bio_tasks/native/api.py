# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package API operations run in their isolated Python or R environment."""

import json
import sys
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute


def python_api(source: str, inputs: Path, work: Path) -> list[dict]:
    script = work / "operation.py"
    script.write_text(source)
    answer = work / "answer.json"
    execute([sys.executable, str(script), str(inputs), str(answer)], work, "api.log")
    return json.loads(answer.read_text())


def r_api(source: str, inputs: Path, work: Path) -> list[dict]:
    script = work / "operation.R"
    script.write_text(
        "args <- commandArgs(trailingOnly=TRUE)\nsetwd(args[1])\n"
        + source
        + '\njsonlite::write_json(answer, args[2], dataframe="rows", auto_unbox=TRUE, digits=16, na="null")\n'
    )
    answer = work / "answer.json"
    execute(["Rscript", "--vanilla", str(script), str(inputs), str(answer)], work, "api.log")
    return json.loads(answer.read_text())


def solve_deseq(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(DESeq2)
x <- as.matrix(read.csv("counts.csv", row.names=1, check.names=FALSE))
factors <- estimateSizeFactorsForMatrix(x)
answer <- data.frame(id=colnames(x), size_factor=unname(factors), eligible_genes=sum(rowSums(x > 0)==ncol(x)))
""",
        inputs,
        work,
    )


def solve_edger(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(edgeR)
x <- as.matrix(read.csv("counts.csv", row.names=1, check.names=FALSE))
cutoff <- scan("threshold.txt", quiet=TRUE)
n <- rowSums(cpm(x, log=FALSE) >= cutoff)
answer <- data.frame(id=rownames(x), n_samples=as.integer(n), keep=as.integer(n>=2))
""",
        inputs,
        work,
    )


def solve_limma(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(limma)
x <- read.csv("patients.csv")
fit <- lmFit(matrix(x$outcome, nrow=1), model.matrix(~treatment+batch, x))
answer <- data.frame(id="cohort", treatment_effect=unname(fit$coefficients[1,"treatment"]),
                     batch_effect=unname(fit$coefficients[1,"batch"]), n_patients=nrow(x))
""",
        inputs,
        work,
    )


def solve_seurat(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(Seurat)
f <- read.delim("features.tsv", header=FALSE)
b <- readLines("barcodes.tsv")
x <- Matrix::readMM("matrix.mtx")[f$V3=="Gene Expression",,drop=FALSE]
rownames(x) <- f$V1[f$V3=="Gene Expression"]
colnames(x) <- b
nonempty <- Matrix::colSums(x) > 0
normalized <- x
normalized[,nonempty] <- LogNormalize(x[,nonempty,drop=FALSE], scale.factor=10000, verbose=FALSE)
answer <- do.call(rbind, lapply(seq_along(b), function(j) data.frame(
    id=paste(rownames(x), b[j], sep=":"), log_count=as.numeric(normalized[,j]))))
""",
        inputs,
        work,
    )


def solve_genomicranges(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(GenomicRanges)
f <- read.delim("features.bed", header=FALSE)
p <- read.delim("peaks.bed", header=FALSE)
features <- GRanges(f$V1, IRanges(f$V2+1L, f$V3))
peaks <- GRanges(p$V1, IRanges(p$V2+1L, p$V3))
hits <- findOverlaps(features, peaks, minoverlap=scan("minimum.txt", quiet=TRUE))
answer <- data.frame(id=f$V4, peak_count=tabulate(queryHits(hits), nbins=nrow(f)))
""",
        inputs,
        work,
    )


def solve_biostrings(inputs: Path, work: Path) -> list[dict]:
    return r_api(
        """
library(Biostrings)
x <- readDNAStringSet("sequences.fa")
answer <- list()
for (i in seq_along(x)) {
  for (strand in c("+", "-")) {
    sequence <- if (strand=="+") x[[i]] else reverseComplement(x[[i]])
    for (frame in 0:2) {
      n <- (length(sequence)-frame)%/%3*3
      peptide <- as.character(translate(subseq(sequence, start=frame+1, width=n), no.init.codon=TRUE))
      answer[[length(answer)+1]] <- list(id=paste0(names(x)[i], ":", strand, frame+1), protein=peptide)
    }
  }
}
""",
        inputs,
        work,
    )


def solve_biopython(inputs: Path, work: Path) -> list[dict]:
    return python_api(
        """
import json, re, sys
from collections import defaultdict
from pathlib import Path
from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation, SimpleLocation, SeqFeature
inputs, output = map(Path, sys.argv[1:])
genome = SeqIO.to_dict(SeqIO.parse(inputs / "genome.fa", "fasta"))
groups = defaultdict(list)
for line in (inputs / "annotations.gtf").read_text().splitlines():
    row = line.split("\\t")
    if row[2] == "exon":
        name = re.search(r'transcript_id "([^"]+)"', row[8]).group(1)
        groups[name].append(row)
answer = []
for name, rows in groups.items():
    strand = 1 if rows[0][6] == "+" else -1
    rows.sort(key=lambda r:int(r[3]), reverse=strand==-1)
    locations = [SimpleLocation(int(r[3])-1, int(r[4]), strand=strand) for r in rows]
    feature = SeqFeature(CompoundLocation(locations))
    sequence = str(feature.extract(genome[rows[0][0]].seq))
    answer.append(dict(id=name, sequence=sequence, length=len(sequence)))
output.write_text(json.dumps(answer))
""",
        inputs,
        work,
    )


def solve_pysam(inputs: Path, work: Path) -> list[dict]:
    return python_api(
        """
import json, sys
from collections import Counter
from pathlib import Path
import pysam
inputs, output = map(Path, sys.argv[1:])
depth = Counter()
with pysam.AlignmentFile(inputs / "reads.sam", "r") as alignment:
    for read in alignment:
        if read.flag & (4 | 256 | 2048):
            continue
        for query, reference in read.get_aligned_pairs(matches_only=True):
            depth[read.reference_name, reference] += 1
answer = []
for line in (inputs / "windows.bed").read_text().splitlines():
    chrom, start, end, name = line.split("\\t")
    counts = [depth[chrom, position] for position in range(int(start), int(end))]
    answer.append(dict(id=name, depth_sum=sum(counts), covered_bases=sum(c>0 for c in counts)))
output.write_text(json.dumps(answer))
""",
        inputs,
        work,
    )


def solve_scanpy(inputs: Path, work: Path) -> list[dict]:
    return python_api(
        """
import json, sys
from pathlib import Path
import scanpy as sc
from scipy.io import mmread
from anndata import AnnData
inputs, output = map(Path, sys.argv[1:])
features = [line.split("\\t") for line in (inputs / "features.tsv").read_text().splitlines()]
barcodes = (inputs / "barcodes.tsv").read_text().splitlines()
genes = [i for i, row in enumerate(features) if row[2] == "Gene Expression"]
data = AnnData(mmread(inputs / "matrix.mtx").tocsr()[genes].T.tocsr())
data.obs_names = barcodes
data.var_names = [features[i][0] for i in genes]
data.var["mt"] = [features[i][1].startswith("MT-") for i in genes]
sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
answer = [dict(id=name, counts=int(row.total_counts), features=int(row.n_genes_by_counts),
               mt_fraction=float(row.pct_counts_mt)/100 if row.total_counts else None)
          for name, row in data.obs.iterrows()]
output.write_text(json.dumps(answer))
""",
        inputs,
        work,
    )


def solve_pybedtools(inputs: Path, work: Path) -> list[dict]:
    return python_api(
        """
import json, sys
from pathlib import Path
from pybedtools import BedTool
inputs, output = map(Path, sys.argv[1:])
rows = [line.split("\\t") for line in (inputs / "annotations.gff3").read_text().splitlines()
        if not line.startswith("#")]
bed = []
for row in rows:
    if row[2] == "exon":
        attrs = dict(item.split("=",1) for item in row[8].split(";"))
        bed.append("\\t".join([row[0], str(int(row[3])-1), row[4], attrs["Parent"], "0", row[6]]))
result = BedTool("\\n".join(bed)+"\\n", from_string=True).sequence(fi=str(inputs/"genome.fa"), s=True, nameOnly=True)
lines = Path(result.seqfn).read_text().splitlines()
answer = []
for i in range(0, len(lines), 2):
    name = lines[i][1:].split("(")[0]
    sequence = lines[i+1]
    answer.append(dict(id=name, sequence=sequence, length=len(sequence)))
output.write_text(json.dumps(answer))
""",
        inputs,
        work,
    )
