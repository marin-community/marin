# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Prepare private fitted references and a public annotation snapshot on a CPU worker.
# Inputs are the unchanged GSE60450 counts and sample metadata; no benchmark data.
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(org.Mm.eg.db))
suppressPackageStartupMessages(library(GO.db))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
input <- args[1]
output <- args[2]
dir.create(output, recursive = TRUE)
stopifnot(as.character(packageVersion("DESeq2")) == "1.50.2")
stopifnot(as.character(packageVersion("org.Mm.eg.db")) == "3.22.0")
stopifnot(as.character(packageVersion("GO.db")) == "3.22.0")
options(digits = 17)

raw <- read.delim(file.path(input, "counts.tsv"), row.names = 1, check.names = FALSE)
samples <- read.delim(file.path(input, "samples.tsv"), stringsAsFactors = FALSE)
stopifnot(nrow(raw) == 27179, nrow(samples) == 12)
stopifnot(!anyDuplicated(samples$sample), !anyDuplicated(rownames(raw)))
stopifnot(setequal(samples$sample, setdiff(colnames(raw), "Length")))
counts <- as.matrix(raw[, samples$sample])
stopifnot(all(is.finite(counts)), all(counts >= 0), all(counts == floor(counts)))
storage.mode(counts) <- "integer"
rownames(samples) <- samples$sample
stage_levels <- c("virgin", "18.5 dP", "2 dL")

write_tsv <- function(value, filename) {
    write.table(value, file.path(output, filename), sep = "\t", quote = FALSE,
                row.names = FALSE, na = "NA")
}

fit_records <- list()
for (population in c("basal", "luminal")) {
    selected <- samples[samples$population == population, , drop = FALSE]
    selected <- selected[order(selected$sample), , drop = FALSE]
    selected$stage <- factor(selected$stage, levels = stage_levels)
    x <- counts[, selected$sample, drop = FALSE]
    stopifnot(ncol(x) == 6, all(table(selected$stage) == 2))
    keep <- rowSums(x) >= 10
    dds <- DESeqDataSetFromMatrix(x[keep, , drop = FALSE], selected, ~stage)
    started <- proc.time()[[3]]
    dds <- DESeq(dds, test = "Wald", fitType = "parametric", sfType = "ratio",
                 betaPrior = FALSE, minReplicatesForReplace = Inf, parallel = FALSE)
    elapsed <- proc.time()[[3]] - started
    stopifnot(attr(dispersionFunction(dds), "fitType") == "parametric")
    write_tsv(data.frame(sample = colnames(dds), size_factor = unname(sizeFactors(dds))),
              paste0(population, "-size-factors.tsv"))
    write_tsv(data.frame(id = rownames(dds), gene_dispersion = mcols(dds)$dispGeneEst,
                         fitted_dispersion = mcols(dds)$dispFit, dispersion = dispersions(dds)),
              paste0(population, "-dispersions.tsv"))
    for (baseline in stage_levels) {
        for (treatment in setdiff(stage_levels, baseline)) {
            result <- results(dds, contrast = c("stage", treatment, baseline),
                              independentFiltering = FALSE, cooksCutoff = FALSE,
                              pAdjustMethod = "BH", alpha = 0.05, parallel = FALSE)
            name <- paste(population, match(baseline, stage_levels),
                          match(treatment, stage_levels), sep = "-")
            write_tsv(data.frame(id = rownames(result), as.data.frame(result)),
                      paste0(name, "-results.tsv"))
            fit_records[[name]] <- list(population = population, baseline = baseline,
                                        treatment = treatment, genes = nrow(dds),
                                        samples = colnames(dds), fit_seconds = elapsed)
        }
    }
}

# Preserve the database's propagated GO membership, deduplicating evidence codes.
mapped <- intersect(rownames(counts), keys(org.Mm.eg.db, keytype = "ENTREZID"))
go <- AnnotationDbi::select(org.Mm.eg.db, keys = mapped, keytype = "ENTREZID",
                             columns = c("GOALL", "ONTOLOGYALL"))
go <- unique(go[!is.na(go$GOALL), c("ENTREZID", "GOALL", "ONTOLOGYALL")])
go <- go[order(go$ONTOLOGYALL, go$GOALL, go$ENTREZID), ]
colnames(go) <- c("gene", "term", "ontology")
write_tsv(go, "mouse-go-membership.tsv")
terms <- AnnotationDbi::select(GO.db, keys = unique(go$term), keytype = "GOID",
                                columns = c("TERM", "ONTOLOGY"))
write_tsv(terms[order(terms$GOID), ], "go-terms.tsv")

capture.output(sessionInfo(), file = file.path(output, "session-info.txt"))
jsonlite::write_json(list(fits = fit_records,
                         packages = list(DESeq2 = as.character(packageVersion("DESeq2")),
                                         org.Mm.eg.db = as.character(packageVersion("org.Mm.eg.db")),
                                         GO.db = as.character(packageVersion("GO.db"))),
                         org_metadata = AnnotationDbi::metadata(org.Mm.eg.db),
                         go_metadata = AnnotationDbi::metadata(GO.db)),
                    file.path(output, "reference-metadata.json"), auto_unbox = TRUE,
                    pretty = TRUE, digits = 16)
