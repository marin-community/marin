# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Native construction reference for the complete observed population-by-stage design.
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2, as.character(packageVersion("DESeq2")) == "1.50.2")
input <- args[1]
output <- args[2]
dir.create(output, recursive = TRUE)
options(digits = 17)
raw <- read.delim(file.path(input, "counts.tsv"), row.names = 1, check.names = FALSE)
samples <- read.delim(file.path(input, "samples.tsv"), stringsAsFactors = FALSE)
samples <- samples[order(samples$sample), ]
rownames(samples) <- samples$sample
stopifnot(nrow(raw) == 27179, nrow(samples) == 12, !anyDuplicated(samples$sample))
stopifnot(setequal(samples$sample, setdiff(colnames(raw), "Length")))
samples$population <- factor(samples$population, levels = c("basal", "luminal"))
samples$stage <- factor(samples$stage, levels = c("virgin", "18.5 dP", "2 dL"),
                        labels = c("virgin", "pregnant", "lactating"))
stopifnot(!anyNA(samples), all(table(samples$population, samples$stage) == 2))
x <- as.matrix(raw[, samples$sample])
stopifnot(all(is.finite(x)), all(x >= 0), all(x == floor(x)))
storage.mode(x) <- "integer"
keep <- rowSums(x) >= 10
design <- model.matrix(~population * stage, samples)
stopifnot(qr(design)$rank == 6)
model <- DESeqDataSetFromMatrix(x[keep, , drop = FALSE], samples, ~population * stage)
started <- proc.time()[[3]]
model <- DESeq(model, test = "Wald", fitType = "parametric", sfType = "ratio",
                betaPrior = FALSE, minReplicatesForReplace = Inf, parallel = FALSE)
stopifnot(attr(dispersionFunction(model), "fitType") == "parametric")
write_tsv <- function(value, filename) {
    write.table(value, file.path(output, paste0("population-interaction-", filename)), sep = "\t", quote = FALSE,
                row.names = FALSE, na = "NA")
}
write_tsv(data.frame(id = samples$sample, size_factor = unname(sizeFactors(model))), "size-factors.tsv")

# Derive contrasts from observed design rows, independently of coefficient names.
group_vector <- function(population, stage) {
    rows <- samples$population == population & samples$stage == stage
    stopifnot(sum(rows) == 2)
    colMeans(design[rows, , drop = FALSE])
}
basal <- group_vector("basal", "pregnant") - group_vector("basal", "virgin")
luminal <- group_vector("luminal", "pregnant") - group_vector("luminal", "virgin")
weights <- list(basal = basal, luminal = luminal, interaction = luminal - basal)
write_tsv(data.frame(id = names(weights), do.call(rbind, weights), check.names = FALSE), "contrasts.tsv")
for (name in names(weights)) {
    result <- results(model, contrast = unname(weights[[name]]), cooksCutoff = FALSE,
                      independentFiltering = FALSE, pAdjustMethod = "BH", parallel = FALSE)
    write_tsv(data.frame(id = rownames(result), as.data.frame(result)), paste0(name, "-results.tsv"))
}
capture.output(sessionInfo(), file = file.path(output, "session-info.txt"))
jsonlite::write_json(list(genes = nrow(model), samples = ncol(model), design_rank = qr(design)$rank,
                         fit_seconds = proc.time()[[3]] - started, coefficients = resultsNames(model),
                         package = as.character(packageVersion("DESeq2"))),
                    file.path(output, "metadata.json"), auto_unbox = TRUE, pretty = TRUE)
