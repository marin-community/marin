# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Fresh input-reading fit; coefficient-name contrasts independently check the
# construction reference's design-row contrasts using the same statistical engine.
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3, as.character(packageVersion("DESeq2")) == "1.50.2")
input <- args[1]
output <- args[2]
options(digits = 17, scipen = 999)
query <- jsonlite::fromJSON(file.path(input, "query.json"))
stopifnot(query$analysis == "real-rnaseq-population-interaction")
samples <- read.delim(file.path(input, "samples.tsv"), stringsAsFactors = FALSE)
samples <- samples[order(samples$sample), ]
rownames(samples) <- samples$sample
samples$population <- factor(samples$population, levels = c("basal", "luminal"))
samples$stage <- factor(samples$stage, levels = c("virgin", "18.5 dP", "2 dL"),
                        labels = c("virgin", "pregnant", "lactating"))
raw <- read.delim(file.path(input, "counts.tsv"), row.names = 1, check.names = FALSE)
stopifnot(!anyNA(samples), !anyDuplicated(samples$sample), !anyDuplicated(rownames(raw)))
stopifnot(setequal(samples$sample, setdiff(colnames(raw), "Length")))
x <- as.matrix(raw[, samples$sample, drop = FALSE])
stopifnot(all(is.finite(x)), all(x >= 0), all(x == floor(x)))
storage.mode(x) <- "integer"
model <- DESeqDataSetFromMatrix(x[rowSums(x) >= query$minimum_total_count, ], samples, ~population * stage)
model <- estimateSizeFactors(model, type = "ratio")
model <- estimateDispersions(model, fitType = "parametric", quiet = TRUE)
stopifnot(attr(dispersionFunction(model), "fitType") == "parametric")
model <- nbinomWaldTest(model, betaPrior = FALSE, quiet = TRUE)
coefficient_names <- resultsNames(model)
basal_name <- "stage_pregnant_vs_virgin"
interaction_name <- "populationluminal.stagepregnant"
stopifnot(basal_name %in% coefficient_names, interaction_name %in% coefficient_names)
weights <- list(basal = as.integer(coefficient_names == basal_name),
                luminal = as.integer(coefficient_names %in% c(basal_name, interaction_name)),
                interaction = as.integer(coefficient_names == interaction_name))
column_names <- c("intercept", "luminal", "pregnant", "lactating", "luminal_pregnant", "luminal_lactating")
design <- model.matrix(design(model), colData(model))
stopifnot(ncol(design) == length(column_names), qr(design)$rank == 6)
write_tsv <- function(value, filename) {
    write.table(value, file.path(output, filename), sep = "\t", quote = FALSE,
                row.names = FALSE, na = "NA")
}
colnames(design) <- column_names
write_tsv(data.frame(id = samples$sample, design), "design.tsv")
contrast_table <- do.call(rbind, weights)
colnames(contrast_table) <- column_names
write_tsv(data.frame(id = names(weights), contrast_table), "contrast_weights.tsv")
write_tsv(data.frame(id = colnames(x), library_counts = unname(colSums(x)),
                     detected_genes = unname(colSums(x > 0)),
                     size_factor = unname(sizeFactors(model))), "sample_qc.tsv")
log_probability <- function(p) -log10(pmax(p, 1e-300))
directions <- list()
testable <- list()
for (name in names(weights)) {
    result <- as.data.frame(results(model, contrast = weights[[name]], cooksCutoff = FALSE,
                                    independentFiltering = FALSE, pAdjustMethod = "BH", parallel = FALSE))
    write_tsv(data.frame(id = rownames(result), base_mean = result$baseMean,
                         log2_fold_change = result$log2FoldChange, standard_error = result$lfcSE,
                         wald_statistic = result$stat, neg_log10_p = log_probability(result$pvalue),
                         neg_log10_padj = log_probability(result$padj)), paste0(name, "_results.tsv"))
    finite <- is.finite(result$pvalue) & is.finite(result$padj) & is.finite(result$log2FoldChange)
    detected <- finite & result$padj <= query$maximum_fdr & abs(result$log2FoldChange) >= query$minimum_effect
    direction <- rep(0L, nrow(result))
    direction[which(detected)] <- as.integer(sign(result$log2FoldChange[which(detected)]))
    directions[[name]] <- direction
    testable[[name]] <- finite
}
one_only <- xor(directions$basal != 0, directions$luminal != 0)
same_direction <- directions$basal != 0 & directions$basal == directions$luminal
calls <- data.frame(id = rownames(model), basal_direction = directions$basal,
                    luminal_direction = directions$luminal, interaction_direction = directions$interaction,
                    all_testable = as.integer(Reduce(`&`, testable)),
                    one_population_only = as.integer(one_only),
                    one_population_only_without_interaction = as.integer(one_only & directions$interaction == 0),
                    both_same_direction = as.integer(same_direction))
write_tsv(calls, "gene_calls.tsv")
answer <- data.frame(id = "comparison", genes_modeled = nrow(model),
                     genes_testable = sum(calls$all_testable),
                     interaction_genes = sum(directions$interaction != 0),
                     one_population_only = sum(one_only),
                     one_population_only_without_interaction = sum(one_only & directions$interaction == 0),
                     both_same_direction = sum(same_direction))
jsonlite::write_json(answer, args[3], dataframe = "rows", auto_unbox = TRUE, digits = 16)
