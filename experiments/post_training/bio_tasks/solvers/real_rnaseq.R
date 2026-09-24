# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Private input-reading oracle. The construction reference uses the DESeq wrapper;
# this script performs its estimation stages explicitly and reads only task inputs.
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
input <- args[1]
output <- args[2]
answer_path <- args[3]
stopifnot(as.character(packageVersion("DESeq2")) == "1.50.2")
options(digits = 17, scipen = 999)
query <- jsonlite::fromJSON(file.path(input, "query.json"))
samples <- read.delim(file.path(input, "samples.tsv"), stringsAsFactors = FALSE)
raw <- read.delim(file.path(input, "counts.tsv"), row.names = 1, check.names = FALSE)
metadata <- samples[samples$population == query$population, , drop = FALSE]
metadata <- metadata[order(metadata$sample), , drop = FALSE]
rownames(metadata) <- metadata$sample
metadata$stage <- factor(metadata$stage, levels = c("virgin", "18.5 dP", "2 dL"))
x <- as.matrix(raw[, metadata$sample, drop = FALSE])
storage.mode(x) <- "integer"
keep <- rowSums(x) >= 10
model <- DESeqDataSetFromMatrix(x[keep, , drop = FALSE], metadata, ~stage)
model <- estimateSizeFactors(model, type = "ratio")
model <- estimateDispersions(model, fitType = "parametric", quiet = TRUE)
stopifnot(attr(dispersionFunction(model), "fitType") == "parametric")
model <- nbinomWaldTest(model, betaPrior = FALSE, quiet = TRUE)
result <- as.data.frame(results(model, contrast = c("stage", query$treatment, query$baseline),
                               cooksCutoff = FALSE, independentFiltering = FALSE,
                               pAdjustMethod = "BH", parallel = FALSE))
log_probability <- function(p) -log10(pmax(p, 1e-300))
write_tsv <- function(value, filename) {
    write.table(value, file.path(output, filename), sep = "\t", quote = FALSE,
                row.names = FALSE, na = "NA")
}
write_tsv(data.frame(id = colnames(x), library_counts = unname(colSums(x)),
                     detected_genes = unname(colSums(x > 0)),
                     size_factor = unname(sizeFactors(model))), "sample_qc.tsv")
write_tsv(data.frame(id = rownames(result), base_mean = result$baseMean,
                     log2_fold_change = result$log2FoldChange, standard_error = result$lfcSE,
                     wald_statistic = result$stat, neg_log10_p = log_probability(result$pvalue),
                     neg_log10_padj = log_probability(result$padj)), "de_results.tsv")
finite <- is.finite(result$pvalue) & is.finite(result$log2FoldChange)
significant <- finite & result$padj <= query$maximum_fdr
up <- significant & result$log2FoldChange >= query$minimum_effect
down <- significant & result$log2FoldChange <= -query$minimum_effect
if (query$analysis == "real-rnaseq-differential-expression") {
    answer <- data.frame(id = "contrast", genes_tested = sum(finite),
                         upregulated = sum(up, na.rm = TRUE), downregulated = sum(down, na.rm = TRUE),
                         strongest_neg_log10_padj = max(log_probability(result$padj), na.rm = TRUE))
} else {
    stopifnot(query$analysis == "real-rnaseq-go-enrichment")
    memberships <- read.delim(file.path(input, "go_membership.tsv"), stringsAsFactors = FALSE,
                              colClasses = "character")
    memberships <- unique(memberships[memberships$ontology == "BP", c("gene", "term")])
    universe <- intersect(rownames(result)[finite], memberships$gene)
    selected <- intersect(rownames(result)[if (query$direction == "up") up else down], universe)
    sets <- split(memberships$gene, memberships$term)
    sets <- lapply(sets, intersect, y = universe)
    sizes <- lengths(sets)
    sets <- sets[sizes >= 10 & sizes <= 500]
    overlap <- vapply(sets, function(genes) length(intersect(genes, selected)), integer(1))
    sizes <- lengths(sets)
    p <- phyper(overlap - 1, sizes, length(universe) - sizes, length(selected), lower.tail = FALSE)
    q <- p.adjust(p, method = "BH")
    enrichment <- data.frame(id = names(sets), overlap = unname(overlap), term_size = unname(sizes),
                             selected_genes = length(selected), background_genes = length(universe),
                             neg_log10_p = log_probability(unname(p)),
                             neg_log10_padj = log_probability(unname(q)))
    write_tsv(enrichment, "enrichment.tsv")
    ranking <- order(p, names(sets))
    answer <- enrichment[head(ranking, query$report_terms), , drop = FALSE]
}
jsonlite::write_json(answer, answer_path, dataframe = "rows", auto_unbox = TRUE, digits = 16, na = "null")
