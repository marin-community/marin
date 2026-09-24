# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Fresh input-reading oracle; package outputs are checked against separately prepared references.
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(apeglm))
suppressPackageStartupMessages(library(clusterProfiler))
suppressPackageStartupMessages(library(org.Mm.eg.db))
suppressPackageStartupMessages(library(GO.db))
args <- commandArgs(TRUE)
stopifnot(length(args) == 2)
input <- args[1]; output <- args[2]
versions <- c(DESeq2="1.50.2",apeglm="1.32.0",clusterProfiler="4.18.4",org.Mm.eg.db="3.22.0",GO.db="3.22.0")
for (name in names(versions)) stopifnot(as.character(packageVersion(name)) == versions[[name]])
options(digits=17, enrichment_force_universe=FALSE)
write_artifact <- function(rows, name) {
  write.table(rows, file.path(output,name), sep="\t", quote=FALSE, row.names=FALSE, na="NA")
}
logp <- function(values) -log10(pmax(values,1e-300))
counts <- read.delim(file.path(input,"counts.tsv"),row.names=1,check.names=FALSE)
metadata <- read.delim(file.path(input,"samples.tsv"),stringsAsFactors=FALSE)
samples <- metadata[metadata$population == "luminal",,drop=FALSE]
samples <- samples[order(samples$sample),,drop=FALSE]
rownames(samples) <- samples$sample
samples$stage <- factor(samples$stage,levels=c("virgin","18.5 dP","2 dL"))
matrix <- as.matrix(counts[,samples$sample,drop=FALSE])
stopifnot(!anyDuplicated(rownames(matrix)),all(matrix >= 0),all(matrix == floor(matrix)),all(table(samples$stage)==2))
storage.mode(matrix) <- "integer"
keep <- rowSums(matrix) >= 10
dds <- DESeqDataSetFromMatrix(matrix[keep,,drop=FALSE],samples,~stage)
dds <- estimateSizeFactors(dds,type="ratio")
dds <- estimateDispersions(dds,fitType="parametric",quiet=TRUE)
dds <- nbinomWaldTest(dds,betaPrior=FALSE,quiet=TRUE)
coefficient <- "stage_2.dL_vs_virgin"
stopifnot(coefficient %in% resultsNames(dds))
result <- results(dds,name=coefficient,alpha=.05,pAdjustMethod="BH",independentFiltering=TRUE,cooksCutoff=TRUE)
shrunk <- lfcShrink(dds,coef=coefficient,res=result,type="apeglm",apeMethod="nbinomCR",parallel=FALSE)
stopifnot(identical(rownames(result),rownames(shrunk)))
write_artifact(data.frame(id=colnames(matrix),library_counts=colSums(matrix),detected_genes=colSums(matrix>0),
                         size_factor=unname(sizeFactors(dds))),"sample_qc.tsv")
write_artifact(data.frame(id=rownames(result),base_mean=result$baseMean,log2_fold_change=result$log2FoldChange,
                         standard_error=result$lfcSE,wald_statistic=result$stat,neg_log10_p=logp(result$pvalue),
                         neg_log10_padj=logp(result$padj),shrunken_log2_fold_change=shrunk$log2FoldChange,
                         shrunken_standard_error=shrunk$lfcSE),"gene_results.tsv")
gene_ids <- rownames(result)
raw <- gene_ids[is.finite(result$pvalue) & result$pvalue < .05]
adjusted <- gene_ids[is.finite(result$padj) & result$padj < .05]
large <- gene_ids[is.finite(shrunk$log2FoldChange) & abs(shrunk$log2FoldChange) >= 1]
unshrunk_large <- gene_ids[is.finite(result$log2FoldChange) & abs(result$log2FoldChange) >= 1]
write_artifact(data.frame(id=rownames(matrix),count_sum=rowSums(matrix),fitted=as.integer(keep),
                         raw_significant=as.integer(rownames(matrix) %in% raw),
                         adjusted_significant=as.integer(rownames(matrix) %in% adjusted),
                         large_shrunken_effect=as.integer(rownames(matrix) %in% large)),"gene_decisions.tsv")
annotation <- AnnotationDbi::select(org.Mm.eg.db,keys=keys(org.Mm.eg.db,keytype="ENTREZID"),
                                  keytype="ENTREZID",columns=c("GOALL","ONTOLOGYALL"))
bp <- AnnotationDbi::Ontology(GOTERM)
valid <- names(bp)[bp == "BP"]
annotated <- sort(unique(as.character(annotation$ENTREZID[!is.na(annotation$GOALL) &
                  annotation$ONTOLOGYALL == "BP" & annotation$GOALL %in% valid])))
backgrounds <- list(adjusted_tested=intersect(gene_ids[is.finite(result$pvalue)],annotated),raw_genome=annotated)
selections <- list(adjusted_tested=intersect(adjusted,backgrounds$adjusted_tested),raw_genome=intersect(raw,annotated))
membership <- data.frame(id=annotated)
significant <- list()
for (view in names(backgrounds)) {
  universe <- backgrounds[[view]]; selected <- selections[[view]]
  membership[[paste0(view,"_background")]] <- as.integer(annotated %in% universe)
  membership[[paste0(view,"_selected")]] <- as.integer(annotated %in% selected)
  set.seed(42)
  enriched <- enrichGO(gene=selected,OrgDb=org.Mm.eg.db,keyType="ENTREZID",ont="BP",universe=universe,
                      pAdjustMethod="BH",minGSSize=10,maxGSSize=500,pvalueCutoff=1,qvalueCutoff=1,readable=FALSE)
  native <- enriched@result
  significant[[view]] <- native$ID[is.finite(native$p.adjust) & native$p.adjust < .05]
  term_size <- as.integer(sub("/.*","",native$BgRatio))
  write_artifact(data.frame(id=native$ID,overlap=native$Count,term_size=term_size,selected_genes=length(selected),
                           background_genes=length(universe),neg_log10_p=logp(native$pvalue),
                           neg_log10_padj=logp(native$p.adjust)),paste0(view,"_enrichment.tsv"))
}
write_artifact(membership,"annotation_membership.tsv")
answer <- list(list(id="audit",genes_fitted=nrow(result),raw_significant=length(raw),adjusted_significant=length(adjusted),
                   large_shrunken_effects=length(large),
                   changed_effect_threshold=length(setdiff(union(large,unshrunk_large),intersect(large,unshrunk_large))),
                   adjusted_tested_significant_terms=length(significant$adjusted_tested),
                   raw_genome_significant_terms=length(significant$raw_genome),
                   shared_significant_terms=length(intersect(significant$adjusted_tested,significant$raw_genome))))
jsonlite::write_json(answer,file.path(output,"answer.json"),auto_unbox=TRUE,pretty=TRUE,na="null",digits=16)
