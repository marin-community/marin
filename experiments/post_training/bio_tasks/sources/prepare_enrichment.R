# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Native input and method preparation; Harbor task validation is a separate step.
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(apeglm))
suppressPackageStartupMessages(library(clusterProfiler))
suppressPackageStartupMessages(library(GOSemSim))
suppressPackageStartupMessages(library(org.Mm.eg.db))
suppressPackageStartupMessages(library(GO.db))
args <- commandArgs(TRUE)
stopifnot(length(args) == 2)
input <- args[1]; output <- args[2]
dir.create(output, recursive=TRUE, showWarnings=FALSE)
options(digits=17, enrichment_force_universe=FALSE)
versions <- c(DESeq2="1.50.2",apeglm="1.32.0",clusterProfiler="4.18.4",DOSE="4.4.0",GOSemSim="2.36.0",org.Mm.eg.db="3.22.0",GO.db="3.22.0")
for (name in names(versions)) stopifnot(as.character(packageVersion(name))==versions[[name]])
write_tsv <- function(x,name) {
 con <- gzfile(file.path(output,paste0(name,".tsv.gz")),"wt")
 on.exit(close(con))
 write.table(x,con,sep="\t",row.names=FALSE,quote=FALSE,na="NA")
}
step <- function(name) {message(name); flush.console()}
step("Read observed counts and sample identities")
raw <- read.delim(file.path(input,"counts.tsv"),row.names=1,check.names=FALSE)
samples <- read.delim(file.path(input,"samples.tsv"),stringsAsFactors=FALSE)
stopifnot(nrow(raw)==27179,nrow(samples)==12,!anyDuplicated(rownames(raw)),!anyDuplicated(samples$sample))
stopifnot(setequal(setdiff(colnames(raw),"Length"),samples$sample))
s <- samples[samples$population=="luminal",,drop=FALSE]
s <- s[order(s$sample),,drop=FALSE];rownames(s)<-s$sample
s$stage<-factor(s$stage,levels=c("virgin","18.5 dP","2 dL"))
counts<-as.matrix(raw[,s$sample,drop=FALSE])
stopifnot(ncol(counts)==6,all(table(s$stage)==2),all(is.finite(counts)),all(counts>=0),all(counts==floor(counts)))
storage.mode(counts)<-"integer";keep<-rowSums(counts)>=10
stopifnot(sum(keep)==16659)
write_tsv(s,"samples")
write_tsv(data.frame(gene=rownames(counts),eligible=keep,count_sum=rowSums(counts)),"eligibility")
step("Fit DESeq2 and apeglm")
started<-proc.time()[[3]]
dds<-DESeqDataSetFromMatrix(counts[keep,,drop=FALSE],s,~stage)
dds<-DESeq(dds,test="Wald",fitType="parametric",sfType="ratio",betaPrior=FALSE,minReplicatesForReplace=Inf,parallel=FALSE)
stopifnot(attr(dispersionFunction(dds),"fitType")=="parametric")
design_matrix<-model.matrix(~stage,s)
stopifnot(qr(design_matrix)$rank == ncol(design_matrix))
coef_index<-which(colnames(design_matrix)=="stage2 dL")
stopifnot(length(coef_index)==1,ncol(design_matrix)==length(resultsNames(dds)))
coef_name<-resultsNames(dds)[coef_index]
res<-results(dds,name=coef_name,alpha=.05,pAdjustMethod="BH",independentFiltering=TRUE,cooksCutoff=TRUE,parallel=FALSE)
contrast_check<-results(dds,contrast=c("stage","2 dL","virgin"),alpha=.05,pAdjustMethod="BH",independentFiltering=TRUE,cooksCutoff=TRUE,parallel=FALSE)
# Character contrasts reset genes with no counts in either compared stage;
# results(name=...) retains the fitted coefficient used by apeglm.
compared <- s$stage %in% c("virgin", "2 dL")
both_groups_zero <- rowSums(counts(dds)[, compared, drop=FALSE]) == 0
write_tsv(data.frame(gene=rownames(res), both_groups_zero=both_groups_zero,
                    named_lfc=res$log2FoldChange, contrast_lfc=contrast_check$log2FoldChange,
                    named_p=res$pvalue, contrast_p=contrast_check$pvalue), "contrast-interface-audit")
stopifnot(isTRUE(all.equal(res$log2FoldChange[!both_groups_zero],
                          contrast_check$log2FoldChange[!both_groups_zero], tolerance=1e-12)))
stopifnot(all(contrast_check$log2FoldChange[both_groups_zero] == 0),
          all(contrast_check$stat[both_groups_zero] == 0),
          all(contrast_check$pvalue[both_groups_zero] == 1))
shrunk<-lfcShrink(dds,coef=coef_name,res=res,type="apeglm",apeMethod="nbinomCR",parallel=FALSE)
stopifnot(identical(rownames(res),rownames(shrunk)),identical(res$pvalue,shrunk$pvalue),identical(res$padj,shrunk$padj))
stats<-data.frame(gene=rownames(res),as.data.frame(res),shrunken_log2FC=shrunk$log2FoldChange,shrunken_lfcSE=shrunk$lfcSE)
write_tsv(stats,"gene-results")
write_tsv(data.frame(sample=colnames(dds),size_factor=unname(sizeFactors(dds))),"size-factors")
write_tsv(data.frame(gene=rownames(dds),gene_dispersion=mcols(dds)$dispGeneEst,fitted_dispersion=mcols(dds)$dispFit,dispersion=dispersions(dds)),"dispersions")
write_tsv(data.frame(gene=rownames(dds),assays(dds)[["cooks"]],check.names=FALSE),"cooks")
fit_seconds<-proc.time()[[3]]-started
step("Export native propagated BP gene sets and both graph sources")
go<-AnnotationDbi::select(org.Mm.eg.db,keys=keys(org.Mm.eg.db,keytype="ENTREZID"),keytype="ENTREZID",columns=c("GOALL","ONTOLOGYALL"))
go<-unique(go[!is.na(go$GOALL)&go$ONTOLOGYALL=="BP",c("ENTREZID","GOALL")])
colnames(go)<-c("gene","term")
valid_bp<-names(AnnotationDbi::Ontology(GOTERM))[AnnotationDbi::Ontology(GOTERM)=="BP"]
go<-go[go$term %in% valid_bp,,drop=FALSE]
write_tsv(go,"full-bp-membership")
write_tsv(AnnotationDbi::select(org.Mm.eg.db,keys=rownames(raw),keytype="ENTREZID",columns="SYMBOL"),"gene-symbols")
sem<-godata(annoDb=org.Mm.eg.db,keytype="ENTREZID",ont="BP",computeIC=FALSE,processTCSS=FALSE)
rel_env<-new.env();utils::data(list="gotbl",package="GOSemSim",envir=rel_env)
bundled<-rel_env$gotbl
write_tsv(bundled,"gosemsim-bundled-graph")
write_tsv(AnnotationDbi::toTable(GOBPPARENTS),"go-db-bp-parents")
annotation_genes<-unique(as.character(go$gene))
finite_p<-stats$gene[is.finite(stats$pvalue)]
raw_selected<-stats$gene[is.finite(stats$pvalue)&stats$pvalue<.05]
adj_selected<-stats$gene[is.finite(stats$padj)&stats$padj<.05]
views<-list(adjusted_tested=list(selected=adj_selected,background=finite_p),raw_genome=list(selected=raw_selected,background=annotation_genes))
view_records<-list()
for (name in names(views)) {
 step(paste("Native ORA and semantic reduction",name))
 v<-views[[name]];background<-intersect(as.character(v$background),annotation_genes)
 selected<-intersect(as.character(v$selected),background)
 stopifnot(length(background)>0,length(selected)>0,all(selected %in% background))
 write_tsv(data.frame(gene=background),paste0(name,"-background"))
 write_tsv(data.frame(gene=selected),paste0(name,"-selected"))
 set.seed(42)
 run_start<-proc.time()[[3]]
 ora<-enrichGO(gene=selected,OrgDb=org.Mm.eg.db,keyType="ENTREZID",ont="BP",universe=background,pAdjustMethod="BH",minGSSize=10,maxGSSize=500,pvalueCutoff=1,qvalueCutoff=1,readable=FALSE)
 stopifnot(!is.null(ora),setequal(ora@universe,background))
 result<-ora@result
 write_tsv(result,paste0(name,"-ora"))
 gene_ratios<-do.call(rbind,strsplit(result$GeneRatio,"/",fixed=TRUE))
 bg_ratios<-do.call(rbind,strsplit(result$BgRatio,"/",fixed=TRUE))
 stopifnot(all(as.integer(gene_ratios[,2])==length(selected)),all(as.integer(bg_ratios[,2])==length(background)))
 k<-as.integer(gene_ratios[,1]);K<-as.integer(bg_ratios[,1]);n<-length(selected);N<-length(background)
 stopifnot(isTRUE(all.equal(result$pvalue,phyper(k-1,K,N-K,n,lower.tail=FALSE),tolerance=1e-12)))
 stopifnot(isTRUE(all.equal(result$p.adjust,p.adjust(result$pvalue,"BH"),tolerance=1e-12)))
 significant<-result[is.finite(result$p.adjust)&result$p.adjust<.05,,drop=FALSE]
 significant_object<-ora;significant_object@result<-significant
 # This diagnostic measures the complete retained-term reduction; no top-k truncation.
 reduced<-if(nrow(significant)>0) simplify(significant_object,cutoff=.7,by="p.adjust",select_fun=min,measure="Wang",semData=sem) else significant_object
 write_tsv(reduced@result,paste0(name,"-simplified"))
 write_tsv(data.frame(term=significant$ID,retained=significant$ID %in% reduced@result$ID),paste0(name,"-retained"))
 view_records[[name]]<-list(background=N,selected=n,tested=nrow(result),significant=nrow(significant),retained=nrow(reduced@result),seconds=proc.time()[[3]]-run_start)
}
metadata<-list(both_compared_groups_zero=rownames(res)[both_groups_zero],versions=as.list(versions),coefficient=coef_name,fit_seconds=fit_seconds,filter_threshold=metadata(res)$filterThreshold,filter_theta=metadata(res)$filterTheta,filter_num_rej=metadata(res)$filterNumRej,views=view_records,raw_significant=sum(is.finite(stats$pvalue)&stats$pvalue<.05),adjusted_significant=sum(is.finite(stats$padj)&stats$padj<.05),native_graph_columns=colnames(bundled),native_graph_rows=nrow(bundled),go_metadata=AnnotationDbi::metadata(GO.db),org_metadata=AnnotationDbi::metadata(org.Mm.eg.db),status="native-preparation-only; no Harbor or coverage validation")
jsonlite::write_json(metadata,file.path(output,"native-result.json"),pretty=TRUE,auto_unbox=TRUE,na="null",digits=16)
capture.output(sessionInfo(),file=file.path(output,"session-info.txt"))
step("Native preparation completed")
