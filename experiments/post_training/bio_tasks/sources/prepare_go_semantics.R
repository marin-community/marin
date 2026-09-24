# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Export native contributions and pair scores for independent full-set reduction.
suppressPackageStartupMessages(library(GOSemSim))
suppressPackageStartupMessages(library(GO.db))
args <- commandArgs(TRUE)
stopifnot(length(args) == 2)
input <- args[1]; output <- args[2]
dir.create(output, recursive=TRUE, showWarnings=FALSE)
options(digits=17)
stopifnot(as.character(packageVersion("GOSemSim")) == "2.36.0",
          as.character(packageVersion("GO.db")) == "3.22.0")
write_tsv <- function(x, name) {
 con <- gzfile(file.path(output, paste0(name, ".tsv.gz")), "wt")
 on.exit(close(con))
 write.table(x, con, sep="\t", row.names=FALSE, quote=FALSE, na="NA")
}
terms <- sort(unique(unlist(lapply(c("adjusted_tested", "raw_genome"), function(view) {
 rows <- read.delim(gzfile(file.path(input, paste0(view, "-ora.tsv.gz"))))
 rows$ID[is.finite(rows$p.adjust) & rows$p.adjust < .05]
}))))
rel_env <- new.env()
utils::data(list="gotbl", package="GOSemSim", envir=rel_env)
relations <- rel_env$gotbl
started <- proc.time()[[3]]
message("Export native S-values for ", length(terms), " significant terms")
values <- lapply(terms, function(term) {
 sv <- GOSemSim:::getSV(term, "BP", relations)
 stopifnot(!anyDuplicated(names(sv)), all(is.finite(sv)))
 data.frame(term=term, ancestor=names(sv), contribution=unname(sv))
})
write_tsv(do.call(rbind, values), "native-s-values")
ancestors <- GOSemSim:::getAncestors("BP")
write_tsv(data.frame(term=terms, ancestor_count=vapply(terms, function(term) length(ancestors[[term]]), integer(1))),
          "native-ancestor-counts")
sv_seconds <- proc.time()[[3]] - started
# Pair samples include every missing-bundle term, identities and seeded pairs.
set.seed(42)
missing <- setdiff(terms, unique(c(relations$go_id, relations$parent)))
pairs <- unique(rbind(data.frame(left=sample(terms, 512, replace=TRUE), right=sample(terms, 512, replace=TRUE)),
                      data.frame(left=terms[seq_len(min(32, length(terms)))], right=terms[seq_len(min(32, length(terms)))]),
                      expand.grid(left=missing, right=terms[c(1, length(terms))], stringsAsFactors=FALSE)))
pairs$score <- mapply(function(a,b) GOSemSim:::wangMethod_internal(a,b,"BP"), pairs$left, pairs$right)
pairs$rounded_score <- round(pairs$score, 3)
stopifnot(all(is.finite(pairs$score)))
write_tsv(pairs, "native-pair-scores")
jsonlite::write_json(list(status="native-intermediates-only", terms=length(terms),
 contribution_rows=sum(vapply(values, nrow, integer(1))), pair_checks=nrow(pairs),
 missing_bundle_terms=missing, contribution_seconds=sv_seconds,
 total_seconds=proc.time()[[3]]-started,
 limits="Full clusterProfiler simplify on the larger set did not complete; these intermediates support a separate equivalent calculation."),
 file.path(output,"native-semantic-result.json"), pretty=TRUE, auto_unbox=TRUE, digits=16)
capture.output(sessionInfo(), file=file.path(output,"session-info.txt"))
message("Native semantic intermediate export completed")
