# DESeq2 source inventory

Investigation started 2026-09-30 15:49:29 UTC. Target: `thelovelab/DESeq2`, pinned revision `c62c60c6ff83fd84ce115cacd1c49827533f85a7`, package DESCRIPTION version 1.53.5. Only the launch file, resolved prompt, source-access instructions, and discovered external repository sources are used. No scientific execution, dataset downloads, installation, builds, cloud jobs, or additional agents.

The recursive GitHub tree was inspected and reports `truncated=false`. Collections: DESCRIPTION/NAMESPACE/package overview; one main vignette and bibliography; 42 API manual files; 14 R source files; C++ backend and generated bindings; developer scripts/notebooks; 29 test files plus test entrypoint; two pasilla example data assets; NEWS/citation and contribution files.

## Inspection queue (checkpoint)

- Inspected: DESCRIPTION, NAMESPACE, man/DESeq2-package.Rd, recursive tree.
- Partly inspected: vignettes/DESeq2.Rmd. Input routes, preprocessing, pasilla DE, shrinkage, IHW and MA plotting inspected; remaining visualization, multifactor, advanced designs, transformations, theory and FAQ portions queued.
- Pending: all individual man/*.Rd manuals, grouped by scientific purpose rather than accessor aliases.
- Pending: relevant R implementations, checked alongside manuals.
- Pending: inst/script/icobra_benchmarks.R, makeSim.R, runScripts.R, testsuite.Rmd, vst.nb; package-version text and PDF/image metadata.
- Pending: tests/testthat/*.R, enumerate and inspect for distinct scientific uses versus repeated demonstrations.
- Pending: pasilla_sample_annotation.csv and linked pasilla provenance; no count-matrix download.
- Pending: tximportData study/setup and independent GENCODE v27 transcript-gene annotation; tximeta example lineage; airway study/setup; linked rnaseqGene workflow and Galaxy wrapper.
- Skipped: .gitignore, CODE_OF_CONDUCT.md, CONTRIBUTING.md, generated Rcpp exports and Makevars as non-scientific operations. Core implementation will be consulted where it resolves scientific semantics.

## Boundaries and evidence policy

Units describe application of existing packages (`tool_use`), even when their implementation is inspected. Composed examples get separate records when their scientific question or data context differs. Raw observations, transformed matrices, fitted objects and annotation products remain explicit. Repeated manual/vignette/test presentations link to the same unit. Public accessibility establishes no biological-data reuse permission. Sources inspected at mutable external revisions will carry their own pinned commit or retrieval date and revision gap.

No failed retrievals so far. Source inspection establishes no execution, numerical validation, grading feasibility, runtime, or Harbor compatibility. Inventory is in progress; final stopping reason, counts and remaining leads will be refreshed after independent record checks.
