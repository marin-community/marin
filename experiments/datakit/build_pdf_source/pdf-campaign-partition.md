# What ships in the PDF pipeline PR, and what stays campaign history

This manifest partitions the PDF campaign's files into **SHIP** — belongs in
[#8023](https://github.com/marin-community/marin/pull/8023), the pipeline as deployed — and
**CAMPAIGN** — kept on this branch as the record behind the reports, and deleted from the PR. A
third label, **PROVISIONAL**, marks files whose disposition depends on a decision that is not yet
made; each one names the decision.

It covers every path under `experiments/**/build_pdf_source/**`, every `lib/**` file the PR touches,
and the tests and reports attached to them.

## The design SHIP is derived from

The pipeline has moved since #8023 was opened, so SHIP is derived from the current design rather
than from what the PR currently contains. Four decisions drive almost every row below.

**pdf-inspector 1.17.0 is the single CPU extractor and Docling is gone.** Docling costs 278 CPU
core-hours per million pages against pdf-inspector's 2.1 — 132× — for corpus-wide quality parity
([pdf-router-v2.md](pdf-router-v2.md), [pdf-inspector-evaluation.md](pdf-inspector-evaluation.md)),
and this cluster is CPU-constrained. That retires `docling_extract/**`, `extract.py`,
`extract_fleet*.py`, `layout_model.py` and the whole OpenVINO INT8 layout effort.

**Router v2 ships on free features only.** `free + detect` is 43 features at 0.12 core-h/M;
`route_features` measured worse than `free` on all five domain splits and the 124-feature FinePDFs
incumbent landed inside the noise, so the entire PyMuPDF router pass is deleted
([pdf-router-v2.md](pdf-router-v2.md), "Free features against paid ones").

**PDFium replaces PyMuPDF as the render feed's rasteriser**, on licensing grounds, with the quality
objection retired: adjudicated blind the two readings are interchangeable at 0.481–0.498
([pdfium-evaluation.md](pdfium-evaluation.md), "Adopt PDFium for the render feed"). The swap is
gated on subprocess isolation, which is not resolved — see the PROVISIONAL block below.

**The VLM operating point is unchanged**: 2,048 visual tokens, 512 in-flight per instance, 15.6
GPU-h per million pages ([ocr-budget-sweep.md](ocr-budget-sweep.md)).

## Three things the current tree does not yet contain

A curator working from this manifest will find that SHIP is not simply a subset of what exists. The
design has outrun the code in three places, and each is a gap rather than a classification problem.

**There is no pdf-inspector extraction step.** pdf-inspector appears only inside evaluation
harnesses (`quality/probe_pdf_inspector.py`, `quality/build_inspector_study.py`). The CPU route the
pipeline is supposed to run does not exist as a pipeline step, so `pipeline.py` cannot be shipped as
written — it wires `fleet_extract_step`, the Docling route.

**`classify.py` still implements router v1.** It loads
`pdf_route_classifier_00757366/route_classifier.ubj` and thresholds `docling_confidence < 0.542031`
against `quality.route_features`. The v2 booster is already fit and calibrated —
`quality/fit_route_v2.py` writes `.../pdf_quality/model/pdf_route_v2/route_v2_classifier.ubj` — but
nothing outside `quality/` imports `route_v2_features`. A v2 `classify.py` has to be written.

**`ocr_extract/render.py` still renders through PyMuPDF.** The overturn commit changed prose only.

## PROVISIONAL: the render swap and the exit 133

[pdfium-evaluation.md](pdfium-evaluation.md) attaches two conditions to adoption, and neither is a
quality condition. The binding one is the unexplained abort:

> The honest statement is **zero aborts in 24,621 PDFium page renders across two architectures, plus
> one unexplained abort in an earlier, unisolated configuration that was never attributed to a
> library.**

The rule-of-three bound is ~1.2 × 10⁻⁴ aborts per page, which the report calls "not a comfortable
bound" over a million-page corpus. The feed rasterises inside a Zephyr map task, where a hard abort
takes the task rather than the page, so the report's condition is to subprocess-isolate the
rasteriser or establish that it does not need it. Until that resolves, three rows below stay
PROVISIONAL: `ocr_extract/render.py`, `pdfium-evaluation.md`, and the `pypdfium2` pin in
`lib/marin/pyproject.toml`.

Two smaller conditions ride along, recorded rather than resolved. The 345 adjudication packets were
produced from MuPDF renderings and want regenerating if the renderer changes — a records-accuracy
cost, not a quality regression. And `tests/datakit/test_ocr_extract.py` authors its fixture PDFs with
`pymupdf.open()`, which `pypdfium2` cannot do, so PyMuPDF stays an AGPL test-path dependency of a
corpus intended for release. The report is explicit that this is not a call to make on technical
grounds.

## experiments/datakit/build_pdf_source/ — the PR's package path

### Pipeline steps and shared contracts

| path | verdict | reason |
|---|---|---|
| `__init__.py` | SHIP | package marker |
| `plan.py` | SHIP | Step 1, the crawl index sample and byte-budgeted fetch plan; unaffected by the extractor and router changes |
| `fetch.py` | SHIP | Step 2, WARC range GETs to Parquet; unaffected |
| `classify.py` | CAMPAIGN | implements router v1 against the retired `docling_ok` label and the deleted `route_features` pass; a v2 replacement has to be written rather than edited |
| `common.py` | SHIP | shared constants, crawl identifiers and HTTP plumbing every step reads |
| `document_record.py` | SHIP | the record contract both routes emit; survives the extractor swap with the docling route renamed |
| `extract.py` | CAMPAIGN | the Docling text route's column and schema contract; retired with Docling |
| `extract_fleet.py` | CAMPAIGN | the persistent Docling converter fleet transport; retired with Docling |
| `extract_fleet_run.py` | CAMPAIGN | entry job for the Docling fleet's production run |
| `extract_fleet_backfill_run.py` | CAMPAIGN | entry job for the Docling backfill over the OCR route |
| `extract_ocr.py` | SHIP | Step 6, the VLM route; retained and unchanged by the CPU-extractor decision |
| `ocr_extract/__init__.py` | SHIP | package marker for the VLM route |
| `ocr_extract/client.py` | SHIP | one page to the OCR endpoint and back; renderer-independent |
| `ocr_extract/fleet.py` | SHIP | the serving fleet at the operating point [ocr-budget-sweep.md](ocr-budget-sweep.md) priced |
| `ocr_extract/render.py` | **PROVISIONAL** | the shipped feed, but it renders through PyMuPDF; needs the PDFium swap plus subprocess isolation, and the Pillow `compress_level=1` encoder, which is renderer-independent and recommended regardless |
| `loop_repair.py` | SHIP | repairs VLM repetition loops; part of the VLM route's output contract |
| `boilerplate.py` | SHIP | strips running headers and footers; extractor-independent by construction |
| `combine_routes.py` | SHIP | unions the two routes into one tagged corpus |
| `dedup.py` | SHIP | exact dedup and decontamination over the combined corpus |
| `quality_label.py` | SHIP | scores the clean corpus and drops documents with no readable window |
| `fuzzy_dedup.py` | SHIP | quality-aware fuzzy dedup, deliberately sequenced after `quality_label.py` |
| `language_label.py` | SHIP | final GlotLID language bucket |
| `lid_th_values.json` | SHIP | GlotLID thresholds `language_label.py` loads |
| `pipeline.py` | **PROVISIONAL** | the DAG ships, but as written it wires `fleet_extract_step` and v1 `classify_step`; both edges have to be rebuilt before it is shippable |
| `layout_model.py` | CAMPAIGN | builds the INT8 OpenVINO layout graph; the layout model is Docling's |

### Docling extraction — retired wholesale

Every path here exists to make Docling read a page well. Docling is dropped, so all thirteen are
CAMPAIGN, and they are the substantive record of that effort.

| path | verdict | reason |
|---|---|---|
| `docling_extract/__init__.py` | CAMPAIGN | package marker for the retired extractor |
| `docling_extract/assemble.py` | CAMPAIGN | span-aware cluster reassembly, the substantive half of the FinePDFs docling fork |
| `docling_extract/backend.py` | CAMPAIGN | PyMuPDF page backend carrying per-span geometry |
| `docling_extract/converter.py` | CAMPAIGN | assembles the converter and its `ExtractionOptions` |
| `docling_extract/fields.py` | CAMPAIGN | attaches four model fields to released docling models |
| `docling_extract/layout_openvino.py` | CAMPAIGN | runs the layout model as an INT8 OpenVINO graph |
| `docling_extract/model_spec.py` | CAMPAIGN | pure-data backend switches for the retired extractor |
| `docling_extract/page_numbers.py` | CAMPAIGN | running page-number recognition in ~50 languages; reusable, but only from Docling's assembled output |
| `docling_extract/postprocess.py` | CAMPAIGN | the four repair passes over the assembled document |
| `docling_extract/reading_order.py` | CAMPAIGN | carries span geometry onto the assembled document |
| `docling_extract/serializer.py` | CAMPAIGN | serialises the assembled document to text rather than Markdown |
| `docling_extract/service.py` | CAMPAIGN | the converter-pool handler and wire format for Docling |
| `docling_extract/tables.py` | CAMPAIGN | PyMuPDF ruling-line table recovery in place of TableFormer |

### Router feature sets

| path | verdict | reason |
|---|---|---|
| `quality/route_v2_features.py` | SHIP | the shipped router's feature contract and cost model; every group's price is declared here |
| `quality/route_features.py` | CAMPAIGN | the paid PyMuPDF pass, measured worse than `free` on all five splits |
| `quality/route_feature_names.py` | CAMPAIGN | the v1 feature contract as pure data; `route_v2_features.py` imports it today, so the SHIP copy has to inline the names it still needs |
| `ocr_features.py` | CAMPAIGN | the 124-feature FinePDFs incumbent extraction; adds nothing measurable at 1.54 core-h/M |
| `ocr_feature_names.py` | CAMPAIGN | the incumbent's pure-data contract |

### Router training, evaluation and labelling

The v1 chain is retired with the router it fit. The v2 chain produced the shipped booster and its
calibrated threshold; whether the training chain travels with the model is the first ambiguity
below.

| path | verdict | reason |
|---|---|---|
| `quality/fit_route_v2.py` | SHIP | fits and calibrates the shipped v2 booster; the v1 analogue `fit_route_booster.py` is already in the PR on the same argument |
| `quality/analyze_route_v2.py` | CAMPAIGN | evaluates the six feature arms and produces the frontier tables; the evidence for the decision, not the decision |
| `quality/build_preference_set.py` | CAMPAIGN | packages blinded two-route packets for the preference label |
| `quality/judge_preference_set.py` | CAMPAIGN | buys the 19,977 verdicts and writes the `escalate` column |
| `quality/build_inspector_output_study.py` | CAMPAIGN | measures pdf-inspector's own output text into the training table |
| `quality/build_route_study.py` | CAMPAIGN | builds the v1 study table around `route_features` and the incumbent |
| `quality/fit_route_booster.py` | CAMPAIGN | fits the v1 booster `classify.py` loads today |
| `quality/analyze_route_study.py` | CAMPAIGN | v1 frontier and cost-matched comparison |
| `quality/train_route_model.py` | CAMPAIGN | v1 candidate-versus-incumbent fit and scoring |
| `quality/route_agreement.py` | CAMPAIGN | the v1 agreement target; router v2 replaced agreement with an adjudicated preference |
| `quality/build_adjudication_set.py` | CAMPAIGN | three-route blind adjudication packets |
| `quality/judge_adjudication_set.py` | CAMPAIGN | judges those packets |

### Quality scorer — the other trained model

| path | verdict | reason |
|---|---|---|
| `quality/__init__.py` | SHIP | package marker; `quality_label.py`'s chain lives here |
| `quality/build_oracle_sample.py` | SHIP | builds the 100k oracle sample the shipped scorer trains on |
| `quality/build_labels.py` | SHIP | turns oracle scores into the scorer's label parquet |
| `quality/train_pdf_scorer.py` | SHIP | trains the scorer `quality_label.py` loads |
| `quality/edu_score_v2_prompt.txt` | SHIP | the oracle rubric the labels were scored under |

### Third-party evaluation harnesses

Each report describes these as probes and study harnesses, not pipeline code. They are the
measurement record behind the extractor and renderer decisions.

| path | verdict | reason |
|---|---|---|
| `quality/probe_pdf_inspector.py` | CAMPAIGN | Stage 0 probe: survival, cost and network dormancy of the Rust extractor |
| `quality/build_inspector_study.py` | CAMPAIGN | adds pdf-inspector as a third route to the study table |
| `quality/compare_inspector_versions.py` | CAMPAIGN | paired 1.14.1-against-1.17.0 comparison |
| `quality/audit_inspector_format.py` | CAMPAIGN | re-checks the agreement normalizer against pdf-inspector's serialization |
| `quality/analyze_inspector_routing.py` | CAMPAIGN | prices pdf-inspector as a feature source and as the cheap route |
| `quality/probe_pdfium.py` | CAMPAIGN | cost, pixel, dimension and failure-taxonomy probe for PDFium |
| `quality/build_render_study.py` | CAMPAIGN | the paired MuPDF-against-PDFium agreement study; deliberately omits the pipeline's `repair_page` |
| `quality/build_render_adjudication_set.py` | CAMPAIGN | packets for adjudicating render direction |
| `quality/judge_render_adjudication_set.py` | CAMPAIGN | judges them; the pass that overturned the rejection |
| `quality/probe_pdf_oxide.py` | CAMPAIGN | prices pdf_oxide for the router pass and the render feed; rejected for both |
| `quality/probe_png_encoders.py` | CAMPAIGN | prices PNG encoders; source of the Pillow `compress_level=1` finding, which does ship as a change to `render.py` |

### Reports

| path | verdict | reason |
|---|---|---|
| `pdf-router-v2.md` | SHIP | documents the shipped router, its operating point and its render policy |
| `pdf-inspector-evaluation.md` | SHIP | the measurement behind "Docling is dropped, pdf-inspector is the extractor" |
| `ocr-budget-sweep.md` | SHIP | prices the VLM operating point that still ships |
| `pdfium-evaluation.md` | **PROVISIONAL** | SHIP if the render swap lands with the PR, CAMPAIGN if it does not; the two travel together |
| `pdf-oxide-evaluation.md` | CAMPAIGN | negative result on both roles it evaluated; its one shipped finding is the PNG encoder, which `pdf-router-v2.md` does not carry — see ambiguity 4 |
| `pdf-extraction-routing.md` | CAMPAIGN | documents router v1, whose cheap route and training label are both retired; `pdf-router-v2.md` opens by replacing it |

## experiments/build_pdf_source/ — the campaign package path

Every path here is **CAMPAIGN**. This tree predates the move under `experiments/datakit/` and none of
it should appear in the PR. It splits into three groups.

**Superseded copies of shipped modules.** Each of these has a later counterpart at
`experiments/datakit/build_pdf_source/<same name>`, and 31 of them differ from it by more than the
package path. They exist so the campaign's own runs stay reproducible against the code that produced
them.

`__init__.py`, `boilerplate.py`, `classify.py`, `common.py`, `dedup.py`, `document_record.py`,
`extract.py`, `extract_fleet.py`, `extract_fleet_run.py`, `extract_fleet_backfill_run.py`,
`extract_ocr.py`, `fetch.py`, `language_label.py`, `layout_model.py`, `lid_th_values.json`,
`loop_repair.py`, `ocr_feature_names.py`, `ocr_features.py`, `pipeline.py`, `plan.py`,
`quality_label.py`, `docling_extract/{__init__,assemble,backend,converter,fields,layout_openvino,
model_spec,page_numbers,postprocess,reading_order,serializer,service,tables}.py`,
`ocr_extract/{__init__,client,fleet,render}.py`, `quality/{__init__,analyze_route_study,
build_adjudication_set,build_labels,build_oracle_sample,build_route_study,edu_score_v2_prompt.txt,
fit_route_booster,route_agreement,route_feature_names,route_features,train_pdf_scorer,
train_route_model}`.

**One-off probes and benchmarks.** Every one carries a `TEMPORARY -- DELETE once ...` docstring and
nothing in the pipeline imports it. They are the record behind the INT8/OpenVINO recipe, the
converter-pool sizing, the NCCL import failure and the OCR route's throughput.

| path | reason |
|---|---|
| `_analyze_page_counts.py` | page-count and bytes-per-page statistics from the routing table |
| `_bench_broker_ceiling.py` | how many GB200 instances one broker and proxy can feed |
| `_bench_ocr_route.py` | the OCR route over a small sample on one GB200 node |
| `_compare_backend_matrix.py` | the full layout × table backend matrix in one run |
| `_compare_layout_backends.py` | what quantizing the layout model costs in extracted text |
| `_compare_matrix_fleet.py` | the 2×2 backend matrix through the converter fleet |
| `_compare_pool_shapes.py` | 64 pods × 4 converters against 256 × 1 |
| `_compare_table_backends.py` | PyMuPDF ruling-line tables against TableFormer |
| `_control_layout_variance.py` | the extractor's FP32-against-FP32 noise floor |
| `_finepdfs_reference.py` | the FinePDFs extractor vendored verbatim to validate the port |
| `_int8_quality_sample.py` | 100 documents converted with INT8 plus TableFormer on a guaranteed-x86 pod |
| `_nccl_import_probe.py` | why `import torch` intermittently dies on an NCCL undefined symbol |
| `_nccl_preload_probe.py` | which NCCL build breaks `import torch` |
| `_probe_arm_render.py` | whether an OCR sender runs on a GB200 node's Grace cores |
| `_probe_int8_threading.py` | root-causing the INT8 layout slowdown seen in the fleet matrix |
| `_probe_int8_wiring.py` | why the INT8 layout backend failed to construct |
| `_probe_raw_pages.py` | raw OCR responses, to settle two questions about surviving fences |
| `_pull_sample_pdf_bytes.py` | on-cluster Zephyr filter of the fetch artifact to a sampled key set |
| `_read_run_errors.py` | tallies per-document error strings from a comparison run |
| `_shard_write_times.py` | whether a run's failed rows were written by an earlier run |
| `_smoke_converter_pool.py` | end-to-end smoke of the brokered converter fleet |
| `_smoke_split_ocr.py` | end-to-end smoke of the two-phase OCR step on 4 GPUs |
| `_tune_layout_quantization.py` | picks the NNCF recipe on the hardware that runs it |
| `_validate_ocr_port.py` | diffs the ported OCR feature extractor against the vendored original |
| `_watch_fleet.py` | terminal dashboard for the fleet extract run |
| `_sample_browser/app.py` | local server for browsing extracted samples |
| `_sample_browser/README.md` | how to run the sample browser |
| `_sample_browser/static/app.js` | sample browser front end |
| `_sample_browser/static/index.html` | sample browser page |
| `_sample_browser/static/style.css` | sample browser styling |
| `quality/merge_ft_scores.py` | merges fast-transformer holdout scores into local sample parquets |

**The all-routes study variants.** A separate corpus build that OCR'd both routes to compare OCR
against CPU extraction. Superseded by the head-to-head the reports carry, and never part of the
production DAG.

| path | reason |
|---|---|
| `extract_ocr_all.py` | OCRs the entire 10% sample, both routes, for the comparison |
| `repair_ocr_all.py` | brings that corpus up to current post-processing and exact-dedups it |
| `fuzzy_ocr_all.py` | quality-aware fuzzy dedup over the quality-filtered all-routes corpus |
| `finalize_ocr_all.py` | quality filter, fuzzy dedup and language labels for the all-routes corpus |
| `combine_docling_all.py` | joins the two Docling passes over the sample into one deduplicated corpus |

## lib/** — the library changes the PR touches

| path | verdict | reason |
|---|---|---|
| `lib/marin/src/marin/inference/converter_pool.py` | CAMPAIGN | a process pool for Docling converters; its only consumer is `docling_extract/service.py` |
| `lib/marin/src/marin/inference/proxy.py` | SHIP | sizes anyio's `to_thread` limiter to the pending budget; the default 40 threads capped any brokered fleet at 40 in flight, and demotes per-request logging to DEBUG |
| `lib/marin/src/marin/inference/dashboard_server.py` | SHIP | sizes httpx's connection pool; the default 100 capped the `/v1` funnel far below a batch engine's `max_num_seqs` |
| `lib/marin/src/marin/inference/worker.py` | SHIP | same connection-pool ceiling on the worker's forwarding path, plus error-response handling |
| `lib/marin/src/marin/inference/broker.py` | SHIP | adds `stats()` for run monitoring; throughput comes from differentiating `completed_total` |
| `lib/marin/src/marin/inference/types.py` | SHIP | `BrokerStats` and the `BrokerStatsProvider` protocol behind that snapshot |
| `lib/marin/src/marin/inference/config.py` | AMBIGUOUS | `uv_with_packages` / `uv_extra_index_urls` for prebuilt FlashInfer kernels on images without nvcc; a CUDA-serving fix the OCR fleet needed, not PDF code |
| `lib/marin/src/marin/inference/vllm_server.py` | AMBIGUOUS | the launcher half of the same FlashInfer plumbing, plus `VllmLauncherWithEnvironment` |
| `lib/marin/src/marin/inference/vllm_backend.py` | AMBIGUOUS | splits `start()` out of the serve context manager so a caller can own HTTP readiness; same serving-path scope question |
| `lib/marin/src/marin/datakit/normalize.py` | SHIP | plumbs `coordinator_resources` through and exports `make_split_writer`; the 1 GB default OOMs at exit 137 near the end of a stage over thousands of small files |
| `lib/marin/src/marin/datakit/decon.py` | SHIP | the same coordinator sizing through the decontamination steps |
| `lib/marin/src/marin/processing/classification/consolidate.py` | SHIP | the same coordinator sizing through `consolidate-filter` |
| `lib/zephyr/src/zephyr/polars_io.py` | SHIP | `scan_parquet_chunk`; polars does not honour the CoreWeave credential helper's env block and bare `https://cwobject.com` returns 400, so the bucket is spliced into the host |
| `lib/zephyr/src/zephyr/external_sort.py` | SHIP | routes merge scans through `scan_parquet_chunk` so external sort addresses CoreWeave correctly |
| `lib/zephyr/src/zephyr/shuffle.py` | SHIP | same addressing through the shuffle path, and drops a schema-unification helper |
| `lib/zephyr/tests/test_polars_io.py` | SHIP | covers the addressing helper |
| `lib/zephyr/tests/test_shuffle.py` | SHIP | one-line follow-on from the shuffle change |
| `lib/marin/pyproject.toml` | **PROVISIONAL** | the `pdf` extra needs curating against this manifest — see below |

The `pdf` extra as it stands declares `warcio`, `pymupdf==1.26.1`, `xgboost-cpu`, `docling>=2.117.0`,
`openvino`, `nncf`, `pdf-inspector==1.17.0`, `pdf-oxide==0.3.77` and `pypdfium2==5.13.0`. Under the
current design `docling`, `openvino` and `nncf` go with the Docling route; `pdf-oxide` is retained
only to keep a rejected candidate's probe runnable and belongs on this branch; `pymupdf` survives as
a test-path dependency at minimum, and as the render path until PDFium lands. One concrete defect to
fix rather than classify: the `pypdfium2` comment still reads "evaluated ... and *not* adopted", which
was written at the rejection commit `5ff241a1e2` and was not updated by the overturn `33534eb1a5`.
The shipped tree should not assert a verdict its own report has retracted.

## tests/**

| path | verdict | reason |
|---|---|---|
| `tests/datakit/test_build_pdf_source.py` | SHIP | covers `plan.py`, `fetch.py`, `common.py` |
| `tests/datakit/test_boilerplate.py` | SHIP | covers `boilerplate.py` |
| `tests/datakit/test_loop_repair.py` | SHIP | covers `loop_repair.py` and the OCR route's repair path |
| `tests/datakit/test_ocr_extract.py` | SHIP | covers `extract_ocr.py`, the OCR client and `render.py`; authors fixtures with PyMuPDF, which is the open AGPL test-path question |
| `tests/datakit/test_combine_routes.py` | SHIP | covers `combine_routes.py` |
| `tests/datakit/test_pdf_dedup.py` | SHIP | covers `dedup.py` |
| `tests/datakit/test_quality_label.py` | SHIP | covers `quality_label.py` |
| `tests/datakit/test_fuzzy_reelect.py` | SHIP | covers `fuzzy_dedup.py`'s quality-aware canonical election |
| `tests/datakit/test_language_label.py` | SHIP | covers `language_label.py` |
| `tests/datakit/test_build_oracle_sample.py` | SHIP | covers the shipped scorer's sample builder |
| `tests/datakit/test_build_labels.py` | SHIP | covers the shipped scorer's label builder |
| `tests/inference/test_converter_pool.py` | CAMPAIGN | covers `converter_pool.py`, which retires with Docling |
| `tests/datakit/test_convert_service.py` | CAMPAIGN | covers the Docling converter and its service handler |
| `tests/datakit/test_docling_extract.py` | CAMPAIGN | covers Docling assembly, backend, fields and serializer |
| `tests/datakit/test_docling_fields.py` | CAMPAIGN | covers the docling model-field patch |
| `tests/datakit/test_page_numbers.py` | CAMPAIGN | covers `docling_extract/page_numbers.py` |
| `tests/datakit/test_pdf_classify.py` | CAMPAIGN | covers v1 `classify.py` and the v1 feature contract |
| `tests/datakit/test_route_features.py` | CAMPAIGN | covers the deleted PyMuPDF router pass |
| `tests/datakit/test_ocr_features.py` | CAMPAIGN | covers the 124-feature FinePDFs incumbent and its pure-data contract |
| `tests/datakit/test_route_agreement.py` | CAMPAIGN | covers the retired agreement target |
| `tests/datakit/test_route_v2.py` | SHIP | covers `route_v2_features`, the preference-set build and the render budget; the only test that exercises the shipped router |
| `tests/datakit/test_combine_docling_all.py` | CAMPAIGN | covers the all-routes study variant |
| `tests/datakit/test_repair_ocr_all.py` | CAMPAIGN | covers the all-routes study variant |

A v2 `classify.py` ships without test coverage unless `test_pdf_classify.py` is rewritten alongside
it. That is a gap to close in the PR, not a classification.

## experiments/b200_ocr/ — cited by a shipped report

CAMPAIGN, all sixteen paths, and load-bearing anyway. [ocr-budget-sweep.md](ocr-budget-sweep.md)
ships and cites this directory by commit:

> The harness and raw results live at commit `39b4095fa` under `experiments/b200_ocr/`.

`39b4095fa` is already on `origin/mark/pdf_processing`, so the citation resolves on GitHub today and
resolves from this branch's first parent. Nothing needs to move into the PR, but
`origin/mark/pdf_processing` must not be deleted or rewritten while
[ocr-budget-sweep.md](ocr-budget-sweep.md) is in the tree. The nine `results/*.jsonl` files are the
only record behind the sweep's numbers, and `results/.gitignore` un-ignores them against the
repo-root `*.jsonl` rule — that negation has to survive any move.

`experiments/b200_ocr/budget_sweep_report.md` is the earlier in-place copy of
[ocr-budget-sweep.md](ocr-budget-sweep.md); the two differ only in the provenance paragraph, and the
`build_pdf_source` copy is the later self-contained one.

## Ambiguities handed back

**1. Does the v2 router's training chain ship with the model?** The PR already contains v1's
equivalent — `build_oracle_sample.py`, `build_route_study.py`, `fit_route_booster.py`,
`train_route_model.py`, `analyze_route_study.py`, `route_agreement.py`, `build_adjudication_set.py`.
This manifest ships only `fit_route_v2.py` and archives the rest of v2's chain, which is a narrower
rule than the PR currently follows. The alternative is to ship `build_preference_set.py`,
`judge_preference_set.py`, `build_inspector_output_study.py` and `analyze_route_v2.py` too, on the
grounds that a router nobody can retrain is a router nobody can move off its operating point. I
lean to shipping them: [pdf-router-v2.md](pdf-router-v2.md) is a SHIP report and its "Reproducing"
block invokes all four by module path, so archiving them leaves a shipped document citing code that
is not in the tree.

**2. Which reports live beside the code, and which are indexed?** `docs/reports/index.md` links
`docs/reports/pdf-extraction-routing.md` — a second copy of the v1 router report, differing from the
`experiments/` copy only in carrying the pre-datakit module path in its repro commands, which no
longer resolves. So the index advertises the one router report the design has retired, with a broken
command, and none of the reports the design depends on. Both copies are CAMPAIGN by content, but the
index entry is a shipped artifact pointing at retired work. Someone has to decide whether
`pdf-router-v2.md` gets an index entry and whether the `docs/reports/` copy is deleted or replaced.

**3. Is the render swap in scope for this PR?** [pdfium-evaluation.md](pdfium-evaluation.md) says
adopt and attaches subprocess isolation as a condition the code does not yet meet. If the swap lands
with the PR, `render.py` and the report ship together and the `pypdfium2` comment is rewritten. If it
does not, `render.py` ships on PyMuPDF, the report is campaign history until the follow-up, and the
stale comment still has to be corrected because it contradicts a report that will be on this branch
either way. This is the decision the concurrent exit-133 investigation resolves.

**4. Where does the PNG encoder finding live?** Pillow at `compress_level=1` is pixel-identical on
3,014 of 3,014 pages and renderer-independent; both the pdf_oxide and PDFium reports recommend taking
it regardless. The change belongs in `render.py`, so it ships. But the only written record of it is
in two reports this manifest sends to CAMPAIGN and PROVISIONAL. Either
[pdf-router-v2.md](pdf-router-v2.md) absorbs the finding, or `pdf-oxide-evaluation.md` ships for one
section of an otherwise negative result. One caveat is unresolved in the source: the Pillow payload
is 2.6% larger and the sweep found API-side CPU to be what sets throughput, so the report asks for
that increase to be confirmed against the serving side before the swap is taken.

**5. Are the vLLM launcher changes PDF work?** `config.py`, `vllm_server.py` and `vllm_backend.py`
carry FlashInfer kernel-artifact plumbing and a `start()`/`serve()` split. The OCR fleet needed them,
but they are serving-path changes with no PDF-specific content and they will be reviewed by a
different owner. They can ship in #8023 or split into a serving PR; splitting makes both easier to
review. The connection-pool and thread-limiter fixes in `proxy.py`, `worker.py` and
`dashboard_server.py` are the same defect class and I would keep those together, wherever they land.

**6. Two PR files outside this manifest's scope.** `experiments/grug/moe/launch_pdf_compare.py` and
`experiments/grug/moe/train.py` are the downstream training comparison that consumes this corpus;
`pyproject.toml`'s `flash-attn-4` constraint entry is a workspace-resolution fix with no PDF content.
All three are in #8023 today and none is PDF pipeline code.

**7. `route_v2_features.py` imports `route_feature_names.py`.** The v1 names module is CAMPAIGN and
the v2 contract is SHIP, so the import has to be broken before the PR is coherent — either by
inlining the names the v2 contract still uses, or by promoting a trimmed names module. Small, but it
is the one place the two partitions touch.

## Appendix: how this branch was consolidated, and the no-loss check

`mark/pdf_processing` and `mark/pdf_inspector_eval` had no shared PDF history — their merge base is
`1373230331`, a plain `main` commit where neither package path exists — so the two trees are
independent additions at different paths. That made a real merge the right mechanism over
cherry-picking or a path-preserving copy: cherry-picking the 30 evaluation commits would have
conflicted on every file they modified rather than added, and a copy would have kept the content
while dropping the commits the reports' provenance rests on. The merge keeps both package paths side
by side and makes every campaign commit reachable from one head.

Twenty-eight files conflicted, all add/add, and all the same shape: a campaign copy against its later
evolved descendant on the PR branch. Every one resolves to the descendant. The campaign copies stay
reachable through the first parent, and 31 of them also survive verbatim in the tree at
`experiments/build_pdf_source/`.

The no-loss check compares the merged tree against both parents:

```bash
git ls-tree -r --name-only mark/pdf_processing@{u} | sort > /tmp/proc
git ls-tree -r --name-only mark/pdf_inspector_eval | sort > /tmp/insp
git ls-tree -r --name-only HEAD | sort > /tmp/merged
comm -23 /tmp/insp /tmp/merged     # empty
comm -23 /tmp/proc /tmp/merged     # 35 paths, all upstream deletions
```

Every path on `mark/pdf_inspector_eval` is present. Thirty-five paths present on the pre-merge
`mark/pdf_processing` are absent afterwards, and all thirty-five were deleted on `main` between the
merge base and the PR branch's base: `lib/finelog/rust/src/query/sidecar.rs` (#8012),
`lib/marin/src/marin/inference/tpu_vllm_pins.py` and `tests/inference/test_tpu_vllm_pins.py` (#7758),
`lib/marin/src/marin/evaluation/samples.py` (#8026), and 31 Iris test files replaced by deterministic
journeys (#7842). None is a campaign artifact; inheriting them is the point of merging forward.

`backup/pdf_processing_pre_integration` holds nothing unique. It is a strict ancestor of
`mark/pdf_processing`, `git log HEAD..backup/pdf_processing_pre_integration` is empty, and the paths
it carries that this branch does not are the same class of upstream deletion (the Grafana dashboard
JSON and `TimeSeriesChart.vue`, removed by `main` before the campaign branch merged it).

The campaign branch is an archive and is not expected to lint or import as a whole. The two package
paths coexist deliberately: `tests/datakit/` imports the datakit package, while the campaign-only
tests still import `experiments.build_pdf_source`.
