# pdf-inspector as a cheap extraction route, at 1.17.0

`pdf-inspector` is a Rust PDF classifier and Markdown extractor proposed as a cheaper front end to
— or replacement for — the Docling CPU route, whose report `pdf-extraction-routing.md` retired with it
to the `mark/pdf_processing` campaign branch.
An earlier pass evaluated 1.14.1 and found it beat Docling in 41.4% of blind head-to-heads, with two
significant losses (RTL and table-heavy documents) and no significant wins. Upstream 1.17.0 rewrites
exactly those two areas across ten commits. This is the paired re-run.

Everything below is measured on the same 100,000-document oracle sample
(`marin/data/pdf_quality/cc_focus_2026_22_sample100k`), the same 178 shards, and the same
domain-disjoint split, with both builds' artifacts kept side by side. The 1.14.1 tables, packets and
verdicts are untouched.

## The wheel now contains an OCR engine, and it stays asleep

From 1.15.0 the crate's `python` Cargo feature implies `ocr = ["render-pdfium", "ocr-oar",
"model-download"]`, so the published wheel carries PDFium bindings, ONNX Runtime bindings and a
`ureq` HTTP client that fetches ONNX weights from a GitHub release. On a fleet of hundreds of
workers an implicit model fetch is an incident, and an extractor that silently OCRs is not a
text-layer extractor at all — the per-page cost model and every comparison against Docling would be
void. This had to be settled before anything else was measured.

Reading 1.17.0's sources says it cannot happen on the entry points this evaluation uses.
`detect_pdf_bytes` and `extract_pages_markdown_bytes` resolve to `detect_pdf_mem` and
`extract_pages_markdown_mem`; neither references `crate::vision`, and the OCR pipeline is reachable
only from `process_pdf_with_ocr` and `process_pdf_with_ocr_bytes`. `PDF_INSPECTOR_MODEL_CACHE`
relocates a cache rather than enabling anything, and no other environment variable touches it.

That is a claim about a call graph, so `probe_pdf_inspector` measures it instead. Over 1,000 crawl
PDFs on each architecture, with every document read from storage before the measurement window
opens:

| | x86_64 | aarch64 |
|---|---|---|
| Worker socket descriptors, peak | **0** | **0** |
| Model-cache files created | **0** | **0** |
| Outbound connections, library window | 702 over 165 s | 479 |
| Outbound connections, idle control | 81 over 20 s (≈668 projected) | — |

The socket census is the decisive one: the worker process never held a socket at any point between
documents, which a pooled `ureq::Agent` could not manage. The namespace counters agree — the pod's
own background traffic projects to ~668 connections over the library window against 702 observed,
and received bytes came in *below* the idle projection. Nothing was written to any cache root. The
OCR entry points are present in the binary (the probe refuses to run against a build without them,
so this is a dormancy result rather than an absence result).

**Verdict: no model download, no network egress, no OCR activation.** The cost model stands.

## Extraction got 1.77× slower

This is the largest change in the release and it is not in its favour. Paired per document over the
whole 89,122-document evaluable set, using the library's own timings:

| Call | 1.14.1 | 1.17.0 | ratio |
|---|---|---|---|
| `detect_pdf_bytes` | 0.439 ms/page | 0.461 ms/page | 1.05× |
| `extract_pages_markdown_bytes` | 4.353 ms/page | **7.709 ms/page** | **1.77×** |

The median document is only 1.20× slower; the p90 is 2.97× and the p99 is 10.50×. 80.0% of documents
got slower by more than 5%. The dedicated single-task probe reproduces the aggregate exactly
(4.874 → 8.602 ms/page on x86_64, 1.77×; 4.578 → 6.98 ms/page on aarch64, 1.53×) and shows the tail
directly: the slowest document went from 4.1 s to 13.1 s. The slowdown is not a function of page
count — median ratio is ~1.15 in every page-count bucket from 1 page to 100+ — so it is the new
table-recovery and banded-layout passes firing on the documents that have tables and columns.

Cheapness was the entire argument for this library, and it survives: 7.7 ms/page is still ~130×
cheaper than Docling's ~1000 ms/page. But the margin against the 35 ms/page route-feature extractor
narrowed from 8× to 4.5×, and the tail is now long enough to hit deadlines (below).

## Survival: the same clean result, plus ten new timeouts

| | 1.14.1 | 1.17.0 |
|---|---|---|
| Documents | 100,000 | 100,000 |
| Panics (`PanicException`) | 0 | 0 |
| Worker deaths (SIGSEGV/SIGABRT/SIGKILL) | 0 | 0 |
| Refusals (`ValueError`) | 253 | 252 |
| Timeouts at 30 s | 10 | **17** |
| Total failures | 263 | 269 |

The transition matrix shows 10 documents that extracted cleanly under 1.14.1 and now exceed the 30-second
deadline, against 3 timeouts recovered and 1 refusal recovered. The new table code is new code on
adversarial input and it did not crash once: no panic, no segfault, no OOM, on either architecture.
What it did was make the worst documents slower, which is the same finding as the speed table read
from the other end. 0.017% of the corpus is a rounding error, but it is a real regression and a
deadline is the only thing bounding it — the library still exposes no page cap or byte cap of its own.

## The normalizer still fits, and page furniture is not double-stripped

`route_agreement` folds away each route's serialization before comparing tokens, and a rule that
stops matching leaks markup into the stream as content one route "added". Re-auditing 1,000
documents against 1.17.0's actual output:

Every construct pdf-inspector emits at meaningful frequency is one the normalizer already handles —
headings (82.7% of documents), emphasis (66.5%), pipe tables with delimiter rows (55.2%), HTML tags
(51.2%), `<u>` wrappers (50.2%), links (37.7%), lists (34.0%), LaTeX math (7.5%). The constructs that
would be new leak classes are absent outright: no markdown images, no footnote markers, no page
comments, no HTML tables. Nothing markup-shaped survives normalization.

What does survive is not markup. Of the 40 most common tokens pdf-inspector produces that the VLM
never does, **38 are short word fragments** — `tion` (10.7% of documents), `ing` (9.1%), `con`
(8.5%), `tions` (7.7%), `ment` (6.9%), `sion`, `ments`, `ity`, `tive`. That is the signature of words
hyphenated at line breaks and not rejoined, and the same fragments appear against Docling as well as
against the VLM, so both other routes join what pdf-inspector splits. Commit `2543abe` (#388) added
hyphen rejoining for exactly this and it has not eliminated the problem on crawl PDFs. This is a
real extraction difference that the metric correctly penalises, not a normalizer defect, and the
normalizer was deliberately left unchanged so the paired numbers stay comparable.

Running headers still reach the output on 12.1% of multi-page documents, which settles the
double-stripping question: `fb45d37` (#427) strips page-edge furniture only when
`strip_headers_footers` is set, and `extract_pages_markdown_mem` passes `false`. It does not overlap
`boilerplate.py`.

## What the metric says: RTL transformed, everything else marginal

Paired bigram recall against the VLM, per document, over 89,122 documents and 2,498 domains.
Intervals are bootstrap over *domains*, not documents, because the crawl holds ~9.8% exact-duplicate
PDFs and many more near-duplicates per publisher. The noise column asks whether the change clears the
~0.012 split-draw noise the published pass measured.

| Stratum | n | domains | 1.14.1 | 1.17.0 | delta | 95% CI | > noise |
|---|---:|---:|---:|---:|---:|---|:--:|
| **rtl** | 321 | 36 | 0.1884 | **0.3789** | **+0.1906** | [+0.047, +0.301] | yes |
| encoding_damage | 16,424 | 1,105 | 0.7277 | 0.7601 | +0.0324 | [+0.015, +0.054] | yes |
| math_dense | 7,883 | 1,071 | 0.8501 | 0.8642 | +0.0141 | [+0.003, +0.031] | yes |
| table_heavy | 11,065 | 1,221 | 0.8604 | 0.8726 | +0.0122 | [+0.008, +0.018] | yes |
| multicolumn | 2,656 | 578 | 0.6193 | 0.6313 | +0.0119 | [+0.002, +0.031] | no |
| latin_text_baseline | 34,379 | 1,912 | 0.8686 | 0.8747 | +0.0061 | [+0.005, +0.008] | no |
| cjk | 3,564 | 147 | 0.4163 | 0.4179 | +0.0016 | [+0.001, +0.003] | no |
| scanned_image_mixed | 6,293 | 884 | 0.1283 | 0.1233 | **−0.0050** | [−0.008, −0.003] | no |
| mutual_agreement_vlm_diverge | 6,537 | 1,115 | 0.4724 | 0.4644 | **−0.0080** | [−0.010, −0.006] | no |
| **ALL** | 89,122 | 2,498 | 0.7307 | 0.7419 | +0.0113 | [+0.007, +0.019] | no |

RTL is transformed. The share of RTL pages destroyed outright (bigram recall below 0.5) fell from
0.898 to 0.558, and unigram recall went 0.300 → 0.513. `00a7579` and `3a3e68f` did what they claimed.

Everything else moves by a hair. The corpus-wide gain of +0.0113 does not clear the noise floor, and
two strata went backwards: documents where both cheap routes fail the label yet agree with each other,
and scanned/image/mixed documents — where precision fell −0.0281 and −0.0111 respectively, both
clearing the noise floor in the wrong direction. On `mutual_agreement_vlm_diverge` the destroyed-page
fraction rose +0.0156 and agreement with Docling fell −0.0223: the new layout passes are actively
changing what these documents read as, and not for the better.

## What the judges say: RTL rescued, tables untouched

345 documents were re-judged **paired** — the same packets, the same three pages, the same blinded
label assignment, with only pdf-inspector's text replaced. All 345 rebuilt; the inspector's page-level
recall moved on 118 of them. A further 260 fresh documents extend the strata that were most
under-powered. The two arms are reported separately and never pooled.

Overall, on the paired 345:

| | 1.14.1 | 1.17.0 |
|---|---|---|
| inspector beats Docling | 41.4% [36.4, 46.7] | **42.6% [37.5, 47.9]** |
| Win rate (ranked first) | VLM 58.6 / Docling 25.5 / inspector 15.9 | VLM 58.3 / Docling 21.7 / **inspector 20.0** |
| Ranked last | inspector 50.7% | inspector **50.7%** |
| Decisive verdicts | 99.1% | 99.4% |
| Inter-judge pairwise κ | 0.844 | 0.844 |

The paired shift is +0.012 — 30 documents flipped to pdf-inspector, 26 flipped to Docling,
McNemar p = 0.689. **The overall head-to-head did not move.** pdf-inspector wins outright more often
(+4.1pp, taken from Docling) but is ranked last exactly as often as before.

Per stratum, paired, with the flip counts that carry the test:

| Stratum | n | domains | before | after | delta | flips → insp / → docl | McNemar p |
|---|---:|---:|---:|---:|---:|---|---:|
| **rtl** | 55 | 36 | 0.182 | **0.400** | **+0.218** | 13 / 1 | **0.002** |
| math_dense | 30 | 28 | 0.600 | 0.733 | +0.133 | 5 / 1 | 0.219 |
| latin_text_baseline | 40 | 35 | 0.575 | 0.600 | +0.025 | 2 / 1 | 1.000 |
| **table_heavy** | 30 | 30 | 0.267 | **0.267** | **0.000** | 1 / 1 | 1.000 |
| scanned_image_mixed | 35 | 32 | 0.371 | 0.343 | −0.029 | 2 / 3 | 1.000 |
| encoding_damage | 30 | 28 | 0.467 | 0.433 | −0.033 | 3 / 4 | 1.000 |
| multicolumn | 30 | 27 | 0.400 | 0.333 | −0.067 | 0 / 2 | 0.500 |
| mutual_agreement_vlm_diverge | 45 | 43 | 0.511 | 0.422 | −0.089 | 3 / 7 | 0.344 |
| cjk | 50 | 32 | 0.440 | 0.340 | −0.100 | 1 / 6 | 0.125 |

RTL is the only significant paired move anywhere, and it is decisive. **`table_heavy` is unmoved to
three decimal places** — one document flipped each way — despite eight upstream commits aimed at
tables and layout and a measurable +0.0122 in bigram recall. The metric improvement is real and too
small for a judge looking at a rendered page to care about.

The table commits are visible in the signals, just not in the verdicts. Pages reported as containing
tables *fell* from 2.741 to 2.608 per document, and pages with columns rose from 4.043 to 4.127.
That is the shape of `0027b04`, `7f982d2` and `4ee9664`, which reject parallel-prose grids, tables
made of running headers, and page-flow projections — most of the release's table work removes false
positives rather than recovering more structure, so it moves precision on documents that were
already being read acceptably and leaves the genuinely table-heavy ones where they were.

The extension arm, which is where the power is:

| Stratum | n | domains | inspector beats Docling | 95% CI | wins outright |
|---|---:|---:|---:|---|---:|
| latin_text_baseline | 80 | 75 | **0.650** | [0.541, 0.745] | 25.0% |
| encoding_damage | 90 | 75 | 0.511 | [0.410, 0.612] | 17.8% |
| table_heavy | 90 | 79 | 0.400 | [0.305, 0.503] | 11.1% |

Reading the two significant losses of the previous pass against 1.17.0:

- **`rtl` is no longer a significant loss.** 0.182 [0.10, 0.30] became 0.400 [0.281, 0.532]; the
  interval now includes parity. It is still not a win, and it is 55 documents on 36 domains, which is
  the number to believe.
- **`table_heavy` is no longer *significant*, but only because the sample grew.** The paired 30 sit
  at 0.267 [0.142, 0.444], still excluding parity; the better-powered extension 90 sit at 0.400
  [0.305, 0.503], which grazes it. The honest reading is that table-heavy documents remain
  pdf-inspector's worst non-scanned stratum and the upstream table work did not change that.

Against zero significant wins in the previous pass, 1.17.0 produces two: `math_dense` at 0.733
[0.556, 0.858] on the paired arm, and `latin_text_baseline` at 0.650 [0.541, 0.745] on the extension.

`cjk` is the regression to watch: −0.100 with 6 documents flipping to Docling against 1 the other
way. p = 0.125 on n=50 over 32 domains is a flag, not a finding, but it points the same way as the
metric's negative deltas on `mutual_agreement_vlm_diverge` and `scanned_image_mixed`.

**The style confound doubled.** Judged in native dialect rather than canonical, pdf-inspector gains
+6.7pp against Docling, up from +3.3pp for 1.14.1 — 1.17.0 recovers more tables, which makes its
Markdown look more like the VLM's. Canonicalization removes it, and every headline number above is
from the canonical arm. Had this pass been judged natively it would have reported a larger
improvement than actually happened.

## Routing: nothing changed

Stage 2 was re-run on the same domain-disjoint split. As a feature source, quality loss at the 50%
VLM budget:

| Arm | ms/page | 1.14.1 | 1.17.0 |
|---|---:|---:|---:|
| `route_features` (shipped) | 35.00 | 0.1112 | 0.1112 |
| inspector detect | 0.46 | 0.1687 | 0.1700 |
| inspector detect + extract | 7.71 | 0.1433 | 0.1410 |
| `route_features` + detect | 35.46 | 0.1111 | 0.1111 |
| `route_features` + detect + extract | 42.71 | 0.1104 | 0.1101 |

Every delta is ≤ 0.0023, an order of magnitude inside the noise floor. pdf-inspector's signals are
still worse than the route features alone and still add nothing on top of them. `detect_pdf_bytes`
also still populates none of its four declared layout signals — `has_encoding_issues`,
`is_complex_layout`, `pages_with_tables`, `pages_with_columns` are constant across all 100,000
documents, exactly as in 1.14.1 — so the cheap tier gained no capability and the layout signals still
cost the full extraction.

As the cheap route itself, against its own proxy label, pdf-inspector improves slightly: it reaches
Docling's 50%-budget quality at a 25.4% document budget (28.9% of pages, 252 crawl-wide GPU-hours),
against 28.2% / 31.2% / 273 GPU-hours for 1.14.1 and 50% / 76.9% / 672 GPU-hours for Docling. Its
label pass-rate rose from 0.6248 to 0.6339. That advantage is measured against agreement with the
VLM, which is the metric the adjudication exists to check, and the adjudication says pdf-inspector is
ranked last on half of all documents. The routing number and the judged number disagree for the
reason they disagreed before: pdf-inspector shares the VLM's dialect.

The split-draw noise floor was re-confirmed incidentally. The shipped `route_features` arm scored
0.1230 in the published pass and 0.1112 here on identical features and identical data — a 0.0118
difference from the split draw alone, which is the ~0.012 quoted throughout.

## Does 1.17.0 change the conclusion?

**No.** pdf-inspector's extraction is still not better than Docling's.

On the paired 345 documents it beats Docling in 42.6% of head-to-heads, CI [37.5, 47.9], which still
excludes parity; it is still ranked last more often than either alternative, on exactly the same
50.7% of documents as before; and the VLM still wins 58.3%. The paired shift of +0.012 is not
significant (McNemar p = 0.689).

What 1.17.0 genuinely delivers is the RTL fix. That is a real, large, significant repair of a defect
the previous pass identified precisely, and it removes the more embarrassing of the two significant
losses. It does not generalise: `table_heavy` did not move at all under paired judging, the
corpus-wide metric gain does not clear the split-draw noise floor, `cjk` moved backwards by enough to
warrant watching, and two strata regressed on precision beyond the noise floor.

The cost is concrete. Extraction is 1.77× slower with a p99 ratio of 10.5× and ten new deadline
failures, bought for a corpus-wide bigram-recall gain of +0.0113 that a blind judge cannot see. If
pdf-inspector is adopted it should be for its price, its clean failure behaviour and now its RTL
handling — not on a claim that it reads documents better than Docling, which two passes have now
failed to support.

## Reproducing

**The modules below are not in this tree.** They stayed with the campaign, on `mark/pdf_processing`
under `experiments/datakit/build_pdf_source/quality/`, so check that branch out to run any of this.

```bash
# Stage 0: survival, latency, and the egress accounting. Once per architecture.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-inspector-probe --extra pdf \
    --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.probe_pdf_inspector

# Stage 1: the study table for this build, at its own prefix.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-inspector-study --extra pdf \
    --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.build_inspector_study

# The paired difference between two builds' study tables.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-inspector-compare --extra pdf \
    --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.compare_inspector_versions
```

`quality/audit_inspector_format.py` re-checks the normalizer,
`quality/analyze_inspector_routing.py` prices the router, and
`quality/build_adjudication_set.py` plus `quality/judge_adjudication_set.py` build and judge the
paired and extension packets.

Artifacts, with the 1.14.1 originals left in place beside them:

| | 1.14.1 | 1.17.0 |
|---|---|---|
| Probe | `.../cc_focus_2026_22_pdf_inspector_probe/{arch}.parquet` | `.../{arch}-1.17.0.parquet` |
| Study table | `.../cc_focus_2026_22_inspector_study` | `.../cc_focus_2026_22_inspector_study_1_17_0` |
| Routing result | `.../cc_focus_2026_22_inspector_routing.json` | `.../cc_focus_2026_22_inspector_routing_1_17_0.json` |
| Adjudication | `.../cc_focus_2026_22_adjudication` | `.../cc_focus_2026_22_adjudication_1_17_0` |

The paired difference is at `.../cc_focus_2026_22_inspector_version_compare.json` and the format
audit at `.../cc_focus_2026_22_inspector_format_audit.json`. Judging 605 packets across three passes
cost $2.88.
