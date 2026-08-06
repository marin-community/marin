# Routing PDFs between Docling and the VLM

Every PDF in the focus crawl goes down one of two extraction routes: Docling, which reads the
embedded text layer on CPU, or `Infinity-Parser2-Flash`, which reads a rendered image on GPU. The
router decides which. This report measures what the router was doing, why it was wrong, and what
replaced it.

Measured on the 100k oracle sample
(`marin/data/pdf_quality/cc_focus_2026_22_sample100k`): 100,000 documents, of which 89,327 (89.3%)
are usable for routing analysis. The rest are dropped rather than labelled — 10.7% because the VLM
extraction is itself damaged (truncated pages, loop repair, pages rendered below the legibility
floor), where a disagreement measures the VLM's failure rather than Docling's.

## The old router answered a different question

The step ran the [FinePDFs](https://github.com/huggingface/finepdfs) XGBoost classifier, trained to
answer *is this a scan?* The decision actually being made is *will Docling read this document the
way the VLM would?* Those come apart badly. A born-digital paper with a broken ToUnicode CMap, an
invisible OCR layer sitting over a bitmap, a two-column layout, or a page of equations all have
healthy "not a scan" statistics and all extract badly.

The consequence is not subtle. The shipped rule routed 31.4% of documents to the VLM and left **28.7%
of the entire corpus on the Docling route with materially degraded text**, catching only 42% of the
documents Docling cannot read. That loss is silent: the text lands in training data truncated,
garbled or misordered, and nothing downstream flags it.

## What the new router measures

`quality/route_features.py` computes ~70 signals per document from 8 sampled pages, decode-free and
model-free, in six groups:

| Group | What it asks |
|---|---|
| `encoding` | Can the glyph→Unicode mapping be believed? Embedded fonts, ToUnicode coverage, Type3 and glyphless fonts, unmappable subsets, replacement and private-use characters, Marker's `detect_bad_ocr` ratios |
| `layer` | Does the text sit where the ink is? Invisible text, invisible text over a bitmap, overlapping and out-of-page lines, duplicated spans |
| `math` | TeX/AMS/STIX font names, fraction of characters in mathematical Unicode blocks |
| `structure` | Ruling lines and the grid they imply; how aligned the text's left edges are |
| `order` | Content-stream order versus a column-aware geometric reading order; column count |
| `script` | CJK, RTL and Latin character fractions |

Cost is **35 ms per page against Docling's ~1000 ms**, so the router stays roughly 30× cheaper than
the extraction it decides to skip.

The target is Docling-versus-VLM agreement (`quality/route_agreement.py`): format-normalized so the
two routes' serialization conventions do not read as disagreement, reported asymmetrically (recall
is content Docling *lost*; precision is content Docling *added*), with bigram variants because
reading-order damage is invisible to unigrams — this repository already retired an INT8 layout
backend that scored 0.935 unigram F1 while splicing multi-column reading order.

## Results

Domain-disjoint held-out documents, at matched VLM budget. Splitting by registered domain rather
than by row matters: the crawl holds ~9.8% exact-duplicate PDFs and many more near-duplicates from
the same publisher.

| Router | At the shipped 29% budget | At the new 50% budget |
|---|---|---|
| Shipped FinePDFs rule | 0.3034 | 0.2681 |
| Same features, retrained on the routing label | 0.2866 | 0.1713 |
| **New route features** | **0.2534** | **0.1230** |
| Both | 0.2536 | 0.1211 |

Asking the right question buys 6% at the old budget. The new features buy 16%, and the gap widens
with budget — at 50% they buy 54% against 36% for retraining alone, because the incumbent's score
runs out of ability to rank documents long before half the corpus is selected. Adding the
incumbent's probability on top of the new features buys nothing: `ocr_prob` ranks 7th by gain when
the model may use everything, behind `mean_latin_ratio`, `mean_left_edge_concentration`,
`mean_char_count`, `page_count`, `mean_alphanum_ratio` and `pages_sampled`.

## The cost/quality frontier, and where it bends

The marginal slope has a direct reading: it is the precision of the *next* documents sent to the
VLM — the share of them Docling would genuinely have botched. Its reciprocal is how many VLM runs
you buy per document actually rescued.

| VLM budget | Quality loss | Catches bad | Marginal precision | VLM docs per rescue |
|---|---|---|---|---|
| 20% | 0.333 | 37% | 0.97 | 1.0 |
| 25% | 0.289 | — | 0.90 | 1.1 |
| 30% | 0.244 | 54% | 0.86 | 1.2 |
| 35% | 0.207 | — | 0.73 | 1.4 |
| 40% | 0.170 | 68% | 0.66 | 1.5 |
| **50%** | **0.114** | **79%** | **0.50** | **2.0** |
| 60% | 0.070 | 87% | 0.35 | 2.9 |
| 70% | 0.040 | 92% | 0.25 | 4.0 |
| 80% | 0.018 | 97% | 0.16 | 6.3 |

Three regimes. Below ~34% almost every additional VLM document is a real rescue, so stopping there
leaves cheap quality unbought — and the old 31.4% operating point sits inside that region. From 34%
to 50% the price rises gently. Past 50% it degrades fast: marginal precision falls below 0.33 at 60%
and below 0.25 at 74%, so the last stretch toward a clean corpus costs 4–6 VLM runs per document
rescued.

The frontier's knee, by maximum distance from the endpoint chord, is **45.5%**.

For contrast the incumbent's own curve is not merely worse but locally degenerate: around 20% its
marginal precision is ~0.00, so a whole band of the documents it adds are pure waste. That is the
garbled-text override pinning 17.4% of documents to a score of exactly 1.0 and clumping the
probability distribution, which leaves the rule unable to rank inside that band at all.

## What shipped

`classify.py` now routes on `docling_confidence < 0.542031`, which sends **50% of documents to the
VLM**. On held-out documents from unseen domains that point catches 79% of the documents Docling
reads badly and leaves 10.8% of the corpus mis-routed, against 30.3% for the FinePDFs rule at its
own 29% budget — and against 26.8% for that rule rethresholded to spend the same 50%.

Weights are pinned by SHA-256 (`pdf_route_classifier_00757366`) and are reproducible: the fit reads
the published study table, trains on every usable row for the round count the domain-disjoint
evaluation stopped at, and writes a JSON sidecar recording the threshold, the feature contract and
the measured operating point.

The threshold is calibrated, not tuned. The score is a probability of a *proxy* label, so only its
rank carries meaning; the threshold is the quantile of the model's own output that yields the target
budget. Recalibrating for a different budget on a new corpus is a quantile, not a retrain.
Regenerate the weights with `quality/fit_route_booster.py`.

## Two things to know before trusting these numbers further

**A page-alignment bug invalidated the first pass, and the fix changed results a lot.** Docling
drops pages it reads nothing from — 7.9% of documents come back with fewer pages than the PDF has.
Pairing pages by index compares every page after a drop against its neighbour. Blind judges reading
those packets reported one extractor "fabricating" content and the other "losing" it on documents
where neither had done anything wrong. Page matching is now content-based
(`route_agreement.align_pages`, Needleman-Wunsch over token overlap with a fast path for equal page
counts); on affected documents recall went from 0.176 to 0.894. Everything above is post-fix.

**The VLM is not a uniform gold standard, and the label is backwards on RTL.** 72 documents were
blind-adjudicated against rendered pages, routes anonymized and order randomized per document (11
excluded as contaminated by the alignment bug). The metric validates directionally: below 0.2 bigram
recall the VLM wins 89% of the time, and among decisive verdicts below 0.5, 93%. But Docling wins
~21% of the time throughout, concentrated where the VLM garbles non-Latin scripts.

| Script | n | Docling better | VLM better | Label says `docling_ok` |
|---|---|---|---|---|
| Latin | 44 | 11% | 75% | 48% |
| CJK | 5 | 0% | 100% | 0% |
| RTL | 4 | 75% | 25% | 0% |

CJK (4.0% of the corpus) is handled correctly. **RTL (0.4%) is not**: the label calls Docling
unacceptable on 95.6% of RTL documents while judges preferred Docling in 3 of 4 cases. `n=4`, so
this is a flag rather than a conclusion — but `mean_latin_ratio` is the model's top feature, so it
is currently learning "non-Latin → VLM" partly for the wrong reason. RTL deserves a carve-out and
its own labels.

The absolute quality-loss values inherit the proxy label's definition (bigram recall ≥ 0.80 against
VLM text, with a page-level floor). The shape of the frontier and the incumbent-versus-candidate
comparison are robust to it.

## Reproducing

```bash
# Build the study table: features, incumbent scores and agreement for every sampled document.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-route-study --cpu 8 --memory 32GB --disk 32GB \
    --enable-extra-resources --extra datakit \
    -- python -m experiments.build_pdf_source.quality.build_route_study

# Refit the shipped booster and recalibrate its threshold.
uv run python -m experiments.build_pdf_source.quality.fit_route_booster
```

`quality/build_adjudication_set.py` packages documents for blind human or model adjudication;
`quality/analyze_route_study.py` and `quality/train_route_model.py` hold the frontier and the
cost-matched comparison.
