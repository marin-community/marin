# PDFium for the VLM render feed, at pypdfium2 5.13.0

`pypdfium2` wraps Chromium's PDFium — BSD-3-Clause renderer, Apache-2.0/BSD bindings — evaluated as
a replacement for MuPDF in the one role PyMuPDF still holds in this pipeline once the router pass
and the Docling route are removed: turning pages into pixels.

**Yes.** PDFium is cheaper, permissively licensed, operationally as clean as MuPDF on this corpus,
and dimensionally exact. It also changes what the model reads: against a self-agreement control of
**0.9954** bigram F1 the same pages rendered through PDFium score **0.9591**, a paired delta of
**−0.0364 ± 0.0032** over 1,795 pages and eight times the noise floor. That divergence is real and it
reproduces. What it is not is a loss.

Adjudicated blind against the rendered page, over 715 pages in eight strata, two judge models and
both engines used as the reference in turn, the model's reading of a PDFium-rendered page is judged
the more faithful one **0.481 to 0.498** of the time corpus-page-weighted. Every one of the five
arms contains 0.500. On **33.7%** of pages the two readings are byte-identical and there is nothing
to choose between; on the rest the estimate is 0.4585 ± 0.0263, which also contains 0.500. The
divergence is **displacement, not degradation**, and a quality record built on readings that turn out
to be interchangeable with the new ones is not invalidated by swapping them.

**The motivation is licensing, not speed, and that is worth saying plainly.** The feed costs ~20 CPU
core-hours per million pages, of which PNG encoding is ~62% and rasterisation ~38%. PDFium rasterises
1.21× faster on x86 and 1.43× faster on aarch64, moving the whole feed by 1.25 and 1.89 core-h/M —
7% and 12%. Nobody would take a corpus-quality risk for that. The case for the swap is that PyMuPDF
is AGPL and this corpus is intended for release, and what this evaluation establishes is that the
quality risk it was weighed against does not exist.

**This reverses the first revision of this document, which recommended against the swap.** That
revision measured divergence, could not measure direction, and treated divergence from MuPDF as the
cost — which silently makes MuPDF's reading the reference and so cannot tell "worse" from
"different". [Question 1b](#question-1b-is-pdfiums-reading-worse-or-only-different) is the pass it
called for and did not run.

[`pdf-oxide-evaluation.md`](pdf-oxide-evaluation.md) rejected a renderer swap on cost — the
candidate was 1.7–1.8× slower, so fidelity never came up. PDFium is not in that position, so this
evaluation had to answer the question that one never reached: **the pixels differ; does the model's
reading differ?** Pixel identity was never available between two rasterisers and was never the test.

Everything below is measured on the cluster over 1,000 documents from the 100,000-document oracle
sample (`marin/data/pdf_quality/cc_focus_2026_22_sample100k`), the same documents in the same order
on both architectures, by [`probe_pdfium`](quality/probe_pdfium.py) and
[`build_render_study`](quality/build_render_study.py); Question 1b re-reads and adjudicates a
715-page draw from that study with
[`build_render_adjudication_set`](quality/build_render_adjudication_set.py) and
[`judge_render_adjudication_set`](quality/judge_render_adjudication_set.py). x86_64 ran on
`cw-us-east-02a`, aarch64 on `cw-us-east-08a` — the freer of the two on both occasions. **The render
study exists on aarch64 only**: the x86_64 run produced no output, so every agreement number here and
in Question 1b is an aarch64 measurement. Pages are sampled evenly across each document rather than
taken from the front, because a document's first pages are covers more often than they are
representative.

## Question 1: does the model read the page differently? Yes, materially

This is the gate, and it is the only question that could have decided the matter either way.

Each page was rendered both ways at identical `smart_resize` dimensions and encoded by the identical
Pillow encoder, then read three times by the same model with the same prompt at temperature 0: the
MuPDF rendering **twice** and the PDFium rendering once. The MuPDF-against-MuPDF pair is the control
— vLLM's greedy decoding is not bit-reproducible across batch compositions, so two requests carrying
byte-identical images do not return identical text, and without knowing that floor an absolute
agreement number means nothing.

| bucket | n | control | treatment | paired delta | changed px |
|---|---|---|---|---|---|
| **all pages** | 1,795 | 0.9954 | 0.9591 | **−0.0364 ± 0.0032** | 0.0423 |
| born_digital | 1,646 | 0.9958 | 0.9599 | −0.0359 ± 0.0033 | 0.0430 |
| scanned | 79 | 0.9844 | 0.9444 | −0.0401 ± 0.0176 | 0.0266 |
| dense_text | 455 | 0.9988 | 0.9597 | −0.0391 ± 0.0064 | 0.0668 |
| line_art | 256 | 0.9951 | 0.9344 | −0.0607 ± 0.0105 | 0.0466 |
| cjk | 77 | 0.9839 | 0.8983 | **−0.0856 ± 0.0269** | 0.0387 |
| small_glyphs | 109 | 0.9945 | 0.9065 | **−0.0881 ± 0.0185** | 0.0421 |
| below_legibility_floor | 32 | 0.9846 | 0.7841 | **−0.2006 ± 0.0586** | 0.0644 |
| most_divergent_pixels | 36 | 0.9935 | 0.9021 | −0.0914 ± 0.0385 | 0.1091 |

The model is essentially deterministic on a fixed image: 0.9954 against itself. Read against that
same reference, the PDFium rendering falls 0.0409 short of perfect agreement, of which 0.0046 is the
model's own irreproducibility and **0.0364 is the renderer**.

### The mean understates it, because the damage is a tail

The median page is **unchanged** — delta exactly 0.0000, and 62.4% of pages score at or above the
control. This is not a uniform softening of every page. It is a heavy one-sided tail:

| | pages | share |
|---|---|---|
| delta < −0.001 | 672 | 37.4% |
| delta < −0.01 | 465 | 25.9% |
| delta < −0.05 | 233 | 13.0% |
| delta < −0.10 | 142 | 7.9% |
| delta < −0.50 | 38 | 2.1% |

The sharpest way to put it: **38 of 1,795 pages (2.12%)** have a control above 0.9 and a treatment
below 0.5 — the model read the MuPDF rendering the same way twice and read the PDFium rendering as
something else. The reverse case, where PDFium is stable and MuPDF is not, occurs **once**. A 38:1
asymmetry is not sampling noise.

Extrapolated to the ~2.0M OCR-routed pages in the 10% sample, 2.1% is roughly 42,000 pages read
differently, and 7.9% is roughly 158,000.

### It is not the truncation confound, and it is not the pixel difference

Two candidate explanations were tested and neither carries the result.

**Repetition loops and token-cap truncation.** The study deliberately does not apply the pipeline's
`repair_page` loop repair, so a model that loops on one rendering shows as total disagreement here
while production would partly recover it. Dropping every page where either side hit the token cap or
where one side's output exceeds three times the other's length:

    n=1,748   control 0.9977   treatment 0.9745   paired delta −0.0233 ± 0.0018   median +0.0000

The confound accounts for roughly a third of the deficit. The remaining **−0.0233 is 13 standard
errors**, and against this subset's own tighter control (0.9977, a floor of 0.0023) it is ten times
the noise floor. The stratum pattern survives intact: cjk −0.0694,
below_legibility_floor −0.0912, small_glyphs −0.0562, line_art −0.0375. Runaway-length pages split
evenly, 11 PDFium against 11 MuPDF, so loops are not systematically a PDFium failing.

**Heavier ink.** PDFium lays down ~2.5% more ink than MuPDF (0.0879 against 0.0856), so the obvious
story is that fatter stems make the model read worse. That story is wrong: the correlation between a
page's agreement delta and its changed-pixel fraction is **−0.032**, i.e. none. The pages that
diverge are not the pages whose pixels differ most. A small local difference is enough to flip the
model into a different reading of a particular glyph, table cell or column, and which pages that
happens on is not predictable from the pixel metric.

### What this does and does not establish

It establishes that the two renderings produce materially different model output. It does **not**
establish that PDFium is worse: bigram recall (0.9622) and precision (0.9629) are symmetric, so
neither renderer systematically loses content relative to the other, and the study measures
divergence rather than correctness.

The first revision of this evaluation argued that the distinction did not matter — that divergence
from MuPDF *is* the cost, whichever reading a judge would prefer, because every quality number the
pipeline holds was established against MuPDF renderings. **That argument was wrong, and Question 1b
is where it fails.** It treats the MuPDF reading as the reference by default and so cannot
distinguish "PDFium is worse" from "PDFium is different", which is the whole question; a corpus
record built on readings that turn out to be *interchangeable* with the new ones is not invalidated
by swapping them. The blind-judging pass the paragraph above called for was run, and the answer is
below.

## Question 1b: is PDFium's reading worse, or only different?

Question 1 measured divergence and could not measure direction. This section adjudicates it, with
[`build_render_adjudication_set`](quality/build_render_adjudication_set.py) and
[`judge_render_adjudication_set`](quality/judge_render_adjudication_set.py), on `cw-us-east-08a`.
The answer is **different, not worse**, and the reason that answer took a two-way design to reach is
the most useful thing in it.

### The study did not keep the text, so the pages were read again

`build_render_study` persists `mupdf_chars`, `pdfium_chars` and the agreement columns and **not the
model's output**, so there was nothing on storage to adjudicate. Every drawn page was rendered both
ways and read again, three times as before — the MuPDF rendering twice and the PDFium rendering once
— so the fresh readings carry their own control and the divergence is shown to have reproduced
rather than assumed to have. It did, to 3% of its own size: over the 715-page draw the study's mean
delta is **−0.0849** and the fresh one **−0.0822**, against a fresh control of 0.9911.

| stratum | pages | study delta | fresh delta | fresh control | fresh treatment | identical text |
|---|---|---|---|---|---|---|
| catastrophic | 38 | −0.8291 | −0.6825 | 0.9380 | 0.2555 | 2.6% |
| large_loss | 104 | −0.2199 | −0.2170 | 0.9827 | 0.7657 | 1.0% |
| moderate_loss | 150 | −0.0344 | −0.0385 | 0.9968 | 0.9582 | 2.7% |
| below_legibility_floor | 22 | −0.0255 | −0.0795 | 0.9794 | 0.8999 | 36.4% |
| cjk | 50 | −0.0151 | −0.0221 | 0.9995 | 0.9774 | 34.0% |
| small_glyphs | 50 | −0.0146 | −0.0182 | 0.9778 | 0.9596 | 30.0% |
| unchanged | 300 | −0.0004 | −0.0025 | 0.9994 | 0.9969 | 64.7% |
| reverse_catastrophic | 1 | +0.9947 | +0.0000 | 1.0000 | 1.0000 | 100% |

The last column earns its own sentence: on **33.7%** of the draw the two renderings produced
**byte-identical** text. On a third of these pages there is nothing for any judge to prefer.

### There is no neutral reference renderer, and the pixels say so

A judge decides by looking at a rendered page, and that page has to be drawn by one of the two
engines under test. If they converged at the judging resolution the choice would not matter. They do
not. All 715 pages, both engines onto identical buffer shapes, at the feed's own dimensions and at
three fixed resolutions:

| resolution | mean DPI | changed-pixel p50 | p90 | mean abs difference |
|---|---|---|---|---|
| feed (what the VLM saw) | 145.7 | 0.0408 | 0.0751 | 2.53/255 |
| **160 (what the judge sees)** | 160.0 | **0.0376** | 0.0698 | 2.40/255 |
| 220 | 220.0 | 0.0277 | 0.0522 | 1.88/255 |
| 300 | 300.0 | 0.0228 | 0.0419 | 1.73/255 |

Divergence falls with resolution and does not collapse: at the judge's 160 DPI it retains **92%** of
the feed's changed-pixel fraction, and even at 300 DPI it retains 56%. Restricting to the strata
where the readings actually moved changes nothing (0.0427 at the feed, 0.0375 at 160 DPI). So the
adjudication was run **twice** — once against each engine's rendering of the same page, with the
text, the blinding and the section order held fixed and only the image moving — and the agreement
between the two arms is reported as a result rather than assumed away.

### What the judge is asked

One page image and two transcriptions of it, labelled A and B, with the engine's *position* and its
*letter* randomised together per page. Both transcriptions come from the same model under the same
prompt at temperature 0, so there is no dialect axis and the route packets' canonicalisation is
deliberately not applied: it would erase the structural differences the judge is meant to see.

The verdict is a **forced pairwise choice**, the half of this instrument that survived human
validation — 0.756 agreement over 45 human verdicts, 1.000 where the human was confident. The
equivalence flag is recorded and reported as description. Nothing is gated or weighted by it, because
the margin is the half that failed validation, at 0.22.

### The draw, and why two headline numbers

715 pages over 577 documents and 294 domains, assigned first-match into disjoint strata, oversampling
the divergent tail because a uniform draw would spend three-quarters of the budget on pages whose
readings already agree.

| stratum | corpus pages | corpus share | drawn | draw share |
|---|---|---|---|---|
| unchanged | 1,218 | 67.86% | 300 | 41.96% |
| moderate_loss | 269 | 14.99% | 150 | 20.98% |
| large_loss | 104 | 5.79% | 104 | 14.55% |
| small_glyphs | 86 | 4.79% | 50 | 6.99% |
| cjk | 57 | 3.18% | 50 | 6.99% |
| catastrophic | 38 | 2.12% | 38 | 5.31% |
| below_legibility_floor | 22 | 1.23% | 22 | 3.08% |
| reverse_catastrophic | 1 | 0.06% | 1 | 0.14% |

The stratified mean is what a corpus made entirely of hard pages would experience; the
corpus-page-weighted estimate is what this corpus would. Both are reported and they are not
interchangeable — in the route adjudication the stratified headline read 0.414 and post-stratifying
to corpus page share put it at ~0.51.

### The judge breaks ties by position, and that has to be taken out

A forced choice means a judge that finds two readings interchangeable still has to name one. This one
names the first. On the 241 judged pages whose two readings are **byte-identical** it called them
equivalent on 240 and picked label A on 240 of 241; across all 715 pages it names the first
extraction 72.2% of the time.

That is not a preference for either renderer — which engine hides behind A is drawn per page — so the
plain rate stays unbiased. It is noise, and on a tie-heavy stratum it is most of the signal:
`unchanged` came out at 0.443 because PDFium happened to draw label A on 44.0% of its pages, not
because anything about the readings differed. Conditioning on the draw removes it exactly. With
`a = P(pick first | first is PDFium's reading)` and `b = P(pick first | first is MuPDF's)`, the
estimate `(a + 1 − b) / 2` cancels any position preference the two halves share; a judge that only
looked at position gives `a = b` and lands on exactly 0.5. Every number below is that estimator, and
it moves `unchanged` from 0.443 to **0.487**, which is what a stratum of near-identical readings
should report.

### Verdicts

P(the PDFium-rendered page's reading is judged the more faithful one). The null is 0.500.

| stratum | corpus share | pages | domains | MuPDF reference | PDFium reference | judged equivalent |
|---|---|---|---|---|---|---|
| catastrophic | 2.12% | 38 | 21 | 0.316 ± 0.073 | 0.474 ± 0.079 | 7.9% |
| large_loss | 5.79% | 104 | 65 | 0.487 ± 0.048 | 0.526 ± 0.048 | 1.0% |
| moderate_loss | 14.99% | 150 | 91 | 0.483 ± 0.041 | 0.505 ± 0.041 | 6.7% |
| below_legibility_floor | 1.23% | 22 | 16 | 0.364 ± 0.084 | 0.364 ± 0.095 | 40.9% |
| cjk | 3.18% | 50 | 17 | 0.466 ± 0.051 | 0.447 ± 0.053 | 38.0% |
| small_glyphs | 4.79% | 50 | 35 | 0.500 ± 0.060 | 0.540 ± 0.056 | 36.0% |
| unchanged | 67.86% | 300 | 171 | 0.487 ± 0.020 | 0.485 ± 0.021 | 76.7% |
| reverse_catastrophic | 0.06% | 1 | 1 | — | — | 100% |

| | MuPDF reference | PDFium reference |
|---|---|---|
| stratified (the oversampled draw) | 0.4583 ± 0.0167 | 0.4778 ± 0.0169 |
| **corpus-page-weighted** | **0.4808 ± 0.0157** | **0.4898 ± 0.0162** |
| 95% interval | [0.450, 0.512] | [0.458, 0.522] |
| judged equivalent | 40.7% | 40.6% |
| names the first extraction | 72.2% | 71.3% |

**Both corpus-weighted intervals contain 0.500.** The stratified numbers sit below it because the
draw is four-fifths hard pages by construction, which is exactly the number not to quote.

### Agreement, and what the reference actually costs

Three agreement numbers, all on the same 715 pages, and the first is the one that makes the other two
readable.

| comparison | same engine named |
|---|---|
| same reference, judged twice (the judge's own noise floor) | **0.9259** |
| MuPDF reference against PDFium reference | 0.9189 |
| two judges, MuPDF reference | 0.8923 |
| two judges, PDFium reference | 0.9020 |

**Swapping the reference renderer moves the verdict barely more than asking the same judge the same
question twice** — 0.9189 against a 0.9259 floor, a difference of 0.007. Two different models
disagree with each other three to five times as much as one model disagrees with itself across
references. Corpus-wide, the reference is very nearly neutral at the level of the verdict even though
its pixels are not.

`gemini-3.7-flash` reaches the same place from a visibly different disposition: it calls the two
readings equivalent on 53.2% of pages against `gpt-5.6-luna`'s 40.7%, and still lands at 0.491 and
0.498. The equivalence flag is worth more here than in the route adjudication, and there is an
objective check on it: of the 241 judged pages whose readings are byte-identical, the primary judge
called 240 equivalent.

### The reference is not neutral where it matters most

The corpus-wide neutrality above hides a local failure, and it is the most useful finding in this
section. The one stratum that looks like a real loss under a MuPDF reference is `catastrophic` —
0.316 ± 0.073, the 38 pages the rejection was written around. Under a PDFium reference the same 38
pages, same readings, same blinding, score **0.474 ± 0.079**.

That is not noise. Paired page by page, the reference swap flips verdicts in one direction on
`catastrophic` and in no direction anywhere else, and the same judge re-asked with the *same*
reference flips symmetrically everywhere:

| stratum | reference swap → PDFium | → MuPDF | McNemar p | same reference, repeat |
|---|---|---|---|---|
| **catastrophic** | **6** | **0** | **0.031** | 1 / 1 |
| large_loss | 12 | 8 | 0.503 | 7 / 5 |
| moderate_loss | 7 | 4 | 0.549 | 7 / 6 |
| unchanged | 8 | 8 | 1.000 | 11 / 9 |

The manipulation moves one stratum one-directionally; the judge's own irreproducibility moves none of
them.

So the reference effect is not a general distortion — it is concentrated exactly where the two
renderings disagree most about what is on the page, which is where a reference has the most
opportunity to agree with one reading and not the other. **This is what the two-way design was built
to catch**: a single-reference adjudication of a renderer pair will favour the incumbent's reading,
by construction, on precisely the pages that decide the question. Had this study been run with a
MuPDF reference alone it would have reported the catastrophic set as a real PDFium loss, and that
conclusion would have been an artifact.

### Answer: different, not worse

Every arm, both judges, both references, corpus-page-weighted:

| arm | corpus-page-weighted | 95% interval |
|---|---|---|
| gpt-5.6-luna, MuPDF reference | 0.4808 ± 0.0157 | [0.450, 0.512] |
| gpt-5.6-luna, PDFium reference | 0.4898 ± 0.0162 | [0.458, 0.522] |
| gpt-5.6-luna, MuPDF reference, repeat | 0.4902 ± 0.0154 | [0.460, 0.520] |
| gemini-3.7-flash, MuPDF reference | 0.4912 ± 0.0169 | [0.458, 0.524] |
| gemini-3.7-flash, PDFium reference | 0.4979 ± 0.0167 | [0.465, 0.531] |

**All five contain 0.500.** The spread across them, 0.481 to 0.498, is smaller than any one
interval's width. Restricting to the 474 pages whose readings are not byte-identical — an objective
cut, not a model's opinion — gives 0.4585 ± 0.0263, which also contains 0.500. Dropping the render
study's truncation and runaway-length confound gives 0.4848 ± 0.0160.

The one cut that excludes parity is the 424 pages the judge declined to call equivalent, at 0.4362 ±
0.0275. That cut conditions on the model's margin, which is the part of this instrument that failed
human validation at 0.22 agreement, and it is reported here for completeness rather than as evidence.

The render study's −0.0364 bigram-F1 delta is real and reproduces at −0.0822 on this draw. What
Question 1b establishes is that it is **displacement, not degradation**: the model reads a
PDFium-rendered page differently, and a judge looking at the page cannot tell which reading is
better. On a third of the corpus it cannot tell because the two readings are the same string.

## Question 2: what the pixels do

Both architectures, at the feed's token-budget dimensions and at the 160 DPI the judging packets use.
Results are near-identical across architectures, so the rasterisers are deterministic per platform.

| | x86_64 feed | aarch64 feed | x86_64 judge | aarch64 judge |
|---|---|---|---|---|
| pages | 1,790 | 1,795 | 1,790 | 1,795 |
| changed-pixel fraction p50 | 0.0416 | 0.0416 | 0.0382 | 0.0382 |
| p90 / p99 | 0.0705 / 0.1051 | 0.0707 / 0.1053 | 0.0662 / 0.0952 | 0.0663 / 0.0966 |
| mean absolute difference | 2.44 / 255 | 2.45 / 255 | 2.33 / 255 | 2.33 / 255 |
| pages over 2% changed | 81.7% | 81.8% | 77.8% | 77.8% |
| ink MuPDF vs PDFium | 0.0872 / 0.0894 | 0.0880 / 0.0904 | 0.0875 / 0.0898 | 0.0883 / 0.0908 |

This is a much *smaller* pixel difference than pdf_oxide's (p50 5.4%, mean |Δ| 8.72/255) — PDFium
and MuPDF are close rasterisers. It is still enough to move the model on 8% of pages, which is the
substantive finding: **at these resolutions the pixel metric does not predict the VLM outcome, so it
cannot be used as a proxy for it.** A future renderer question has to be settled by running the
model, not by diffing images.

## Question 3: cost, with the PNG encoder held constant

Both renderers rendering the same pages at the same 2.070 MP, both encoding through Pillow at
`compress_level=1`, so the only thing that varies is the rasteriser.

| | x86_64 ms/page | x86 core-h/M | aarch64 ms/page | aarch64 core-h/M |
|---|---|---|---|---|
| **MuPDF feed total** | **68.63** | **19.06** | **57.55** | **15.98** |
| — rasterise | 25.88 (37.7%) | 7.19 | 23.13 (40.2%) | 6.42 |
| — PNG encode (Pillow c1) | 42.23 (61.5%) | 11.73 | 34.15 (59.3%) | 9.49 |
| — base64 | 0.52 | 0.14 | 0.27 | 0.07 |
| **PDFium feed total** | **64.12** | **17.81** | **50.73** | **14.09** |
| — rasterise | 21.45 (33.5%) | 5.96 | 16.23 (32.0%) | 4.51 |
| — PNG encode (Pillow c1) | 42.12 (65.7%) | 11.70 | 34.22 (67.5%) | 9.51 |
| — base64 | 0.54 | 0.15 | 0.28 | 0.08 |

PDFium rasterises **1.21× faster on x86 and 1.43× faster on aarch64**, and its tail is tighter
(per-page p99 268 ms against MuPDF's 535 on x86; 214 against 437 on aarch64). Whole-feed that is
1.25 core-h/M on x86 and 1.89 on aarch64 — **7% and 12%**.

The harness cross-validates against the published figures. Timing MuPDF's own PNG encoder alongside
(46.17 ms/page on x86) reconstructs the incumbent path at 25.88 + 46.17 + 0.52 = 72.57 ms/page =
20.16 core-h/M, against the 72.79 ms/page and 20.22 core-h/M measured independently in
[`pdf-oxide-evaluation.md`](pdf-oxide-evaluation.md). Agreement to 0.3% across separate runs on
separate days.

## The PNG finding survives, and it is the whole prize here

Encoding is the expensive half of the feed and PNG is lossless, so this is the one lever that moves
cost with **provably identical pixels**. Measured over 3,014 pages, all three encoders fed the same
PDFium buffers:

| encoder | x86 ms/page | x86 core-h/M | aarch64 ms/page | aarch64 core-h/M | KiB/page | pixel-identical |
|---|---|---|---|---|---|---|
| MuPDF `tobytes("png")` | 47.75 | 13.26 | 48.51 | 13.48 | 437 | — |
| **Pillow `compress_level=1`** | **41.07** | **11.41** | **33.31** | **9.25** | 458 | **3014 / 3014** |
| Pillow `compress_level=6` | 72.21 | 20.06 | 70.05 | 19.46 | 481 | 3014 / 3014 |

Confirmed on PDFium's buffers as well as MuPDF's: 1.16× faster on x86 and 1.46× on aarch64, decoding
byte-identical on 3,014 of 3,014 pages, for a 4.8% larger payload. Level 6 remains slower than MuPDF
on both architectures for a payload no smaller than level 1, so the compression knob still has no
useful middle.

**This is renderer-independent and should be taken regardless.** Against the current incumbent
(MuPDF rasteriser + MuPDF encoder, 20.16 core-h/M on x86) the encoder swap alone lands the feed at
19.06 on x86 and 15.98 on aarch64. That is a larger saving than changing the rasteriser would buy,
and it costs nothing in quality because the pixels are provably the same.

## Failure taxonomy

Crawl PDFs are adversarial and a native extension fails in ways a Python library cannot — a panic
compiled under `panic = "abort"`, an unbounded allocation, a content stream that never returns.
None of those are catchable in the calling process, so every document went through a subprocess the
driver was willing to lose, over evenly spaced pages rather than front-loaded ones.

| operation | arch | ok | exception | timeout | panic | worker death |
|---|---|---|---|---|---|---|
| `render_mupdf` | x86_64 | 1000 | 0 | 0 | 0 | 0 |
| `render_mupdf` | aarch64 | 1000 | 0 | 0 | 0 | 0 |
| `render_pdfium` | x86_64 | 1000 | 0 | 0 | 0 | 0 |
| `render_pdfium` | aarch64 | 1000 | 0 | 0 | 0 | 0 |
| `pixels` | x86_64 | 996 | 0 | **4** | 0 | 0 |
| `pixels` | aarch64 | 1000 | 0 | 0 | 0 | 0 |
| `encoders` | both | 1000 | 0 | 0 | 0 | 0 |
| study render (both libraries) | aarch64 | 1000 | 0 | 0 | 0 | 0 |

**PDFium's record on this corpus is as clean as MuPDF's**: no exceptions, no panics, no fatal
signals, on either architecture. The four `pixels` timeouts are the 30-second driver deadline on an
operation that performs eight full-page renders plus numpy comparisons on PDFs of 1.6–12.3 MB; the
operation calls both libraries and the timeouts are not attributable to either.

One behavioural difference worth recording even though it did not show up as a failure here: PDFium
refuses an encrypted document at open where MuPDF opens it and hands back blank pages. That would
turn a silently-empty document into a counted `render_failed`, which is an improvement, but it is a
change in corpus composition.

### Reconciling the exit 133

An earlier revision of the render study rendered in-process and died at **exit 133 (SIGTRAP)** about
a hundred documents in. I reported that as PDFium's `IMMEDIATE_CRASH`. **That inference was
unsupported and it did not survive contact with the data.**

The rewritten study ran the identical 1,000 documents in the identical order at the identical evenly
spaced page indices, with rendering isolated in a subprocess, and reported `render ok 1000/1000`
with one worker spawn — zero worker deaths. Summing the three runs, PDFium performed **24,621 page
renders with zero aborts** — 9,628 feed, 7,170 pixel-comparison, 6,028 encoder and 1,795 study
renders — and MuPDF performed 18,593 with the same clean record.

- **"It depends on the page-selection change" is excluded.** Even spacing was introduced for the
  probes; both study revisions already used it, on the same documents in the same order.
- **"It was the harness, not PDFium" is supported but not proven.** The failing revision rendered in
  the driver process alongside 64 HTTP request threads, the OpenAI client, a live fleet session, a
  1,000-document Polars frame and 128 encoded pages in flight, with no `faulthandler` and no
  per-library attribution. SIGTRAP points at a native trap rather than an OOM kill, which arrives as
  SIGKILL — but the failing process could not say which of PDFium, MuPDF, Pillow or NumPy trapped.
- **"It is real but rarer than one in ~1,800 pages" cannot be excluded.** Zero events in 24,621
  page renders bounds the per-page abort rate at roughly 1.2 × 10⁻⁴ (rule of three, 95%), which over
  a million-page corpus is not a comfortable bound — but it is a bound on an event never attributed
  to PDFium in the first place.

The honest statement is **zero aborts in 24,621 PDFium page renders across two architectures, plus
one unexplained abort in an earlier, unisolated configuration that was never attributed to a
library.** The recommendation is now to adopt, so the subprocess-isolation question this raised is
live and is the first of the two conditions attached to it: the feed renders inside the Zephyr map
task, and a hard abort would take the task rather than the page.

## The second-order effect: labels judged against MuPDF renderings

Every quality number in this evaluation was judged against MuPDF-rendered page images, so a renderer
change moves the judges' ground truth too. Assessed rather than assumed away, it splits in two:

**The 19,977 preference labels are moot.** Their consumer is the router, and Router v2 retired it —
`route_features` measured worse on all five domain-disjoint splits. Labels whose consumer is being
deleted do not need regenerating whatever the renderer does.

**The adjudication verdicts are not moot, and this is the expensive part.** They compare three
routes' *extractions* against a 160-DPI page image. Two things move:

1. *The VLM extraction itself changes.* This is Question 1, and at −0.0233 to −0.0364 bigram F1 with
   2.1% of pages read differently it is larger than effects this pipeline already treats as
   decisive: pdf-inspector 1.14.1→1.17.0 moved corpus-wide bigram recall by +0.0113 and that was
   judged *inside* the noise floor; Router v2's +0.0127 against a 0.0096 paired floor was judged a
   real finding. The VLM arm of every adjudication packet has to be re-extracted and re-judged.
2. *The judge's reference image changes.* At 160 DPI the two renderers differ by a changed-pixel p50
   of 0.0376 and a mean absolute difference of 2.40/255. Question 1b measured what that is worth at
   the level of a verdict — 0.9189 agreement against a 0.9259 same-reference floor — so it is
   genuinely second-order next to (1) corpus-wide, though not on the pages where the two renderings
   disagree most.

Question 1b changes the *reason* this work is needed, not the amount. It is no longer "the readings
may be worse and the record has to be rebuilt"; it is "the readings are different and the record
should describe the renderer that produced the corpus". The re-extraction and re-adjudication is the
price of adoption, and the price does not include a quality regression.

## Licence and wheels

Verified from the published wheel, not the PyPI blurb, because that is the whole basis of the case.

`pypdfium2` 5.13.0 declares `BSD-3-Clause, Apache-2.0, dependency licenses` and ships
`LICENSES/{Apache-2.0,BSD-3-Clause,CC-BY-4.0}.txt` plus a per-component `BUILD_LICENSES` directory
for the single shared object it vendors (`pypdfium2_raw/libpdfium.so`, 7.7 MB). Every component is
permissive: abseil (Apache-2.0), agg23, fast_float (MIT), freetype (FTL, with no GPL text in the
shipped file), icu (Unicode-3.0), lcms (MIT), libjpeg-turbo (BSD-style), libopenjpeg (BSD-2),
libpng, libtiff, llvm-libc (Apache-2.0 with LLVM exception), pdfium-binaries (MIT), pdfium
(BSD-3-Clause), simdutf, zlib.

Two files mention the GPL and neither creates an obligation: ICU's licence file reproduces the terms
of the `aclocal.m4` and `config.guess` **build scripts** under the standard Autoconf exception, and
those are not compiled into anything; llvm-libc's mention is its *exception* clause, permitting
combination with GPLv2 software rather than requiring anything. There is no copyleft in the shipped
artifact. PyMuPDF is AGPL-3.0.

Wheels are `py3-none-<platform>` — a pure Python tag with the binary vendored — including
`manylinux_2_17_x86_64.manylinux2014_x86_64` and `manylinux_2_17_aarch64.manylinux2014_aarch64`.
That is the same glibc baseline as pdf-inspector and older than pdf-oxide's `manylinux_2_28`, so it
is the least demanding of the three. Verified rather than assumed: both probe runs installed and
imported the wheel in a stock Marin worker image, logging `pypdfium2 5.13.0 (pdfium 153.0.7999.0)`
on x86_64 and on Grace aarch64.

**It is a promotion, not an addition.** `uv.lock` already carried `pypdfium2` 5.12.1 as a transitive
dependency of `docling-slim` — docling ships a pypdfium2 backend. Since docling is being dropped,
adopting PDFium turns that transitive into a direct dependency rather than introducing an unknown
wheel: known-good in this workspace, on a friendlier glibc baseline than pdf-oxide's.

## pdf-inspector's bundled PDFium is not reachable

pdf-inspector 1.17.0's wheel compiles `render-pdfium` in through the `ocr` cargo feature, so a
PDFium binary is already in the worker environment. Whether a second native dependency could be
avoided by reaching it was cheap to check, and the answer is no. The module exports 33 names —
`classify_pdf`, `detect_pdf`, `extract_pages_markdown`, `extract_structure_elements`, `extract_text`,
`extract_text_in_regions`, `extract_text_with_positions`, `process_pdf`, `process_pdf_with_ocr`,
their `_bytes` variants, and 15 result types. Filtering that list for anything containing `render` or
`image` returns the **empty set**. The crate links PDFium for its own OCR pipeline and exports no
rendering API, so the binary is reachable only by running `process_pdf_with_ocr_bytes`, which is a
different operation with a model download attached.

## PDFium can hit production's dimensions exactly

Recorded because it is what made a fair comparison possible at all, and because it is where
pdf_oxide failed outright (0.33% dimension match).

The feed scales a page with a **non-uniform** matrix straight onto `smart_resize` dimensions: each
side a multiple of 32, filling the 2048-visual-token budget. `pypdfium2`'s `PdfPage.render` takes a
scalar `scale` and cannot express that. But it is a convenience wrapper over
`FPDF_RenderPageBitmap`, which takes `size_x` and `size_y` independently and derives the display
matrix from them, so passing the aligned pair rasterises straight to the target with no
decode/resize/re-encode round trip. Letter → 1280×1632, A4 → 1216×1696, exactly as production asks.

Page *sizing* agrees too, which is the other way a renderer swap could have changed output without
changing a single drawn pixel: over 3,585 pages across both architectures, the number where PDFium's
own `get_size()` would have produced different `smart_resize` dimensions than MuPDF's page rectangle
is **0 (0.00%)**, and buffer-shape mismatches are 0. Over synthetic pages at all four `/Rotate`
values the two agree on dimensions and place content identically, a marker rectangle's centroid
matching to the pixel, with a changed-pixel fraction of ~0.005 that is anti-aliasing alone.

## A trap worth recording for whoever touches a PDFium binding next

`PdfBitmap.new_native(..., rev_byteorder=True)` **does not reverse the byte order.** It records the
claim on the Python wrapper so that `to_numpy()` and `to_pil()` label the buffer `RGB`. What
actually makes an `FPDFBitmap_BGR` buffer hold RGB is passing `FPDF_REVERSE_BYTE_ORDER` in the
render flags. Omit the flag and every page comes back with red and blue swapped.

The reason this deserves a paragraph is that **a page of black text on white is symmetric under a
red/blue swap**, so the bug is invisible on every page anyone would think to check first. It
survived a fixture sweep and surfaced only on a single coloured fixture, as a changed-pixel fraction
of 0.80 and a mean absolute difference of 125/255 — and had it not, it would have travelled silently
into every colour page of the corpus and shown up as an unexplained quality regression much later.
The guard is a fixture that is neither black nor white nor grey, asserting the blue channel
dominates; it was mutation-checked by removing the flag and confirming the test fails.

## Test fixtures — a question for the maintainer, not a decision made here

Recorded because it outlives this evaluation: removing PyMuPDF from the runtime path would not have
removed it from the repository. Six test modules use it today, but four are deleted along with the
code they test.

| module | uses PyMuPDF for | fate |
|---|---|---|
| `test_docling_extract.py` | authoring fixture PDFs | deleted with `docling_extract/` |
| `test_ocr_features.py` | authoring fixture PDFs, including encrypted ones | deleted with `ocr_features.py` |
| `test_route_features.py` | authoring fixture PDFs | deleted with `quality/route_features.py` |
| `test_pdf_classify.py` | authoring fixture PDFs | deleted with `classify.py` |
| `test_pdf_dedup.py` | **nothing** — `"pymupdf"` appears only as a string in an import blocklist asserting the pipeline DAG does not import worker-only dependencies at driver scope | survives, no dependency |
| `test_ocr_extract.py` | authoring fixture PDFs | **survives, and needs it** |

So it is **one** test module, not six. `tests/datakit/test_ocr_extract.py` builds fixtures with
`pymupdf.open()` / `new_page()` / `insert_text()`, because a render test needs a real PDF with known
content and `pypdfium2` does not author PDFs. Three ways out: keep PyMuPDF as a test-only dependency;
check in fixture PDFs, removing the dependency at the cost of opaque binaries nobody can regenerate;
or author with reportlab (BSD), pikepdf (MPL-2.0) or fpdf2 (LGPL), each with its own licence
question.

**This is not a call to make on technical grounds.** A test-only AGPL dependency is materially
different from a runtime one — never distributed with the corpus, never linked into anything
released — but it is a question about what "the toolchain is releasable" promises, and that belongs
to whoever makes the release.

## Recommendation

**Adopt PDFium for the render feed.** It is cheaper, permissively licensed, operationally clean and
dimensionally exact, and the quality objection does not survive adjudication: blind, two-way and
corpus-weighted, its readings are preferred 0.481–0.498 of the time against a null of 0.500, with all
five arms containing parity. It changes what the model reads on 8% of pages and does not read them
worse. The AGPL dependency goes away for the price of a re-extraction.

**Two conditions attach, and neither is a quality condition.**

*Subprocess-isolate the rasteriser, or establish that it does not need it.* PDFium performed 24,621
page renders here with zero aborts, but the feed rasterises inside the Zephyr map task, where a hard
abort takes the task rather than the page — and PDFium is a native extension whose failure modes are
not catchable in the calling process. The [exit 133 above](#reconciling-the-exit-133) was never
attributed to any library and the rule-of-three bound over a million-page corpus is not comfortable.
This is the question the first revision deferred because it recommended against adopting; adopting
makes it due.

*Re-extract and re-adjudicate.* The 345 adjudication packets compare three routes' extractions
against a rendered page, and the VLM arm of every one of them was produced from a MuPDF rendering.
Those verdicts need regenerating — not because the readings are worse, but because they are
different and the record should describe the renderer that produced the corpus. The 19,977 preference
labels remain moot: Router v2 retired their consumer. See
[the second-order section](#the-second-order-effect-labels-judged-against-mupdf-renderings).

**Do take the PNG encoder.** Pillow at `compress_level=1` is pixel-identical on 3,014 of 3,014 pages
on both architectures and on both renderers' buffers, and moves the feed from 20.16 to 19.06
core-h/M on x86 and to 15.98 on aarch64 — a larger saving than the rasteriser was ever going to buy,
at zero quality risk. It is renderer-independent and survives this result untouched.

**PyMuPDF leaves the runtime path and stays in the test path.** Its only runtime role after the
router and Docling deletions is `ocr_extract/render.py`, reached through one function, so the swap is
small. `tests/datakit/test_ocr_extract.py` still authors fixture PDFs with it and `pypdfium2` cannot
author PDFs; whether a test-only AGPL dependency is acceptable is
[a release question, not a technical one](#test-fixtures--a-question-for-the-maintainer-not-a-decision-made-here).

**What not to do is settle a renderer question on pixels, and now also not on one reference.** The
changed-pixel fraction correlates with the agreement delta at −0.032, and PDFium's pixels are five
times closer to MuPDF's than pdf_oxide's were for no benefit. The second trap is newer: at the
judging resolution the two engines still differ on 3.8% of pixels, and adjudicating against a single
engine's rendering flipped the `catastrophic` stratum 6–0 in that engine's favour. **Any future
renderer comparison has to be run through the model, and judged against both renderings.**

**Worth revisiting if the model changes.** The finding is a property of *this* model's insensitivity
to this rasteriser pair, not a property of PDFium. The measurements to repeat are
[`build_render_study`](quality/build_render_study.py) and
[`build_render_adjudication_set`](quality/build_render_adjudication_set.py); together they cost one
GPU for about half an hour and roughly $6 of judging.
