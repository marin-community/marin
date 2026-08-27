# PDFium for the VLM render feed, at pypdfium2 5.13.0

`pypdfium2` wraps Chromium's PDFium — BSD-3-Clause renderer, Apache-2.0/BSD bindings — evaluated as
a replacement for MuPDF in the one role PyMuPDF still holds in this pipeline once the router pass
and the Docling route are removed: turning pages into pixels.

**No.** PDFium is cheaper, it is permissively licensed, its failure record on this corpus is as
clean as MuPDF's, and it can hit production's exact page dimensions. It also changes what the model
reads. Against a self-agreement control of **0.9954** bigram F1, the same pages rendered through
PDFium score **0.9591** — a paired delta of **−0.0364 ± 0.0032** over 1,795 pages, about eight times
the noise floor and 11 standard errors from it. **2.1% of pages** the model reads identically twice
from a MuPDF rendering come back essentially unrecognisable from a PDFium one. The licensing win is
real and it is not worth that.

**The motivation was licensing, not speed, and that is worth saying plainly.** The feed costs ~20
CPU core-hours per million pages, of which PNG encoding is ~62% and rasterisation ~38%. PDFium
rasterises 1.21× faster on x86 and 1.43× faster on aarch64, which moves the whole feed by 1.25 and
1.89 core-h/M respectively — 7% and 12%. Nobody would take a corpus-quality risk for that. The case
for the swap was that PyMuPDF is AGPL and this corpus is intended for release, and that case is
sound; it is just outweighed.

[`pdf-oxide-evaluation.md`](pdf-oxide-evaluation.md) rejected a renderer swap on cost — the
candidate was 1.7–1.8× slower, so fidelity never came up. PDFium is not in that position, so this
evaluation had to answer the question that one never reached: **the pixels differ; does the model's
reading differ?** Pixel identity was never available between two rasterisers and was never the test.

Everything below is measured on the cluster over 1,000 documents from the 100,000-document oracle
sample (`marin/data/pdf_quality/cc_focus_2026_22_sample100k`), the same documents in the same order
on both architectures, by [`probe_pdfium`](quality/probe_pdfium.py) and
[`build_render_study`](quality/build_render_study.py). x86_64 ran on `cw-us-east-02a`, aarch64 on
`cw-us-east-08a` (the freer of the two at the time: 3 running tasks against 43, and 13 idle nodes
against 9). Pages are sampled evenly across each document rather than taken from the front, because
a document's first pages are covers more often than they are representative.

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
divergence rather than correctness. Adjudicating which reading is right would need a separate
blind-judging pass.

That distinction does not rescue the swap. Every quality number this pipeline holds was established
against MuPDF renderings, so divergence from MuPDF *is* the cost regardless of which reading a judge
would prefer — it invalidates the record rather than moving it in a known direction.

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
library.** Because the recommendation is
not to adopt, the subprocess-isolation question this raised does not arise. Were PDFium adopted, it
would need answering first, since the feed renders in the map task and a hard abort would take the
task rather than the page.

## The second-order effect: labels judged against MuPDF renderings

Every quality number in this evaluation was judged against MuPDF-rendered page images, so a renderer
change moves the judges' ground truth too. Assessed rather than assumed away, it splits in two:

**The 19,977 preference labels are moot.** Their consumer is the router, and Router v2 retired it —
`route_features` measured worse on all five domain-disjoint splits. Labels whose consumer is being
deleted do not need regenerating whatever the renderer does.

**The adjudication verdicts would not be moot, and this is the expensive part.** They compare three
routes' *extractions* against a 160-DPI page image. Two things would move:

1. *The VLM extraction itself changes.* This is Question 1, and at −0.0233 to −0.0364 bigram F1 with
   2.1% of pages read differently it is larger than effects this pipeline already treats as
   decisive: pdf-inspector 1.14.1→1.17.0 moved corpus-wide bigram recall by +0.0113 and that was
   judged *inside* the noise floor; Router v2's +0.0127 against a 0.0096 paired floor was judged a
   real finding. The VLM arm of every adjudication packet would have to be re-extracted and
   re-judged.
2. *The judge's reference image changes.* At 160 DPI the two renderers differ by a changed-pixel
   p50 of 0.0382 and a mean absolute difference of 2.33/255. Genuinely second-order next to (1).

So the label question is not an argument against adopting PDFium so much as a cost attached to it: a
full re-extraction and re-adjudication, on top of a corpus whose quality record would have to be
rebuilt. Since the recommendation is not to adopt, nothing needs regenerating.

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

**It would have been a promotion, not an addition.** `uv.lock` already carried `pypdfium2` 5.12.1 as
a transitive dependency of `docling-slim` — docling ships a pypdfium2 backend. Since docling is being
dropped, adopting PDFium would have turned that transitive into a direct dependency rather than
introducing an unknown wheel: known-good in this workspace, on a friendlier glibc baseline than
pdf-oxide's. That is a genuinely favourable starting position, and it is why the pin is retained here
for the probes even though the answer is no.

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

**Do not adopt PDFium for the render feed.** It is cheaper, permissively licensed, operationally
clean and dimensionally exact, and it still changes what the model reads on 8% of pages and rewrites
2.1% of them outright. A corpus whose entire quality record was built on MuPDF renderings cannot
absorb that for a 7–12% saving on one stage and a licence change.

**Do take the PNG encoder.** Pillow at `compress_level=1` is pixel-identical on 3,014 of 3,014 pages
on both architectures and on both renderers' buffers, and moves the feed from 20.16 to 19.06
core-h/M on x86 and to 15.98 on aarch64 — a larger saving than the rasteriser was ever going to buy,
at zero quality risk. It is renderer-independent and survives this result untouched.

**PyMuPDF stays in the runtime path**, and the AGPL question stays open. What this evaluation
narrows is where it lives: after the router and Docling deletions, PyMuPDF's only runtime role is
`ocr_extract/render.py`, reached through one function, and its only test role is one module's
fixtures. Whoever reopens this has a small surface to attack.

**Worth revisiting if the model changes.** The finding is a property of *this* model's sensitivity to
this rasteriser pair, not a property of PDFium. A different VLM, a higher token budget (the
`below_legibility_floor` stratum lost 0.2006, four times the corpus figure, which points at
resolution rather than renderer), or a model fine-tuned across renderers would all move it. The
measurement to repeat is [`build_render_study`](quality/build_render_study.py); it costs one GPU for
twelve minutes.

**What not to do is settle it on pixels.** The changed-pixel fraction correlates with the agreement
delta at −0.032. PDFium's pixels are five times closer to MuPDF's than pdf_oxide's were, and that
bought nothing. Any future renderer question has to be run through the model.
