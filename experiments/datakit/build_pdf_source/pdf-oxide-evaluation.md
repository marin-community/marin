# pdf_oxide for the router pass and the VLM render feed, at 0.3.77

`pdf-oxide` is a Rust PDF toolkit — `tiny-skia` rasteriser, no MuPDF, no PDFium, MIT OR Apache-2.0 —
evaluated for two independent roles. Marin's cluster is CPU-constrained rather than GPU-constrained,
which inverts the usual economics: with Docling dropped, the CPU that matters is the router's PyMuPDF
feature pass and the VLM feed's render path, and the feed alone is ~500 core-h crawl-wide at a 50%
VLM budget.

The two questions are separate and so are the answers.

**Router pass: no.** pdf_oxide cannot supply six of the router's signals, and they are the six the
router exists for. It also fails outright on 3.5% of crawl documents that PyMuPDF reads without
complaint. The PyMuPDF pass stays.

**Render feed: no.** pdf_oxide's rasteriser is **5× slower** than MuPDF's, making the whole feed
1.7–1.8× more expensive, and it draws different pixels on 86% of pages. Its PNG encoder is genuinely
10× faster than MuPDF's, but that is not separable — the encoder is only reachable through the
renderer.

**What the measurement did find:** PNG encoding is ~65% of the feed's cost, and swapping MuPDF's
encoder for Pillow at `compress_level=1` is pixel-identical and worth **1.7 core-h/M on x86, 4.4 on
aarch64**, at a 2.6% larger payload. Smaller than it first appeared, and it has nothing to do with
pdf_oxide.

Everything below is measured on the cluster over 1,000 documents from the 100,000-document oracle
sample (`marin/data/pdf_quality/cc_focus_2026_22_sample100k`), the same documents in the same order
on both architectures, by [`probe_pdf_oxide`](quality/probe_pdf_oxide.py) and
[`probe_png_encoders`](quality/probe_png_encoders.py). x86_64 ran on `cw-us-east-02a` (Emerald
Rapids), aarch64 on `cw-us-east-08a` (Grace). Mean document length in the sample is 17.74 pages.

## Licence and wheels

Confirmed from the published artifacts, not the PyPI blurb. The wheel ships `LICENSE-MIT` and
`LICENSE-APACHE` and a `NOTICE` reading "licensed under the terms of the MIT license OR the Apache
License, Version 2.0, at your option". The `PDFOxide`/`pdf_oxide` trademark is reserved separately
and does not restrict code use.

That matters beyond speed: PyMuPDF is AGPL and this corpus is intended for release. The wheel also
carries a CycloneDX SBOM, and **all 361 components in it are permissive** — 259 MIT/Apache dual, the
rest MIT, Apache-2.0, BSD, ISC, Zlib, Unlicense or CC0. No GPL, no LGPL, no MPL, and no component
with an unspecified licence.

Wheels are `cp38-abi3`, one per platform across every supported Python. `manylinux_2_28` is a newer
glibc baseline than pdf-inspector's `manylinux_2_17`, so it was verified rather than assumed: both
probe runs installed and imported the wheel in a stock Marin worker image, on x86_64
(`manylinux_2_28_x86_64`) and on Grace aarch64 (`manylinux_2_28_aarch64`). The 2_28 baseline is not a
problem for this fleet. `musllinux_1_2` wheels exist for both arches but were not exercised.

## Question 1: the router pass

[`route_features`](quality/route_features.py) computes ~70 signals in six groups. Its own docstring is
explicit that the discriminative power is in the signals that are *not* text facts — a born-digital
paper with a broken ToUnicode CMap, an invisible OCR layer over a scan, two columns and a page of
equations has healthy text-layer statistics and extracts badly.

### Coverage

Per page signal, against pdf_oxide 0.3.77's actual read API — the type stub, the PyO3 binding source,
and measurement on real crawl documents where neither settled it.

| group | signal | pdf_oxide | evidence |
|---|---|---|---|
| encoding | `char_count`, 9 character-ratio signals | available | extracted text |
| | `fonts_total` | derivable, different semantics | distinct `TextSpan.font_name` counts fonts *used*, not the page's font resource list: 138 against PyMuPDF's 175 over the same 34 pages |
| | `fonts_not_embedded` | **absent** | no font-resource API; `/FontFile` presence never exposed |
| | `fonts_without_tounicode` | **absent** | `TextSpan.provenance` is the only route and is `None` on **2,815 of 2,815 spans** |
| | `fonts_unmappable` | **absent** | same |
| | `fonts_type3` | **absent** | Type3 parsed internally (`page_renderer.rs`), never surfaced on the read API |
| | `fonts_glyphless` | available | `font_name` substring match |
| layer | `invisible_char_ratio` | available, page-level only | `classify_page().signals.invisible_text_ratio` |
| | `invisible_over_image_ratio` | **absent** | no per-span or per-char visibility flag, so invisible spans cannot be intersected with image placements |
| | `text_over_image_ratio` | available | span bboxes against `page_images` placements |
| | `overlapping_line_ratio`, `out_of_page_line_ratio`, `duplicate_span_ratio` | available | span and line bboxes against the media box |
| | `rotated_char_ratio` | available | `TextChar.rotation_degrees` |
| math | `math_font_ratio` | derivable, different denominator | span-weighted, not font-resource-weighted |
| | `math_unicode_ratio` | available | text |
| structure | `ruling_line_count`, `rule_grid_cells`, `ruled_area_ratio` | available | `extract_paths` tracks `get_cdrawings` closely (381↔381, 430↔430, 1761↔1778, 68↔72). Note `extract_lines`/`extract_rects` are pdf_oxide's *interpretation* and do not: they returned 1 and 0 on a page with 29 drawings |
| | `left_edge_concentration` | available | span bboxes |
| order | `text_block_count` | available | `extract_structured` regions |
| | `column_count` | derivable with work | `extract_structured`'s own `column_index` is `None` on **421 of 425 regions**; usable only by porting the router's geometric column sweep onto pdf_oxide bboxes |
| | `stream_order_inversion_ratio` | available | `TextSpan.sequence` is content-stream emission order |
| | `interleaved_column_ratio` | derivable with work | needs the ported columns above |
| script | `cjk_ratio`, `rtl_ratio`, `latin_ratio` | available | text |

Rolled up:

| group | available | derivable with work | absent |
|---|---|---|---|
| encoding | 10 / 16 | 1 | **5** |
| layer | 6 / 7 | 0 | **1** |
| math | 1 / 2 | 1 | 0 |
| structure | 4 / 4 | 0 | 0 |
| order | 2 / 4 | 2 | 0 |
| script | 3 / 3 | 0 | 0 |

### Can the PyMuPDF router pass be eliminated? No.

Six of the 36 page signals are absent, and they are not a random six.

**The encoding group's whole point is gone.** The router does not check for U+FFFD and stop — MuPDF
only emits U+FFFD when it *knows* it failed. The signal it exists for is a subsetted, embedded font
with a custom encoding and no ToUnicode, which yields confident nonsense no character-ratio check can
see. pdf_oxide's `TextSpan.provenance` is exactly the right idea — it reports which ISO 32000-1
§9.10.2 mapping tier the font offered, including a `fallback` tier meaning "no mapping resource, the
text is a fabricated glyph-index echo" — but **it returned `None` on every span measured**, through
both `extract_spans` and `extract_page_text`, with `set_preserve_unmapped_glyphs` on and off. The
feature is declared and inert. Type3 and not-embedded have no API at all.

**`invisible_over_image_ratio` is gone.** `classify_page` reports a page-level Tr-mode-3 ratio and a
page-level image-area ratio, but nothing joins them, because no span or char carries a visibility
flag. "This page has invisible text" plus "this page has a bitmap" is not the same fact as "invisible
text is drawn on top of the bitmap" — the latter is a scan carrying somebody else's OCR of unknown
quality, precisely the case whose text-layer statistics otherwise look healthy.

A partial replacement saves nothing, because the cost is dominated by opening and parsing the
document: a router that still calls PyMuPDF for one group pays the PyMuPDF parse in full and adds a
second parse on top. That is a cost increase.

### Cost, for completeness

Per-document milliseconds, and CPU core-hours per million *corpus* pages. The router samples 8 pages
per document however long the document is, so the conversion divides by the sample rather than
multiplying by every page — the same arithmetic that makes `route_pymupdf` land at the ~1.9 core-h/M
this pipeline already budgets for `route_features`.

| | x86_64 p50 / p90 / p99 ms | aarch64 p50 / p90 / p99 ms | x86 core-h/M | aarch64 core-h/M | failures |
|---|---|---|---|---|---|
| `route_pymupdf` (all 36 signals) | 50.3 / 184.3 / 2080.8 | 55.2 / 207.4 / 1646.5 | **1.86** | **1.84** | **0 / 1000** |
| `route_oxide` (strict subset) | 40.3 / 172.3 / 1449.7 | 36.8 / 158.5 / 1240.4 | 1.67 | 1.39 | **35 / 1000** |

pdf_oxide's calls are cheaper per page, and it does not matter. Two caveats keep that number from
being a like-for-like win even on its own terms: `route_oxide` times the *library calls only*, while
`route_pymupdf` includes the full Python feature arithmetic — the character loops, the quadratic line
overlap, the column sweep — that any implementation would pay identically; and it calls
`extract_lines`/`extract_rects` where a correct implementation would call the more expensive
`extract_paths`. The honest reading is that the two libraries extract page data at broadly comparable
cost, and pdf_oxide extracts less of it.

## Question 2: the VLM render feed

The feed in [`ocr_extract/render.py`](ocr_extract/render.py) renders each page under a 2048
visual-token budget with a 300-DPI upscale cap, encodes PNG, and base64s it into a data URI. pdf_oxide
does render pages, so this was the more promising question. It fails on four independent counts.

### It is 1.7–1.8× more expensive, because the rasteriser is 5× slower

Both paths rendered the same pages at the same 2.07 MP/page.

| stage | x86_64 ms/page | x86 core-h/M | aarch64 ms/page | aarch64 core-h/M |
|---|---|---|---|---|
| **MuPDF feed total** | **72.79** | **20.22** | **72.98** | **20.27** |
| — rasterise | 25.81 | 7.17 | 23.53 | 6.53 |
| — PNG encode | 46.47 (64%) | 12.91 | 49.19 (67%) | 13.66 |
| — base64 | 0.51 | 0.14 | 0.27 | 0.07 |
| **pdf_oxide feed total** | **134.07** | **37.24** | **121.45** | **33.74** |
| — rasterise | 128.52 (96%) | 35.70 | 117.06 (96%) | 32.52 |
| — PNG encode | 4.61 (3%) | 1.28 | 4.05 (3%) | 1.12 |
| — base64 | 1.11 | 0.31 | 0.58 | 0.16 |

The two libraries sit at opposite ends of the same trade: **pdf_oxide rasterises 5.0× slower and
encodes PNG 10.1× faster**. The fast encoder is not separable — it is reachable only through
`render_page`, which rasterises first — so there is no configuration of pdf_oxide that buys the
encoder without the rasteriser.

The tail is worse than the median suggests: p90 per document is 1376 ms against MuPDF's 644, and p99
is 7601 ms against 1924.

### It cannot produce matched output

Marin scales a page with a **non-uniform** matrix straight onto `smart_resize` dimensions: each side a
multiple of 32, filling the token budget exactly. pdf_oxide offers only scalar DPI (`render_page`,
`render_pixmap`) or a fit-inside-a-box preserving aspect ratio (`render_page_fit`). Neither can
express an arbitrary aligned pair, and the nearest scalar DPI misses on every common paper size:

| page | `smart_resize` W×H | effective DPI | nearest uniform-DPI W×H |
|---|---|---|---|
| Letter | 1280×1632 | 149.5 | 1266×1639 |
| A4 | 1216×1696 | 146.1 | 1207×1707 |
| Legal | 1120×1856 | 132.2 | 1122×1848 |
| A5 | 1184×1696 | 204.2 | 1189×1687 |
| A3 | 1216×1696 | 103.3 | 1204×1703 |

Measured over 4,789 rendered pages, pdf_oxide hit production's dimensions on **16 (0.33%)**. The sides
are not multiples of 32, so the serving path would resize every page — costing CPU on the serving
pods, where the budget sweep already found the bottleneck, and resampling an already-rasterised image
instead of rasterising at the target, softening exactly the glyph detail the budget exists to preserve.

### At matched DPI it draws different pixels

To separate renderer fidelity from sizing policy, `Op.PIXELS` renders both at the *same* integer DPI,
where dimensions agree, and compares pixel for pixel. pdf_oxide's pixmap is premultiplied RGBA and its
alpha is 255 everywhere on these pages, so compositing onto white is a no-op and the comparison is
direct. Results were **bit-identical across the two architectures**, so the rasteriser is at least
deterministic.

Over 1,794 pages from 999 documents:

| | |
|---|---|
| dimensions identical at matched DPI | 1,736 / 1,794 (96.8%) |
| changed-pixel fraction (>16/255 on any channel) | p50 **5.4%**, p90 14.3%, p99 89.2% |
| mean absolute difference | 8.72 / 255 |
| pages differing on more than 2% of pixels | **1,540 / 1,794 (85.8%)** |

This is not anti-aliasing noise and it is not localised: rendered as a difference image, **every glyph
on the page lights up**. The two rasterisers place text in the same positions and draw each glyph
differently — pdf_oxide lays down about 5% more ink (ink fraction 0.0839 against MuPDF's 0.0800), i.e.
systematically heavier stems.

This is the gate. A faster renderer that produces different pixels changes VLM output and invalidates
every extraction-quality number the pipeline has. pdf_oxide is not faster *and* differs on 86% of
pages, so the question of whether the difference is tolerable never arises.

### Its payload is more than twice as large

`render_page` always emits **RGBA** PNGs (colour type 6); MuPDF emits RGB (colour type 2). Over the
same pages the encoded payload is **932 KiB against 430 KiB, 2.17×**. The feed base64s that and ships
it to the VLM, so this more than doubles both the bytes on the wire and the API-side decode the serving
pods pay.

## The encoder is the real lever, and it is smaller than it looks

PNG encoding is ~65% of the feed. PNG is lossless, so unlike every other lever here the encoder can be
swapped with **provably identical pixels** — asserted per page by decoding both outputs and comparing
bytes, not assumed.

Measured over 3,014 pages on each architecture:

| encoder | x86 ms/page | x86 core-h/M | aarch64 ms/page | aarch64 core-h/M | KiB/page | pixel-identical |
|---|---|---|---|---|---|---|
| MuPDF `tobytes("png")` (incumbent) | 47.33 | 13.15 | 48.57 | 13.49 | 427 | — |
| **Pillow `compress_level=1`** | **41.07** | **11.41** | **32.82** | **9.12** | 438 | **3014 / 3014** |
| Pillow `compress_level=6` | 74.01 | 20.56 | 72.00 | 20.00 | 441 | 3014 / 3014 |
| MuPDF `tobytes("jpg", 85)` | 69.11 | 19.20 | 108.95 | 30.26 | 270 | 17 / 3014 |

Pillow at `compress_level=1` is **1.15× faster on x86 and 1.48× on aarch64**, pixel-identical, for a
2.6% larger payload — worth **1.7 core-h/M on x86 and 4.4 on aarch64**, or roughly 50–125 core-h
crawl-wide, with no re-evaluation of extraction quality required.

Two things close off the alternatives. `compress_level=6` is *slower* than MuPDF on both architectures
while producing a payload no smaller than level 1, so the compression-level knob has no useful middle.
And MuPDF's JPEG encoder is slower than its PNG encoder on both architectures — 1.5× on x86, 2.2× on
aarch64 — so the obvious lossy escape is not even fast, quite apart from being lossy.

This result also shows why the measurement had to run on the cluster. On a development Mac the same
comparison put `compress_level=6` at 1.6× *faster* than MuPDF and `compress_level=1` at 2.3× with a 36%
payload increase; both the ranking and the magnitude were wrong, because the zlib builds behind MuPDF
and Pillow differ between platforms.

## Failure taxonomy

Crawl PDFs are adversarial and a native extension fails in ways a Python library cannot, so every
document went through a subprocess the driver was willing to lose. Both architectures produced
identical counts.

| operation | ok | exception | timeout (30 s) | panic | worker death |
|---|---|---|---|---|---|
| `route_pymupdf` | 1000 | 0 | 0 | 0 | 0 |
| `route_oxide` | 965 | **35 (3.5%)** | 0 | 0 | 0 |
| `render_pymupdf` | 1000 | 0 | 0 | 0 | 0 |
| `render_oxide` | 996 | 1 | **3 (0.3%)** | 0 | **3** |
| `pixels` | 999 | 1 | 0 | 0 | 0 |

The good news first: **no panics and no fatal signals.** No `PanicException`, no SIGSEGV, SIGABRT or
SIGKILL on either architecture. The four worker respawns per run were all driver-initiated replacements
after the three render timeouts. A Stage 1 built on this library would not need process isolation for
memory-safety reasons.

The bad news is the refusal rate. `route_oxide` failed on **3.5% of documents that PyMuPDF read
without complaint** — 22 of them `RuntimeError: Invalid object type: expected Stream, found Array`, most
of the rest FlateDecode recovery failures ("stream is labeled as compressed but all decompression
attempts failed"), and one `Catalog missing /Pages entry`. These are recoverable-malformation cases:
MuPDF repairs them and pdf_oxide raises. On a crawl corpus that is 3.5% of documents lost, or a
PyMuPDF fallback path that reparses them — either way it is not a saving.

The three render timeouts are unbounded documents: pdf_oxide exposes no page cap, byte cap or deadline
of its own, so a 30-second driver deadline is the only thing bounding them. The library also floods
stderr with per-object diagnostics on crawl input (over a thousand FlateDecode warnings in a five-minute
window), which on a fleet of hundreds of workers is a log-volume problem in its own right;
`disable_logging()` exists for it.

## Recommendation

In CPU core-hours per million pages, against the pipeline's existing table:

| route | today | with pdf_oxide | recommended |
|---|---|---|---|
| pdf-inspector 1.17.0 (full extract) | 2.1 | — | unchanged |
| router features | 3.4 (1.86 of it `route_features`) | **not possible** — 6 signals absent, 3.5% refusal rate | **keep PyMuPDF** |
| VLM feed (render + PNG + base64) | 20.2 measured (17.8 budgeted) | 37.2 x86 / 33.7 aarch64 — **1.7–1.8× worse** | **keep the MuPDF rasteriser; swap the PNG encoder** → 18.7 x86 / 15.7 aarch64 |
| Docling | 278 | — | being dropped |

**Do not adopt pdf_oxide for either role.** It cannot supply the router's discriminative signals, and
it is substantially more expensive and pixel-divergent as a renderer.

**Do swap the feed's PNG encoder** for Pillow at `compress_level=1`. It is pixel-identical on 3,014 of
3,014 pages, saves 1.7 core-h/M on x86 and 4.4 on aarch64, and costs 2.6% more payload. It is worth
confirming that payload increase against the serving side before taking it, since the budget sweep found
API-side CPU to be what sets throughput.

**Worth revisiting if upstream changes.** The licensing argument is real — MIT/Apache against PyMuPDF's
AGPL, with a fully permissive dependency tree — and pdf_oxide's `provenance` field is a better idea than
anything PyMuPDF exposes for the encoding group. If it starts returning values, and if a font-resource
API and a per-span visibility flag appear, the router question is worth reopening. The renderer question
is not: a 5× slower rasteriser is not a tuning gap.
