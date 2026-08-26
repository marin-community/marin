# Routing PDFs between pdf-inspector and the VLM

Every PDF in the focus crawl goes down one of two extraction routes: pdf-inspector, which reads the
embedded text layer on CPU, or `Infinity-Parser2-Flash`, which reads a rendered image on GPU. The
router decides which. This report replaces [the first router](pdf-extraction-routing.md), whose
cheap route and whose training label have both been retired.

## What changed under the router

**Docling is gone.** It cost 278 CPU core-hours per million pages against pdf-inspector's 2.1 — 132× —
for corpus-wide quality parity, ~0.51 page-weighted in the blind head-to-head
([pdf-inspector-evaluation.md](pdf-inspector-evaluation.md)). The decision is now binary: keep the
document on pdf-inspector, or escalate it to the VLM.

**The cost axis is CPU.** This cluster is CPU-constrained and GPU-rich, so a frontier drawn against
GPU time optimizes the resource that is spare. Per million crawl pages: pdf-inspector 2.1 core-h, the
router's PyMuPDF feature pass 3.4 core-h, and the VLM's feed path — PyMuPDF render, PNG encode,
base64 — 17.8 core-h on top of 15.6 GPU-h. Every number below is CPU core-hours per million crawl
pages, with GPU hours carried alongside and never optimized. Escalation is charged per *page*,
because the feed path and the model are; page counts run p50 6, p90 38, p99 207, so a document budget
and a page budget are different numbers and only one of them is money.

**The label is a preference against the rendered page.** This is the part that mattered most.

## The old label did not rank quality

Router v1 trained on `docling_ok`: bigram recall ≥ 0.80 against the VLM, with a page-level floor.
Blind adjudication of 605 documents against rendered pages showed the label does not measure what its
name says. On documents `docling_ok` called fine, the other route still won 41–43% of head-to-heads,
and the label separated preference by 0.015 — 0.404 True against 0.419 False, inside the noise.

The failure is structural, not a threshold in the wrong place. Agreement measures distance from one
of the two candidates. It has no opinion about which of them is right, so it cannot rank them, and it
inherits every format-normalization and dialect problem of whichever route it treats as the
reference. The ground truth that *can* rank them is the rendered page, which is what both routes were
reading.

## The label

`quality/build_preference_set.py` packages each document as its rendered pages plus both routes'
transcription of those same pages, blinded and order-randomized per document. A judge sees
"Extraction A" and "Extraction B" against the page image and says which reproduces it better.
`quality/judge_preference_set.py` buys the verdicts and writes one column: `escalate`, true where the
judge preferred the VLM.

Three properties of the draw and one of the verdict are worth stating.

**Domains, not documents, are the sample size.** The crawl holds ~9.8% exact duplicates and many more
near-duplicates per publisher, and the evaluation split is domain-disjoint, so a deep publisher
contributes almost no independent evidence past its first few documents while a domain left out of
the draw costs a whole unit. The draw takes every domain's first document before any domain's second,
capped at 15 per domain.

**VLM-damaged documents are labelled, not dropped.** Every earlier pass filtered to the
VLM-trustworthy subset, because on a damaged row an agreement number measures the VLM's failure
rather than the cheap route's. Under a preference label that reasoning inverts: the packet shows the
VLM's actual production output, truncation and loop repair included, so a judge comparing a truncated
transcription against a complete one prefers the complete one, and the router learns not to escalate
a document the VLM will botch. That is 16.7% of the sample, and it is exactly the behaviour v1 had to
keep as a separate gate.

**Rank only; the model's margin is discarded.** A 45-document human-judged subset calibrates the
judge, and the two halves of that calibration point in opposite directions. Pairwise agreement with
the human is 0.756 overall, and it tracks the human's own confidence the way a real signal should:
1.000 where the human called the gap large (n=6), 0.760 small (n=25), 0.643 where the human called
the two equivalent (n=14), which is chance where people genuinely differ. Aggregate rates match
closely, and the human ranked pdf-inspector *worse* than the model does, so the judge is not biased
toward the cheap route.

The margin fails the same test. The human called 6 of 45 gaps large and 14 equivalent; the model
called 36 large and zero equivalent, for a margin agreement of 0.22. So the target is the pairwise
call. Where a graded target is wanted it comes from agreement between two judge models on the same
document, which at least measures something external to either of them.

## Two gates are arithmetic, and they run before the score

<!-- MEASURED: corpus split table -->

## The frontier

<!-- MEASURED: per-arm frontier, knee, marginal precision -->

## Free features against paid ones

<!-- MEASURED: gain split and equal-quality CPU comparison -->

## The legibility floor: gate it, or render it bigger?

<!-- MEASURED: budget option table and recommendation -->

## Operating points

<!-- MEASURED -->

## A recurring defect: unsized client concurrency limits

The judging pass ran at roughly a quarter of its achievable rate until its HTTP client's connection
pool was sized. `httpx.AsyncClient` defaults to 100 pooled connections and 20 keep-alive, so at 256
in-flight requests 156 of them queued *inside the client* waiting for a connection — and that wait
counts against the request timeout, so raising concurrency past 100 bought queueing rather than
throughput and presented as the judge having got slower. Sizing the pool to the concurrency took the
measured rate from ~1–2 verdicts/s to 4.7.

That is the fourth instance of the same defect class in this repository, and the first three are all
in one unrelated subsystem:

| where | ceiling | symptom |
|---|---|---|
| inference proxy (`a191db3c56`) | anyio's 40-thread `to_thread` limiter | any brokered fleet capped at ~40 in-flight requests total |
| inference worker's forwarding path (`a191db3c56`) | httpx's 100-connection pool | forwarding capped at 100 against a 512-thread pool |
| dashboard `/v1` funnel (`a191db3c56`) | httpx's 100-connection pool | same ceiling on the dashboard proxy |
| this judging pass | httpx's 100-connection pool | 256-way concurrency delivered ~100-way throughput |

[`ocr-budget-sweep.md`](ocr-budget-sweep.md) already records the first three as "three separate
library-default concurrency ceilings this campaign hit in the serving path". The fourth is not in the
serving path at all — it is an offline labelling job — which is what makes it a pattern rather than a
property of one component. In every case the library's default is an order of magnitude below the
concurrency the caller configured, and nothing fails: the work completes, slowly, and the
misconfiguration is invisible unless someone measures throughput against the concurrency they asked
for.

Two shapes of fix suggest themselves, and this task deliberately did neither: a shared helper that
constructs an `httpx.AsyncClient` with its limits derived from the caller's concurrency, or a lint
rule in `infra/lint/` that flags an `httpx.AsyncClient` constructed without explicit `limits`
alongside a semaphore or worker count. Both belong to whoever owns the serving path rather than to
the PDF router.

## Reproducing

```bash
# Package the blinded two-route packets over the 100k oracle sample.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-preference-set --extra pdf \
    --cpu 8 --memory 24GB --disk 16GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.build_preference_set

# Measure pdf-inspector's own output text, one narrow row per document.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-inspector-output-study --extra pdf \
    --cpu 8 --memory 24GB --disk 16GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.build_inspector_output_study

# Buy the verdicts and write the label table.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-preference-judge --extra pdf \
    --cpu 16 --memory 32GB --disk 16GB --enable-extra-resources \
    -e OR_KEY_SCALE_UP "$OR_KEY_SCALE_UP" \
    -- python -m experiments.datakit.build_pdf_source.quality.judge_preference_set

# Evaluate the arms, then fit and calibrate the shipped booster.
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
    --job-name pdf-route-v2 --extra pdf \
    --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \
    -- python -m experiments.datakit.build_pdf_source.quality.analyze_route_v2
```

`quality/route_v2_features.py` holds the feature contract and the cost model; every group's price is
declared there, so replacing the PyMuPDF pass with a cheaper producer is an edit to one table rather
than to the router.
