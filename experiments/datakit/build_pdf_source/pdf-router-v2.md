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
judge preferred the VLM. Both are campaign-only and live on `mark/pdf_processing`; see
[Reproducing](#reproducing).

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

## What the labels cost and cover

19,977 of 20,000 drawn documents carry a label, across **2,588 of the corpus's 2,589 domains**.
17,923 came from a judge; 2,054 are decided without one, because pdf-inspector produced no text and a
packet would have asked a judge to compare a transcription against an empty string. 23 documents
(0.115%) have no label: 22 were packets above 30 MB that the judge's API rejected with a 502 on every
attempt, and one exhausted its retries on a malformed response. Those are recorded `unjudged` rather
than defaulted to a decision.

Total spend was **$55.74** for 19,497 verdicts, against a $140 cap.

**The escalation rate is 0.762** over all labelled documents, 0.735 over judged ones alone, and 0.768
page-weighted. That number governs everything below: the VLM is preferred on roughly three documents
in four, so the router's job is to find the remaining quarter rather than to ration a scarce resource.

Two checks on the label:

**Inter-judge agreement is 0.878** (κ = 0.757) over 1,152 documents from 854 domains judged by a
second vendor's model. The two disagree on their aggregate rate by less than two points — 0.739
against 0.758 — so the label is not one model's idiosyncrasy.

**Labelling the VLM's own failures worked.** Documents whose VLM extraction is damaged — truncated,
loop-repaired, or rendered below the floor — escalate at **0.613**, against **0.781** for clean ones
(n = 2,271 and 17,706). The judges do penalize the VLM for its own output, which is the behaviour
router v1 had to encode as a separate gate and this label captures directly.

One caveat measured rather than assumed: **8.0% of verdicts cite evidence the prompt asked the judge
to ignore** — chart, axis or figure text, where the two systems are under different instructions.
Those verdicts are kept, because the reason field is a summary rather than the whole basis of the
call, but it bounds how much of the label could be a figure-policy artifact.

## One gate is arithmetic; the other was refuted

| class | documents | share of docs | share of pages |
|---|---|---|---|
| forced escalate — pdf-inspector produced no text | 12,396 | 12.40% | 9.19% |
| routable — the learned score decides | 87,604 | 87.60% | 90.81% |
| ...of which flagged for a raised render budget | 1,630 | 1.63% | — |

**The no-text gate is exact.** pdf-inspector returns nothing for 12,396 documents (269 of them
library exceptions, the rest successful extractions of a scan). Every one of the 2,054 that carry a
label was escalated by the judge: **escalation rate 1.000**. There is no cheap route to keep, and the
score never sees them.

**The legibility gate was wrong, and the label says so.** Router v1 skipped documents whose pages
render below 100 DPI on the grounds that the VLM cannot read a page the render underresolves, so
escalating buys a transcription of a blur. Judges escalated **79.0%** of those documents (n = 558).
The premise does not hold on this corpus: these are large-format scans where pdf-inspector produces
nothing usable, and the VLM reading a 50-DPI render still recovers more of the page than that. The
floor stops gating routing. What it still earns is a *render* policy, below.

## The frontier

Held out on 4,428 documents from **633 unseen domains**, for the shipped arm. Budgets are quoted both
ways: *net* of the forced escalations, which is what the score actually ranks, and *gross* against the
whole corpus, which is what the pipeline pays. The two differ by 9.19% of pages before the score
allocates anything.

| net docs | gross pages | loss/pg | misroute/pg | catches | marginal | core-h/M | crawl core-h | crawl GPU-h |
|---|---|---|---|---|---|---|---|---|
| 5% | 16.5% | 0.6813 | 0.6828 | 10.4% | 0.96 | 5.16 | 289 | 144 |
| 10% | 22.7% | 0.6171 | 0.6221 | 18.8% | 0.91 | 6.25 | 350 | 198 |
| 20% | 36.2% | 0.4876 | 0.5128 | 35.9% | 0.93 | 8.67 | 486 | 317 |
| 30% | 45.9% | 0.4011 | 0.4458 | 47.2% | 0.70 | 10.38 | 582 | 401 |
| 40% | 54.2% | 0.3243 | 0.3842 | 57.3% | 0.89 | 11.87 | 665 | 474 |
| 50% | 62.7% | 0.2552 | 0.3390 | 66.4% | 0.90 | 13.37 | 749 | 547 |
| 60% | 69.9% | 0.1911 | 0.2909 | 74.9% | 0.76 | 14.67 | 821 | 611 |
| 70% | 78.9% | 0.1229 | 0.2527 | 83.8% | 0.69 | 16.26 | 910 | 689 |
| 80% | 85.2% | 0.0835 | 0.2438 | 89.0% | 0.66 | 17.39 | 974 | 744 |
| **90%** | **93.0%** | **0.0252** | **0.2129** | **96.7%** | 0.45 | **18.77** | **1,051** | **812** |

**Marginal precision never collapses.** Router v1's fell to 0.16 by an 80% budget, so its last stretch
cost 6 VLM runs per document rescued. Here it is 0.45 even at 90%, or 2.2 pages escalated per page
rescued, and above 0.65 everywhere below 80%. That is not the router being better — it is the base
rate being 0.768. When three pages in four genuinely want the VLM, almost any additional escalation is
a real rescue, and the frontier stays close to linear in cost across its whole range.

**Which is why one-sided loss is the wrong metric here, and this report reports a second one.** Router
v1 treated a needless escalation as recoverable CPU rather than quality damage, because the VLM was
better on essentially everything it was sent. Under a preference label that is false: pdf-inspector
wins 26.5% of documents, and escalating those puts the VLM's *worse* transcription into the corpus.
One-sided loss is therefore monotone in budget and always recommends escalating everything.
`misroute/pg` counts both errors, and it has a real minimum.

## Free features against paid ones

The pipeline budgets 3.4 CPU core-hours per million pages for "router features". That is two separate
PyMuPDF extractions: `route_features`' own 36 page signals at **1.86 core-h/M**, and the 124-feature
FinePDFs incumbent extraction behind `ocr_prob` at another **1.54**. Everything pdf-inspector reports
is free, because the extraction runs on every document whether or not it is escalated.

| arm | features | core-h/M | loss@25% | loss@50% | loss@70% | largest clump | free share of gain |
|---|---|---|---|---|---|---|---|
| rule: incumbent `ocr_prob` | 2 | 1.54 | 0.5875 | 0.3926 | 0.2410 | 0.14% | — |
| rule: `pages_needing_ocr` fraction | 12 | 0.00 | 0.3028 | 0.3028 | 0.0000 | **79.0%** | — |
| free (inspector output + extract + shape) | 31 | **0.00** | 0.4492 | 0.2687 | 0.1557 | 0.05% | 100% |
| **free + detect** | 43 | **0.12** | 0.4534 | 0.2552 | 0.1229 | 0.05% | 98.7% |
| free + `route_features` | 105 | 1.86 | 0.4845 | 0.2958 | 0.1315 | 0.05% | 45.9% |
| free + `route_features` + incumbent | 107 | 3.40 | 0.4799 | 0.2925 | 0.1231 | 0.05% | 45.9% |
| `route_features` only (v1's set) | 74 | 1.86 | 0.4878 | 0.2526 | 0.1194 | 0.07% | — |
| everything | 119 | 3.52 | 0.4618 | 0.2916 | 0.1449 | 0.05% | 46.8% |

**The split-draw noise floor is 0.0608** page-weighted (sd 0.0251) over five domain splits at a 50%
budget. Every difference in that table is inside it. Read unpaired, all six trained arms are tied.

So the arms are compared **paired within each split**, which cancels the split-to-split variance —
page counts are heavily skewed, so which long documents land in the test half dominates the metric.
Differencing two arms on the same split leaves the part attributable to the feature set:

| arm, differenced against `free` | mean Δ loss/pg | sd | range | verdict |
|---|---|---|---|---|
| free + detect | **−0.0014** | 0.0094 | −0.0118 … +0.0123 | indistinguishable |
| free + `route_features` + incumbent | +0.0104 | 0.0111 | −0.0061 … +0.0217 | indistinguishable |
| free + `route_features` | **+0.0127** | 0.0091 | +0.0003 … +0.0227 | **worse on every split** |
| everything | +0.0192 | 0.0179 | −0.0102 … +0.0341 | indistinguishable |
| `route_features` only (v1's set) | **+0.0273** | 0.0223 | +0.0050 … +0.0618 | **worse on every split** |

**The paid pass does not earn its cost, and adding it is mildly harmful.** `free + route_features` is
worse than `free` on all five splits — 105 features against 17,922 training rows is enough to overfit
the noise. The incumbent's extraction adds another 1.54 core-h/M and lands inside the noise. Deleting
the whole PyMuPDF router pass saves **190 crawl core-hours** and costs nothing measurable.

Gain ranking alone would have said the opposite. Given `route_features`, the model spends 51.9% of its
total gain on it — it looks like the dominant feature set right up until you hold quality fixed and
read off cost. Gain measures how much explanatory work a group did; it cannot say whether the same
quality was reachable without it.

Holding quality fixed, in corpus CPU core-hours per million pages:

| loss ≤ | free | free + detect | free + `route_features` | free + rf + incumbent | `route_features` only |
|---|---|---|---|---|---|
| 0.20 | 15.58 | **14.67** | 18.17 | 18.35 | 17.52 |
| 0.15 | 16.87 | **16.26** | 18.17 | 19.94 | 18.97 |
| 0.10 | **16.87** | 17.39 | 19.43 | 21.24 | 20.26 |
| 0.07 | **18.53** | 18.77 | 20.56 | 21.24 | 20.26 |
| 0.05 | **18.53** | 18.77 | 20.56 | 22.35 | 21.13 |

**The clumping check does its job.** The untrained `pages_needing_ocr` rule ties **79.0%** of documents
at one value — the same pathology that made router v1's incumbent locally degenerate — and its
"frontier" is two points, which is why its equal-quality column is a constant. Every trained arm's
largest clump is 0.05%, and the shipped model's corpus-wide score has 90,334 distinct values over
99,726 documents with 12.7% of documents tied anywhere at all.

## The legibility floor: render it bigger

2,234 documents render below 100 DPI at the shipped 2,048-token budget — 2.23% of documents and 0.55%
of pages. 77% are single-page, and their implied paper area is a median 787 square inches against US
Letter's 93.5. These are posters, maps and plans, not damaged files.

Since the judges escalate 79% of them, the question is not whether to send them but what to send.

| budget | Letter median DPI | gated median DPI | gated p90 | gated pages at cap | rescued | still gated | global GPU | **targeted GPU** | throughput |
|---|---|---|---|---|---|---|---|---|---|
| 2048 (shipped) | 146 | 51.6 | 94 | 0.0% | — | 2,234 | ×1.000 | ×1.0000 | measured |
| 4096 | 206 | 117.3 | 139 | 0.0% | 762 | 1,472 | ×1.246 | ×1.0014 | measured |
| 8192 | 292 | 144.5 | 193 | 0.0% | 1,249 | 985 | ×1.377 | ×1.0021 | measured |
| **16384** | 300 (capped) | 148.7 | 269 | 0.0% | **1,890** | **344** | ×1.530 | **×1.0029** | extrapolated |

**The published sweep's reason for stopping at 8192 does not apply to this population.**
[`ocr-budget-sweep.md`](ocr-budget-sweep.md) skipped larger budgets because the 300-DPI upscale cap
binds there, so a bigger budget re-renders near-identical payloads. That is measured on ordinary
paper: a Letter page has been pinned at 300 DPI since 8192. These sheets are still at 149 DPI median
at 16384, and **0.0% of their pages reach the cap at any budget**. 16384 × 1024 pixels is exactly
`render.MAX_PIXELS`, so it is the largest budget the render path can express, and its throughput is
extrapolated from the sweep's curve rather than measured — that is the one number here worth
confirming before shipping.

Raising the budget globally costs ×1.53 GPU on every page in the corpus to rescue 0.47% of them.
Raising it **only for the flagged documents** costs **×1.0029** — 0.29% more GPU crawl-wide — and
rescues 1,890 of 2,234. That is the recommendation.

The 344 that remain render below ~35 DPI even at the maximum budget. They are 0.34% of the corpus and
they are lost: neither route reads them, and they stay on pdf-inspector with whatever it produced.

## What to ship

**Route on the `free + detect` arm at 90% of routable documents.** The full configuration:

- **Gate first:** pdf-inspector returned no text → escalate. Exact, validated at an escalation rate of
  1.000 over 2,054 labelled documents, and it covers 12.4% of the corpus.
- **Score the rest** on 43 features costing **0.12 core-h/M**: pdf-inspector's extraction signals, the
  statistics of the text it produced, the document's shape, and `detect_pdf_bytes`. Escalate at or
  above the calibrated threshold.
- **Render policy:** a document whose mean render DPI falls below 100 at the 2,048-token budget is
  rendered at 16,384 instead. 1,630 documents, ×1.0029 GPU crawl-wide.
- **Delete the PyMuPDF router pass**, both halves of it.

Crawl-wide at 56M pages, this configuration costs **1,051 CPU core-hours and 812 GPU-hours**, escalates
93.0% of corpus pages, catches 96.7% of the documents that want the VLM, and misroutes 0.2129 of pages
in either direction.

| policy | misroute/pg | crawl core-h | crawl GPU-h |
|---|---|---|---|
| no router — escalate everything | 0.2320 | 1,114 | 874 |
| **`free + detect` at 90%** | **0.2129** | **1,051** | **812** |
| `free + detect` at 50% | 0.3390 | 749 | 547 |
| v1's feature set, at its own optimum | 0.2388 | 1,100 | 861 |

**The honest headline: the router is worth having, but the margin is thin.** Against simply escalating
every document it removes 10.1% of the routing error (0.2320 → 0.2129) and saves 63 crawl core-hours
and 62 GPU-hours. That is a real win on both axes at once, and it costs 0.12 core-h/M to obtain — but
it is a tenth of the error, not most of it. At a 0.768 page-weighted escalation rate there is simply
not much room between "escalate everything" and the best achievable policy.

**The preference labels are learnable, but only weakly, and the features that help are the free ones.**
The frontier is well-ordered, marginal precision is high and stable, the score does not clump, and
domain-disjoint held-out performance is consistent across five splits. What none of that buys is a
large gain over the trivial policy. Anyone hoping the router would pay for itself by rationing the VLM
should read the base rate first: three documents in four want it.

If the goal is corpus quality rather than CPU, the defensible alternative is to escalate everything
and skip the score entirely, accepting 0.2320 misroute for 63 more crawl core-hours. The router's case
rests on being better on both axes simultaneously, not on being dramatically better on either.

**The biggest single saving in this report is not the operating point.** It is deleting the router's
PyMuPDF feature pass: **190 crawl core-hours**, for no measurable quality cost, because the signals
that carry the decision come free with an extraction the pipeline already runs.

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

The two packets that surfaced this were also a lesson in where a barrier hides. They are 51.6 MB and
68.6 MB and return 502 on every attempt, which on its own would cost 22 of 17,946 packets. But packet
ids sort by draw order, those two land near the head of the pending list, and the pending list's head
is exactly what the spend-projection pilot samples -- so every restart put the same two doomed
requests in the pilot and held the entire run behind their retry chains before a single production
verdict was bought. The cost of a deterministic failure is not its own rate; it is whatever waits on
the barrier it happens to sit behind.

Two shapes of fix suggest themselves, and this task deliberately did neither: a shared helper that
constructs an `httpx.AsyncClient` with its limits derived from the caller's concurrency, or a lint
rule in `infra/lint/` that flags an `httpx.AsyncClient` constructed without explicit `limits`
alongside a semaphore or worker count. Both belong to whoever owns the serving path rather than to
the PDF router.

## Reproducing

**The four modules below are not in this tree.** They produced the shipped booster and then stayed
with the campaign, on `mark/pdf_processing` under `experiments/datakit/build_pdf_source/quality/`,
along with `fit_route_v2.py` which the last of them calls. What ships instead is the booster and its
JSON sidecar, which record the threshold, the feature contract and the operating point -- everything
the pipeline needs to route, and everything a reader needs to check what it was calibrated against.
Refitting is a campaign activity, not a pipeline step, so check that branch out to run any of this:

```bash
git checkout mark/pdf_processing
```

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
than to the router. It is the one module of this chain that does ship, because `classify.py` reads
the contract from it at runtime to check the booster it loaded was fit on the columns being fed to
it.
