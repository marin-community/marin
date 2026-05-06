# May 2026 Internal Review Outline

Drafted on May 6, 2026 from `origin/main` merged PRs since the late March 2026 review. The talk is scheduled for May 7, 2026, so every claim below should get a final status pass tomorrow.

## Working Thesis

Since late March, Marin moved from "Iris is being rolled out" to "Ray is gone from Marin." The strongest review narrative is not a single model result. It is the operating system around model work getting more real: Iris owns execution, CoreWeave has recurring tests, finelog makes logs queryable, datakit/Zephyr has a cleaner data path, and eval coverage is wide enough to start driving source decisions.

The Delphi blogpost work is the likely model-science anchor if the final figures are available.

## Slide Skeleton

1. Cover: May 2026 Marin Review.
2. Agenda: scorecard, systems, data/evals, training/scaling, next-review targets.
3. April goal scorecard: grouped readout from the preregistered April milestone epics.
4. Since late March: 5-6 bullets that name the main shifts.
5. Iris replaced Ray.
6. CoreWeave and GPU path.
7. Finelog and observability.
8. Datakit and Zephyr.
9. Eval story: data gaps, not one score.
10. Agent traces: patch and observation gaps.
11. Perplexity gap: long-tail data.
12. 3e18 MoE progress over time.
13. MoE and training mechanics.
14. Delphi scaling + blog work.
15. Decisions for tomorrow.
16. What should be true by next review.

## High-Impact Landed Changes

## April Goal Scorecard Grouping

The original April milestone has too many issue rows for an exec scorecard, so the deck groups the user-provided core epics into four clusters:

- Execution platform + observability: `#4269`, `#4273`, `#4474`. Ray removal is closed; workqueue/dev-TPU/resource visibility landed; Levanter store, export memory, and historical utilization remain follow-through.
- Data + library foundations: `#4272`, `#4271`. Both closed; this is the cleanest "landed" row.
- MoE scaling + MFU: `#4281`, `#4283`. Partial: clean MoE isoflop and 1e23 preregistration landed, but key ablations and H100/MFU parity remain open.
- Agentic + post-training readiness: `#4282`, `#3192`. Agentified experimentation closed; synthetic-data strategy remains open. The eval work makes this a more concrete data-readiness story than the raw issue state alone.

Dropped from the scorecard as separate rows:

- CoreWeave/GPU, because the April epic set only directly represents it through the H100/MFU part of `#4283`; the separate CoreWeave slide handles that story.
- Evals as a standalone scorecard row, because the provided April goal list does not include `#4963`/`#5005`; evals are used as evidence for agentic/post-training readiness instead.

### 1. Ray Was Removed From Marin

Use this as the main systems headline unless tomorrow's data says otherwise.

- `#5138` landed "No more Ray in Marin."
- Related cleanup removed Ray cluster templates, Ray operator tooling, `ray_run.py`, Ray+Iris integration tests, dead Marin Ray glue, and Levanter Ray TPU infra: `#5132`, `#5131`, `#5089`, `#5087`, `#5068`, `#5028`, `#5031`.
- Docs and skills were routed to Iris: `#5076`, `#5202`, `#4985`, `#4742`.
- This is a clean before/after story from the March review, which still talked about turning down Ray over the next few weeks.

Evidence to fill in tomorrow:

- How many production jobs ran on Iris since the cutover.
- Whether any active workflows still depend on a Ray fallback outside Marin.
- Babysitting load before/after Ray retirement.

### 2. Iris Became The Scheduling Surface

The change is bigger than "more reliability fixes." Iris now exposes scheduling concepts users can reason about.

- Budgets, priorities, and preemption: `#4096`, `#5081`, `#5083`, `#5240`.
- Manual slices and routing: `#5078`, `#5418`, `#5426`.
- Scheduler hot paths and query load were repeatedly attacked: `#4181`, `#4209`, `#4261`, `#4264`, `#4558`, `#4703`, `#5025`, `#5384`, `#5454`, `#5412`.
- Controller state became more typed and testable: `#4280`, `#4644`, `#5147`, `#5164`, `#5165`.
- Dashboard/CLI gained job state filters, child job sorting, task summaries, status markdown, resource history, endpoint proxying, and job request details: `#4228`, `#4229`, `#4592`, `#4614`, `#4668`, `#5187`, `#5283`, `#5336`.

Open caveat:

- The same period includes many race, heartbeat, endpoint, and capacity bugfixes. The slide should say Iris is carrying more real load, not that it is done.

### 3. CoreWeave Shifted From Bring-Up To Recurring Validation

March framed CoreWeave as initial rollout. May can frame it as a path being hardened.

- Iris PR CoreWeave workflow: `#4174`.
- Canary routing/timeouts/manual runs/docs: `#5112`, `#5125`, `#5429`, `#5431`, `#5463`, `#5479`.
- RNO2A / USW09B clusters and `cwobject` S3 wiring: `#5420`.
- JAX 0.10 / CUDA 13 for GPU installs: `#5428`.
- Native vLLM by default and Docker sidecar removal: `#4753`, `#5326`.
- NCCL fixes: `#5379`, `#5461`.

Evidence to fill in tomorrow:

- Current CoreWeave canary result.
- Whether multi-host JAX is still the main blocker.
- Any measured throughput on H100/GH200 workloads.

### 4. Finelog Became A Real Log System

This deserves a slide if the review includes operational maturity.

- Iris moved from heartbeat logging to push LogService: `#4274`.
- The log store moved into `lib/finelog`: `#5212`.
- Follow-on work added stable segments, DuckDB/Parquet compaction, zstd, namespace stats, catalog copy workers, and deploy plumbing: `#4518`, `#4881`, `#5290`, `#5441`, `#5457`, `#5459`, `#5456`.
- Debuggability improved through CPU/memory profiles, RPC stats, job-profile summaries, status markdown, and linked log lines: `#4186`, `#4194`, `#4935`, `#5284`, `#5443`, `#5174`.

Evidence to fill in tomorrow:

- One incident or debugging loop where finelog changed the outcome.
- Query latency or storage numbers if available.

### 5. Datakit And Zephyr Got A Cleaner Data Path

The change is a more explicit data lifecycle: download -> normalize -> dedup -> consolidate -> tokenize.

- Datakit bootstrap: `#4142`.
- Normalize and canonical source registry: `#4188`, `#5105`.
- Daily/weekly smoke ferries and consistent workflows: `#4598`, `#4966`, `#5450`.
- Normalized Parquet, split main/dup outputs, exact dedup, MinHash/FuzzyDups separation, per-shard resume: `#4596`, `#4610`, `#4876`, `#4761`, `#4893`, `#5397`.
- Zephyr execution and shuffle work: `#4522`, `#4695`, `#4782`, `#5282`, `#5340`.
- Better counters and schema diagnostics: `#4189`, `#4212`, `#5063`, `#5136`, `#5142`.
- Tokenization reliability and memory bounds: `#4332`, `#4341`, `#4454`, `#5158`, `#5231`.

Data/source examples:

- Six rollout data pipelines: `#4329`.
- NSF grant abstracts: `#4516`.
- HPLT likely non-duplicates: `#4326`.
- BHL page stitching: `#5408`.
- GAIR/daVinci-Dev, Molmo2-Cap, nyuuzyou/svgfind: `#5252`, `#5299`, `#5304`.
- Public diagnostic logs: `#5121`.

Evidence to fill in tomorrow:

- Tokens processed since late March.
- Ferry pass/fail history.
- Any cost or throughput improvement from Zephyr changes.

### 6. Eval Work Turned Into A Data Gap Map

The deck should avoid reciting every new slice. The story is that evals are now concrete enough to say what data Marin is missing.

#### Agent-trace PPL work (`#4963`)

- Motivation: make base-model pretraining more "agentic RL ready" without waiting for a full pretrain -> posttrain -> downstream benchmark loop on every candidate.
- Method: score coding-agent traces by span type: assistant text, final assistant text, patch, tool call, and observation. For text spans, report BPB. For patches, report patch gain: how much trace context helps the model predict the final patch.
- Main finding: Marin is not broadly broken on trace prose. It is roughly comparable on assistant text and unusually strong on tool-call spans, but weak on observations and very weak on patches.
- The sharpest number: peer models have patch BPB around `0.22-0.25` and positive patch gain around `+0.16` to `+0.20`; Marin 8B has patch BPB around `1.87-1.93` and patch gain around `-1.43`.
- Interpretation: Marin can emit the shape of tool calls, but does not yet model tool results or patch consequences well. That is exactly the kind of pretraining gap that will make posttraining for coding agents harder.
- Status: `#5248` packages the trace-masked agent probe into a repeatable eval suite.

#### Perplexity-gap portfolio (`#5005`)

- Motivation: broaden checkpoint confidence from Paloma-style held-out loss into a portfolio of places where the model might fail: raw technical text, multilingual text, code-adjacent artifacts, structured text, scientific notation, and agent traces.
- Method: compare Marin against peer models in BPB, including cross-tokenizer realignment so the report can drill below doc-level summaries.
- Main finding: Marin looks good on edited English prose but has clear gaps on non-English text, messy web artifacts, package metadata, structured tables, code-ish surfaces, and scientific notation.
- Concrete numbers from the public report:
  - Marin 8B vs Qwen3 8B: Paloma `-0.0272` BPB, Uncheatable `+0.0074`, FineWeb2 multilingual `+0.2431`, runnable long tail `+0.0911`, package metadata `+0.0823`.
  - Marin 32B vs Qwen3 32B: Paloma `-0.0878`, Uncheatable `-0.0260`, FineWeb2 multilingual `+0.184`, runnable long tail `+0.1017`, bio/chem `+0.0824`, game/music `-0.1998`.
- Interpretation: this is not a single quality score. It is a source-selection tool. The next data mix should explicitly buy back long-tail structured and technical surfaces, not just add more clean English web text.
- Caveat: no decontamination has been attempted, and we cannot decontaminate peer models. Use these as directional source diagnostics, then validate with hard evals.

Figures wanted:

- Agent trace span heatmap/table from `#4963` or `#5248`: rows for Qwen/Llama/Marin base/instruct, columns for assistant text, tool, observation, patch, and patch gain. Highlight lower BPB and positive patch gain.
- Patch gain bar chart sorted by model. This is the cleanest one-slide visual because Marin is negative while peers are positive.
- Perplexity-gap family bar chart from <https://marin.community/analysis/perplexity-gap/> with Paloma, Uncheatable, FineWeb2 multilingual, runnable long tail, package metadata, structured text, bio/chem, and game/music.
- Optional if there is time: a 32B drilldown chart for GH/log/API/URL/diff/patch surfaces or worst pattern buckets, since this connects directly to coding-agent data needs.

### 7. MoE Work Continued Across Recipe And Kernels

This is the training-performance section.

- New progress figure uses finished `marin-community/dial_moe` runs in the 3e18-scale band from April 7 through May 5.
- The band is `2.5e18-4.0e18` FLOPs. Exact 3e18 runs only cover May 2-3, so the band preserves the month-long progress story while keeping the compute scale comparable.
- Pull result: 57 finished runs with `eval/paloma/c4_en/bpb` in the band.
- The Delphi AdamH ladder winners at 3e18, 2e19, and 3e19 are now teal reference lines on the chart: `1.087110`, `0.994790`, and `0.972027` BPB.
- Story read: the 3e18 MoE best of `0.988889` BPB is below the Delphi 2e19 point and `0.016862` BPB above the Delphi 3e19 point.
- The full Delphi by-band C4 table is in `public/charts/moe-progress-frontier/delphi_c4_by_flop_band.txt`.
- The March chart's best-so-far point was `1.040673` BPB at March 24. The current best is `isoflop-k5e256-d768-3e+18` at `0.988889` BPB, a `0.051784` BPB drop.
- Configurable MoE implementation: `#4964`.
- `MoeAdamHHeuristic`, dense layer removal, and sharding fixes: `#4636`.
- Router logits upcast to fp32: `#4234`.
- Grug MoE XSA grouped-query fix: `#4315`.
- Muon MoE orthogonalization: `#3902`.
- Ragged all-to-all capacity clipping and sender offsets: `#4359`, `#4867`.
- Pallas API fix: `#5347`.
- Triton `ragged_dot` grouped matmul and reported **3.1x Grug MoE backward speedup over main**: `#4297`, `#5350`.

Evidence to fill in tomorrow:

- The current recommended MoE implementation and capacity factor.
- Whether the GPU speedup changes the next Grug/CoreWeave plan.

### 8. Delphi Blogpost Work Exists, But The Post Is Not In This Repo

Use a placeholder slide until the draft path is known.

- `#4591` added seed sweeps for the Delphi runs.
- `#5168` added `experiments/exp1337_eval_suite.py`, with the module docstring saying it generates data for the Delphi scaling-suite blog post.
- The referenced target is `content/blog/delphi.md`.
- I searched `/Users/dlwh/src/marin`, `/Users/dlwh/.codex/worktrees/d044/marin`, `/Users/dlwh/.codex/worktrees/marin-community-site`, and `/Users/dlwh/notes`; I did not find `content/blog/delphi.md` or a local `*delphi*.md` blog draft.

What the slide can safely say now:

- The eval suite covers Delphi IsoFLOP winners plus 1e21 / 1e22 / 1e23 optimal runs.
- It evaluates MMLU, HumanEval, and GSM8K through both hard metrics and soft logprob/bpb proxies.
- It compares against Qwen3 Base, Llama 2, Llama 3.1/3.2, OLMo 2, and Marin 8B Base.

Evidence to fill in tomorrow:

- Final 1e23 training loss/eval loss and forecast error.
- Blog figure path for `mmlu-emergence`.
- Whether the downstream eval projection agrees with the scaling-law story.

## Candidate One-Slide Summary

Marin's main progress since late March is operational. Ray is gone from Marin, Iris owns scheduling, CoreWeave has recurring validation, finelog made logs queryable, datakit/Zephyr has a clearer data lifecycle, and eval coverage is broad enough to debug source-level model gaps. Delphi is the model-science anchor if the 1e23 and downstream-eval figures are ready.

## Open Data Requests For Tomorrow

- Iris production job count, failure count, and top incident classes since March 26.
- CoreWeave canary status and latest multi-host GPU run status.
- Datakit/Zephyr throughput for the main smoke ferry and any large production data run.
- Source table or W&B report for the agent-trace span losses in `#4963` / `#5248`.
- Current perplexity-gap dashboard exports or summary paths for the new raw slices.
- Delphi final figures and blogpost draft location.
