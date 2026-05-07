---
theme: slidev-theme-open-athena
title: Marin Review
info: |
  May 2026 Marin internal review.
  Skeleton drafted on May 6, 2026 from merged PRs since the late March review.
layout: cover
transition: none
mdc: true
---

# ![](/icons/chart.svg) Marin Review

May 2026

---
layout: agenda
---

::title::

# Agenda

::default::

## Scorecard since late March

## Systems: Iris, CoreWeave, finelog

## Data and eval coverage

## Training, MoE, and Delphi

## What should be true by next review

---
layout: default
---

# ![](/icons/calendar.svg) April Goal Scorecard

| April goal cluster | Status | Read |
|---|---|---|
| Execution platform + observability (`#4269`, `#4273`, `#4474`) | **Mostly landed** | Ray is gone from Marin; Iris workqueue, dev-TPU, and resource visibility landed. Left: Levanter store, export memory, historical utilization. |
| Data + library foundations (`#4272`, `#4271`) | **Landed** | Canonical data pipeline and Marin-as-a-library epics closed. Watch whether new datasets are now simple in practice. |
| MoE scaling + MFU (`#4281`, `#4283`) | **Partial** | Clean MoE isoflop and 1e23 preregistration landed; key ablations and H100/MFU parity remain open. |
| Agentic + post-training readiness (`#4282`, `#3192`) | **Started** | Experiment-agentification closed; synthetic-data strategy remains open. Evals made code/tool/patch gaps concrete. |

---
layout: default
---

# ![](/icons/flag.svg) Since Late March

- We got the compute grant!

## Infra
- Ray was removed from Marin as an execution path.
- Iris got budgets, priorities, preemption, better scheduler hot paths, and a real status/debug surface.
- CoreWeave moved from bring-up into recurring smoke/canary coverage.
- Datakit and Zephyr now have a clearer download -> normalize -> dedup -> consolidate -> tokenize path.

## Modeling

- MoE recipe continued to improve. 1e23 MoE still running
- Delphi release+blog post almost done!
- Lots more training data source ingested, including new code-adjacent synthetic data.
- Lots of new perplexity evals to identify gaps in our existing coverage.

---
layout: section
---

# Systems

Iris, CoreWeave, and logs

---
layout: default
---

# ![](/icons/servers.svg) Iris Replaced Ray

TODO: russell

- `#5138` removed Ray from Marin, with related cleanup in `#5132`, `#5131`, `#5089`, `#5087`, `#5076`, `#5031`, and `#5028`.
- Iris now carries user-facing scheduling concepts: budgets and priorities (`#4096`, `#5081`), preemptible jobs (`#5083`), manual slices (`#5078`), same-variant preemption (`#5240`), and routing docs (`#5418`, `#5426`).
- Scheduler and API hot paths were tightened: lightweight job-state polling (`#4209`), raw SQL / denormalized scheduling rows (`#4181`, `#4264`), ListJobs/ListWorkers pagination and filtering (`#4558`, `#4703`, `#5025`, `#5384`, `#5454`), and cached resource scalars in `can_fit` (`#5412`).
- Dashboard work moved debugging closer to the jobs: state filters, child job sorting, task summaries, status markdown, endpoint proxying, and task-resource history.

---
layout: default
---

# ![](/icons/gpu.svg) CoreWeave / GPU Path

TODO: russell

- CoreWeave CI and canaries continued after the March rollout: Iris PR workflow (`#4174`), canary routing and timeouts (`#5112`, `#5125`, `#5429`, `#5463`, `#5479`), and sharp-edge docs (`#5431`).
- New CW cluster wiring landed for RNO2A / USW09B plus `cwobject` S3 paths (`#5420`).
- GPU stack moved with JAX 0.10 / CUDA 13 (`#5428`), native vLLM mode (`#4753`, `#5326`), and CoreWeave NCCL fixes (`#5379`, `#5461`).
- Grug MoE GPU work got a Triton `ragged_dot` path and a reported **3.1x** backward speedup over main (`#4297`, `#5350`).

<Box>

TODO: Fill in whether multi-host JAX performance is still the blocking issue, and whether the canary is green enough to trust per-PR changes.

</Box>

---
layout: default
---

# ![](/icons/chart.svg) Finelog and Observability

TODO: russell

- Iris log delivery moved from heartbeat-based logging to a push LogService (`#4274`), then into `lib/finelog` (`#5212`).
- Storage and query behavior improved through stable segments, DuckDB/Parquet compaction, zstd batches, a leveled compactor, catalog-driven copy workers, namespace stats, and dashboard plumbing (`#4518`, `#4881`, `#5290`, `#5441`, `#5457`, `#5459`, `#5456`).
- Debugging loops got more concrete: CPU/memory profiling, job profile summaries, RPC stats, task-resource history, status markdown, and linked log lines (`#4186`, `#4194`, `#4935`, `#5284`, `#5443`, `#5174`).

---
layout: section
---

# Data + Evals

More sources, better pipelines, broader diagnostics

---
layout: default
---

# ![](/icons/big-data.svg) Datakit and Zephyr

TODO: russell

- `datakit` was bootstrapped for consolidated downloads (`#4142`) and now has normalize, source registry, staged workflows, and smoke ferries (`#4188`, `#4598`, `#5105`, `#5450`).
- The pipeline now writes normalized Parquet, supports split main/duplicate outputs, exact dedup in normalize, MinHash / fuzzy dedup job separation, and per-shard resume for MinHash attrs (`#4596`, `#4610`, `#4876`, `#4893`, `#5397`).
- Zephyr got stronger execution semantics: subprocess-per-shard and later inline shard execution (`#4522`, `#5282`), shuffle/external-sort fixes (`#4695`, `#4782`), stage counters (`#4189`, `#4212`, `#5063`), schema diagnostics (`#5136`, `#5142`), and byte-based scatter heuristics (`#5340`).

---
layout: default
---

# Training Data

TODO: will

- Landed/normalized sources now include rollout pipelines for six datasets (`#4329`), NSF abstracts (`#4516`), PleIAs/common_corpus (`#4606`), HPLT likely non-duplicates (`#4326`), BHL stitching (`#5408`), GAIR/daVinci-Dev (`#5252`), Molmo2-Cap (`#5299`), nyuuzyou/svgfind (`#5304`), and public diagnostic logs (`#5121`).
- Will's in-flight source work adds more code/agent-adjacent surfaces: Stack v2 stitching (`#5009`), SWE-Rebench Contree traces (`#5276`), Hermes agent reasoning traces (`#5300`), SEC-EDGAR (`#5305`), and MASSIVE multilingual tool use (`#5339`).



---
layout: default
---

# ![](/icons/chart.svg) Perplexity Gaps

- In `#4693` and `#5005`, we developed a thesis that Marin's weakness at posttraining is actually observable on the pretrained artifacts. We sought to identify weak points for our v1 Marin data mix.
- In `#4693`, we look at agentic traces to see if there are particular spans (prose, tool calls, tool results, patches) where Marin is weak.
- In `#5005` we broaden the search to every data source we could think of that was easy to add. We looked at ~100 new naturalish data sources (with ~80-90 more wired up) and our classic set of ppl evals.
- The broad English story is not the problem. Marin 8B is roughly on par with Llama 3.1 8B on Paloma / Uncheatable (`+0.0029` / `+0.0049` BPB) and beats Qwen3 8B on Paloma while staying close on Uncheatable (`-0.0272` / `+0.0074` BPB).

---
layout: default
---

# ![](/icons/code.svg) Agent Traces: Patch And Observation Gaps

<div style="display: flex; align-items: center; justify-content: center; height: 420px;">
  <img src="/charts/agent-traces/trace-labels.png" alt="Agent trace span labels" style="max-width: 96%; max-height: 405px; object-fit: contain;"/>
</div>

---
layout: default
---

# ![](/icons/code.svg) Agent Traces: Patch And Observation Gaps

<div style="font-size: 0.72em; line-height: 1.15">

| Model | Assistant | Tool | Observation | Patch | Patch gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B base | 0.482 | 0.796 | 0.428 | 0.220 | +0.161 |
| Llama3.1-8B base | 0.532 | 0.619 | 0.459 | 0.237 | +0.183 |
| Marin-8B base | 0.543 | 0.435 | 0.752 | 1.867 | -1.432 |
| Marin-8B instruct | 0.564 | 0.477 | 0.902 | 1.934 | -1.429 |

</div>

- BPB: lower is better. Patch gain: how much the trace helps predict the final patch; positive means the trace helped.
- Marin looks basically fine on prose/chat and surprisingly good on tool-call spans.
- The failures are in understanding tool results and producing patches. More trace context helps peer models predict patches, but hurts Marin.
- This is now a repeatable agent-trace PPL suite (`#4963`, `#5248`) and should directly inform the next data mix. (Good evidence for us needing to add program traces, log files, etc. to the mix!)

---
layout: default
---

# ![](/icons/chart.svg) Perplexity Gap: Long-Tail Data

<div style="font-size: 0.72em; line-height: 1.15">

| Slice family | 8B gap vs Qwen3 | 32B gap vs Qwen3 | Read |
| --- | ---: | ---: | --- |
| Paloma | -0.027 | -0.088 | Edited English is not the main gap |
| Uncheatable Eval | +0.007 | -0.026 | Prose-heavy held-out slices mostly hold up |
| FineWeb2 multilingual | +0.243 | +0.184 | English-only training shows up immediately |
| SVG Markup | +0.098 | +0.104 | Markup in general is weak |
| Package metadata | +0.082 | +0.038 | Repo-adjacent structured text is undercovered |
| Bio / chem notation | TODO | +0.082 | Science notation is a clear blind spot |
| Game / music notation | TODO | -0.200 | Not every weird slice is bad |

</div>

- The recurring weak families are code, messy web artifacts, structured text, code-adjacent metadata, tables, multilingual prose, and scientific notation.
- The dashboard is live at [marin.community/analysis/perplexity-gap](https://marin.community/analysis/perplexity-gap/).

---
layout: section
---

# Training + Scaling

MoE, Delphi, and open questions

---
layout: split-left-green
---

# ![](/icons/chart.svg) 3e18 MoE Progress Over Time

<img src="/charts/moe-progress-frontier/dial_moe_3e18_progress.png" alt="3e18-scale dial_moe finished run progress over time" style="width:620px;"/>

::right::

- Orange is cumulative best; teal lines are Delphi AdamH refs: **3e18 1.0871**, **2e19 0.9948**, **3e19 0.9720**.
- Current best is **0.9889** BPB at exact **3e18**: `isoflop-k5e256-d768-3e+18`, finished **May 2**.
- That is below Delphi **2e19** and within **0.0169** BPB of Delphi **3e19**. March 24 MoE best was **1.0407**.

---
layout: split-left-green
---

# ![](/icons/chart.svg) 1.7e18 MoE Progress Over Time

<img src="/charts/moe-progress-frontier/dial_moe_1p7e18_progress.png" alt="1.7e18-scale dial_moe finished run progress over time" style="width:620px;"/>

::right::

<div style="font-size: 0.84em; line-height: 1.14">

- **217** finished `dial_moe` runs in the **1e18-2e18** FLOP band; **187** are in the **1.65e18-1.75e18** focus band.
- Orange is cumulative best within the 1.7e18 focus band; gray points show the broader 1e18-2e18 context.
- Best focus run is **1.0151** BPB: `may-arch-lr0.8x-d768-1.70e+18`, finished **May 1**.
- The 1.7e18 best beats the March 3e18 MoE best by **0.0256** BPB at about **57%** of the compute.
- Current 3e18 best is still another **0.0262** BPB lower, so this separates recipe progress from scale.

</div>

---
layout: default
---

# April MoE improvements

From 60+ experiments, these are the ones that stuck.


* [Expert Sparsity](https://github.com/marin-community/marin/issues/5387).  64 -> 256 experts. 17% speedup at 2e18, and 25% speedup at 9e18.
* [Partial RoPE](https://github.com/marin-community/marin/issues/4946). Applying RoPE to only half of the head dims. 9% at 2e17, 12% at 2e18, 4% at 9e18, and 6% at 3e19.
* [Partial Key Offset.](https://github.com/marin-community/marin/issues/4976) Shifting half of the head dims for each key by one position on the long windows gives a 19-22% speedup across all 4 compute scales, building on partial RoPE.
* [Embed to AdamH](https://github.com/marin-community/marin/issues/5184). The main benefit of moving the embed from AdamW to AdamH is to give cleaner parameter and gradient norm scaling over training.

Many null results besides, and lots more work in progress!


---
layout: default
---

# ![](/icons/tpu.svg) Misc Training stuff

- Kernel work landed on both TPU and GPU: ragged all-to-all capacity clipping (`#4359`), sender offsets (`#4867`), Pallas API fixes (`#5347`), and Triton `ragged_dot` (`#4297`, `#5350`).
- Training reliability improved around W&B failures and resume symmetry (`#5332`, `#5415`), checkpoint roots (`#4387`, `#5066`), shuffle defaults (`#5246`, `#5259`), and tokenizer migration (`#4405`, `#4451`, `#4977`).


---
layout: default
---

# MoE 1e23  Progress

<div style="display: flex; align-items: center; justify-content: center; height: 420px;">
  <img src="/charts/moe-1e23-progress/cross_entropy_loss_crop.png" alt="1e23 MoE train cross entropy loss over time" style="max-width: 92%; max-height: 410px; object-fit: contain;"/>
</div>


---
layout: default
---

# The Death Throes of Ray

- Node instability (I think caused by Ray, but may have been GCP) led to a run crash that we had a hard time recovering from.
- Interaction of a bug in our logging and wandb's quirky resume behavior meant we couldn't reuse run ids -> auto-recovery didn't work.
- Also had some issues where we had checkpoint corruption due to a botched launch and checkpointing policy.
- We need a "hero run" skill/playbook to mitigate.


---
layout: split-left-green
---

# ![](/icons/chart.svg) Delphi Scaling + Blog Work

TODO: will

<img src="/charts/delphi_scaling_suite_results.png" alt="Delphi scaling suite placeholder" style="width:100%;"/>

::right::

- `#4591` added the seed sweep for Delphi 1e21 / 1e22 / 1e23 runs.
- `#5168` added downstream evals for the Delphi scaling-suite blogpost.
- `experiments/exp1337_eval_suite.py` says it generates the data behind the `mmlu-emergence` figure for `content/blog/delphi.md`.
- I did not find `content/blog/delphi.md` locally. Treat this slide as a placeholder until the draft/post path is available.

<Box>

TODO: Replace March figure with final blog figure and summarize the 1e23 result, MMLU/HumanEval/GSM8K projection, and any mismatch from the scaling forecast.

</Box>


---
layout: default
---

# ![](/icons/rocket.svg) What Should Be True By Next Review


- Scaling recipe (#5358, @ClassicLarry) — isoFLOP results integrated from April, LR retuned, possible long-context  extension; output is a forecast for the June run.
- Data mix — @Helw150 launches an active swarm over all datakit/sources.py (#5359, target launch May 15) that must   beat proportional baselines on UncheatableEval/HumanEval/MMLU/GPQA + David's PPL sets; @ravwojdyla lands the upstream pipeline (#5360: dedup params + contamination detection p0, quality scores p1) in time to feed it; @dlwh + @Helw150 identify the perplexity gaps that drive mixture decisions (#5367).
- GPU training — @rjpower gets a June-sized MoE running ~1k steps across 2+ H100 hosts on CoreWeave (#5356), while @dlwh chases Nemotron-parity MFU on the H100 kernels (#5357).
- Eval + infra — @yonromai stands up a preemption-resilient vLLM eval service on Iris (#5368, P0 = MMLU + HumanEval on the 1e22 MoE); #5369 is the catch-all infra tune-up (unified queries, zero-trust proxy, GH→Iris).
