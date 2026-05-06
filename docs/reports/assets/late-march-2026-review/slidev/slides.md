---
theme: slidev-theme-open-athena
title: Marin Review
info: |
  Late March 2026 Marin review.
  Drafted from milestone 6, recent engineering threads, and meeting notes.
layout: cover
transition: none
mdc: true
---

# ![](/icons/chart.svg) Marin Review

Late March 2026

---
layout: agenda
---

::title::

# Agenda

::default::

## Scorecard and milestone status

## Infra rollout and data tuneup

## Training and scaling

## What should be true by next review

---
layout: default
---

# ![](/icons/flag.svg) The Scorecard We Set Up Last Time

- The December review framed February and March around two goals:
  - training progress
  - infra reliability and automation
- The deeper goal was to make repeated large runs and scaling decisions more trustworthy.

---
layout: default
---

# ![](/icons/calendar.svg) Status Snapshot on March 25, 2026

| Area | Status | Read |
|---|---|---|
| Infra | **Improving** | Iris is carrying real traffic, but Ray still carries more volume |
| Data + Zephyr | **Mixed** | pipeline execution is better, but need more data |
| Scaling science | **Pretty good** | Delphi at **1e21** and **1e22** close to forecast. 1e23 still training. |
| MoE recipe | **Narrowing** | Recipe at ~1e22 is pretty clear. Need to continue to evaluate at scale / do scaling laws. |
<!--| Evals / post-training | **One clear success** | NemotronTerminal reproduction closed above the reference result |-->

---
layout: default
---

# ![](/icons/flag.svg) Milestone 6 on March 25, 2026

## Kick off a 32B-A4B 10T token MoE run, advance scaling laws, and get 15T+ tokens ready


| Issue | Status | Notes |
|---|---|---|
| CoderForge reproduction | Closed |  |
| Delphi | Open | 1e23 still in flight, ~10 days out |
| NemotronTerminal reproduction | Closed |  |
| infra for 50B MoE | Open | 14/16 P0+P1's closed |
| 32B-A4B 10T MoE launch | Open | Still figured out recipe, pending hardware |
| data sources | Open | small amount of progress |
| synthetic data | Open | some progress from students  |


---
layout: section
---

# Infra

What happened since last time?

---
layout: default
---

# ![](/icons/chart.svg) Iris Cluster Rollout

Iris fully deployed for our TPU clusters and in heavy use now. We’ve run into some (expected) issues as we scale up.

- Worked through a lot of performance and stability issues
- Migrated to a database for cluster state
- Better logging system
- Live worker & task diagnostics, CPU & memory profiles
- Numerous fixes around FD exhaustion, locking, checkpoint/restore

<Box>

We’ll start turning down our Ray clusters over the next few weeks, and moving users towards Iris entirely. We’ll need to dedicate some time to ensuring this goes smoothly, but should as a whole be a lower amount of effort than our Ray cluster babysitting.

</Box>

---
layout: default
---

# ![](/icons/gpu.svg) CoreWeave GPU Rollout

Marin runs on CoreWeave GPUs now, including initial multi-host GPU tests. This involved porting Iris to work on K8S (surprise!), working through MoE kernel issues, and setting up filesystem and data processing pipelines in the new CW cluster.

### Highlights

- Daily MoE canary ferry for regression testing
- Full nemotron download & tokenization in R2 - this highlighted numerous issues with our tokenization performance
- Hardened networking: tunnel reconnects, S3/R2 conditional-write locking

### Moving forward

- We suffered from near-daily breakages of our CW canary runs as they were not effectively tested in CI. We need to get better per-PR testing of GPU runs.
- Multi-host JAX performance is bad, and we’ll need to aggressively work on this.
- More work on R2 performance on CW or using CW S3

---
layout: default
---

# ![](/icons/big-data.svg) Data Tuneup

Dedup scaled to ~10T tokens, tokenization & downloads now reliable

- Fuzzy and exact dedup both at Nemotron scale
- Major Zephyr shuffle rewrite: custom vortex format → Parquet with streaming writers, dynamic batch sizing
- Fixed coordinator hangs and OOM handling
- Nemotron-CC downloads stream as `.jsonl.zst`; tokenization tuned for reliability

---
layout: default
---

# ![](/icons/servers.svg) Stability and Testing

We’ve had a lot of churn with our clusters recently, and need to work on our testing and monitoring infrastructure — [retro doc](https://docs.google.com/document/d/1b9qyytEbg11rqZlJY2V4z4NvKMbAx-ZpC4YReaTPIn4/edit?tab=t.0#heading=h.si4eug1vsap5)

- Added skills for babysitting, debugging, and profiling Iris/Zephyr jobs
- Revamped triage and cleanup workflows
- Need better testing moving forward to work with agents without stress
- MirrorFileSystem for cross-region access with 10GB transfer budget guard


---
layout: section
---

# Training + Scaling

---
layout: three-cols
---

# ![](/icons/tpu.svg) Kernel and Performance Work

::left::

## MoE Block Path

- Expert-parallel dispatch: ring vs ragged-all-all profiling
- On an A3B-like block shape, `#2710` moved from **24.3 ms** at `EP=1` to **17.8 ms** at `EP=8`: about **1.36x** faster on the current path. Also int8 work.
- 32B-A4B bring-up on `v5p-64`: `topk=4` matched-active gave about **+8.8%** achieved FLOP/s over `topk=8`; Capacity Factor another possible knob.
- On GPU, backend-dispatched GMM paths and GPU execution for the routed-expert block. Hooked up DeepEP and got Codex to improve perf.

::center::

## Fused CE

- Fused computation of `cross_entropy(dot(X, W), Y)`
- A separate bucket was fused cross-entropy: block-size retuning, autotune cache/offload, and several TPU/GPU crash fixes under shard_map and tracing.
- This was less about new model capability and more about making the loss path fast and dependable.
- Initial support for autotuning block size when there's a missing block size.

::right::

## Mamba-3

- Mamba is a state space model architecture based on fancy math.
- Initial draft XLA implementation of Mamba3 (both SISO and MIMO variants)


<Box bold>

All of this mainly driven by the new `agent-research` and `agent-profiling` skills.  Still imperfect, but very useful.

</Box>


---
layout: split-left-green
---

# ![](/icons/chart.svg) Delphi Scaling Suite Results

<img src="/charts/delphi_scaling_suite_results.png" alt="drawing" style="width:100%;"/>

::right::


- The **1e21** point landed about **+0.013** macro loss from forecast across three seeds.
- The **1e22** point landed about **+0.005** from forecast.
- The **1e23** run is still training, so the high-compute end is informative but not settled yet.
- Interestingly, the old Marin 32B run is only a little bit off this prediction!

---
layout: two-cols
---

# ![](/icons/chart.svg) 1e23 Progress

::left::

### Train loss

<img src="/charts/delphi-1e23-progress/train_loss.png" alt="train loss" style="width:400px;"/>

::right::

### Eval loss

<img src="/charts/delphi-1e23-progress/eval_loss.png" alt="eval loss" style="width:100%;"/>

---
layout: split-left-green
---

::default::

# ![](/icons/brain.svg) MoE Recipe Work Is Narrowing

- There is now a validated base configuration rather than a pile of disconnected ideas.
- The current recipe has been stress-tested at larger scales, including the question of whether it stays stable at **1e21** and **1e22**.
- The optimizer picture is clearer than it was a few weeks ago, with AdamH still looking better than AdamHR in the matched comparison.


::right::

## Performance follow-ups

- Quantile balancing looks usable again after the original performance penalty was reduced.
- Memory overhead from routing and sorting is still an active area of work.
- Will need more tuning of EP for launched runs and probably some profiling.

---
layout: split-left-green
---

# ![](/icons/chart.svg) MoE Progress Frontier Over Time


<img src="/charts/moe-progress-frontier/paloma_c4_frontier.png" alt="drawing" style="width:600px;"/>

::right::

- Dashed line is the best dense / non-MoE result under **3e18** model FLOPs: **1.1564**.
- Bucketed into **6-hour** time windows, the MoE frontier keeps improving through **March 24**.
- Best-so-far loss reaches **1.0407** in the **March 24 00:00 PT** bucket.

<Box bold>

Next step: kick off the **1e22** run this week.
</Box>

---
layout: split-left-green
---

# ![](/icons/chart.svg) The New MoE Recipe Is Better At Matched Step

<!--![](/charts/moe_step_compare.png)-->
<img src="/charts/moe_step_compare.png" alt="drawing" style="width:470px;"/>

::right::

- At roughly **10.7k** steps, the current recipe is at about **1.0167**.
- The current baseline is about **1.0288** at the same step.
- An earlier comparison point is about **1.0478**.

<Box bold>

This is the cleaner apples-to-apples read: the current recipe is better at the same training budget, not just better on wallclock frontier.

</Box>

---
layout: two-cols
---

# ![](/icons/hammer.svg) What Should Be True by the Next Review?

::left::



## Close the March gaps

Finish the remaining milestone 6 surfaces:
  - Large-run infra
  - MoE launch readiness
  - Data sources
  - Synthetic data


## Solidify infra

- Fewer manual recoveries
- Cleaner launch paths
- More repeatable full runs
- Automated babysitting

::right::

## Next work
  - Improve GPU performance and stability, especially multinode
  - Fully automated babysitting and agent-driven testing of ideas
  - Continue to improve MoE recipe, get scaling curves
  - Hopefully actually kick off the 30B-A4B 10T run on MoE recipe!
