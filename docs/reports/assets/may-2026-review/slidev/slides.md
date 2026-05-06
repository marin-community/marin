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

# ![](/icons/calendar.svg) Status Snapshot on May 7, 2026

| Area | Status | Read |
|---|---|---|
| Infra | **TODO** | Iris is now the default path; Ray was removed from Marin. Fill in current production reliability. |
| CoreWeave / GPU | **TODO** | Multi-host GPU canaries and CW cluster wiring landed; current NCCL/JAX status needed. |
| Data + Zephyr | **TODO** | Datakit normalized around standard Parquet, source registries, and smoke ferries. Fill in latest throughput. |
| Evals | **TODO** | Perplexity-gap and raw slice coverage expanded materially. Decide which slices changed decisions. |
| Training + scaling | **TODO** | MoE kernel/config work and Delphi eval/blog work landed. Fill in current run outcomes. |

---
layout: default
---

# ![](/icons/flag.svg) Since Late March

- Ray was removed from Marin as an execution path.
- Iris got budgets, priorities, preemption, better scheduler hot paths, and a real status/debug surface.
- CoreWeave moved from bring-up into recurring smoke/canary coverage.
- Datakit and Zephyr now have a clearer download -> normalize -> dedup -> consolidate -> tokenize path.
- Evals grew from a few headline suites into a broader diagnostic/perplexity-gap matrix.
- Delphi moved from scaling-ladder training toward downstream evals and blog figures.

<Box>

TODO: Convert this into 3-4 measured claims once tomorrow's run/job status is known.

</Box>

---
layout: section
---

# Systems

Iris, CoreWeave, and logs

---
layout: default
---

# ![](/icons/servers.svg) Iris Replaced Ray

- `#5138` removed Ray from Marin, with related cleanup in `#5132`, `#5131`, `#5089`, `#5087`, `#5076`, `#5031`, and `#5028`.
- Iris now carries user-facing scheduling concepts: budgets and priorities (`#4096`, `#5081`), preemptible jobs (`#5083`), manual slices (`#5078`), same-variant preemption (`#5240`), and routing docs (`#5418`, `#5426`).
- Scheduler and API hot paths were tightened: lightweight job-state polling (`#4209`), raw SQL / denormalized scheduling rows (`#4181`, `#4264`), ListJobs/ListWorkers pagination and filtering (`#4558`, `#4703`, `#5025`, `#5384`, `#5454`), and cached resource scalars in `can_fit` (`#5412`).
- Dashboard work moved debugging closer to the jobs: state filters, child job sorting, task summaries, status markdown, endpoint proxying, and task-resource history.

---
layout: default
---

# ![](/icons/gpu.svg) CoreWeave / GPU Path

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

- `datakit` was bootstrapped for consolidated downloads (`#4142`) and now has normalize, source registry, staged workflows, and smoke ferries (`#4188`, `#4598`, `#5105`, `#5450`).
- The pipeline now writes normalized Parquet, supports split main/duplicate outputs, exact dedup in normalize, MinHash / fuzzy dedup job separation, and per-shard resume for MinHash attrs (`#4596`, `#4610`, `#4876`, `#4893`, `#5397`).
- Zephyr got stronger execution semantics: subprocess-per-shard and later inline shard execution (`#4522`, `#5282`), shuffle/external-sort fixes (`#4695`, `#4782`), stage counters (`#4189`, `#4212`, `#5063`), schema diagnostics (`#5136`, `#5142`), and byte-based scatter heuristics (`#5340`).
- Data sources added or normalized include rollout pipelines for six datasets (`#4329`), NSF abstracts (`#4516`), HPLT likely non-duplicates (`#4326`), BHL stitching (`#5408`), GAIR/daVinci-Dev (`#5252`), Molmo2-Cap (`#5299`), and public diagnostic logs (`#5121`).

---
layout: default
---

# ![](/icons/chart.svg) Evals and Perplexity Gap

- The raw/perplexity-gap matrix grew across FineWeb2 multilingual, long-tail reruns, npm metadata, UWF Zeek, game/music, Common Crawl WARC/WAT, GH Archive, ASR/OCR noisy text, bio/chem notation, structured text, formal methods, RTL, and raw LM-eval bridge slices (`#5008`, `#5074`, `#5075`, `#5124`, `#5126`, `#5193`, `#5192`, `#5119`, `#5118`, `#5127`, `#5129`, `#5128`, `#5196`).
- Per-model scores are cached before gap diffs (`#5169`), and raw perplexity-gap reports for pairwise LM comparisons landed in Levanter (`#4962`).
- Served LM eval moved toward a cleaner native handoff (`#5285`, `#5322`, `#5325`), and several eval paths migrated off Ray / Fray v1.

<Box>

TODO: Choose the one or two eval findings that changed data or model decisions. Avoid listing every slice in the live deck.

</Box>

---
layout: section
---

# Training + Scaling

MoE, Delphi, and open questions

---
layout: default
---

# ![](/icons/tpu.svg) MoE and Training Mechanics

- MoE recipe work continued through configurable MoE implementations (`#4964`), `MoeAdamHHeuristic` (`#4636`), router-logit fp32 upcasting (`#4234`), Grug grouped-query XSA fixes (`#4315`), and Muon MoE orthogonalization (`#3902`).
- Kernel work landed on both TPU and GPU: ragged all-to-all capacity clipping (`#4359`), sender offsets (`#4867`), Pallas API fixes (`#5347`), and Triton `ragged_dot` (`#4297`, `#5350`).
- Training reliability improved around W&B failures and resume symmetry (`#5332`, `#5415`), checkpoint roots (`#4387`, `#5066`), shuffle defaults (`#5246`, `#5259`), and tokenizer migration (`#4405`, `#4451`, `#4977`).

---
layout: split-left-green
---

# ![](/icons/chart.svg) Delphi Scaling + Blog Work

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

# ![](/icons/flag.svg) Decisions For Tomorrow

| Question | Evidence Needed | Owner |
|---|---|---|
| Is Iris stable enough to keep Ray fully retired? | job failure rate, babysitting load, stuck-job incidents | TODO |
| Is CoreWeave ready for model work beyond smoke tests? | multi-host canary status, NCCL/JAX failures, throughput | TODO |
| Which data pipeline improvements changed throughput or cost? | Zephyr/datakit stage timings, dedup resume data, ferry history | TODO |
| Which eval slices changed a training/data choice? | perplexity-gap deltas by source, raw slice outliers | TODO |
| What is the Delphi headline? | final 1e23 loss/evals, blog figure, forecast error | TODO |

---
layout: default
---

# ![](/icons/rocket.svg) What Should Be True By Next Review

- TODO: Iris reliability target, with a measured incident/babysitting threshold.
- TODO: CoreWeave readiness target, with one named multi-host workload.
- TODO: Data target, with tokens staged and a cost/throughput constraint.
- TODO: Eval target, with a published diagnostic matrix and one action taken from it.
- TODO: Training target, with the next MoE/Delphi milestone and the evidence needed to stop debating it.

