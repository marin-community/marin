# Notebook: do reasoning-length and give-up filters reduce runaway reasoning?

Working record for the data-patch experiment that follows marin-community/marin#11188.
The approved plan and decision rule are summarised in the first entry; later entries are appended.

## 2026-11-13: question, baseline and design

**Question.** The 10.05 SFT model runs past 100k characters on 27-32% of AIME24 outputs and 45-53% of
OlympiadBench outputs, and most of those outputs are wrong. Does removing long-reasoning and give-up
documents from the SFT mix reduce this?

**Baseline evidence** (existing eval rollouts, scanned with the scripts in gist
`5ce8d65b840d6a7c273977fde8b9a1cf`; settings temperature 0.7, 73,664-token cap):

| Checkpoint | AIME24 outputs >= 100k chars | AIME24 no boxed answer | OlympiadBench outputs >= 100k chars |
|---|---|---|---|
| Repro step 222 | 37.0% | 25.3% | 53.3% |
| Repro step 471 | 43.0% | 29.7% | 50.7% |
| Repro step 1024 | 34.3% | 20.7% | 47.2% |
| Repro step 2048 | 32.3% | 18.7% | 51.3% |
| 10.05 (seven runs) | 27-32% | 12-17% | 45-53% |

The habit is at full strength by step 222 of 4,750, so a 222-step run is a valid readout.

**Design.** Three arms of the historical 262,144-context cold-start recipe, 222 updates each, same seed
and weights, differing only in the cache catalog:

| Arm | Catalog |
|---|---|
| `control` | `full-mixture/2026.10.24.2/cache_catalog.json`, unchanged |
| `cap32k` | `math_v3`, competitive coding (Python, C++) and OpenThoughts4 with documents over 32,000 reasoning tokens removed |
| `cap32k-giveup` | as `cap32k`, plus give-up endings removed from `math_v3` and competitive coding |

**Decision rule.** Primary: the share of outputs >= 100k characters on AIME24 and OlympiadBench at step 222
is at most half the control's, and the gap exceeds the gap between the fresh control and the published
step-222 checkpoint. Guards: AIME24, OlympiadBench, MATH-500, GSM8K, HumanEval+ and IFBench accuracy show
no decisive loss under the #10197 pairwise rule.

**Changes from the approved plan.**

- Filtering is done on the tokenized stores (`reasoning_store_filter.py`), not by rebuilding sources. Kept
  documents are byte-identical to the control's.
- The third arm does not include the Terminus or multilingual hygiene patches; they are not built.
- Replay junk removal is out of scope; it is tested under marin-logbooks#40.
- Permanent checkpoints land at steps 100, 200 and 222 (the recipe keeps every 100), not 50, 100 and 222.

## 2026-11-13: filter audit on sample shards

Source commit `c6a5b1d520` (filter), dry runs from a workstation on shards `math_v3` 7 and 101,
`competitive_coding_python` 13, `competitive_coding_cpp` 29, OpenThoughts4 0.

Reasoning length (longest think span, tokens):

| Source | Rows over 32k | Tokens in those rows |
|---|---|---|
| `math_v3` | 8% | 31% |
| competitive coding | 51% | 84% |
| OpenThoughts4 | 78% | 95% |

Give-up rules, read by hand (12 hits per shard):

- Coding: about 21 of 24 real ("I cannot produce a correct solution", placeholder code). Two false positives
  came from `# placeholder` in ordinary code and "hope it's okay"; both patterns were narrowed afterwards.
- Math: the rule mostly matches "Given the time, I'll ...". About 17 of 24 are unsupported guesses; the rest
  stop verification early after a real derivation. Hit rate 3.7-3.9% of rows and 8.0-8.6% of tokens. The rule
  is therefore defined as "reasoning ends by citing time pressure or guessing", not "answer is wrong".
- Math give-up hits are short (median 4.3k reasoning tokens); coding hits are long (median 86k). Give-up and
  length are separable in math and not in coding.

**Interpretation.** A 32k cap is a large intervention on coding and OpenThoughts4. It is kept for this
experiment because a strong dose gives the clearest test. For a production mix it implies reweighting.

## 2026-11-13: filtered stores

Experiment commit `5b48d94244`. Jobs `/dml/sft-filter-<variant>-<source>` on `cw-us-east-08a`; outputs under
`s3://marin-us-east-02a/tmp/ttl=30d/sft-data-patch-11188/stores/<variant>/<source>/train`, each with a
`selection.json` listing every dropped row and its reason.

| Store | Rows kept / total | Tokens kept / total |
|---|---|---|
| `cap32k/ot4` | 8,264 / 36,666 | 0.169B / 3.354B |
| `cap32k/cp_python` | 163,637 / 333,399 | 2.091B / 13.056B |
| `cap32k/cp_cpp` | 164,092 / 331,927 | 2.133B / 12.983B |
| `cap32k-giveup/cp_python` | 162,806 / 333,399 (831 give-up) | 2.079B / 13.056B |
| `cap32k-giveup/cp_cpp` | 162,889 / 331,927 (1,203 give-up) | 2.120B / 12.983B |

With weights pinned to the control's, 222 steps draw about 0.21B positions from OpenThoughts4, more than
the 0.169B the filtered store holds, so that source wraps about 1.2 times in the patched arms.

## 2026-11-13: cluster and priority

The long-loop runner pins `cw-rno2a` at production priority. Reno had 3 of 504 H100 free with 376 held by
production work, so no band could place the gang there. `cw-us-east-02a` had 22 free and 224 held by batch
jobs. The arms run there on 16 nodes (128 H100, the geometry of the original step 0-471 run) at the default
interactive priority, through `patch_arm.py`, which calls the long-loop's `sft_run` directly.

## 2026-11-13: launch attempts for the control arm

| Attempt | Cluster, priority | Outcome |
|---|---|---|
| `/dml/sft-patch-11188-control` | `cw-us-east-02a`, interactive | Never accepted. The peer accepted no handoff submitted after 18:04 UTC (Echo wiki 984). Cancelled at 19:41 UTC. |
| `...-control-cw-rno2a` | `cw-rno2a`, interactive | Gang started 19:44 UTC, evicted 19:57:58 UTC by the 256-H100 production gang `/benfeuer/long-loop-exp2-full-2026.11.13.11` before the first step. |
| `...-control-cw-rno2a-r2` | `cw-rno2a`, interactive, resumable | Gang never admitted behind production work. Cancelled when production priority on Reno was approved. |
| `...-control-cw-rno2a-r3` | `cw-rno2a`, production | Step 1 committed at 20:33 UTC, then `TypeError: wait_at_barrier(): incompatible function arguments` in the first-checkpoint restore check. Cancelled at 20:54 UTC. |
| `...-control-cw-rno2a-r4` | `cw-rno2a`, production | Running from commit `ae0bc4fb54`, which has the checkpoint fix; resumes from the step-1 checkpoint. |

**Changes made because of these attempts.**

- `patch_arm.py train` restores from the arm's own checkpoints and carries a preemption retry budget, so an
  eviction costs at most the time since the last half-hourly checkpoint.
- `lib/levanter/src/levanter/checkpoint.py` rounds the async checkpoint timeout to an integer. The long-loop
  branch made it a float config field; JAX multiplies it by 1000 and passes it to `wait_at_barrier`, which
  rejects a float. Main passes an integer constant and does not have the bug.

**Evaluation recipe.** The eval launcher on this branch caps OlympiadBench output at 32,768 tokens, below the
baseline's average of 38.9k generated tokens. Evaluation therefore uses the baseline launcher at `e0cd563`
with the campaign's six Evalchemy configs (AIME24 and OlympiadBench at temperature 0.7, seed 42, 10 repeats;
MATH-500, GSM8K 0-shot, HumanEval+, IFBench at the server default), served DP8 on one 8xH100 node with
`max_model_len` 73,728. The published step-222 checkpoint was evaluated without a `generation_config.json`;
these exports write one with stop tokens, so comparison with the published numbers is looser than between arms.

## 2026-11-14: control arm finished and evaluated

`/dml/sft-patch-11188-control-cw-rno2a-r4` committed permanent step 222 at 05:55 UTC (8 h 49 min of training,
no failures or preemptions; final logged loss 0.583). Exported in the stacked-expert layout to
`runs/control/serve-step222/hf` and evaluated as run group `sft11188-control-s222` on Reno.

| Benchmark | Fresh control, step 222 | Published repro, step 222 |
|---|---|---|
| AIME24 accuracy / outputs >= 100k chars / no boxed | 71.3% / 33.3% / 24.3% | 72.3% / 37.0% / 25.3% |
| OlympiadBench accuracy / outputs >= 100k chars / no boxed | 22.3% / 48.3% / 24.7% | 25.0% / 53.3% / 22.3% |
| MATH-500 | 95.0% | |
| GSM8K 0-shot | 81.2% | |
| HumanEval+ pass@1 | 68.3% | |
| IFBench strict prompt | 19.7% | |

The fresh control reproduces the published checkpoint within 3.7 points (AIME24) and 5.0 points
(OlympiadBench) on the long-output share. Passing the primary rule therefore needs at most 16.7% (AIME24)
and 24.2% (OlympiadBench) long outputs in a patched arm.

**Rehearsal on control step 100** (run group `sft11188-control-s100b`): AIME24 69.7% / 36.0% / 26.3%,
OlympiadBench 22.0% / 49.0% / 23.7%, MATH-500 92.8%, GSM8K 86.3%, HumanEval+ 65.2%, IFBench 19.7%. The habit
is at full strength by step 100.

**Pipeline fixes found by the rehearsal.**

- Export memory peaks at about 420 GB; 64 GB and 400 GB were OOM-killed. CPU nodes on `cw-us-east-02a` are
  full of other jobs' file cache and stall allocation (40+ minutes); a Reno GPU node restores in 5-12 minutes.
- The baseline launcher at `e0cd563` is rejected by the controller as too old; evals run from main at
  `12f0ceff06`, the last commit with the Evalchemy route. The six benchmark configs are unchanged.
- The eval stack's vLLM release rejects per-expert tensors (`Unexpected GrugMoE weight
  'model.layers.2.mlp.experts.72.down_proj.weight'`); `patch_arm.py export` now writes stacked tensors.
