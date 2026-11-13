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
