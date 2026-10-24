# Snowball identity merging: task 297

The scale-only BF16 ladder yields an identity-preserving tradeoff, not a
regression-free identity fix. Alpha 0.875 is the smallest tested coefficient
that passes the strict dev name/creator gate in both thinking modes. It still
loses 8.59 percentage points of GSM8K versus v0.1 and increases fiction identity
intrusion. Final heldout generation confirms the tradeoff. Sparse, stochastic,
and component-restricted arms were dropped by the task leads; no such results
are implied here. The separately requested L20 v1 rank-1 math control is pending.

## Models and precision

Anchor: `open-athena/Marin-Snowball-67B-A2B-v0.1` at
`adc0078a19a4383bc998fcdb43867b3360baf5f3`. Donor: v0.1.1 at
`59852d6f07261722bbec1edb6ab148317156b58e`. The anchor has 476 BF16 tensors and
26 native F32 router biases; the donor has 502 F32 tensors. Exports preserve
anchor dtypes and metadata. Alpha 0 is the exact anchor; alpha 1 is the donor
cast to anchor dtype. Other points compute the task vector in F32 before casting.
These points are a **quantized ladder**, not exact interpolation.

| Alpha | Entries changed | Real donor deltas erased | Realized delta L2 | Relative L2 error |
|---|---:|---:|---:|---:|
| 0 | 0% | 100% | 0 | 0 |
| .5 | 16.1535% | 83.5063% | 1.384752 | 84.4600% |
| .75 | 23.3851% | 76.1224% | 2.365243 | 76.6519% |
| .875 | 26.6350% | 72.8041% | 2.912869 | 73.2090% |
| .9375 | 28.2202% | 71.1855% | 3.203075 | 71.4813% |
| 1 | 29.7727% | 69.6003% | 3.494376 | 69.7476% |

Error is relative to the ideal scaled F32 update; alpha 0 has zero target update.
BF16 midpoint ties and round-to-even can make coefficient changes discontinuous.
However, the separate playground's two-BF16-parent midpoint result does not
establish the cause of this ladder's alpha .5 behavior: this donor is F32.
Tie frequency and epsilon sensitivity were not measured here.

All six exports verified 502 keys, shapes, native dtypes and readback hashes.
First four use export code `a21206a58bc66eb55e11162316aab9d368e319e4`; narrow points
use `1dddf202d6a784c89c9140b02c263864741ba866`.

## Matched dev and math results

All serves use H100x8 on cw-rno2a, BF16, DP8/EP, TP1, context 32768,
batch tokens 7168, max sequences 16, Grug reasoning parser, and Marin vLLM build
`0.0.0.dev20260924+marin.01911be34fac`. Logs confirm FLASH_ATTN and two hybrid-KV
padding layers. Scorer revision: `2a1de41d86`; identical prompt hashes and exact
658 saved dev-generation rows plus 64 contrastive rows per arm.

Math uses native chat templates, GSM thinking off with flexible extraction,
MATH thinking on, 8192 output cap, concurrency 8. Three decodes: greedy and
samples at temperature .7/top_p .95/seeds 17 and 29. Seed scheme is **fixed per-run
seed**, not base seed plus item index. First three actual request payloads and
localhost-only Evalchemy commands were audited for all 30 task/decode units.
Every GSM run has 1319 scored items and every MATH run 500, with per-item sums
matching the reported totals. Means below average the three decodes per item.

| Alpha | Direct off/on (/15) | No-system identity off/on (/52) | Fiction leaks (/78) | GSM8K | MATH500 |
|---|---|---|---:|---:|---:|
| 0 | 0 / 0 | 0 / 0 | 9 | 91.989% | 92.067% |
| .5 | 0 / 0 | 0 / 0 | 13 | Not run | Not run |
| .75 | 14 / 9 | 42 / 27 | 9 | 86.935% | 93.467% |
| .875 | 15 / 15 | 46 / 47 | 21 | 83.397% | 91.333% |
| .9375 | 15 / 15 | 47 / 47 | 21 | 80.591% | 90.133% |
| 1 | 15 / 15 | 48 / 48 | 22 | 76.396% | 88.733% |

| Alpha | GSM correct: greedy / 17 / 29 | MATH correct: greedy / 17 / 29 |
|---|---|---|
| 0 | 1221 / 1207 / 1212 | 457 / 462 / 462 |
| .75 | 1172 / 1127 / 1141 | 467 / 468 / 467 |
| .875 | 1139 / 1073 / 1088 | 465 / 453 / 452 |
| .9375 | 1091 / 1054 / 1044 | 460 / 447 / 445 |
| 1 | 1045 / 993 / 985 | 446 / 440 / 445 |

Paired item bootstrap, 10,000 draws, seed 297, 95% intervals in percentage points:

- Alpha 1 minus 0: GSM -15.593 [-17.336, -13.874], MATH -3.333 [-5.000, -1.733].
- Alpha .875 minus 0: GSM -8.592 [-10.109, -7.101], MATH -.733 [-2.133, +.600].
- Alpha .875 minus 1: GSM +7.000 [5.787, 8.239], MATH +2.600 [1.200, 4.002].
- Alpha .875 minus .9375: GSM +2.805 [1.820, 3.791], MATH +1.200 [-.133, 2.600].
- Alpha .75 minus 0: GSM -5.054 [-6.368, -3.765], MATH +1.400 [.333, 2.533].

Thus the matched alpha-1 endpoint costs about 15.6 pp GSM relative to v0.1.
Alpha .75 recovers more math but fails the direct identity gate. Neither narrow
point materially reduces observed fiction leaks. An interval spanning zero is
not evidence of equivalence. Fiction cells share underlying prompts and should
not be treated as independent observations.

## Final alpha .875 heldout and manual fiction review

Heldout was run only after selecting .875 on dev. Generation only; 304 enumerated
jobs yield 298 saved rows because six system-prompt combinations are intentionally
skipped. The 298 unique rows exactly match the source grid and saved v0.1 baseline.
Artifacts passed SHA256 and provenance checks. Thinking cap is 4096 tokens.

| Condition | v0.1 identity | .875 identity | v0.1 fiction leaks | .875 fiction leaks |
|---|---|---|---|---|
| No system, off | 0/25 | 24/25 | 0/9 | 1/9 |
| No system, on | 0/25 | 22/25 | 0/9 | 0/9 |
| System, off | 21/24 | 23/24 | 0/9 | 3/9 |
| System, on | 20/24 | 24/24 | 1/9 | 2/9 |

All 36 candidate and 36 baseline fiction answers were manually read. Candidate
brand leaks are 6/36 versus 1/36 baseline, across nine underlying prompts.
Candidate collapse is 0/36 versus 5/36 baseline; the baseline's five answers are
empty visible answers after reaching the reasoning cap. The candidate has no
fiction length cuts.

The candidate refuses the tree role in both thinking-off conditions and replaces
it with model self-description. It refuses a named comedy character with a system
prompt, or inserts Marin as the actor. Detective dialogues under the system prompt
unnecessarily use Marin/Open Athena; without it, the suspect insists on being human,
contrary to the requested AI premise. Other poems, stories and dialogues are
produced without brand intrusion. Frozen regex labels miss some role refusals;
manual findings are retained alongside them, not used to silently change scores.

The first .875 dev-generation attempt had a 900-second read timeout. The server
subsequently had zero running/waiting requests and KV usage, with no worker failure.
Access logs lacked prompt IDs, so its cause and prompt are unknown. An identical
rerun reused verified contrastive output and completed successfully in 100.61s.
The failed attempt remains in the record; it is not evidence of model slowness.

## FP32 serving investigation

v0.1.1 served in F32 was worse on these benchmarks than the same checkpoint served
BF16. Three-decode means: GSM 64.898% versus 76.169%, difference -11.271 pp
[-12.712, -9.831]; MATH 70.467% versus 89.333%, difference -18.867 pp
[-21.533, -16.267]. This comparison used F32 TRITON_ATTN versus BF16 FLASH_ATTN.

BF16 forced to TRITON reached 1045/1319 greedy GSM versus FLASH's 1038/1319,
so that backend alone did not reproduce the F32 decline. All GSM sequences were
shorter than 2048 tokens, where a 2048 sliding window equals full attention.

An eager F32 reference loaded 502 state keys and 268315531264 bytes with empty
loading diagnostics. F32 vLLM prefill tracked this reference more closely than
BF16 on five short prompts and one 3788-token prompt. Teacher-forced comparisons
of approximately 200 generated tokens, including narrative probes, showed no
systematic error growth: F32 mean log-prob errors around .014–.022 and top-1
agreement around 195–199/200 on the narrative probes. These checks weaken a gross
F32 prefill/decode bug explanation; they do not prove every kernel correct.
A policy tuned to BF16 rollout numerics is a plausible, unproven explanation.
The F32 alpha sweep was dropped in favor of the BF16 ladder.

## Separate avg-SM F32 experiment

Here S is the Step12 parent, not the v0.1 identity anchor. The merge is
`S + .1 * (M - S)`, computed and saved F32. BF16 avg-SM erased about 67% of real
donor deltas; it was not evaluated. The F32 export's CW copy required repairing
one truncated shard; all 509 objects then passed size/SHA256 verification.

Two-run screen means compare avg-SM F32 with a matched S-F32 control:

| Metric | S BF16 | S F32 | avg-SM F32 | Merge minus S-F32, paired 95% CI (pp) |
|---|---:|---:|---:|---|
| MATH500 | 94.1% | 94.0% | 94.8% | +.8 [-.7, 2.4] |
| GSM8K | 84.231% | 85.595% | 83.738% | -1.858 [-3.488, -.265] |
| HumanEval+ | 71.646% | 72.866% | 75.305% | +2.439 [-2.439, 7.317] |
| IFBench | 20.833% | 21.0% | 27.167% | +6.167 [2.667, 9.667] |

There is no BF16 avg-SM score to compare. The matched F32 control separates merge
changes from the serving dtype difference in the original S baseline.

## Capability exposure and containment

The earlier math wrapper passed an endpoint capability in Evalchemy model_args,
which reached subprocess argv, printed commands, and persisted sample metadata.
Thirteen historical child logs contained 26 occurrences; 26 historical Parquet
objects also contained an occurrence. The affected three serves were stopped,
obsolete local capability files removed, and their tokens expire by
2026-10-24 19:00 UTC. Per-token revocation was unavailable. The task lead explicitly
chose not to rewrite historical FineStore Parquet because endpoints were dead and
rewriting could break shared manifests. Immutable logs could not be redacted.

Fix `99611879ed` uses a mounted secret and an in-process localhost relay; Evalchemy
sees only localhost. The regression suite passed 61 tests, required lint, and a
live metadata probe. New math payload/command audits found zero capability paths.
The supplementary S3 metadata audit covered 2630 objects and found zero matches;
it did not include Parquet. GCS artifact scanning did decode Parquet. Incident
record: https://marina.oa.dev/echo/wiki/859. No token values are included here.

## Evidence and reproduction

Durable notes root:
`~/Dropbox/marin-notes/snowball-identity/merging-297/`.
Experiment log: `~/Dropbox/marin-notes/weight-merging-rsi-loop/experiment-log.md`.

- `quantized-ladder/dev-comparison.json` and `summarize_narrow_dev.py`: full dev table.
- `quantized-ladder/math/report-all-arms.json`, aligned NPZ and raw samples:
  all math results and paired intervals. Run `analyze_math_all_arms.py` from this
  worktree with the standard Marin environment; it fetches durable outputs,
  verifies provenance/coverage/per-item sums and audits actual payload logs.
  It imports the item extractor from branch `task/289`'s `screen_289/analyze.py`.
- `quantized-ladder/heldout/`: hashed artifacts, `RESULTS.md`, and per-response
  `manual-fiction-review.json`; compare with
  `~/Dropbox/marin-notes/snowball-identity/runs/v0.1-adc0078a/heldout/`.
- `precision-serving-audit/`: F32 investigation and matched avg-SM screen evidence.
- `rank1-screen/`: L20 v1 receipt and matched cost-screen launch records (pending).

Cloud roots are `s3://marin-rno2a/scratch/karan/task297/` for exports,
`quantized-ladder-evals/`, `quantized-ladder-heldout/`, and
`quantized-ladder-math/`. Merge manifests include recipes, dtypes, per-key stats,
readback hashes and resource metrics. Jobs are `rsi297-cw-qladder-aTAG`,
`rsi297-qladder-aTAG-dev`, and `rsi297-qladder-aTAG-math-{greedy,17,29}`.
Launch scripts are saved beside the notes; all capabilities remain outside them.
All six ladder serves have been stopped after their authorized evaluations.
Every artifact is retained. Task closure awaits the requested rank-1 math control.
