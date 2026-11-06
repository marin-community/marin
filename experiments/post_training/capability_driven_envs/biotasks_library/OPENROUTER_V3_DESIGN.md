# BioTasks v3 reference/path/contract wave

Frozen internal-only v3 TaskSpecs:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v3/snapshots/reference-path-review-1/
```

`taskspecs.jsonl` SHA-256:
`b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6`.
There are 305 records: **230 provisionally eligible**, 58 originally disputed,
and 17 newly quarantined consensus-disagreement leads. Four frozen references
were independently corrected: microarray preprocessing, CIGAR arithmetic,
variance-stabilizing transformation, and alpha diversity. The alpha diversity
task now explicitly uses the standard Hill q=2 definition (`1/lambda`).
None of the other 226 are independently reference-verified. Every edited
TaskSpec has a new ID/digest; v1 and v2 snapshots and scores are untouched.

The corrected working directory is `/` for 170 eligible tasks whose prompts
use `workspace/...` relative paths; files continue to live at
`/workspace/...`. The other 60 retain `/workspace`. The v3 answer format
`json_object_in_final` accepts one unambiguous JSON object in a terminal reply
and discards surrounding prose or fences. It **rejects** multiple independent
objects, duplicate keys, nonfinite JSON, missing JSON, and incomplete
responses; it never inspects the private reference to select a candidate. The
model-visible instruction remains unchanged from v2, so the existing
generation can be reused when *all* model-visible inputs and sandbox settings
are identical. It is a more permissive grading contract, not a retroactive
change to any v2 score.

Generation deduplication proof (`run_openrouter_first_wave_v3.load_cohort`):
compare v2/v3 conversation events, submission instruction, final/interaction
tools and bindings, resources, working directory and all environment
requirements, private grader, and output paths. Only the new task ID and
reference-blind JSON extractor may differ for reusable pairs. **60 tasks × 12
models** qualify structurally, but an old pair is reused only if it has a
route-verified non-infrastructure result. The offline reuse audit found **690
reusable generations**; **30** old pairs need a fresh attempt due to missing
infrastructure. Among 690 reused results, 66 grades changed under v3's
extractor without new model calls; original raw records and scores remain
addressable through the new response ledger. The other **170 tasks × 12
models** have changed sandbox settings and require new generations. Total
intended new model-task pairs: **2,070**; expected final coverage: **2,760**
selected pair records (including missing/ungraded, never miscast as zero).

The same 12 pinned model/provider routes, eight turns, 4,096 output tokens
per turn, 32,768 total tokens, temperature 0, top_p 1, and ShellSim apply.
One smoke trial per route was recorded separately before the main wave. Those
12 attempts are reused by the main run, not repeated. The new OpenRouter
campaign has a **$50 pre-batch spend ceiling** with a frozen initial key-usage
baseline; do not construe this as a real-time per-token cutoff inside a batch.
The BaseTen Nemotron route returned a 429 even in its single-request smoke;
its route remains pinned and incomplete responses count as missing, not
incorrect. The adduct-correction smoke also exposed a residual wording issue:
Qwen used lowercase `dbe` where the frozen grader expects `DBE`; its v3 score
is zero, not repaired by guesswork. Approximately 36 eligible instructions
lack a literal spelling of every expected top-level key. Treat these as a
follow-up review queue; v3 does not claim that every instruction is fully
validated. No CoreWeave/gVisor population is mixed into ShellSim.

Private new attempt root:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v3/calibration/openrouter-first-wave-v3/
```

The finalizer publishes `snapshots/complete-1/` **last**, after auditing every
new, smoke and reused pair and verifying source checksums, identities,
routes, and eligibility. This is a second *diagnostic* wave; IRT and scientific
release remain blocked on independent task validation, unresolved formatting
and tool failure categories, and residual infrastructure missingness.
