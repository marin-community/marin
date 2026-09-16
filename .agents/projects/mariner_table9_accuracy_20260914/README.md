# Table-9 Accuracy Backfill

## Scope

The user approved accuracy backfill for the completed Proportional and UniMax-8
1e21 checkpoints, then explicitly deferred the 17 multilingual MBPP components
whose exact Table-9 dataset has no execution tests. No companion multilingual
benchmark is substituted. No new training is requested.

The target is 34/51 component counterparts: 14 existing overlap components,
11 Basic Skills/gen2mc tasks, seven Minerva subjects, HumanEval, and Python MBPP.
The overlap uses its historical lm-eval prompts. The 20 new tasks preserve
native BPB prompts and documents. Reports distinguish these protocols and do
not publish a partial Table-9 accuracy macro.

## Frozen Artifacts

- Current backfill plan: `plan_v6e4_r2.json`, canonical SHA256
  `3e429ef8afb9abeb3fc7b2e76327b8f4bbe22a33fac2f8853beeb1c9b0603956`;
  file SHA256 `a1091d3a6c9db164f6b72d73106ec8f9f858f56dabbd2958f8a9bb8a85baede7`.
- Backfill protocol SHA256:
  `e2fe42b0875d66d27c387b15a55230a1980ee78649523460a447aa253a13671b`.
- Superseded v6e plan: `plan_v6e4.json`, canonical SHA256
  `df4b68205435b4793d677498248d37cd43d18cbbd5bc518ba6eb5c513ba227b8`.
- Superseded v5p plan: `plan_v2.json`, canonical SHA256
  `ae071b2b44a8c23319addc0151bb96f333dfe80b89719927339d5e9b05e1dfe8`.
  No artifact identities were overwritten or transferred between hardware plans.
- Existing overlap plan: `../mariner_ladder_accuracy_20260914/plan_v5p_cpu_staging.json`.
- Local coverage: `coverage/coverage.json` and `coverage/components.csv`.
- Durable outputs: `gs://marin-us-east5/experiments/table9_accuracy_20260914`.
- `requests/manifest.json` pins 27,153 documents across 20 tasks, source revision,
  formatting, sampling, primary MC metrics, counts, and payload hashes.
- `future_plan_smoke/` is preparation validation only. Do not launch it: it
  deliberately re-inventories an already evaluated checkpoint.

## Live State At 2026-09-15 01:50 UTC

Both v6e r2 parents passed regional exact prompt/gold verification and the new
tokenizer preflight for both actual checkpoints, then released their two
children. All four children are queued for v6e-4 in us-east5-b.
No new model accuracy or generations are complete yet; no capacity-based ETA.
Full evaluations have not been released and do not start automatically.
The scheduler reports no free matching TPUs and is waiting for worker scale-up.

- `/calvinxu/table9-accuracy-choices-v6e4-canary-20260915-r2`
- `/calvinxu/table9-accuracy-generation-v6e4-canary-20260915-r2`

The original v6e canaries failed at tokenizer startup, not for memory. Direct
GCS checkpoint URIs were incorrectly passed to Marin's Hub/local tokenizer
loader. All four failed before loading weights or running memory probes, and
neither backfill output root existed at recovery. The fix stages tokenizer
assets with the existing fsspec-capable HF loader into a persistent local
cache, then opens that same directory through Marin's loader. CPU parents
preflight both interfaces and full-window tokenization before releasing TPUs.
Only the runner source pin changes in r2; hardware and scientific settings are
unchanged. Fieldbook retains the six failed jobs and all retry links.
Exact current commands: `launch_v6e4_r2_commands.txt`.

The two r2 v5p roots and their four children were cancelled after verifying all
children had attempt ID -1 and no assigned worker. No completed TPU work was
discarded. Exact replacement commands are in `launch_v6e4_commands.txt` and
Fieldbook, with retry links for both parents and all four children. Parents
remain on nonpreemptible east5-a CPU; child TPUs are explicitly east5-b.
Both uploaded workspace bundles were 23.2 MiB.

Only TPU type, child zone, and the runner source pin changed from `plan_v2`.
Checkpoint objects, requests, prompts, generation settings, batch 8, length
8192, and runtime versions are unchanged. The runner also fixes its stale call
to the exact-continuation token counter; normalization is unchanged.

The original parents without `-r2` were cancelled while all four children were
queued. Their declared batch size was not applied to the four JAX devices of
v5p-8. The replacement enforces actual batch size 8. Original plans and output
identities remain separate. No completed TPU work was discarded.

Fieldbook experiment: `exp_01kvvvv6zxrf0j7tkp4f7k6y66`.
Proportional result run: `run_01m2g8d93rsaq6p46egp60khvf`.
UniMax-8 result run: `run_01m2g8d99y7rgmb8954nbrzw32`.
Current parent jobs: `job_01m2hbzfs1zvmzavgwrs7tkp3h` (choices) and
`job_01m2hbzgk988nzgwbdksp1d3wn` (generation). Exact commands are in Fieldbook.

## Validation And Next Actions

Sixteen focused tests pass, including canary/full isolation, artifact tamper
rejection, incomplete-sample rejection, generation-versus-grading state,
native continuation normalization, memory-probe request lengths, and
checkpoint/hardware isolation of probe markers. The tokenizer regression
reproduced the exact old failure through an in-memory remote filesystem; the
fix preserves token IDs, special tokens, and local assets without downloading
weights. Touched-file lint/type
checking passes, and both exact launch commands passed the east5 guard.
Repository-wide tests were not run: the test selector includes all tests
because the existing branch/worktree has 5,027 changed files.
The future-checkpoint preparer previously ran
twice against the existing permanent step-22056 export and produced identical
plans. No duplicate evaluation was submitted for that smoke test.

Before real task canaries, each child probes the actual checkpoint with eight
full-window requests. Choices use eight 8,192-token likelihood sequences;
generation uses eight 8,128-token prompts and 64 decode tokens, with the
original decoder configuration. This tests long-context memory, not every
possible full-corpus dynamic batch. TPU backend/device count and memory stats
are recorded separately under `memory_choices` / `memory_generation`; these
markers never count as task scores. Live memory fit remains unverified.
If v6e-4 fails specifically for memory, prepare and freeze a v6e-8 plan with
the same scientific settings, preserve failed-job lineage, and rerun canaries.
Do not duplicate queued jobs or reduce batch/context to obtain a pass.

The current grader passed 18 known-correct references: two per math/code task.
Separate sandbox probes distinguished correct, incorrect, timeout, and blocked
network cases. Early reference-audit artifacts from older grader identities
are superseded: one used exact-match formatting as a correctness gate; another
mistook a stopped Docker daemon for a failed program. The corrected grader
inspects container state and errors on infrastructure failure. Math reports
retain both exact_match and math_verify, using math_verify for correctness.
This is a reference smoke test, not a full reference-corpus audit or TPU test.

When canaries finish, inspect every retained task output and grade the two-
document generation canaries before releasing full evaluations. Reuse the
registered v6e commands, remove `--canary-documents 2`, and assign fresh full-run
parent names. Validate east5 placement and register those jobs before launch.
The runner requires both the memory marker and all two-document inference
artifacts before full release. Grading and retained-output review remain
manual release gates.
Release both selected checkpoint rows concurrently; do not duplicate a queued
child to bypass capacity. Hash-valid completed tasks are reused on retry.

Grade with `experiments.domain_phase_mix.grade_table9_accuracy --plan PLAN`;
add `--canary-documents 2` for canaries. Docker must be running. Use
`uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1`
before `python -m` for grading and reporting. Generated Python executes only
inside the pinned, credential-free, network-disabled Docker sandbox.

Refresh coverage with `experiments.domain_phase_mix.report_table9_accuracy
--plan PLAN --overlap-plan OVERLAP_PLAN --output OUTPUT`. Generation markers
alone do not count as scores. At this snapshot both checkpoints are 14 scored,
20 missing, 17 deferred.

For subsequent Qwen3 ladder checkpoints, run
`experiments.domain_phase_mix.prepare_table9_checkpoint --help`. It freezes
both suites from an explicit permanent training step and region-local HF
export; use `plan_v6e4_r2.json` as the backfill template. Submit the overlap suite
plus both backfill modes. This is a reusable
entry point, not an automatic callback added to running training jobs.

## Full release at 2026-09-15 16:10 UTC

Calvin released the full backfill for both completed checkpoints after both r2 canaries succeeded (choices 02:32 UTC, generation 13:47 UTC; the generation reference audit graded both canary sets). Commands in `launch_v6e4_full_commands.txt`: the r2 canary commands with `-full-20260915` job names and no `--canary-documents`, validated by the east5 guard; bundles 23.2 MB. Parents `/calvinxu/table9-accuracy-choices-v6e4-full-20260915` and `/calvinxu/table9-accuracy-generation-v6e4-full-20260915`, each releasing one v6e-4 child per checkpoint in us-east5-b (27,153 documents per checkpoint per mode). Submission logs `submit_table9-accuracy-*-full-20260915.log`. After the generation children finish: grade locally (`grade_table9_accuracy --plan plan_v6e4_r2.json`, Docker required), then `report_table9_accuracy --plan plan_v6e4_r2.json --overlap-plan ../mariner_ladder_accuracy_20260914/plan_v5p_cpu_staging.json --output ...`.

## Generation release blocked by a protocol drift, fixed in r3 (2026-09-15 23:12 to 23:22 UTC)

`/calvinxu/table9-accuracy-generation-v6e4-full-20260915` failed its release guard: "Full release requires the two-document canary: proportional_1e21-2f1a48/minerva_math_algebra". Cause: `evaluate_task` passed the plan's own `spec["generation"]` dict to each lm-eval `Instance`; the harness's `_process_and_tokenize_stop_sequences` calls lm-eval's `handle_stop_sequences`, which appends the EOS string to the shared `until` list in place, so `digest(protocol(plan))` drifted after every generation task. Reproduced exactly: appending `<|end_of_text|>` once to each processed task's `until` gives the r2 canary markers' hashes (0eabcf84 for minerva algebra after one task, fb838886 for HumanEval after eight) against the plan's e2fe42b0. Every generation marker therefore lives under a path no fresh process can compute; choices markers (no `until`) are unaffected, and the r2 full choices run continues.

Fix: `evaluate_task` now hands the harness `copy.deepcopy(spec["generation"])` (regression test `test_generation_requests_never_alias_the_frozen_plan_spec` in `tests/test_table9_accuracy.py`). Because the runner is a source pin, `plan_v6e4_r3.json` was prepared with the same checkpoint plan and requests (canonical SHA256 1c914ee17c7ad20b6db2e341714ef4dcfeae450605d532515081e4f9b6ccafec); it differs from r2 only in the runner pin. Both r3 canaries were submitted at 23:22 UTC (`launch_v6e4_r3_commands.txt`, bundles 23.2 MB): `/calvinxu/table9-accuracy-choices-v6e4-canary-20260915-r3` and `/calvinxu/table9-accuracy-generation-v6e4-canary-20260915-r3`. After they succeed, release both full modes under r3 (the r2 choices results stay as a cross-check; the report takes one plan).

## Choices complete under r2; r3 choices released (2026-09-15 23:36 to 23:37 UTC)

The r2 full choices children finished in 15 minutes each with no preemptions (all 11 choice tasks, 21,489 documents per checkpoint); `coverage_r2_choices/` holds `report_table9_accuracy` output for r2 (25 of 51 components scored per checkpoint: 14 overlap + 11 choices; 9 generation missing; 17 deferred). The r3 choices canary passed, so the full r3 choices run was released as `/calvinxu/table9-accuracy-choices-v6e4-full-20260915-r3` (`launch_v6e4_r3_full_choices_command.txt`) to keep the final report on one plan; r2 stays as a cross-check. The r3 generation canary is running; its logs show the TPU ragged paged attention kernel rejecting this Qwen3 layout (`num_combined_kv_heads=40 can not be XLA fully tiled`) and levanter falling back to the reference attention, which decodes at about 1.8 tokens/s at batch 2 (r2 canary: MBPP 194 tokens in 111 s).

## Kernel fix and plan r4 (2026-09-15 23:50 UTC)

Generation on v6e was running at 1.75 tokens/s because the TPU ragged paged attention kernel rejects Qwen3's 20 KV heads in bfloat16 (`num_combined_kv_heads=40 can not be XLA fully tiled`: the kernel packs the interleaved K/V head axis two-per-lane and needs the packed count to be 1, 2, 4, 8 or a multiple of 8) and levanter fell back to `default_ragged_paged_attention`. Fix in `lib/levanter/src/levanter/layers/attention.py`: `_do_tpu_ragged_paged_attention` pads the KV-head axis of `q` and `kv_pages` to the smallest tileable count (`_tpu_rpa_padded_kv_heads`, 20 to 24 for bfloat16) with zero heads and slices them off the output; numerics of the real heads are unchanged and the reference path is untouched. CPU tests in `lib/levanter/tests/inference/test_paged_attention.py` check the padding rule and compare the padded kernel path against the reference (23 tests pass). The runner now pins `attention.py` and `kv_cache.py`, so `plan_v6e4_r4.json` (canonical SHA256 d09155cefe9707f0be8d8d129d04115c1d5c5609e92bc5ba75c2d516012f6e58) differs from r3 only in those three pins. Both r4 canaries submitted (`launch_v6e4_r4_commands.txt`); the r3 generation canary keeps running on the slow path as a fallback until r4's canary proves the kernel path, then it is cancelled. Full r4 runs for both modes follow the canaries.

r3 generation canary cancelled at 00:32 UTC to free two v6e-4 slices for the r4 canaries (it was on the slow fallback path and is superseded by r4).

r3 full choices cancelled at 00:39 UTC: r4 supersedes r3 for both modes, and its two v6e-4 slices were blocking the r4 canaries (the pool appears to hold only two v6e-4 slices for us).

## r4 result and plan r5 (2026-09-16 00:49 to 01:05 UTC)

r4 proved the padding takes the kernel path (no "Falling back to reference" warnings) but the Proportional generation child died compiling the kernel during the memory probe: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space smem. Used 1.00M of 1.00M smem. Exceeded smem capacity by 4.6K`. The kernel prefetches the int32 page table `page_indices[max_seqs, pages_per_seq]` into 1 MiB of scalar memory, and the harness's hard-coded `max_seqs=256` with 8192-token contexts in 8-token pages is exactly 1 MiB. Fix in `lib/levanter/src/levanter/eval_harness.py`: `_paged_attention_max_seqs` sizes `max_seqs` to an 896 KiB budget (224 for this configuration; unchanged elsewhere), unit-tested in `lib/levanter/tests/test_eval_harness.py`. The r4 choices canary had succeeded; the r4 generation parent was cancelled (its UniMax child would hit the same compile error). `plan_v6e4_r5.json` (canonical SHA256 c799909e7b7a1adf8fa9455d58a835b9c484dae99b0391ff1b4f16f6fc614abe) differs from r4 only in the `eval_harness.py` pin. r5 canaries submitted at `--priority batch` so they cannot preempt a ladder parent (the r4 generation parent had evicted the MARINER suite 3e20 retry7 parent from the full east5 CPU pool at 16:4x PDT).

## r5 result and plan r6 (2026-09-16 01:55 to 02:50 UTC)

r5 compiled and ran the kernel path (no fallback warnings, the r5 choices canary passed) but generation decoded at 0.64 tokens/s: every 64-token iteration took 99.26 s, all in output extraction (the blocking device wait), with only a weak dependence on context length (139 s for the 8 x 8192-token memory probe). The per-call KV-head padding copied the whole page cache (0.6 to 0.7 GB per layer, 26 layers) on every decode step. r6 moves the padding to allocation: `KvPageCache.init(..., cache_kv_heads=...)` allocates the kernel's tileable head count (`paged_cache_kv_heads`, 24 for Qwen3 in bfloat16, only when the TPU kernel will run), `update` writes the model's 20 heads into the first 40 interleaved slots, the TPU wrapper pads only the per-token query, and the reference path slices the extra heads. Tests: 28 pass in `lib/levanter/tests/inference/test_paged_attention.py`. `plan_v6e4_r6.json` (canonical SHA256 659f1bdada7e04f2d4c5905a35c9a6254899e7c77eb4e0cbf5ed49248b346e7c) differs from r5 only in the attention.py and kv_cache.py pins. r5 generation canary cancelled; r6 canaries submitted at batch priority (`launch_v6e4_r6_commands.txt`).

## r6 result and plan r7 (2026-09-16 03:40 to 04:10 UTC)

r6 (cache-level padding) decoded at exactly r5's rate: 98.5 s per 64-token iteration (0.65 tokens/s) on 2-document Minerva, 98 to 103 s per 32-token iteration on the 8 x 8192 memory probe. The per-round cost (about 3.1 s on the kernel path, about 1.1 s on the reference path in r2) therefore depends neither on the cache copy nor on the number of live sequences or their length. The engine pads every decode round to `max_tokens_per_round` query tokens, which defaults to `max_seqs` (224 in r5/r6, 256 before), and the kernel's grid scales with that. r7 caps harness generation at `GENERATION_MAX_SEQS = 32` sequences and 32 tokens per round (`lib/levanter/src/levanter/eval_harness.py`); if the hypothesis holds the round cost should fall several-fold. r6 generation canary cancelled; `plan_v6e4_r7.json` (canonical SHA256 9f227ccf15365eb566bc658fdca5e3f9f0719575da47977b0f1619fc50b90c2a) differs from r6 only in the harness pin; r7 canaries submitted at batch priority (`launch_v6e4_r7_commands.txt`). If r7 does not move the round cost, the next step is a TPU profile of the engine or moving generation to vLLM.
