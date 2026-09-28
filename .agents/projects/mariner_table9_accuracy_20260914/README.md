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

## r7 result: the round size was the bottleneck (2026-09-16 05:31 UTC)

With `max_seqs = max_tokens_per_round = 32`, the r7 generation canary decodes 64 tokens in 3.44 s (18.6 tokens/s with two live sequences) against 98.5 s in r5/r6 and 36 s on the reference path: about 107 ms per decode round for 32 padded tokens, so the per-round cost scales with the padded round size and is roughly 3.3 ms per padded token. At full occupancy (32 live sequences) that is about 300 tokens/s per v6e-4, which puts the full generation backfill (roughly 1.5 million generated tokens per checkpoint) at a few hours per checkpoint. Kernel path active, no fallback warnings. Full r7 commands for both modes are in `launch_v6e4_r7_full_commands.txt`, to be released when the r7 generation canary passes.

## r7 full release failed preflight; plan r8 on the merged tree (2026-09-16 06:42 to 06:50 UTC)

The r7 generation canary succeeded at 06:42 UTC and both full r7 parents were released, but each failed its preflight within a minute with "Frozen evaluation code/runtime differs": the branch had merged origin/main (d7307fcca8) after r7 was frozen, changing five pinned sources (levanter data/loader.py, eval_harness.py, inference/engine.py, trainer.py, uv.lock). `plan_v6e4_r8.json` (canonical SHA256 627479e1b43da705383cf2a7f7170062aaeb442751949582ebd9611ed9f9253e) was frozen on the merged tree; it differs from r7 only in those pins (runtime versions and the lm-eval revision are unchanged). r8 canaries submitted for both modes at batch priority (`launch_v6e4_r8_commands.txt`); the full runs (`launch_v6e4_r8_full_commands.txt`, east5-validated) are released automatically when both canaries succeed.

## r8 generation invalid: the engine admits one prefill batch per call; r9 batches requests (2026-09-16 10:44 to 11:35 UTC)

The full r8 runs completed (choices valid for both checkpoints, `coverage_r8/`), but the generation outputs are empty beyond the first 10 to 16 requests of every task (12 Minerva algebra, 10 geometry, 14 HumanEval, 16 MBPP, identical for both checkpoints). `InferenceEngine.generate` admits requests to prefill once per call, until `max_seqs_in_prefill` (16) sequences or `max_prefill_size` (8192) prompt tokens are queued, and never admits the remainder during decoding, so they return with no tokens; the two-document canaries could not see this. The r8 grading and `accuracy_vs_bpb_r8/` therefore report near-zero math and code accuracy for both models and must not be used. Fix: `eval_harness.generate_until` splits requests into batches the engine admits whole (`_admissible_request_batches`, tested) and calls `generate` per batch. `plan_v6e4_r9.json` (canonical SHA256 a246408c52621975aaebbb465019398e1011a2cefb37cfa08449e8c91b0c90fb) differs from r8 only in the harness pin. r9 canaries submitted (`launch_v6e4_r9_commands.txt`); the full runs (`launch_v6e4_r9_full_commands.txt`) release on their success and are graded automatically.

## r9 starved by preemption; r10 resumes tasks at 64-request chunks (2026-09-16 15:20 UTC)

The full r9 choices run completed for both checkpoints, but both r9 generation children were preempted about 75 minutes into Minerva algebra (1,187 documents at 106 tokens/s with 12 live sequences) and restarted it from scratch: with task-level durability a long task cannot outlast the v6e-4 pool's preemption cadence. r10 makes `evaluate_row` resumable: `evaluate_task_resumably` evaluates each task in 64-request chunks and saves a doc_id-prefix `partial.json.gz` under the task's result root after every chunk, resuming there on the next attempt and discarding it once `SUCCESS.json` is written (crash-and-resume test in `tests/test_table9_accuracy.py`). `plan_v6e4_r10.json` (canonical SHA256 9612aaa7b2a4cd3f538136f4190a0285da2638926382b1df61936cc2133acd78) differs from r9 only in the runner pin. r9 generation cancelled to free the slots; r10 canaries submitted at batch priority; full r10 releases and grades automatically.

## r10 too slow under spot reclaims; r11 shards generation by task and admits 32 sequences (2026-09-16 19:40 UTC)

r10's resumable runs progressed but the v6e-4 spot VMs were reclaimed about every 30 minutes, so each checkpoint completed one 64-request chunk per roughly 50 minutes: 2 to 3 days for the generation set. r11 changes three things: full generation runs launch one child per (checkpoint, task) so every task holds its own slice and resumes independently (`child_task_groups`; canaries and choice runs keep one child per checkpoint, and the memory-probe marker write tolerates concurrent children); the harness engine admits a full 32-sequence round per generate call (`max_seqs_in_prefill = 32`, `max_prefill_size = 16384`, `hbm_utilization = 0.6`); and the resume chunk is 32 requests. `plan_v6e4_r11.json` (canonical SHA256 71f6758d35445f88fe8fd551b4eac89d4bed9c7bce60300fee3ea622f575f169) differs from r10 in the runner and harness pins. r11 canaries submitted at batch priority; full r11 releases on their success (18 generation children plus 2 choices children), after which the r10 generation run is cancelled, and grading runs automatically.

## r11 canaries starved by a us-east5-b v6e-4 stockout; r10 generation cancelled; detached release chain (2026-09-16 20:40 to 21:00 UTC)

The interactive session that held the r11 auto-release and grading jobs restarted, which killed both. At 20:40 UTC the four r11 canary children had zero attempts after 90 minutes: the `tpu_v6e-preemptible_4-us-east5-b` scaling group reports "no more capacity in the zone" with 34 consecutive create failures (two slices were booting at 20:40), and the batch band queue for v6e-4 held, ahead of the canaries by root submission time, the two r10 generation children (15:35 UTC) and 26 workers of another user's cross-region eval pools (18:14 UTC). r10's placements today lasted 9 to 59 minutes each. The r10 generation parent was cancelled at 20:50 UTC (superseded by r11, whose runner and harness pins differ, so r10's partial outputs cannot be reused; Fieldbook job_01m2ndpvdte8y2ttnqw0e8j9pb marked killed). The release chain now runs detached from any session as `auto_release_r11.sh` (log `auto_release_r11.log`): it polls both canary parents every 5 minutes, validates and submits `launch_v6e4_r11_full_commands.txt` from a secrets subshell with redacted logs, registers the parents in Fieldbook, waits for both full parents (child counts logged every 10 minutes), then grades in Docker, writes `coverage_r11/` and `accuracy_vs_bpb_r11/`. It stops and logs instead of retrying if a canary or full parent fails. Open question for Calvin: the accuracy parents run at `--priority batch` (chosen so they cannot evict ladder parents), which puts their v6e-4 children behind every earlier batch submission; resubmitting the canaries at the default interactive band would place them ahead of the other user's batch pools and is his call.

## Canaries moved to the interactive band as r11i (2026-09-16 21:30 UTC)

At 21:20 UTC the first us-east5-b v6e-4 slice of the afternoon went to an interactive-band ops job while the batch-band r11 canaries stayed unplaced, so Calvin approved moving the accuracy runs to the default interactive band. The r11 canary parents were cancelled and resubmitted unchanged except for `--priority interactive` and the job names `table9-accuracy-{choices,generation}-v6e4-canary-20260915-r11i` (commands `launch_v6e4_r11i_commands.txt`, both validated region-local; Fieldbook job_01m2p24bg85rmdzfa7xk57pkan and job_01m2p24h9fbhywwhww5katj2kp). The full runs will use `launch_v6e4_r11i_full_commands.txt` with names `...-full-20260915-r11i`; the plan is still `plan_v6e4_r11.json`, so the canary artifacts satisfy the full-release guard. `auto_release_r11.sh` was restarted against the r11i names. Interactive parents cannot evict the interactive ladder parents in the CPU pool by the documented band rules; if a ladder parent is preempted anyway, the r4 incident's explanation was wrong and the accuracy parents go back to batch.

## r11i generation canary failed the memory probe; plan r12 reverts the two memory settings (2026-09-16 22:17 to 22:40 UTC)

The r11i choices canary succeeded at 22:17 UTC, but both r11i generation children failed the full-window memory probe within a minute of starting: `RESOURCE_EXHAUSTED` allocating 685 MB with 226 MB free inside `engine.reset`. r11 had raised `max_prefill_size` from 8192 to 16384 tokens and `hbm_utilization` from 0.5 to 0.6 (KV budget 18.68 GB against 15.57 GB under r10, whose probe passed); the probe is the worst case by design, so the release guard did its job. Plan r12 (`plan_v6e4_r12.json`, canonical SHA256 e04bbd3d476c9d0d32df37ca5fdd5314d055114507aab2e22e75c27f488c9c16, prepared with the same checkpoint rows and request set) differs from r11 only in the harness pin: `GENERATION_PREFILL_TOKENS = 8192` and `GENERATION_HBM_UTILIZATION = 0.5` again, keeping r11's task-sharded children, 32-sequence admission and 32-request resume chunks. Canaries `table9-accuracy-{choices,generation}-v6e4-canary-20260915-r12` submitted at the interactive band (`launch_v6e4_r12_commands.txt`); full commands in `launch_v6e4_r12_full_commands.txt`; `auto_release_r11.sh` restarted against the r12 names.

## Full r12 backfill complete; Proportional vs UniMax-8 accuracy reported (2026-09-17 05:55 UTC)

Both r12 full parents succeeded (choices 00:45 UTC; the 18 task-sharded generation children between 01:40 and 05:45 UTC, none failed after the interactive-band move), grading ran in Docker at 05:45 UTC, and `report_table9_accuracy` plus `analyze_table9_accuracy_vs_bpb.py` wrote `coverage_r11/` and `accuracy_vs_bpb_r11/` (directory names kept from the chain script; the plan is r12). Coverage 34 of 51 components per checkpoint (14 overlap, 11 choices, 9 generation); 17 MT-MBPP components BPB-only by design. Unweighted mean accuracy over the 34 scored components: Proportional 42.76%, UniMax-8 44.15% (+1.39 pp); native macro BPB over the 51 components 0.6731 vs 0.6253; BPB over the 34 scored 0.7518 vs 0.7114. Groups (Proportional -> UniMax-8): basic skills 68.2 -> 74.4, math 3.7 -> 8.6, code (HumanEval, MBPP) 3.4 -> 2.4, MMLU 34.9 -> 33.9, QA 58.2 -> 56.9. Sign agreement between BPB and accuracy changes 23 of 33 decided components (Spearman 0.56 across the 34). Per-component table in `accuracy_vs_bpb_r11/components.md`.

## MARINER OlmoBaseEval Easy 1e21 checkpoint: family-resumable overlap suite and backfill submitted (2026-09-21 13:23 PDT)

Calvin asked for the accuracy evaluation of the landed MARINER suite 1e21 checkpoint
(`lwspu_t9_snc_cap08_1e21_seed662005-e8e9d7`, permanent step 22,056, audited by the ladder-watch session) and, before
submitting, for a check that the evaluation is idempotent and resumes after preemption.

Findings: the backfill runner already was (task-level `SUCCESS.json` markers with hash and provenance checks, 32-request
`partial.json.gz` chunks, one child per generation task, Fray's default 100 preemption retries per child, the durable
memory-probe marker). The overlap suite was idempotent per row only: one `run_eval_harness_main` call evaluated all 67
leaves and persisted at the end, so a preempted v5p-8 child (the east5 v5p pool is preemptible) restarted from scratch;
the baselines' outputs show it (UniMax-8 done 36 minutes after staging, Proportional 3 h 18 min).

Fix in `evaluate_mariner_ladder_accuracy.py`: the model is loaded once and the eleven families are evaluated one at a
time with `run_lm_eval_harness`; each family's validated output is written to `<row root>/partial/<family>.json.gz`
(readback-verified), a new attempt loads and re-validates saved families and evaluates only the rest, and the family
outputs are merged (disjoint union of the task-keyed sections, `averages` recomputed with Levanter's own helper) into
the same results dictionary the whole-suite call returned, then persisted and the partials removed. Tests in
`tests/test_mariner_ladder_accuracy.py` (families partition the 67 leaves; merge reproduces sections and averages;
partials round-trip, reject coverage drift and tampering). Because the evaluator is a source pin, MARINER's overlap plan
carries the new hash; the baselines' results stay under their own plan.

Plans (`prepare_table9_checkpoint`, templates `plan_v5p_cpu_staging.json` and `plan_v6e4_r12.json`, training experiment
`exp_01m1zy8yths4dqp5bgc0ffzztp`): `mariner_t9_1e21/overlap_plan.json` (canonical SHA256
435347315824e90f72efd54f3b11fb738d959402df8e8e9f003ff5b57ab4f1ee) and `mariner_t9_1e21/backfill_plan.json`
(c3cd371bc417c550cc18654c6b0f89b396f0d269c2467fd5b1e999f3d8162cfa); row name `lwspu_t9_snc_cap08_1e21-e8e9d7`.
All five launch commands passed the east5 guard (`mariner_t9_1e21/launch_*.txt`). Submitted at the interactive band
from a secrets subshell with redacted logs (`mariner_t9_1e21/submit_*.log`), Fieldbook experiment
`exp_01kvvvv6zxrf0j7tkp4f7k6y66`:

- `/calvinxu/mariner-ladder-accuracy-mariner-t9-1e21-v5p-20260921` (overlap suite, one v5p-8 child in us-east5-a; job_01m32t75n0myarjybeb940c8qb)
- `/calvinxu/table9-accuracy-choices-v6e4-canary-mariner-t9-1e21-20260921` (job_01m32t76149jhnfyjphfjm6f3e)
- `/calvinxu/table9-accuracy-generation-v6e4-canary-mariner-t9-1e21-20260921` (job_01m32t76ctm1kdaepxsygpr3mf)

`mariner_t9_1e21/auto_release.sh` runs detached (log `auto_release.log`): it releases `launch_full_commands.txt` when
both canaries succeed (`full.released` marker), waits for both full parents and the overlap parent, then grades in
Docker and writes `mariner_t9_1e21/coverage/`. It stops and logs on any failed parent; resubmitting the same command
resumes from the durable partials. The accuracy-vs-BPB analysis against Proportional is a manual step afterwards.

## MARINER OlmoBaseEval Easy 1e21: evaluation complete, graded and compared (2026-09-22 00:27 PDT)

All Iris jobs succeeded without a failed attempt: the overlap suite (v5p-8, family-resumable evaluator), the choices
parent and the nine task-sharded generation children (the last finished 2026-09-21 19:57 PDT). The chain's grading step
failed at 19:59 PDT because Docker (OrbStack) was not running on the Mac (`docker create` could not reach the socket);
Calvin approved starting OrbStack, and `mariner_t9_1e21/stage4_rerun.sh` reran grading and the report
(`auto_release.log`: `DONE: coverage/ written` at 00:27 PDT). Coverage is 34 of 51, the same target as the baselines'
r12 report (17 multilingual MBPP components deferred). `mariner_t9_1e21/coverage/{coverage.json,components.csv}`.

Accuracy-vs-BPB analysis (`mariner_t9_1e21/accuracy_vs_bpb/`): the native summary entry for MARINER was built from the
audited Table-9 result (`frozen_scaling_update_20260913/mariner/lwspu_t9_snc_cap08_1e21_seed662005-e8e9d7_table9.json`,
51 components, macro 0.5894, checkpoint URI equal to the coverage row's) instead of a W&B eval run, and merged with the
baselines' coverage rows (`coverage_merged.json`, `native_summaries_with_mariner.json`). Equal-weight means over the
34 scored components:

| | Proportional | UniMax-8 | MARINER |
|---|---:|---:|---:|
| accuracy (pp) | 42.76 | 44.15 | 46.36 |
| BPB, scored components | 0.7518 | 0.7114 | 0.6912 |

MARINER is best on 22 of the 34 components, below both baselines on 3 (winogrande, hellaswag, coqa; naturalqs is below
Proportional only). Against Proportional: BPB improved on 21 components, accuracy also improved on 19 of those; sign
agreement 25/33; Spearman(-dBPB, dAcc) 0.76; gains concentrate in basic skills (+10.4 pp), code (+7.4 pp; HumanEval
6.1 -> 17.1, MBPP 0.8 -> 4.6) and Minerva math (+6.2 pp), while the 15 QA components are flat (-0.4 pp) with slightly
worse BPB (+1.1%). Against UniMax-8: +2.2 pp overall, better on every group; sign agreement 24/31, Spearman 0.45. The
README's rule stands: no partial Table-9 accuracy macro is published as the suite's accuracy; the paper's Table 2
accuracy column is Calvin's decision.

Paper (2026-09-22 02:45 PDT): Table 2's accuracy column carries the 34-component mean (42.8 / 44.2 / 46.4); the
group breakdown with scored/unscored BPB is the last display of Section 5.1; Appendix A.4 gives the protocol and a
51-row per-component BPB and accuracy table built by
`experiments/domain_phase_mix/exploratory/two_phase_many/build_accuracy_component_table_20260922.py` from this
directory's `accuracy_vs_bpb/{coverage_merged.json,native_summaries_with_mariner.json}`.

## Olmix OlmoBaseEval Easy 1e21: staged, gated on OLM-U (2026-09-25 21:30 PDT)

Calvin: wait for Olmix's Uncheatable 1e21 run (OLM-U) to land before submitting this evaluation, so the v6e-4 jobs
cannot hold back its v6e-64 slices; he considered running it in another region and chose to keep waiting. Plans
(`prepare_table9_checkpoint`, same templates as MARINER's, row `olmixq_t9_kl0p005_cap04_1e21-3f95f2`, HF export
`.../delphi_matched_olmix_scaling_v6e_20260910/olmixq_t9_kl0p005_cap04_1e21_seed662005-3f95f2/hf/step-22056`, training
experiment `exp_01m21dyb8mxjg2aqhtbjncejpc`): `olmix_t9_1e21/overlap_plan.json` (70a397d8...) and
`olmix_t9_1e21/backfill_plan.json` (1d964319...). They differ from MARINER's plans only in the pin of
`lib/marin/src/marin/evaluation/eval_dataset_cache.py` (the 24 Sep return-type fix of the cache step; no effect on
scoring). The five launch commands (`olmix_t9_1e21/launch_*.txt`, job names ending `olmix-t9-1e21-20260926`) pass the
east5 guard. `olmix_t9_1e21/auto_release.sh` runs detached (log `auto_release.log`): stage 0 waits until the
matched-Olmix scaling parent succeeds (or the audit marks OLM-U measured), then submits the overlap suite and both
canaries, releases the full runs on canary success, waits, grades and reports as MARINER's chain did. Grading needs
Docker (OrbStack) running on the Mac. Stop the chain with `pkill -f olmix_t9_1e21/auto_release.sh`.

## Olmix OlmoBaseEval Easy 1e21: choices and generation moved to us-east1-d (2026-09-26 05:49 PDT)

OLM-U was still pending: all 16 of its v6e-64 tasks were waiting for capacity after six preemptions, and no v6e-64 slice
existed in the cluster. Calvin asked to run the accuracy evaluation elsewhere, approving a one-time copy of the checkpoint
and nothing else: evaluation inputs must come from their sources, not from another region. The backfill therefore runs
on v6e-4 in us-east1-d (five idle slices at launch, a separate regional pool from OLM-U's us-east5-b requests), the same
hardware and kernel path as the MARINER, Proportional and UniMax-8 rows. v4 in us-central2-b was not used: generation
depends on the TPU paged-attention path that took rounds r4 to r12 to make work on v6e.

- **Checkpoint:** copied server-side to the same path in `gs://marin-us-east1` (13.55 GB, eight objects, every one equal
  to its source in size and CRC32C; `olmix_t9_1e21_east1/copy_checkpoint.{py,log}`).
- **Requests:** uploaded from the local frozen copy in `requests/` (manifest SHA256 f97331...; each file checked against
  it). The native OLMo-Eval request set exists only in us-east5, so the us-east1 parent does not read it; this manifest's
  exact prompt/gold parity already passed in us-east5 for the r12 and MARINER 1e21 parents, recorded in
  `NATIVE_PARITY_VERIFIED` in `evaluate_table9_accuracy.py`.
- **Code:** `evaluate_table9_accuracy.py` now takes the region from the plan (`REGION_BUCKETS`, per-region `TPU_ZONES`) and
  gains `--relocate-from` to re-home a frozen plan after its checkpoints are copied; `report_table9_accuracy.py` matches
  backfill and overlap rows by checkpoint content (`same_checkpoint`: name, step, provenance and every object's size and
  CRC32C) and reads overlap results with the overlap plan's own row. Tests in `tests/test_table9_artifacts.py`.
- **Plan:** `olmix_t9_1e21_east1/backfill_plan.json` (digest 83549a5b...), relocated from `olmix_t9_1e21/backfill_plan.json`.
  It differs only in region, zone, bucket paths, the checkpoint row (copy URI, object generations, `copied_from`) and the
  evaluator's own source pin; TPU type, batch 8, length 8192, request manifest, runtimes and all other pins are unchanged.
- **Overlap suite:** unchanged plan (`olmix_t9_1e21/overlap_plan.json`) on v5p-8 in us-east5-a beside the original
  checkpoint; it does not use v6e.
- **Launch:** `olmix_t9_1e21_east1/launch_*.txt` (us-east1 guard for the backfill, east5 guard for the overlap suite; four
  extra bundle excludes from the 2026-09-14 list bring the bundle to 20.8 MB). `olmix_t9_1e21_east1/auto_release.sh` runs
  detached: it submitted the overlap suite and both canaries at 05:48 PDT, releases the full runs when both canaries
  succeed, waits, then grades (Docker required) and writes `olmix_t9_1e21_east1/coverage/`. The gated east5 chain in
  `olmix_t9_1e21/` was stopped before it submitted anything.

## Olmix OlmoBaseEval Easy 1e21: evaluation complete and graded (2026-09-26 10:51 PDT)

Every Iris job succeeded without a failed attempt. The full runs were released at about 06:25 PDT. The choices child and
the overlap suite finished early, and the last generation child (`minerva_math_algebra`, preempted once and resumed from
its 32-request chunks) finished at 10:47. OrbStack's Docker engine had hung while the last child ran; it was stopped and
started at about 10:25 with Calvin's approval, before grading began. The chain then graded in three minutes and wrote
`olmix_t9_1e21_east1/coverage/`, with 34 of 51 components, the same scope as the other rows.

Equal-weight mean accuracy over the 34 scored components: Proportional 42.76, UniMax-8 44.15, Olmix 46.06, MARINER 46.36.
Olmix minus MARINER is -0.29 pp (unpaired binomial SE over documents 0.28 pp, 95% interval -0.84 to +0.25; trainer-seed
variance not included). Olmix is higher on 18 components and lower on 16. MARINER leads on basic skills (coding -6.5,
arithmetic -5.0, pattern -4.5 for Olmix) and Minerva math (algebra -5.2, prealgebra -4.6). Olmix leads on csqa (+5.3),
hellaswag (+3.6), lambada (+3.3), winogrande (+3.2) and common knowledge (+2.8).

## Name-agnostic MBPP regrade (2026-09-26; adopted 12:45 PDT)

The native MBPP prompt describes each task in words and never names the function the MBPP asserts call, so the
frozen grading fails any generation that chooses another name: only 37–55 of 500 generations per 1e21 checkpoint
define the tested name. `mbpp_regrade/regrade_mbpp_name_agnostic.py` binds the tested name to the generation's
function (`tested = chosen`, inserted between the generation and the asserts; among top-level functions that accept
every call shape in the asserts, the entry points, last defined) and re-executes those programs in the grader's
sandbox. Generations that already define the tested name keep their frozen result.

Controls (`mbpp_regrade/results/summary.json`): 40/40 re-executed named programs reproduce their frozen result;
MBPP's reference solutions pass 498/498 as released and 498/498 after renaming the tested function (the two tasks
whose asserts call two names are left unbound); 0/198 references of a different problem pass when bound; the
tie-break never applies.

| Mixture | frozen pass@1 | regraded pass@1 (95% CI) | call-incompatible | pass among callable |
|---|---|---|---|---|
| Proportional | 0.8% | 6.4% (4.4–8.6) | 136 | 9.4% of 342 |
| UniMax-8 | 1.8% | 14.2% (11.2–17.4) | 104 | 18.9% of 375 |
| Olmix | 3.0% | 15.2% (12.2–18.4) | 109 | 19.9% of 381 |
| MARINER | 4.6% | 18.2% (14.8–21.6) | 100 | 23.6% of 386 |

The ordering is unchanged and now matches the MBPP BPB ordering; MARINER − Olmix is +3.0 points (paired bootstrap
95% CI +0.2 to +5.8). Call-incompatible generations define functions whose parameters cannot take the asserts'
arguments, usually because MBPP passes an extra argument (an array length, a count) that the task text never
mentions; binding cannot rescue them. Effect on the paper if adopted: MBPP 0.8/1.8/3.0/4.6% becomes
6.4/14.2/15.2/18.2%; Code group 3.4/2.4/7.9/10.8% becomes 6.2/8.6/14.0/17.6%; the 34-component mean
42.8/44.2/46.1/46.4% becomes 42.9/44.5/46.4/46.8% (MARINER − Olmix 0.30 → 0.34 points). Rebuilt tables are in the
session scratchpad, not in `reference_outputs/`.

Adopted at Calvin's request: `grade_table9_accuracy.py` carries the binding (`mbpp_binding`; new grader identity, so the
frozen gradings stay on GCS beside the new ones). `mbpp_regrade/adopt_chain.sh` regraded all four checkpoints
(HumanEval, Minerva and every other component reproduce exactly; MBPP matches the exploratory regrade sample by sample),
rewrote `coverage_r11/`, `mariner_t9_1e21/coverage/`, `olmix_t9_1e21_east1/coverage/`, both merged coverage files,
`mariner_t9_1e21/accuracy_vs_bpb/vs_{proportional,unimax8}` (19 of 21 still; Spearman 0.77 and 0.47) and the paper
tables in `reference_outputs/accuracy_component_table_20260922/`. Paper pushed in Overleaf 55b0d13.
