# Phase A unverified-item resolutions — Mochi loop guards (#12264)

Resolved 2026-11-25 in worktree `mochi-loop-guards` (branch `benfeuer/mochi-loop-guards`). Evidence paths are relative to the repo root unless absolute.

---

## (a) Trace storage format — `trajectory_retention {phases: [eval], required: true}`

**RESOLVED from the vendored SkyRL code.** Files: `vendor/skyrl/skyrl-train/skyrl_train/trajectory_runners/trajectory_retention.py` (`trajectory_retention.py`), `trajectory_retention_config.py`, `skyrl_train/evaluate.py`, `skyrl_train/utils/trainer_utils.py`, `skyrl_train/trainer.py`.

### On-disk layout (what retention produces)

- **Root**: `output_path` from the retention config. The trainer gives the eval runner its own sink with `output_path` suffixed `/validation`: `trainer.py:298-306` — `evaluation_config.trajectory_retention.output_path = join_resource_path(..., "validation")`. (The base `output_path` is supplied by launch tooling per run; the Micro210 `policy.json` does not pin one — its `trajectory_retention` section has `run_id: "2026.11.09"` and no `output_path` key.)
- **Archives**: `schema_v{record.schema_version}/archives/phase={phase}/step={global_step:08d}|step=eval_only/{sha256(zip_payload)}.zip` — `trajectory_retention.py:1109-1118` (`_archive_path`), with `TRAJECTORY_RECORD_SCHEMA_VERSION = 7` (`:53`) and `_ARCHIVE_DIRECTORY = "archives"` (`:58`). For eval-only sessions (global_step `None`) the step segment is literally `step=eval_only`; for in-training evals it is the training step.
- **Zip contents**: one `records/{record_id}.json.gz` entry per record — a **gzip of the canonical JSON bytes** of the serialized record (`_encode_record`, `:995-999`; `_archive_payload`, `:511-536`, zip written `ZIP_STORED` with fixed 1980 timestamps) — plus `_ARCHIVE_MANIFEST = "manifest.json"` inside the zip, listing `{record_id, step ("{phase}:{global_step}"), reasons, sample_score, bytes, entry}` per record (`:521-535`).
- **Ledger**: `_LEDGER_NAME = "_retention_ledger.json"` (`:54`) at the output_path root, written atomically (`_publish_archives`, `:719-728`). Ledger fields: `schema_version` (1), `total_bytes`, `step_bytes` (keyed `"{phase}:{global_step}"`), `records` (record_id → `{path, bytes, step, reasons, sample_score}`), `archives` (zip path → `{bytes, step, record_ids}`) (`:62-119`). On restart, `_reconciled_ledger` re-adds orphaned zips/json.gz files found on storage (`:680-710`).

### Record format (`TrajectoryRecord`, `trajectory_retention.py:206-224`)

`schema_version` (7), `record_id` (sha256 of the redacted canonical record, `:488-489`), `run_id`, `global_step`, `phase`, and:

- `trajectory` (`_TrajectoryIdentity`, `:142-148`): `instance_id`, `repetition_id`, `row_indices` (the batch rows composing the trajectory), `environment_class`, `environment_extras` (`{data_source, teacher_route}`).
- `prompt` (`_PromptTrace`, `:151-154`): `messages` (the input conversation) + `token_ids`.
- `response` (`_ResponseTrace`, `:167-177`): `messages` (multi-turn), `text`, `token_ids` (concatenated across steps), `loss_mask`, `trainable_spans`, `step_boundaries`, `stop_reason`, `generation_limit`, `loop_spans`.
- `reward` (`_RewardTrace`, `:180-184`): `outcome`, `shaped`, `components`.
- `disposition` (`_DispositionTrace`, `:187-191`): `exception_type`, `error_treatment`, `server_error`.
- `verifier`, `verification_result`, `metrics` (rollout metrics), `provenance` (`_ProvenanceTrace`, `:194-203`: `runner`, `inference_backend`, `model_path`, `model_source_identity`, `resume_path`, `model_version_step`, `sampling`, `reward_shaping_schema_version`).

### Where the per-step action/observation text lives

`response.messages` — built at `:442-461` from `output["evidence_messages"][final_index]` when the batch carries evidence messages ("The parsed conversation the grader read: tool calls, observations and the final answer" — same source as the JSONL `messages` field, `trainer_utils.py:315-316`), else the single decoded assistant text (`:449`). Per-step structure is in `response.step_boundaries` (`_StepBoundary`, `:157-164`; `_step_boundaries`, `:328-347`): each step's `row_index`, `prompt_token_ids`, and `token_start`/`token_end` span into the concatenated `response.token_ids`, with that step's `stop_reason` and `is_last_step`. So one record holds the whole multi-turn trajectory with per-step action/observation text in `messages` and per-step token spans in `step_boundaries`.

### Selection under `{phases: [eval], required: true}` + cohort-size sampling

- Config: `trajectory_retention_config.py:8-34` — `phases` (must be ⊆ {train, eval}, `:101-102`), `sample_count_per_step` (default 1), `sample_fraction`, `eval_sample_fraction`, `required` (default False), `max_bytes_per_step` (800 MiB) / `max_bytes_per_run` (25 GiB), `publish_timeout_seconds` (120) / `shutdown_timeout_seconds` (30).
- A record is selected when it is **mandatory** (`always_retain_failures` outcome ≤ 0, `always_retain_non_terminating` stop not in accepted set, `always_retain_loops`, reward bounds; `:623-639`), or by **fraction** (for eval, `eval_sample_fraction` if set else `sample_fraction`; hash-threshold on the sample score, `:642-644`), or by **count** (the `sample_count_per_step` lowest-hash samples per step key `"{phase}:{global_step}"`, with displacement of the largest; `:1051-1076`, `:666-678`). A cohort sized ≥ `sample_count_per_step` per step is therefore fully covered by the count rule; with `eval_sample_fraction: 1.0` everything is covered by the fraction rule regardless. The Micro210 policy actually pins `{phases: [eval], required: true, eval_sample_fraction: 1.0, sample_count_per_step: 0}` (policy.json `.evaluation.config.generator.trajectory_retention`), i.e. **all** eval records retained, count sampling off.
- Byte bounds (`max_bytes_per_step`/`max_bytes_per_run`) can drop selected records (`dropped_by_bounds`, `:960-976`); with `required: true` these are still silent drops (bounds are config, not failure). The policy sets both to `null` (unbounded).

### `required: true` semantics — where retention failure fails the run

1. `TrajectorySink.__init__`: initialization failure → `_raise_publication_error` raises `TrajectoryRetentionPublicationError` (or `...Timeout`) — `:808-815`.
2. `_ensure_ledger`: when required, re-runs initialization synchronously and raises on error — `:874-885`.
3. `_publish_queued`: when required, publishes synchronously (no async queue) and raises on error — `:887-897`.
4. `retain_trajectories` (the rollout-worker entry): any retention exception is re-raised when `sink.config.required` — `:1126-1148` (`if sink.config.required: raise`). The comment at `:1140-1143` states retention runs inside the rollout worker, "whose own handler exits the process" — so with `required: true` a retention failure kills the worker and fails the run. With `required: false` it logs and continues with `write_errors` metrics.

### How this differs from the `dumped_evals` JSONL rows

`evaluate.py` dumps flat JSONL **independently of retention** when `cfg.trainer.dump_eval_results` is on (`evaluate.py:366-383`), to `evaluation_dump_dir(export_path, global_step, attempt_id)` = `{trainer.export_path}/dumped_evals/{attempt_id}/{eval_only|global_step_{N}_evals}` (`evaluate.py:77-80`), via `dump_per_dataset_eval_results` (`trainer_utils.py:266-352`):

- One file per data source: `{sanitized_data_source}.jsonl` + an `aggregated_results.jsonl` of eval metrics (`:296-352`).
- Row fields (`:303-342`): `uid`, `attempt_id`, `step`, `ordinal`, `is_last_step`, `identity_disposition` ("task_spec"|"preference_input"), `task_id`, `prompt_token_ids`, `response_token_ids`, `input_prompt` (decoded), `output_response` (decoded), `messages` (= `evidence_messages`, the parsed conversation), `score`, `stop_reason`, `exception_type`, `error_treatment`, `server_error`, `env_class`, `subset_id`, `data_source`, `env_metrics`, `verification` (trimmed: `status`, `reason`, `score`, `passed`, `score_min`, `score_max`, filtered `diagnostics`).

Differences: retention records are **one per trajectory** (rows grouped by trajectory id, `:283-305`) with full token ids + loss masks + step boundaries + reward shaping + provenance, selected by the mandatory/fraction/count rules, ledger-tracked, gzip-inside-zip at `output_path`; JSONL rows are **one per rollout row**, flat, decoded text, unselected (every row), no ledger, no loss masks/shaping/provenance, under `trainer.export_path/dumped_evals/...`. `loop_audit.py` consumes the **JSONL** rows (`read_rows`, `:479-486`); the retention archives are the trace-level record.

---

## (b) TB timeouts in the Micro210 `2026.11.20-together` eval policy

**RESOLVED: agent 1800 s confirmed; verifier 300 s NOT set by the policy (no verifier override).**

- **Download & verify**: `https://huggingface.co/datasets/open-athena/micro210-policy/resolve/fa38e3eacc25137c9febe898119f08d65cc5ce88/policy.json` → HTTP 200, **121,292 bytes**, sha256 **`8f7ef8170fbef2880214cbf03f70d80cffa3114276e024389617777bf07f53e5`** — exactly `POLICY_DIGEST` and `byte_size` in `experiments/evaluation/micro210_policy.py:19,46`. (`POLICY_REVISION` "2026.11.20-together", commit `fa38e3e...`.)
- **Row identification**: the policy's TB2 row is benchmark **`tb2-recovery`**, the OT-TBLite row is **`ot-tblite-recovery`** (`.selection.attempts_per_benchmark` / `.attempts_per_data_source`; task ids like `tb2-recovery/terminal-bench/raman-fitting`, `ot-tblite-recovery/malicious-package-forensics` in `.selection.identities`). Both run through the single `terminal_bench` config section (`.evaluation.config.config_groups` = `{"terminal_bench_config": "terminal_bench"}`) with the Pi installed agent.
- **Population**: `evaluation.tasks_per_benchmark = 10`; `evaluation.attempts_per_selected_task` = `{"tb2-recovery": 3, "ot-tblite-recovery": 3}` → **10 tasks × 3 attempts = 30 attempts each**.
- **Agent timeout = 1800 s** ✓:
  ```json
  "harbor": { "eval_timeout_override_sec": 1800, "override_timeout_sec": 1800,
              "trial_attempt_timeout_sec": 10800, "verifier_override_timeout_sec": null,
              "n_concurrent_trials": 8, "max_retries": 0, ... }
  "installed_agent": { "agent": { "name": "pi", "kwargs": { "version": "0.87.0",
        "thinking": "high", "thinking_format": "chat-template", "compaction_reserve_tokens": 16384,
        "model_info": { "max_input_tokens": 65536, "max_output_tokens": 16384 } },
        "override_setup_timeout_sec": 360 }, "run_timeout": 1800, "setup_timeout": 360, ... }
  ```
  (`eval_timeout_override_sec`/`override_timeout_sec`/`run_timeout` = 1800 = `CAMPAIGN_JOB.agent_timeout`, `agentic.py:211`.) Note the keys are consumed by the Micro210 runner image (`.dataset.repo` = `open-athena/Marin-Micro-Eval-210` @ `4564ed081c316bc3231ae5c3a53a46b39248c819`); they do not appear in this checkout, so the per-key semantics above follow the key names and the Harbor settings they mirror.
- **Verifier timeout ≠ 300 s**: `harbor.verifier_override_timeout_sec` is **`null`** — no verifier override at run time; each prepared task keeps its own verifier timeout. The only `300` anywhere in the policy is the **LLM judge's** `request_timeout: 300.0` (`.evaluation.judge`, Together serverless judge HTTP timeout) — not a task verifier. So the rows do **not** equal `CAMPAIGN_JOB = (agent 1800 s, verifier 300 s)` on the verifier half.
- **Other limits that differ from the campaign TB2 protocol**: `trial_attempt_timeout_sec: 10800` (3 h, vs the panel's `TERMINAL_BENCH_ATTEMPT_TIMEOUT = 8 h`, `agentic.py:45`); `context_budget: {"request_window_tokens": 65536, "max_new_tokens_per_turn": 16384, "max_turns": 1024}` (vs `CAMPAIGN_MAX_INPUT_TOKENS = 32768`).
- **Sampling** (`.evaluation.config.generator`): `eval_sampling_params = {"temperature": 0.7, "top_p": 0.95, "stop_token_ids": [128001, 128009], "skip_special_tokens": false}`, `eval_n_samples_per_prompt: 1`, `max_num_seqs: 8`.
- **Conclusion**: the DESIGN's claim holds **only for the agent timeout** (1800 s). Verifier 300 s, attempt timeout 28800 s, and the 32768 input cap are *not* what the Micro210 rows ran.

---

## (c) The #11723 SWE-smith screen path (469 retained attempts)

**PARTIALLY RESOLVED — the exact storage path is not stated in any receipt I can reach; here is what is established and the concrete query recipe.**

Established from **#11536 comment 6941306305** (the "Mochi nested-edit screen" readout, fetched via `gh api repos/marin-community/marin/issues/comments/6941306305`) and **#11723** §2.3 (body + 13 comments, fetched via `gh issue view 11723 --repo marin-community/marin --json body,comments`):

- What ran: "The screen ran Mochi SFT step 400407 on the same 128 SWE-smith tasks × 8 tries as the Snowball pass-rate screen. Setup: Pi 0.87.0, the nested `edit` tool, one DP16 GB200 serve in cw-us-east-08a, 32 trials at once, a 900 s agent cap and no retries."
- The comparator run: Snowball SFT `grug-datakit-sft-262k-20261005-ac60d940-r1`, **run 43604**.
- "r6" is the **sixth serve start** (r1–r6 GB200-hours table in the readout), not a dataset revision.
- The count: "We cancelled the screen at about 06:42Z, with **469 of 1,024 attempts recorded**." #11723 §2.3 snapshot counts: 469 retained, 463 scored, 184 passed (39.74%), 289/469 `AgentTimeoutError` at 900 s.
- Retention: "Two retained-trace snapshots, taken about 15 minutes apart after the stop, agree exactly. That is an observation, not proof of final retention. **Private traces are not published.**"
- The only public access is the trace viewer Marina app `https://public.applets.marina.oa.dev/a/193c0a18-17c8-4348-9df3-765f3811a5d5/` (run "Pass-rate 43604" — the **Snowball** screen), which "reads from temporary storage: the traces expire about 2026-11-28 and the exercise data about 2026-12-18". The app's backing storage URI is **not** given in the issue.

Not found: no `s3://`/`gs://` path for the 469 Mochi attempts appears in #11723 (body or comments), #11536 (body or comments), or comment 6941306305. No evalrecords `run_id` for the Mochi screen is named either. Confirming it needs marina-app/cluster metadata access (not GPU work).

**How the 469 attempts would be queried** (best recipe, in order of likelihood):

1. **Marina app storage.** The viewer app `193c0a18-17c8-4348-9df3-765f3811a5d5` exposes both the Snowball screen and the edit-exercise rows; the Mochi screen run would be a sibling run in the same app. Read the app's release metadata / Postgres package (marina-app tooling) to get its backing storage prefix, then list the run's attempt objects there. Note the TTL: traces expire ≈ 2026-11-28 — read soon.
2. **evalrecords record scan.** The eval-run record layout is `{prefix}/{run_id}/record.json` with `DEFAULT_SCAN_PREFIXES = ("gs://marin-eval-metadata/evals", "s3://marin-us-east-02a/marin/evals", "gs://marin-eval-metadata/runs", "s3://marin-us-east-02a/marin/eval-metadata/runs")` (`lib/evalrecords/src/evalrecords/records.py:36-44`). Scan those prefixes (`list_record_paths`, `records.py:861`), read each `record.json` (`read_record`, `:818`), and filter for the Mochi step-400407 model identity + an SWE-smith/Pi evaluation created ≈ 2026-11-19/21 (the screen window). Then read the run store with `finestore.reader.ReadView(run_root)`:
   - `iter_rows("evaluation_attempts")` — `ATTEMPTS_TABLE = "evaluation_attempts"`, `RUN_OBJECT = "evaluation/run.json"` (`lib/evalrecords/src/evalrecords/rollout_attempts.py:11-12`); `Attempt` fields: `dataset`, `task_id`, `source_row`, `trial`, `retry`, `phase` (started/finished), `seed`, `outcome`, `error`, `result_blob`, `prompt_tokens`, `completion_tokens` (`:20-34`). Filter `dataset`/`task_id` and `phase == "finished"`; result payloads come from `read_blob(attempt.result_blob)`.
   - `"rollouts_v1"` — `ARCHIVE_ROLLOUTS_TABLE = f"rollouts_v{ROLLOUT_SCHEMA_VERSION}"` (`lib/finestore/src/finestore/eval.py:216`), written alongside attempts at `lib/marin/src/marin/rollouts/evaluation.py:623-626`; holds the request/turn records the screen audit parsed ("incomplete request/turn records", "unreadable records" in the readout).
   - This matches the screen's vocabulary (attempt records, request/turn records, 555 assigned pairs "with no retained attempt record", 171 "unmappable records").
3. **SkyRL trajectory-retention archives.** If the screen driver instead retained through the vendored SkyRL sink, the run's `trajectory_retention.output_path` would hold `_retention_ledger.json` + `schema_v7/archives/phase=eval/...zip` (see (a)); the ledger's `records` map would enumerate the retained record ids and the zips hold the per-attempt multi-turn records.

---

## Panel build

**Done 2026-11-25.** Builder script: `experiments/post_training/mochi_loop_guards/panel.py` (uncommitted). Manifest written to **`cohorts/manifest.json`** (new repo-root directory, uncommitted).

- **manifest_sha256 (canonical JSON of the manifest without that field): `562bd617cd6d38369ae1c8d534b2df769888a8eec59d0017578bd001be6ef985`** — recomputed independently from the written file and matches; `frozen_at` = 2026-11-25T12:22:17+00:00.
- **Trials total 823** ✓: `tb2` 89 tasks × 3 = 267; `tblite` 100 × 2 = 200; `swe100` 100 × 1 = 100; `swesmith-ho` 128 × 2 = 256.
- **tb2/tblite/swe100 rows** are `{"task_id": "<hf-repo>@<revision>:<index>", "prompt_hash": null}` placeholders in pinned source order, with `"task_selection": "pinned-source-order"` — the real Harbor task ids require lowering the pinned packages (agentic.py does not enumerate them), so none were fabricated. Cohort-level loop-audit fields for `loop_audit.py --manifest`: `max_period_tokens: 4096`, `budget: 32768`, `minimum_occurrences: 8` (the frozen `LoopCreditConfig(4096, 32768, 8)`), `interrupt: true` for tb2/tblite.
- **tb2/tblite/swe100 protocol** (recorded per cohort): agent_timeout 1800 s / verifier_timeout 300 s (`CAMPAIGN_JOB`, `agentic.py:211`), max_input_tokens 32768 (`CAMPAIGN_MAX_INPUT_TOKENS`, `:216`), max_output_tokens 8192 (`MAX_OUTPUT_TOKENS`, `harbor.py:77`), model_turn_timeout 600 s (`MODEL_TURN_TIMEOUT`, `harbor.py:74`), attempt_timeout 28800 s for tb2 (`TERMINAL_BENCH_ATTEMPT_TIMEOUT`, `agentic.py:45`) and 14400 s for tblite/swe100 (`ATTEMPT_TIMEOUT` default, `harbor.py:75`), sampling `{temperature/top_p/top_k: null, max_tokens: 8192, stop: [], chat_template_kwargs: {}}` (`harbor_protocol`, `harbor.py:172-191`), harness `terminus-2`.
- **swesmith-ho selection**: read `s3://marin-us-east-02a/marin/rl-data/inputs/swe-verified-task-spec-setup/2026.11.19.1/agentic-train.parquet` (465.3 MB, single cross-region read; under the 500 MB stop threshold). Verified its sha256 `ae53e7a3678a3ef05c5364834644d8eefb5bdc258f78f3b4c4d316fa956a0192` against the sibling release `manifest.json` `output_sha256`; verified all 5,608 rows against the release manifest's per-row `task_id` (exact index alignment) and that every row shares source revision `0e8137a6a1eca1e2c8c12eeb90a16c5d7f1f750c` (dataset `long-loop-smartie-swesmith`). Row hash = sha256 of `json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)` of the parquet row dict (single `task_spec` column); selected the 128 rows with the smallest `sha256(f"20261125:{row_hash}")` (seed 20261125). The manifest stores the rule text, seed, source URI + sha256 + row count + revision, and 128 rows `{"row_hash", "task_id", "index"}` sorted by row_hash (indices span 10–5582; all 128 task ids distinct).
- **rl_training_exclusions**: the same 128 row hashes under `swesmith_ho_row_hashes` with reason "held-out cohort swesmith-ho; excluded from the I3 training pool".
- **swesmith-ho protocol** (recorded): Pi 0.87.0 (`SWESMITH_PI_VERSION`), thinking high / chat-template / setup 360 s (`rl_agentic_recipe.py:36-41`), trial_timeout 14400 s, agent_timeout 10800 s, stall_timeout 28800 s, memory_mb 4096 (`mochi_rl.py:190-195`), verifier_timeout 900 s (`swesmith_training.py:50`), daytona ttl 240 min, max_turns 128 (`agentic_preset`, `rl_agentic_recipe.py:94`), task machine daytona/allow/0 GPUs/user `longloop-agent`/startup 1800 s/cleanup 120 s (`SWESMITH_MACHINE`), sampling "pi-own" (Pi chooses its own sampling, per `PiAgent`).
- **Checks**: `./infra/pre-commit.py --all-files` passes on the new file (4 ruff fixes applied and re-verified); manifest hash re-verified from the file. No GPU jobs, no cluster submissions, nothing committed — working tree left with the changes uncommitted (`cohorts/`, `experiments/post_training/mochi_loop_guards/panel.py` are untracked; pre-existing modified/added tracked files are the parent session's loop-audit work, untouched).
- **Blockers**: none for the panel build. Item (c)'s exact 469-attempt storage path remains unconfirmed (needs marina-app/cluster metadata access; see above — and note the viewer's temp traces expire ≈ 2026-11-28). Item (b) resolved with a discrepancy: the Micro210 rows do not set the 300 s verifier override.
