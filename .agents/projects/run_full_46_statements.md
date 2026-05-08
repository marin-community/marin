# Plan — Full 46-statement run with Grok-opposite + Claude judge (overnight)

**Status:** in progress, started 2026-05-08.

## Goal

Bring the Claude-ensemble + Grok-opposite-generator pipeline from 8-statement
coverage to full 46-statement coverage. Use Anthropic Batch API for Claude
(50% off) since it's the dominant cost.

## What's already done — DO NOT REPEAT

| artifact | scope |
|---|---|
| GPT + Gemini judges on 3 original generators (gpt-5.1, Qwen, gemini) | All 46 statements × 4 conditions (already in `grounding/per_judgment.jsonl`, 32,912 rows) |
| Claude judge on 3 original generators | **8 statements only** (`claude_judge_v0/<sid>/{bare,phase_4}_claude.jsonl`) |
| Grok-opposite generator responses | **8 statements only** (`e9_opposite_mode_responses.jsonl`, 160 rows) |
| All 3 judges on Grok-opposite | **8 statements only** (`per_judgment_opposite.jsonl` + `claude_judge_v0_opposite/`) |
| 0-6 anchored rubrics + pilot | side-experiment, not used by this plan |

The 8 already-done statements are:
```
do_not_make_unprompted_personal_comments
be_professional
no_erotica_or_gore
present_perspectives
be_thorough_but_efficient
avoid_hateful_content
no_topic_off_limits
be_clear
```

The remaining 38 statements need: Grok-opposite responses, then GPT + Gemini +
Claude judges on those responses, then Claude judge on the existing 3 generators.

## Phases

### Phase 1 — Grok-opposite generation on 38 remaining statements

Sync run. 38 × 20 = 760 cells via xAI `grok-4-1-fast-non-reasoning` with the
calibrated subtle-deviation system prompt from `opposite_mode.py`.

- Script: `e9_run_opposite_mode_generation.py` with `--missing-only` flag
  (existing script extended to skip statement_ids that already appear in
  `e9_opposite_mode_responses.jsonl`)
- Output: appends to `e9_opposite_mode_responses.jsonl`
- Estimated cost: ~$2 (extrapolating from $0.50 for 160 cells)
- Estimated wall: ~15-20 min
- Mode: **synchronous; runs first**

### Phase 2 — GPT-5.1 + Gemini-3-flash sync judging on the new 760 Grok cells

Extends `e9_judge_opposite_mode.py` to filter for not-yet-judged cells. Uses the
existing 1-5 prompts (`JUDGE_A_SYSTEM` for bare, `JUDGE_RUBRIC_PLUS_SPEC_SYSTEM`
for phase_4) and the original `e8_rubrics.jsonl` rubrics.

- Script: `e9_judge_opposite_mode_remaining.py` (new — sync GPT/Gemini only,
  Claude routed through batch in Phase 3)
- Calls: 760 cells × 2 conditions × 2 judges = **3,040 calls**
- Estimated cost: GPT $7.60 + Gemini $1.52 = **~$9**
- Estimated wall: ~1-2 hours (parallel ThreadPoolExecutor max_workers=6)
- Mode: **synchronous; runs after Phase 1**

### Phase 3 — Claude batch submissions (parallel, fire-and-forget)

Uses Anthropic Batch API (50% discount, up to 24h SLA, typically <1h for batches
this size). Two separate batch submissions for organizational clarity:

- **Batch 3a:** Judge new Grok-opposite responses on 38 statements
  - 760 cells × 2 conditions = 1,520 requests
- **Batch 3b:** Judge existing 3 generators (gpt-5.1, Qwen, gemini) on 38 statements
  - 38 × 60 cells × 2 conditions = 4,560 requests

Both batches use:
- Model: `claude-sonnet-4-6`
- `thinking: {"type": "disabled"}`
- Tool-use forcing with `JUDGMENT_TOOL_1_5` (the tool that recovered 7/7 in
  earlier retries)
- 1-5 anchored prompts (`JUDGE_A_SYSTEM` / `JUDGE_RUBRIC_PLUS_SPEC_SYSTEM`)
- Rubrics from `e8_rubrics.jsonl`

- New module: `batch_anthropic.py` — Anthropic-equivalent of `batch_lib.py`
- Submission script: `e9_submit_claude_batches.py` (builds requests, submits both batches, persists state)
- Total Claude calls: **6,080**
- Estimated cost at 50% batch discount: **~$30** (Anthropic portion)
- Estimated wall: minutes to submit; processing happens server-side over hours
- Mode: **submit-and-go; results fetched in Phase 4**

### Phase 4 — Morning: fetch Claude batches + integrate

Run by hand or via cron when I wake up. The batches will likely be done well
before then (Anthropic typically completes batches of this size within 1-2 hours).

- Script: `e9_fetch_claude_batches.py` (poll → download → parse → integrate)
- Polls each batch_id until `processing_status == "ended"`
- Downloads `results_url`, parses each row's `tool_use.input`
- Splits into:
  - `claude_judge_v0/<sid>/{bare,phase_4}_claude.jsonl` (38 new statements × 60 cells)
  - `claude_judge_v0_opposite/<sid>/{bare,phase_4}_opposite_claude.jsonl` (38 new statements × 20 cells)
- Mode: **synchronous; only ~minutes wall once batches are done**

### Phase 5 — Morning: full-population analysis

After Phase 4, all data is in place. Re-run analysis scripts on the full 46-statement
data:

- `e9_kappa_with_opposite.py` extended to 46 statements
- Compute per-statement α / k2 / k3 with and without Grok-opposite
- Update `.agents/logbooks/claude_judge_spec_repair.md` with full-pop results

## Cost summary

| phase | API | calls | est. cost |
|---|---|--:|--:|
| 1 — Grok generation | xAI | 760 | ~$2 |
| 2 — GPT judge sync | OpenAI | 1,520 | ~$7.60 |
| 2 — Gemini judge sync | Google | 1,520 | ~$1.52 |
| 3a — Claude batch (Grok cells) | Anthropic batch | 1,520 | ~$7.60 |
| 3b — Claude batch (existing 3 gens) | Anthropic batch | 4,560 | ~$22.80 |
| **Total** | | **9,880** | **~$41.50** |
| **Anthropic portion** | | | **~$30.40** |

(Anthropic sync would have been ~$60; batch saves ~$30.)

Running tally for the session prior to this plan: $11.20 Anthropic + $3 OpenAI
+ $1 other = **~$15**. After this plan: **~$56 total session**.

## Wake-up state expectations

By morning, I expect:
- ✅ Phase 1 (Grok generation): **DONE**, outputs in `e9_opposite_mode_responses.jsonl`
- ✅ Phase 2 (GPT + Gemini judges on new Grok cells): **DONE**, outputs in `per_judgment_opposite.jsonl`
- ✅ Phase 3 (Claude batches): **likely DONE** (Anthropic typically completes <2 hours), waiting on Phase 4 fetch
- ⏳ Phase 4 (fetch + integrate): **manual step**, run `e9_fetch_claude_batches.py`
- ⏳ Phase 5 (analysis): **manual**, run after Phase 4

Worst case: a Claude batch fails or stalls. Phase 4 script should handle
partial results and report what's missing for resubmission.

## Failure modes + mitigations

1. **Grok rate limit during Phase 1.** Generation script has retries; if it
   stalls, resume from where it left off.
2. **GPT/Gemini failures during Phase 2.** Each cell logged via RawAPILogger;
   missing cells can be reprocessed.
3. **Anthropic batch rejection.** If a batch is rejected (size too large,
   schema invalid), submission script reports the error and saves partial
   batch state for retry.
4. **Anthropic batch stalls > 24h.** Per Anthropic docs, batches can take
   up to 24h. If still pending in the morning, extend wait or resubmit.
5. **Tool-use forcing fails on some cells.** From earlier work, tool-use forcing
   was 100% reliable on the 8-statement run; expect same.

## Persistence

All artifacts are committed to git as they're produced (matching the precedent
from commits 036025005 + e62dd237e). Raw API dumps and JSONL outputs both
persistent for cross-worktree handoff.
