# Marin Belay runbook

This file is the only set of instructions for a Belay session. Follow it exactly. Every command below ends by
printing a line that starts with `next:`. Run that line next, unchanged except for the `<placeholders>` it names.
When it says `next: stop`, end the session with the final message described in section 7.

When any of these happen, run the escalate command in section 6 and stop:

- an output does not match what this file says to expect;
- a command refuses (`refusing: ...`) and its message does not give a `next:` line;
- a step needs a decision this file does not make for you.

A human decides what happens next. Never work around a refusal.

`S` below means `uv run python -m experiments.belay.steps`. Type it out in full every time.

## 0. Every session starts here

1. Run `S start --events <events file from your prompt>`.
   - **Expect:** a `recipe:` line, then either a `do these in order:` list or `next: S week`.
   - **If `next: S week`:** do section 4, then run step 0.1 again.
   - **If `no events`:** stop.
2. Work through the numbered list from top to bottom. Each item names a section and its first command. An item
   that starts with `skip:` needs nothing; go to the next item.

## 1. New approved idea (events `approved`, `work_pending`)

1. **House rules check.** Read the idea's title and body in the events file. Does the idea ask for anything on
   the list in section 8?
   - **If yes, or you cannot tell:** run section 6 with `--idea-id <N>` and the title
     `"Idea #<N> may break house rule: <rule>"`. Stop working on this idea.
   - **If no:** go on.
2. **Register it.** Pick the category from this list whose name best matches the idea. If none clearly matches,
   use `Unsorted`. Categories: `Attention variants`, `Linear attention / KDA`, `Depth connections`,
   `Embeddings and inputs`, `MoE and routing`, `Optimizers`, `Normalization`, `Output head and loss`,
   `Training dynamics and schedules`, `Systems and speed`, `Recipe stacks`, `Unsorted`.
   - **Run:** `S new-idea --idea-id <N> --category "<category>"`
   - **Expect:** the entry's path, then `idea #<N> is X####`, then `next: S flags`.
   - Use that `X####` from here on.
3. **Find the settings.** Run `S flags`. It lists every field a run may change: `--model-set` for model fields
   and `--opt-set` for optimizer fields.
   - **If the idea names exact settings** (for example `--opt-set adam_ademamix_alpha=0`), use them as written.
   - **If the idea describes a change that one or two listed fields express directly** (for example "turn off
     AdEMAMix" with field `adam_ademamix_alpha`), use those fields.
   - **If no listed field does what the idea asks** (it needs new code), run section 6 with `--idea-id <N>` and
     the title `"X#### needs new code: <one line on what>"`. Stop working on this idea.
4. **Write the run spec.** In `<variant>`, say what changes, in at most 60 characters.
   - **Run:** `S spec X#### --variant "<variant>" --opt-set <name>=<value>` (repeat `--opt-set` or
     `--model-set` once per setting).
   - **Expect:** the spec YAML (a `seed0` variant run and a `control` run), the `git` commit and push, and
     `next: S launch X####`.
5. **Launch.**
   - **Run:** `S launch X####`
   - **Expect:** `checks passed: ...`, the plan being reported and posted, one `launching belay-X####-<run>`
     block per run, and `next: stop (...)`.
   - **If it prints `held:`:** stop. The idea stays approved, and a later session launches it.

## 2. Run finished (event `run_finished`, Iris state `succeeded`)

1. **Run:** `S record X####`
   - **If `still running: [...]`:** stop. The last run to finish starts a new session.
   - **If `next: S fail X#### --run <run>`:** do section 3.
   - **Otherwise, expect:** the registry entry being written and pushed, then the result reported and posted with
     a table and one verdict, `win`, `noise` or `loss`, by the fixed rule in section 9. For a `win`, an
     escalation is posted as well. Then `next: stop`.

## 3. Run failed (event `run_finished` with any Iris state other than `succeeded`)

1. **Run:** `S fail X#### --run <run>`
   - **Expect:** an escalation posted with the run's state and log command, then `next: stop`.
   - Never relaunch a failed run. A human reads the logs and decides.

## 4. Weekly rollover (step 0.1 printed `next: S week`)

1. **Run:** `S week`
   - **Expect** one of these:
     - `cutting <this week's branch> from origin/<last week's branch>`, then a merge of the base branch;
     - `cutting ... from origin/<base branch>`, when last week's branch does not exist;
     - `... exists on origin; checking it out`.
   - Then the push, `on <branch>, which contains origin/<base branch>`, and `next: S start --events ...`.
   - **If it prints `merge of ... conflicts`:** run the escalate command it prints, then stop.
2. Go back to step 0.1.

## 5. Recipe change accepted (maintainers only; a session never does this)

A session that is asked to change the recipe runs section 6 and stops. When maintainers accept a new
checkpoint, a maintainer:

1. Pushes the new recipe's code to a new branch.
2. Edits `experiments/belay/config.py`, the only file with fixed values:
   - `recipe_name`, `recipe_branch` and `base_branch`;
   - `launcher`, adding it to `LAUNCHERS` in `submit.py` if new;
   - `rung_minutes`.
3. Commits and pushes that change on the code branch.
4. At the next rollover, runs `S week --fresh` once, so the new week's branch is cut from the new base instead
   of last week's branch.

## 6. Escalate (any mismatch, refusal or judgment call)

1. **Run:** `S escalate --title "<one line>" --body "<what you ran, what it printed, what you expected>"`
   - Add `--idea-id <N>` when it concerns an idea. Later sessions then skip that idea until a maintainer acts.
   - Add `--registry-id X####` when you have one.
   - **Expect:** `posted escalation: ...` (and `reported idea #<N> ...`), then `next: stop`.

## 7. Final message

End with two to five plain lines:

- what you recorded (`X####`: verdict);
- what you launched (run names);
- what you escalated;
- what is still running.

## 8. House rules (an idea that asks for any of these goes to section 6)

- Removing the final-logit (lm_head) z-loss, or Muon momentum below float32.
- Tokenizer changes, or changes that only affect evaluation.
- Fewer than 8 active routed experts, larger shared experts, or more than 64x as many experts as active experts.
- Changing the MoE latent (0.5x hidden) or the MLA KV latent. Expert or shared widths that are not multiples of
  hidden. Anything that would stop d1024 fitting on 8xH100. A full-width routed write outside the final two
  layers.
- Sequence-length or batch-size warmups, token-superposition training, or Inkling `rel_extent` above 2048.
- Anything touching GB200 or the hero run.
- Starting, stopping or changing an Iris cluster, or touching a job Belay did not submit.

## 9. Fixed values (from `experiments/belay/config.py`; the scripts enforce them)

| What | Value |
|---|---|
| Recipe | Belay Ckpt Oct 20 2026, branch `marin_belay_v1_clean_s7`, launcher `s7n` |
| Weekly branch | `<week_branch_prefix><Monday, YYYY-MM-DD>`; demo `marin_belay_v1_loomdemo_w2026-10-26` |
| Weekly branch base | must contain `base_branch` (demo `marin_belay_v1_loomdemo`; production `marin_belay_v1_clean_s7`) |
| Registry branch | demo `marin_belay_v1_registry_demo` (never `marin_belay_v1_registry` in the demo) |
| Runs | rung d512, seed 0, variant plus a same-commit control, on `cw-us-east-02a` at `batch` priority |
| Budget | soft cap 8 nodes across all Belay runs |
| Metric | `eval/paloma/macro_loss`; Δ = variant − control (same seed, same commit) |
| Verdict | Δ ≤ −0.006 `win`; Δ ≥ +0.006 `loss`; otherwise `noise` |
| Posts | the Belay page; also #9451 only when an approver turns GitHub posting on |
