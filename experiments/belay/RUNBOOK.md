# Marin Belay runbook

This file is the only set of instructions for a Belay session. Follow it exactly.

`B` below means `uv run python -m experiments.belay`. Type it out in full every time.

Each section below is one command, `B <section name>`. Every command ends by printing a line that starts with
`next:`. Run that line next, unchanged except for the `<placeholders>` it names. When it says `next: stop`, finish
with the final message in section 7.

Run section 6 (escalate) and stop when:

- an output does not match what this file says to expect;
- a command refuses (`refusing: ...`) without printing a `next:` line;
- a step needs a decision this file does not make for you.

A human then decides what happens. Never work around a refusal.

## 0. start

1. **Run:** `B start --events <the events file named in your prompt>`
2. **Expect:** a `recipe:` line, then one of:
   - `next: B weekly-rollover`: do section 4, then run step 1 again;
   - `no events`: stop;
   - a numbered `do these in order:` list. Work through it from the top. Each item names its section and its
     first command. An item marked `skip:` needs nothing; go on to the next one.

## 1. new-idea (events `approved` and `work_pending`)

1. **Run:** `B new-idea --idea-id <N>`
   - **Expect:** the idea's text, three checks, a list of settings, and a `next:` command with placeholders.
2. **Check 1, house rules.** Does the idea ask for anything in section 8?
   - **If yes, or you cannot tell:** run section 6 with `--idea-id <N>` and the title
     `"Idea #<N> may break house rule: <rule>"`. Stop working on this idea.
3. **Check 2, category.** Pick the listed category whose name best matches the idea. If none clearly matches,
   use `Unsorted`.
4. **Check 3, settings.** Find the listed field or fields that make the change the idea asks for.
   - **If the idea names exact settings** (for example `--opt-set adam_ademamix_alpha=0`), use them as written.
   - **If no listed field makes the change** (it needs new code), run section 6 with `--idea-id <N>` and the
     title `"Idea #<N> needs new code: <one line on what>"`. Stop working on this idea.
5. **Run** the `next:` command from step 1 with the placeholders filled in:
   `B new-idea --idea-id <N> --category "<category>" --variant "<what changes>" --opt-set <name>=<value>`
   (one `--opt-set` or `--model-set` per setting).
   - **Expect**, in this order:
     1. the registry entry `X####` written and pushed;
     2. the run spec written, committed and pushed;
     3. `checks passed: ...`;
     4. the plan reported and posted;
     5. one `launching belay-X####-<run>` per run;
     6. `next: stop (...)`.
   - **If it prints `held:`:** stop. The idea stays approved, and a later session reruns this step.
     Running it again resumes where it stopped.

## 2. run-finished (event `run_finished`)

1. **Run:** `B run-finished X####`
   - **If it prints `still running: [...]`:** stop. The last run to finish starts a new session.
   - **If it prints `next: B run-failed X#### --run <run>`:** do section 3.
   - **Otherwise, expect** these, then `next: stop`:
     1. the registry entry written with the losses and pushed;
     2. the result posted with a table and one verdict, by the fixed rule in section 9: `win`, `noise` or `loss`;
     3. for a `win`, an escalation as well.

## 3. run-failed (event `run_finished` for a run that did not succeed)

1. **Run:** `B run-failed X#### --run <run>`
   - **Expect:** an escalation posted with the run's state and its log command, then `next: stop`.
   - Never relaunch a failed run. A human reads the logs and decides.

## 4. weekly-rollover (when start prints `next: B weekly-rollover`)

1. **Run:** `B weekly-rollover`
   - **Expect:** `checking out <this week's branch> from origin/<source>`. The source is last week's branch, or
     the base branch when there is no last week. Then the push, `on <branch>, which contains origin/<base>`, and
     `next: B start ...`.
   - **If it prints `merging ... conflicts`:** run the escalate command it prints, then stop.
2. Go back to section 0.

## 5. recipe-change (maintainers only; a session never does this)

A session asked to change the recipe runs section 6 and stops. When maintainers accept a new checkpoint, a
maintainer does the following:

1. Pushes the new recipe's code to a branch.
2. Edits `experiments/belay/config.py`, the one file with fixed values: `recipe_name`, `recipe_branch`,
   `base_branch`, `launcher`, `launch_module` and `rung_minutes`.
3. Runs `B recipe-change`. It checks that the branches and the launch module exist.
4. Commits and pushes `config.py`.
5. At the next weekly rollover, runs `B weekly-rollover --fresh` once. This cuts the new week's branch from the
   new base.

## 6. escalate (any mismatch, refusal or judgment call)

1. **Run:** `B escalate --title "<one line>" --body "<what you ran, what it printed, what you expected>"`
   - Add `--idea-id <N>` when it concerns an idea. Later sessions then skip the idea until a maintainer acts.
   - Add `--registry-id X####` when there is one.
   - **Expect:** `posted escalation: ...`, then `next: stop`.

## 7. Final message

End with two to five plain lines:

- what you recorded (`X####`: verdict);
- what you launched (run names);
- what you escalated;
- what is still running.

## 8. House rules (an idea that asks for any of these goes to section 6)

- Removing the final-logit (lm_head) z-loss, or keeping Muon momentum in less than float32.
- Tokenizer changes, or changes that only affect evaluation.
- Fewer than 8 active routed experts, larger shared experts, or more than 64x as many experts as active experts.
- Changing the MoE latent (0.5x hidden) or the MLA KV latent.
- Expert or shared widths that are not multiples of hidden.
- Anything that would stop d1024 from fitting on 8xH100.
- A full-width routed write outside the final two layers.
- Sequence-length or batch-size warmups, token-superposition training, or Inkling `rel_extent` above 2048.
- Anything touching GB200 or the hero run.
- Starting, stopping or changing an Iris cluster, or touching a job Belay did not submit.

## 9. Fixed values (from `experiments/belay/config.py`; the commands enforce them)

| What | Value |
|---|---|
| Recipe | Belay Ckpt Oct 20 2026: branch `marin_belay_v1_clean_s7`, launcher `s7n` (`belay_s7.py --ladder s7n`) |
| Weekly branch | `<week_branch_prefix><Monday as YYYY-MM-DD>`. Demo: `marin_belay_v1_loomdemo_w2026-10-26` |
| Weekly base | Must contain `base_branch`. Demo: `marin_belay_v1_loomdemo`. Production: `marin_belay_v1_clean_s7` |
| Registry branch | Demo: `marin_belay_v1_registry_demo`. The demo never touches `marin_belay_v1_registry` |
| Runs | Rung d512, seed 0: one variant plus a same-commit control, on `cw-us-east-02a` at `batch` priority |
| Budget | Soft cap of 8 nodes across all Belay runs; each run takes one node |
| Metric | `eval/paloma/macro_loss`. Δ = variant − control (same seed, same commit) |
| Verdict | Δ ≤ −0.006 is a `win`; Δ ≥ +0.006 is a `loss`; anything between is `noise` |
| Posts | Always the Belay page. Also #9451, but only when an approver turns GitHub posting on |
