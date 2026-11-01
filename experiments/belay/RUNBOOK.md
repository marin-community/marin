# Marin Belay runbook

This file is the only set of instructions for a Belay session. Follow it exactly.

`B` below means `uv run python -m experiments.belay`. Type it out in full every time.

Each section below is one command, `B <section name>`. Every command ends by printing a line that starts with
`next:`. Run that line next, unchanged except for the `<placeholders>` it names. When it says `next: stop`, go on
with the drain loop (section 0b) unless it says the session is done.

A session drains the queue, then exits:

1. `B start --events <file>` and work through every item it lists (sections 1 to 3, 6).
2. `B wait --events <that file>`, then follow its `next:` line, repeating from step 1 with the file it prints.
3. When `wait` prints `drained:`, finish with the final message (section 7).

Run section 6 (escalate) and stop when:

- an output does not match what this file says to expect;
- a command refuses (`refusing: ...`) without printing a `next:` line;
- a step needs a decision this file does not make for you.

A human then decides what happens. Never work around a refusal.

## 0. start

1. **Run:** `B start --events <the events file named in your prompt>`. A scheduled session whose prompt names no
   events file runs `B claim` first; it claims the queued events and prints the file to pass.
2. **Expect:** a `recipe:` line, then one of:
   - `next: B weekly-rollover`: do section 4, then run step 1 again;
   - `no events`: stop;
   - a numbered `do these in order:` list. Work through it from the top. Each item names its section and its
     first command. An item marked `skip:` needs nothing; go on to the next one.

## 0b. wait (the drain loop)

1. **Run:** `B wait --events <the events file you last worked through>` (give the shell command a 10-minute
   timeout; a scheduled session adds the `--session <id>` that `claim` printed). It polls the Belay page once a minute and spends no model turns while it waits.
2. **Expect** one of:
   - `claimed <n> event(s) into <file>`: new work (a finished run, an overdue run, or a newly approved idea). Run
     `B start --events <file>` and work through it, then run `wait` again with that file.
   - `still waiting on <n> run(s) ...`: runs due before this session's hard stop are still going. Run the same
     `wait` again.
   - `drained: ...`: nothing is queued and nothing short is running. Runs due later (long ladder rungs) are left to
     a later session, which the page starts when they finish or run late. Finish with section 7.
3. **Budget:** after 3 hours or 10 ideas, `wait` takes only run results (finished or overdue runs), so the session
   finishes what it launched; at 3 hours 45 minutes it prints `drained:` regardless.

## 1. new-idea (events `approved` and `work_pending`)

1. **Run:** `B new-idea --idea-id <N>`
   - **Expect:** the idea's text, its Track and baseline, three checks, the settable fields, and a `next:` line.
   - **If it prints `posted escalation: Idea #<N> was already tested as X####`:** the idea has prior trials and
     goes to a maintainer. Stop working on this idea. It runs again only when a maintainer re-queues it with the note
     "retest on current recipe".
   - **If it refuses because of the Track** (unknown, `custom` or `hero-mfu`): run the escalate command it prints,
     then stop working on this idea. A missing Track is `speedrun`.
   - **If it prints a `MAINTAINER NOTE`:** a maintainer re-queued the idea with instructions. Follow them; they
     settle any check they address (for example, how a port keeps a house rule), and list each choice they fix in
     `--defaults`.
2. **Check 1, house rules.** Does the idea ask for anything in section 8? A maintainer note that says how to meet
   the rule settles it.
   - **If yes, or you cannot tell:** run section 6 with `--idea-id <N>` and the title
     `"Idea #<N> may break house rule: <rule>"`. Stop working on this idea.
3. **Check 2, category.** Pick the listed category whose name best matches the idea. If none clearly matches,
   use `Unsorted`.
4. **Check 3, settings.** Find the listed field or fields that make the change the idea asks for.
   - **If the idea names exact settings** (for example `--opt-set adam_ademamix_alpha=0`), use them as written.
   - **If no listed field makes the change:** the idea needs code. Run
     `B new-idea --idea-id <N> --name "<short name>" --category "<category>"`, which registers `X####` and prints
     `next: B code-change X#### --start`. Do section 1b.
   - **If the idea leaves a setting open** (a value, a range, a schedule length), choose a sensible value and list it
     in `--defaults`, for example `--defaults "final momentum 0.98; warmup over the first 10% (idea suggests 10-20%;
     chose 10%)"`. Escalate only when no sensible value exists. With nothing left open, pass `--defaults none`.
5. **Run:** `B new-idea --idea-id <N> --name "<short name>" --category "<category>" --variant "<what changes>"
   --defaults "<chosen values, or none>" --opt-set <name>=<value>`. Give one `--opt-set` or `--model-set` per
   setting. `--name` is the registry title, at most 50 characters, such as "MuonH momentum warmup"; an idea that
   already has an `X####` keeps its name.
   - **Expect**, in this order:
     1. the registry entry `X####` written and pushed (skipped if it already exists);
     2. the run spec written, committed and pushed (with `reusing the running control <run>` when another
        idea's control at the same code and seed is still running: the two share it);
     3. `checks passed: ...`;
     4. the idea's #9451 comment posted (status `plan`, the planned runs with their steps);
     5. one `launching belay-X####-<run>` per run;
     6. `next: stop (...)`.
   - **If it prints `held:`:** the launch would pass the node cap (set on the admin page) or the idea's node grant.
     Stop working on this idea; it stays approved, and a later session reruns this step. Running it
     again resumes where it stopped.

## 1b. code-change (an idea that needs new code)

The change goes straight onto this week's branch. No human reviews it: passing `--finish` is the only gate.

1. **Run:** `B code-change X#### --start`
   - **Expect:** the rules below and `next: B code-change X#### --finish --model-set <flag>=<on value>`. For an
     idea with recorded code that this branch lacks, it first prints `applied X####'s recorded code (uncommitted)`;
     check it, then go to step 4.
2. **If the idea points at a pull request** (`#NNNN`), read it with `gh pr view NNNN --json title,body,files` and
   `gh pr diff NNNN`, and the PRs it builds on the same way. Port the change, and everything it depends on, into
   `experiments/grug/fast_track/`:
   - all of it behind the idea's one default-off flag;
   - the GPU path only (skip TPU-specific code);
   - data the change needs (for example a table built from the current tokenizer's vocabulary) as files in
     `experiments/grug/fast_track/`.
   Size is not a reason to stop: the six gates are the only test. Run section 6 only if the code cannot run on the
   H100 fast_track model at all.
3. **Edit** only `experiments/grug/fast_track/` and `tests/grug/test_fast_track_*.py`:
   - add one new field to `GrugModelConfig` (`model.py`) or `GrugMoeMuonHConfig` (`optimizer.py`) whose default
     is off (`False`, `None`, `0` or `0.0`);
   - put every behavior change behind that field;
   - add `tests/grug/test_fast_track_<flag>.py`, a test that turns the flag on.
   Do not run the test yourself; `--finish` runs it (gate 5).
4. **Run:** `B code-change X#### --finish --model-set <flag>=<on value>` (use `--opt-set` for an optimizer flag).
   These are the tests that gate the change, in order. All must pass:
   1. **Paths:** only the two allowed paths changed, including a test file.
   2. **Default off:** at least one new config field; every new field defaults to off; no existing default changed.
   3. **Flag off is unchanged:** with every flag off, a tiny CPU model's loss and every gradient are bit-identical to
      the commit before the change.
   4. **No leakage** (`--model-set` flags): with the flag on, changing future tokens leaves past hidden states
      unchanged (at most 1e-6).
   5. **Its own tests** pass on CPU.
   6. **`./infra/pre-commit.py`** passes on the changed files.
   - **Expect:** six numbered `ok` lines, then the commit `X####: <flag> (default off)` pushed to the weekly branch,
     and `next: B new-idea --idea-id <N> ... --model-set <flag>=<on value>`. Fill in the category and the variant,
     then run it (section 1, step 5).
   - **If it refuses:** follow its `next:` line. It never commits a failing change. To give up on the change, run
     `git checkout -- .` and `git clean -fd experiments/grug/fast_track tests/grug`, then run section 6.

## 2. run-finished (event `run_finished`)

Each idea has exactly one #9451 comment. `new-idea` posts it; every later step edits it in place (title status
and one section per stage). Its link is stored on the idea's registry trials (`comment_url`).

1. **Run:** `B run-finished X####`
   - **If it prints `still running: [...]`:** stop. The last run to finish starts a new session.
   - **If it prints `next: B run-failed X#### --run <run>`:** do section 3.
   - **Otherwise** it records the stage that finished and, for the `speedrun` track, starts the next stage by
     itself. The stages are fixed:
     1. **Equal data:** the variant and its control at the baseline's steps.
     2. **The rule after equal data**, from Δt = variant s/step ÷ control s/step − 1 (the comment states which
        branch applied under stage 1):
        - |Δt| ≤ 1%: equal data already is equal time; stage 1 is final ("<verdict> at equal time (step time
          within 1%)").
        - Variant slower (Δt > +1%): stage 2 only after a stage-1 win, since fewer steps can't turn noise or a
          loss into a win. Otherwise final: "<verdict> at equal data; slower, equal time not run".
        - Variant faster (Δt < −1%): stage 2 always, since it gets more steps at equal time.
     3. **Timing probe:** a 150-step variant run that measures its s/step.
     4. **Equal time:** the variant at 480 s ÷ its s/step steps, against the equal-data control. This verdict is
        final.
   - Other Tracks have one stage: equal data.
   - **Expect**, at each recorded stage:
     - the registry entry and the trial's analysis report pushed together
       (`pushed ... as <sha> to origin/<registry branch>`);
     - `edited <comment url>`;
     - then either the next stage's `launching ...` lines, or `next: stop`.
   - A final win also escalates to the admin page for the adoption decision.

## 2b. analyze (rewrite a trial's analysis report)

`run-finished` writes the report itself. Run this only when asked to backfill or redo one:
`B analyze X####`. It rewrites, commits and pushes the reports of every finished stage.

## 2d. score (after every `run-finished` that records a stage, and after an escalation of a registered idea)

1. **Run:** `B score X#### --score <0-10> --rationale "<one line>" --model <your model name>`. It stores
   `potential` (score, rationale, model, date, and a digest of the results it saw) on the registry entry. The
   Status page shows it; nothing else writes it.
2. **Score with this rubric**, from the entry's trials and events. A win's validations are equal-time speedrun,
   the competitive-ladder rungs (scaling), `hero-mfu` (MFU cost) and a vanilla-track (Dense) win; the noise band is ±0.005 Paloma macro loss.
   | Score | When |
   |---|---|
   | 0–2 | Clear loss |
   | 3 | Noise: no stage beyond ±0.005 |
   | 4 | Noise with a consistent sign, or mixed results with a clear fix |
   | 5 | No result yet |
   | 6 | Small win (−0.005 to −0.010) with one validation (equal data only, or equal time only) |
   | 7 | Larger win (≤ −0.010) with one validation, or a small win holding at equal time and on one more axis |
   | 8 | Win holding at equal time **and** on the scaling ladder (or with an acceptable `hero-mfu` cost) |
   | 9 | Large win holding at equal time, on the scaling ladder **and** on `hero-mfu` |
   | 10 | Adopted into the live recipe |
   An entry that is not an idea (a bundle of other entries) gets `--score na`: no score, sorted last.
   The rationale names the win size and which validations it passed, e.g. "−0.0148 at equal time; ladder and MFU
   not run → 7".

## 2c. launch (reproduce or show one run)

Every launch goes through one path and records the exact submitted command in the spec under `launched:`.
`B launch X#### --run <run> --print` prints the full command a run would submit (module, flags, steps, cluster,
priority, run id) and submits nothing. Without `--print` it submits that run with the same checks as `new-idea`;
sessions run it only when a maintainer asks. Each post shows a `Reproduce:` line with the commit and command.

## 3. run-failed (event `run_finished` for a run that did not succeed)

1. **Run:** `B run-failed X#### --run <run>`
   - **Expect:** an escalation posted with the run's state and its log command, then `next: stop`.
   - Never relaunch a failed run. A human reads the logs and decides.

## 4. weekly-rollover (when start prints `next: B weekly-rollover`)

Each week's branch starts fresh from `marin_belay_v1_recipe`, the last accepted recipe, and carries only the code
of ideas still in play: ideas in progress (approved, planned or running, or with a stage still pending) and wins
not yet adopted. Their code commits come from the registry (`code_commits`). Losses, noise, declined and closed
ideas stay behind on the old branch; settings-only ideas carry nothing.

1. **Run:** `B weekly-rollover`
   - **Expect:**
     - `carrying <n> idea(s): X####, ...`, then `cutting <branch> from origin/marin_belay_v1_recipe` and one
       `carrying <commit> X####: <flag> (default off)` line per carried commit;
     - the push, then `on <branch>: origin/marin_belay_v1_recipe plus <n> carried commit(s)`;
     - `next: B start ...`.
   - **If a carried commit conflicts, or the carried code fails its code-change gates on the recipe:** it pushes
     nothing; run the escalate command it prints, then stop.
2. Go back to section 0.

To retest an idea that stayed behind, a maintainer re-queues it with the note "retest on current recipe". If it
needs its code, `B code-change X#### --start` applies its recorded commits (uncommitted), and `--finish` runs the
gates again and commits them.

## 5. recipe-change (maintainers only; a session never does this)

Only this step changes the recipe, and only a maintainer runs it. A session asked to change the recipe runs
section 6 and stops.

1. **Run:** `B recipe-change --from <branch>`. It does the following:
   - checks that every Track's launch module exists;
   - merges the branch into `marin_belay_v1_recipe` (fast-forward when possible);
   - tags the new tip `belay-ckpt-<YYYY-MM-DD>`, named "Belay Ckpt <Mon DD YYYY>";
   - pushes both.
2. Set `recipe_tag` in `experiments/belay/config.py` to the new tag, then commit and push.
3. The next weekly rollover cuts the new week's branch from the new recipe.

## 6. escalate (any mismatch, refusal or judgment call)

Escalations go to the admin page, never to #9451: only plans and results are public. The admin page shows the idea
as "needs maintainer" with this session's transcript. No session takes the idea up again until a maintainer
re-queues it (with a note) or declines it.

1. **Run:** `B escalate --title "<one line>" --body "<what you ran, what it printed, what you expected>"`
   - Add `--idea-id <N>` when it concerns an idea. Later sessions then skip the idea until a maintainer acts.
   - Add `--registry-id X####` when there is one.
   - **Expect:** `posted escalation: ...`, then `next: stop`.

## 7. Final message

End with two to five plain lines:

- what you recorded (`X####`: verdict);
- what you launched (run names);
- which flags you committed;
- what you escalated;
- what is still running.

A scheduled session (one that ran `B claim`) then reports these lines with
`B finish --session <id from claim> --summary "<the lines>"`, which frees the queue for the next session.

## 8. House rules (an idea that asks for any of these goes to section 6)

- Removing the final-logit (lm_head) z-loss, or keeping Muon momentum in less than float32.
- A new or modified tokenizer, or any change to tokenization. Reading the current 16k vocabulary (for example
  grouping tokens by their decoded bytes) is allowed.
- Changes that only affect evaluation.
- Fewer than 8 active routed experts, larger shared experts, or more than 64x as many experts as active experts.
- Changing the MoE latent (0.5x hidden) or the MLA KV latent.
- Expert or shared widths that are not multiples of hidden.
- Anything that would stop d1024 from fitting on 8xH100.
- A full-width routed write outside the final two layers.
- Sequence-length or batch-size warmups, token-superposition training, or Inkling `rel_extent` above 2048.
- Anything touching GB200 or the hero run.
- Starting, stopping or changing an Iris cluster, or touching a job Belay did not submit.

## 9. Tracks and fixed values (from `experiments/belay/config.py`; the commands enforce them)

Every idea may name its Track on a `Track:` line; with no such line it is `speedrun`. The Track sets the baseline
it runs against and the rule it is judged by. The variant and its control (`belay-X####-control`) run the same
baseline command at the same commit and seed. The only difference is the variant's flags.

`B tracks` prints the one definition of every Track: its code branch, exact launch command, hardware, priority,
metric and rule. Nothing else spells out a launch.

| Track | Baseline | Metric | Rule |
|---|---|---|---|
| `vanilla-d512` … `vanilla-d1280` | Vanilla Track at that rung | Paloma macro loss | Δ = variant − control; ≤ −0.005 win, ≥ +0.005 loss |
| `moe-d512` … `moe-d1280` | Basic MoE at that rung | Paloma macro loss | same |
| `competitive-d512`, `-d768`, `-d1024`, `-d1280` | The current recipe (Belay Ckpt Oct 20 2026) at that rung | Paloma macro loss | same |
| `speedrun` (the default) | The 8-minute d512 run of the current recipe (`belay-8min-ckpt-oct20`: 3,030 steps, Paloma macro 3.0441) | Paloma macro loss | same |
| `hero-mfu` | The one-rack GB200 MFU stack `mfl-mp-qall` (11.465 s/step, 36.8% MFU), on `marin_belay_v1_hero_mfu` | s/step | Δ = (variant − control) / control, same rack, commit and PGLE profile; ≤ −0.5% win, ≥ +0.5% loss. A maintainer approves each launch |
| `custom` | Stated by the submitter | stated | escalate |

| What | Value |
|---|---|
| Recipe | `marin_belay_v1_recipe`, tagged `belay-ckpt-2026-10-20` (Belay Ckpt Oct 20 2026). Only section 5 moves it |
| Weekly branch | `<week_branch_prefix><Monday as YYYY-MM-DD>`. Demo: `marin_belay_v1_loomdemo_w2026-10-26` |
| Weekly base | Must contain `base_branch`. Production: `marin_belay_v1_recipe`. Demo: `marin_belay_v1_loomdemo` |
| Registry branch | Demo: `marin_belay_v1_registry_demo`. The demo never touches `marin_belay_v1_registry` |
| Runs | Seed 0; cluster, priority and nodes per Track (`B tracks`) |
| Budget | A soft cap on nodes across all Belay runs, set on the admin page (default 8); a maintainer may grant one idea more nodes at approval. Sessions read both from the page and never change them |
| Posts | Always the Belay page. Also #9451, but only when an approver turns GitHub posting on |

## 10. Known limits

- Sessions run with an allowlisted environment (`demo.SESSION_ENV`) and no storage keys. Training jobs they launch
  still get the cluster's shared `iris-task-env` credentials; a Belay-scoped key or workload identity is a later fix.
- `code-change` probes a tiny CPU model. It proves the flag-off path is unchanged and that the flag-on path does not
  leak future tokens. It does not prove an optimizer flag is inert when off: only its default is checked.
- Ideas run from the text as of their first approval (GitHub 👍 or the admin page). A comment edited after approval is refused until it is re-approved.
  Hidden HTML and invisible characters are removed before a session reads the text, and the plan says so.
