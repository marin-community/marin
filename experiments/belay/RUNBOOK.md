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
   - **Expect:** the idea's text, its Track and baseline, three checks, the settable fields, and a `next:` line.
   - **If it refuses because of the Track** (missing, unknown, `custom` or `hero-mfu`): run the escalate command
     it prints, then stop working on this idea.
2. **Check 1, house rules.** Does the idea ask for anything in section 8?
   - **If yes, or you cannot tell:** run section 6 with `--idea-id <N>` and the title
     `"Idea #<N> may break house rule: <rule>"`. Stop working on this idea.
3. **Check 2, category.** Pick the listed category whose name best matches the idea. If none clearly matches,
   use `Unsorted`.
4. **Check 3, settings.** Find the listed field or fields that make the change the idea asks for.
   - **If the idea names exact settings** (for example `--opt-set adam_ademamix_alpha=0`), use them as written.
   - **If no listed field makes the change:** the idea needs code. Run
     `B new-idea --idea-id <N> --category "<category>"`, which registers `X####` and prints
     `next: B code-change X#### --start`. Do section 1b.
5. **Run:** `B new-idea --idea-id <N> --category "<category>" --variant "<what changes>" --opt-set <name>=<value>`.
   Give one `--opt-set` or `--model-set` per setting.
   - **Expect**, in this order:
     1. the registry entry `X####` written and pushed (skipped if it already exists);
     2. the run spec written, committed and pushed;
     3. `checks passed: ...`;
     4. the plan reported and posted;
     5. one `launching belay-X####-<run>` per run;
     6. `next: stop (...)`.
   - **If it prints `held:`:** stop. The idea stays approved, and a later session reruns this step. Running it
     again resumes where it stopped.

## 1b. code-change (an idea that needs new code)

The change goes straight onto this week's branch. No human reviews it: passing `--finish` is the only gate.

1. **Run:** `B code-change X#### --start`
   - **Expect:** the rules below and `next: B code-change X#### --finish --model-set <flag>=<on value>`.
2. **If the idea points at a pull request** (`#NNNN`), read it with `gh pr view NNNN --json title,body,files` and
   `gh pr diff NNNN`, and port only its change to the model or optimizer, behind the new flag.
   If you cannot tell which part of the PR the idea asks for, run section 6 and stop.
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

1. **Run:** `B run-finished X####`
   - **If it prints `still running: [...]`:** stop. The last run to finish starts a new session.
   - **If it prints `next: B run-failed X#### --run <run>`:** do section 3.
   - **Otherwise, expect** these, then `next: stop`:
     1. the registry entry written with the results and pushed;
     2. the result posted with a table and one verdict, by the track's rule in section 9: `win`, `noise` or `loss`;
     3. for a `win`, an escalation as well.

## 3. run-failed (event `run_finished` for a run that did not succeed)

1. **Run:** `B run-failed X#### --run <run>`
   - **Expect:** an escalation posted with the run's state and its log command, then `next: stop`.
   - Never relaunch a failed run. A human reads the logs and decides.

## 4. weekly-rollover (when start prints `next: B weekly-rollover`)

1. **Run:** `B weekly-rollover`
   - **Expect:**
     - `checking out <this week's branch> from origin/<source>`. The source is last week's branch, which keeps its
       flags, or `marin_belay_v1_recipe` when there is no last week.
     - The push, then `on <branch>, which contains origin/<base>`.
     - The list of flag commits since the recipe. A maintainer deletes the flags that never won; sessions do not.
     - `next: B start ...`.
   - **If it prints `merging ... conflicts`:** run the escalate command it prints, then stop.
2. Go back to section 0.

## 5. recipe-change (maintainers only; a session never does this)

Only this step changes the recipe, and only a maintainer runs it. A session asked to change the recipe runs
section 6 and stops.

1. **Run:** `B recipe-change --from <branch>`. It does the following:
   - checks that every Track's launch module exists;
   - merges the branch into `marin_belay_v1_recipe` (fast-forward when possible);
   - tags the new tip `belay-ckpt-<YYYY-MM-DD>`, named "Belay Ckpt <Mon DD YYYY>";
   - pushes both, and lists the flag commits since the old recipe.
2. Set `recipe_tag` in `experiments/belay/config.py` to the new tag, then commit and push.
3. At the next weekly rollover, run `B weekly-rollover --fresh` once. This cuts the new week's branch from the new
   recipe.

## 6. escalate (any mismatch, refusal or judgment call)

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

## 9. Tracks and fixed values (from `experiments/belay/config.py`; the commands enforce them)

Every idea may name its Track on a `Track:` line; with no such line it is `speedrun`. The Track sets the baseline
it runs against and the rule it is judged by. The variant and its control (`belay-X####-control`) run the same
baseline command at the same commit and seed. The only difference is the variant's flags.

`B tracks` prints the one definition of every Track: its code branch, exact launch command, hardware, priority,
metric and rule. Nothing else spells out a launch.

| Track | Baseline | Metric | Rule |
|---|---|---|---|
| `vanilla-d512` … `vanilla-d1280` | Vanilla Track at that rung | Paloma macro loss | Δ = variant − control; ≤ −0.006 win, ≥ +0.006 loss |
| `moe-d512` … `moe-d1280` | Basic MoE at that rung | Paloma macro loss | same |
| `competitive-d512`, `-d768`, `-d1024`, `-d1280L16` | The current recipe (Belay Ckpt Oct 20 2026) at that rung | Paloma macro loss | same |
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
| Budget | Soft cap of 8 nodes across all Belay runs |
| Posts | Always the Belay page. Also #9451, but only when an approver turns GitHub posting on |

## 10. Known limits

- Sessions run with an allowlisted environment (`demo.SESSION_ENV`) and no storage keys. Training jobs they launch
  still get the cluster's shared `iris-task-env` credentials; a Belay-scoped key or workload identity is a later fix.
- `code-change` probes a tiny CPU model. It proves the flag-off path is unchanged and that the flag-on path does not
  leak future tokens. It does not prove an optimizer flag is inert when off: only its default is checked.
- Ideas run from the text as of their first approval (GitHub 👍 or the admin page). A comment edited after approval is refused until it is re-approved.
  Hidden HTML and invisible characters are removed before a session reads the text, and the plan says so.
