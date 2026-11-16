# Contributing an idea to Marin Belay

Marin Belay tests pretraining ideas against a fixed recipe and records every result, win or loss. You describe an
idea in a GitHub comment; a maintainer approves it; an agent implements it behind a default-off flag, runs it against
a baseline, and posts the verdict. This page is the whole contributor workflow. Hand it to your agent as context.

**Every experiment is compared to the official recipe by default** (currently Belay Ckpt Oct 20 2026, tag
`belay-ckpt-2026-10-20`): the `speedrun`, `ladder` and `competitive-*` Tracks all run against it. The `vanilla-*`
and `moe-*` Tracks are the opt-in exceptions. Ideas that pass `speedrun`, `hero-mfu` and the `ladder` can be merged
into the official recipe by a maintainer, and later ideas are then measured against the improved recipe.

## Links

| What | Where |
|---|---|
| Recipe code (build on this) | branch [`marin_belay_v1_recipe`](https://github.com/marin-community/marin/tree/marin_belay_v1_recipe), model in [`experiments/grug/fast_track/model.py`](https://github.com/marin-community/marin/blob/marin_belay_v1_recipe/experiments/grug/fast_track/model.py), launched by `experiments/grug/fast_track/belay_s7.py --ladder s7n` |
| Submit ideas | [marin#9514](https://github.com/marin-community/marin/issues/9514), one idea per comment |
| Status page | [Belay Status](https://public.applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/#status); one idea: `…/#status/X####` |
| Front page (gates, tracks, scaling starter kit) | [Belay](https://public.applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/) |
| Experiment registry | branch [`marin_belay_v1_registry`](https://github.com/marin-community/marin/tree/marin_belay_v1_registry/experiments/belay/registry), one `X####-<slug>.yaml` per idea, reports under `experiments/belay/analysis/X####/` |
| W&B runs | project [`marin-community/marin_moe`](https://wandb.ai/marin-community/marin_moe) |
| Experiment code | this week's branch `marin_belay_v1_w<monday>`, e.g. `marin_belay_v1_w2026-11-16`; every run pins a commit |

## Recommended flow

### 1. Check the idea against the current recipe, locally

Check out `marin_belay_v1_recipe` and point your agent at `experiments/grug/fast_track/model.py` (and
`belay_s7.py` for the ladder configs). Before writing anything up, work out:

- **Is it already there, or already tried?** The recipe already has MLA + KDA attention, a 384-expert top-8 MoE,
  gated norms, MuonH, z-loss and a 16k tokenizer. Search the registry for earlier attempts; a past loss is not a
  verdict on the idea, but read how it was implemented.
- **Does it replace, add to, or adapt an existing feature?** Say which, and which optimizer and learning rate its
  new parameters use.
- **Does it scale cleanly?** Ideas that join the recipe must be sized by a rule (e.g. "width 0.5·d", "only the
  last two layers"), not a constant tuned at one size.
- **Out of scope for now:** tokenizer and data changes, eval-only changes, and ideas that win only by adding
  parameters or compute (the ladder compares at matched compute per step).

Iterate on the design until it fits in a short, unambiguous description. Optional: sketch the change as a
default-off config flag in `model.py` and link it in your comment. The agent still implements from your
description, so the description must stand on its own.

### 2. Post the idea on marin#9514

One idea per comment, with a `Track:` line. Without one, the idea runs on `speedrun`. Say exactly what changes,
where, at what size, and how new parameters are initialised and optimised; the agent implements what you write.

```
Track: speedrun
Replace the sigmoid router scoring with sqrt-softplus, on every MoE layer. Keep the 2.5× renormalisation.
```

Pick the Track by what you want to learn:

| Track | Runs against | Use it when |
|---|---|---|
| `speedrun` (default) | the current recipe at d512 for 8 minutes (3,030 steps), with a same-commit control | The normal first test: is the idea compatible with the recipe, and is there signal? |
| `vanilla-d512` … `vanilla-d1280` | a plain dense transformer at that size | A radical idea that would not survive the tuned recipe yet: look for signs of life first. |
| `moe-d512` … `moe-d1280` | a basic MoE at that size | MoE-specific ideas (routing, experts) without the rest of the recipe in the way. |
| `hero-mfu` | step time on one GB200 rack at hero scale | The idea changes speed or memory materially (new kernels, big extra compute). A maintainer runs it. |
| `ladder` | all four `competitive-*` rungs (d512 → d1280), launched together | After a promising speedrun: does the gain hold as the model scales? |
| `competitive-d512` … `competitive-d1280` | the recipe's finished run at one rung | Re-checking one rung. |
| `custom` | a baseline you state | A study answered by analysing runs (no win/loss verdict). The approver sets its budget. |

A maintainer reviews the comment on the admin page; a 👀 reaction means it was reviewed (approved or declined).
Edits made after approval are ignored, so post a new comment instead of editing.

### 3. Follow along and check the implementation matches your intent

Open the Status page and find your idea (search its title, or go to `#status/X####` once it has a registry id).
Its row expands to:

- **Code**: the commit with the agent's implementation. Read the diff. This is the most important check: the flag
  should do what you described, default off, and touch nothing else.
- **Runs**: each run with its progress, loss and a **W&B ↗** link. On `speedrun` the pair is
  `belay-X####-seed0` (your idea) and `belay-X####-control` (same commit, flag off). In W&B, check the run config
  has your flag set in the variant and unset in the control, and that the parameter count changed as you expect.
- **Agent log**: the agent's plan (flags, settings, what it compares) and, when done, the verdict.

The registry entry (`X####-<slug>.yaml` on `marin_belay_v1_registry`) records the same thing permanently: code
commits, every trial's runs, loss, Δ and step-time change. Each finished trial has a report with loss curves at
`experiments/belay/analysis/X####/t00N.md`.

**Reading the verdict.** The metric is Paloma macro loss (lower is better). Δ ≤ −0.005 is a win, ≥ +0.005 a loss,
anything between is noise. Trials are single seed, and seed-to-seed spread is about the size of the noise band, so
treat a result near ±0.005 as weak evidence. On `speedrun`, the verdict is at equal wall-clock time, so an idea
that slows each step pays for it.

### 4. If the result looks promising, submit the ladder

Post a new comment on #9514 that names the idea's registry id:

```
Track: ladder
Follow-up to X0226 (speedrun −0.0112): run the 4-rung ladder with the same implementation.
```

The ladder shows whether the gain holds from d512 to d1280 and projects it to hero scale. Until the idea has a
`hero-mfu` result, ladder rungs run at the baseline's step counts and the verdict is **provisional**: an idea needs a
hero-mfu measurement before it can join the recipe. If the idea changes step time noticeably, ask for `hero-mfu`
first; the ladder's step counts are then adjusted for the measured speed change.

### 5. Once it passes all three, a maintainer merges it into the official recipe

An idea that wins on `speedrun`, has a `hero-mfu` result, and holds up on the `ladder` (with steps adjusted for that
MFU result) is a candidate for the recipe. A maintainer reviews it, weighing anything else that matters at hero
scale (stability, routing balance, inference cost), and merges it: its flag turns on in the recipe, the recipe branch
gets a new `belay-ckpt-<date>` tag, and the Status page marks it "in the current recipe". From then on every new
experiment is compared against the recipe that includes it. You don't need to do anything for this step beyond
answering questions on your idea.

### 6. If the implementation didn't match your intent, submit a new idea

Don't argue with a verdict on the wrong code. Post a new comment that names the old registry id, says what the
implementation got wrong, and states the corrected design precisely:

```
Track: speedrun
Redo of X0123: the implementation applied the gate after the norm; it should gate the residual input before the
norm, initialised to 1.
```

The same applies to a design you want to change after seeing results: new comment, new attempt.
