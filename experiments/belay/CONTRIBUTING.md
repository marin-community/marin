# Contributing to Marin Belay

Marin Belay tests pretraining ideas against the official recipe and records every result. You post an idea; a
maintainer approves it; Belay's agent implements it behind a default-off flag, runs it, and posts the verdict.
Every experiment is compared to the official recipe unless its Track says otherwise.

**Links:** recipe branch [`marin_belay_v1_recipe`](https://github.com/marin-community/marin/tree/marin_belay_v1_recipe)
(model: [`experiments/grug/fast_track/model.py`](https://github.com/marin-community/marin/blob/marin_belay_v1_recipe/experiments/grug/fast_track/model.py))
· intake [marin#9514](https://github.com/marin-community/marin/issues/9514)
· [front page](https://public.applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/) (recipe, tracks, gates)
· [Status](https://public.applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/#status)
· [registry](https://github.com/marin-community/marin/tree/marin_belay_v1_registry/experiments/belay/registry)
· W&B [`marin_moe`](https://wandb.ai/marin-community/marin_moe)

## Flow

1. **Check fit with the recipe.** Read `model.py` (yourself or with an agent) and search the registry for earlier
   attempts. Decide whether the idea replaces, adds to or adapts an existing feature, and how it scales with model
   size.

2. **Post it on #9514**, one idea per comment, with a `Track:` line (default `speedrun`). Plain-text ideas are
   preferred. You can also link a paper, a GitHub repo, or a branch that applies the idea directly to the Belay
   architecture.

   ```
   Track: speedrun
   Replace the sigmoid router scoring with sqrt-softplus on every MoE layer.
   ```

   | Track | Use it for |
   |---|---|
   | `speedrun` (default) | First test: a short run of the recipe at the smallest size. Is it compatible, is there signal? |
   | `vanilla-<rung>` / `moe-<rung>` | Radical ideas: look for signs of life against a plain dense model or a basic MoE first. The rung is the model width, `d512`, `d768`, `d1024` or `d1280` (e.g. `moe-d768`). |
   | `hero-mfu` | Ideas that change speed or memory: step time at hero scale. Run by a maintainer. |
   | `ladder` | After a promising speedrun: every recipe rung, to see whether the gain holds with scale. |
   | `custom` | A study answered by analysis, with no win/loss verdict. It can involve generating plots from checkpoint weights. |

3. **Follow along on Status** (`…/#status/X####` once it has a registry id). Its row links the **code commit** (check
   the diff matches your intent), the **W&B runs** (variant vs. same-commit control), and the agent's plan and verdict.
   Results are single seed; treat anything near the noise band as weak evidence.

4. **If it looks good, post `Track: ladder`** naming its registry id. A ladder without a `hero-mfu` result is
   provisional; ask for `hero-mfu` too if the idea changes step time.

5. **Merge.** Once an idea holds up on `speedrun`, `hero-mfu` and the `ladder`, a maintainer can merge it into the
   official recipe. Depending on the magnitude of the change, further tests may be run first, such as larger
   scaling ladder rungs. Later ideas are compared against the improved recipe.

6. **If the implementation missed your intent, post a new idea** naming the old registry id and what to change.
