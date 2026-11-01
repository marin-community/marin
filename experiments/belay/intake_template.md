# Marin Belay intake template

Copy this into a comment on the intake issue ([marin#9514](https://github.com/marin-community/marin/issues/9514)), one
idea per comment. Only **Title** and **Idea** are required; the rest helps the agent run it the way you intend.

Every idea runs against the baseline its **Track** names: one variant against a same-commit control at seed 0.
**Track** is optional and defaults to `speedrun`, the 8-minute d512 run of the current recipe (Belay Ckpt Oct 20
2026). The fastest ideas to run name exact settings (for example `--opt-set adam_ademamix_alpha=0`);
an idea that needs new code waits for a maintainer to add a default-off flag. More seeds or other rungs are a
maintainer's call.

```markdown
**Title:** <one line, e.g. "Sqrt-softplus router combine">
**Track:** <optional; default speedrun | competitive-d512 | competitive-d768 | competitive-d1024 | competitive-d1280L16 |
moe-d512 | moe-d768 | moe-d1024 | moe-d1280 | vanilla-d512 | vanilla-d768 | vanilla-d1024 | vanilla-d1280 | hero-mfu |
custom>
**Category:** <Attention variants | Linear attention / KDA | Depth connections | Embeddings and inputs | MoE and routing |
Optimizers | Normalization | Output head and loss | Training dynamics and schedules | Systems and speed | Recipe stacks |
Unsorted>
**Idea:** <what to change, concretely>
**Why:** <the hypothesis, or what it should improve>
**Settings:** <exact --model-set / --opt-set flags, if you know them>
**Papers:** <arXiv ids, optional>
```

Example: `**Track:** moe-d512` then `**Idea:** Replace the sigmoid router scoring with sqrt-softplus.`
