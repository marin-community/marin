# Marin Belay intake template

Copy this into a comment on the intake issue ([marin#9514](https://github.com/marin-community/marin/issues/9514)), one
idea per comment. Only **Title** and **Idea** are required; the rest helps the agent run it the way you intend.

Every idea runs against the current recipe, Belay Ckpt Oct 20 2026, at d512: one variant against a same-commit
control at seed 0. The fastest ideas to run name exact settings (for example `--opt-set adam_ademamix_alpha=0`);
an idea that needs new code waits for a maintainer to add a default-off flag. More seeds or other rungs are a
maintainer's call.

```markdown
**Title:** <one line, e.g. "Sqrt-softplus router combine">
**Category:** <Attention variants | Linear attention / KDA | Depth connections | Embeddings and inputs | MoE and routing |
Optimizers | Normalization | Output head and loss | Training dynamics and schedules | Systems and speed | Recipe stacks |
Unsorted>
**Idea:** <what to change, concretely>
**Why:** <the hypothesis, or what it should improve>
**Settings:** <exact --model-set / --opt-set flags, if you know them>
**Papers:** <arXiv ids, optional>
```
