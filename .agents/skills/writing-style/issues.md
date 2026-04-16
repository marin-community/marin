# Issues

Use this file for GitHub issues. Marin uses two issue styles with different expectations.

## Pick The Right Issue Type

- Use a standard OSS issue for code behavior, bugs, feature requests, and refactors.
- Use an experiment issue for learning that evolves over time.

## Standard OSS Issues

See also the [file-issue](../file-issue/SKILL.md) skill which focuses
more on what should be included rather than tone.

### Assume This Reader

- Assume the reader is a contributor or user who knows the repo but not this problem.

### Write This Way

- State the problem clearly and early.
- Include the smallest useful repro, logs, stack traces, or examples.
- Avoid unexplained shorthand.
- Keep the discussion focused and actionable.
- Use a direct, practical tone with no narrative padding.

### Example Shape

- `Fails when chunk size > 256 due to X (see logs below).`

## Experiment Issues

Treat experiment issues as part of Marin's scientific record. Use them as a summary layer, a coordination surface, and a long-lived artifact.

They are frequently created using the [agent-research](../agent-research/SKILL.md) skill.

### Assume This Reader

- Assume the reader understands LLM systems broadly.
- Assume the reader was not present for the work.
- Assume the reader may revisit the issue much later.
- For issue comments, assume the reader has read the summary and recent comments, but not necessarily every comments in the thread.

### Keep The Key Principle

- Write so someone else can understand what you did, reproduce it, and evaluate it without extra context.

### Keep The Expected Structure

- Maintain a clear TL;DR and evolving conclusion.
- Link the research logbook, W&B runs, commits, and tags.
- Track the decision log and negative results.
- Reflect current understanding instead of pasting raw logs.
- Treat the logbook as the detailed record and the issue as the interpreted summary.

### Write Updates This Way

- Answer four questions in each update: what changed, what was the result, how confident are you, and what is next.
- Keep updates concise, append-only, and link-heavy.
- Emphasize what changed, what was learned, and current belief.

### Keep The Tone

- Stay calm and factual.
- Be comfortable reporting negative results.
- Avoid hype and exaggerated reactions.

### Avoid These Failure Modes

- Do not let the issue become a chat log.
- Do not forget to update the TL;DR or conclusion.
- Do not lose the baseline.
- Do not overly clutter the issue. The log book from the agent-research skill should be the full log.
- Do not assume knowledge of the conversation/thread or the logbook. Each comment should be understandable in the context of the issue.
