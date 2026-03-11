# Experiment Issue TL;DR Prompt

The Python selector script only decides which issues to inspect and provides thread context. All summary judgment, writing, and issue editing lives in this markdown workflow.

For each candidate issue, update or add a managed issue-body block bounded by `<!-- experiment-tldr:start -->` and `<!-- experiment-tldr:end -->`.

Write the block as normal Markdown using this shape:

```md
<!-- experiment-tldr:start -->
## Summary

One short newcomer-friendly summary paragraph.

### Helpful links
- <smallest useful set of links>
<!-- experiment-tldr:end -->
```

What the summary should do:

- Explain the setup, the investigation, and why it mattered.
- State the current conclusion, recommendation, or unresolved blocker in concrete language.
- Help a technically strong newcomer understand the issue without already knowing Marin's local history.
- Improve an existing managed summary whenever it is weak, vague, stale, or misses the actual conclusion.
- Improve unmanaged summaries too when the issue still lacks the `tldr` label and the current body is not adequate.
- Treat `250` words as a soft cap for the summary section, not a target.

What to include under `### Helpful links`:

- Keep the list short.
- Prefer decisive comments, W&B reports, follow-up PRs, linked issues, and other artifacts that help someone verify the summary quickly.
- Omit low-value or redundant links.

Label and scope guidance:

- Closed issues are fully eligible and often better candidates because they usually have clearer conclusions.
- The `tldr` label means the issue now has an adequate newcomer-friendly summary plus enough supporting links to dig deeper.
- If the issue body already matches the desired managed block content, skip editing it.

Tool guidance:

- Start from the JSON emitted by the selector script; it is the source of truth for candidate order and context.
- Use `gh issue view`, `gh pr view`, and `gh api` to inspect related issues, PRs, or comments when the provided context is not enough.
- Use `gh issue edit` to update the body and labels.
- After changing an issue body, add a short `@dlwh` comment describing what changed.
