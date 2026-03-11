# Experiment Issue TL;DR Prompt

Maintain a managed issue-body block with this exact envelope:

```md
<!-- experiment-tldr:start -->
## Summary

<newcomer-friendly summary>

### Helpful links
- <relevant link>
<!-- experiment-tldr:end -->
```

Writing guidance:

- The summary is for a technically strong newcomer who understands ML systems or research, but not Marin's local codebase or prior issue history.
- The goal is not to restate the title. Explain the setup, what was being investigated, why it mattered, and what the current conclusion or recommendation is.
- Use concrete language. If the issue reached a conclusion, say what the conclusion is and what configuration or follow-up seems recommended.
- Treat `250` words as a soft cap for the summary section. Shorter is fine when the issue is simple.
- Helpful links should be the smallest useful set. Prefer W&B reports, decisive comments, follow-up PRs, and closely related issues.
- If the issue already has a managed summary block but it is weak, vague, stale, or missing the point, replace it instead of preserving it.
- The `tldr` label means the issue is now understandable to a newcomer and has enough supporting links to dig deeper.
- Closed issues are still eligible and should often be better summarization targets because they are more likely to have conclusions.
- After editing an issue body, add a short comment tagging `@dlwh` describing the refresh.
