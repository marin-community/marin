---
name: scrub-experiment-issue-tldrs
description: Scheduled scrub workflow for maintaining newcomer-friendly TL;DR blocks on experiment issues.
schedule_cron: "0 8 * * *"
schedule_tz: America/New_York
---

# scrub-experiment-issue-tldrs

Use this skill on scheduled scrub turns that maintain `experiment` issue summaries in `marin-community/marin`.

The Python selector script only decides which issues to inspect and provides thread context. All summary judgment, writing, and GitHub issue editing lives in this markdown workflow.

## Focus

- Keep experiment issues understandable to a technically strong newcomer who does not know the local project history.
- Prefer issues whose current body lacks a managed TL;DR block or whose existing summary is weak, stale, vague, or unlabeled.
- Treat closed issues as fully eligible. They often have the clearest conclusions and are good summary targets.

## Managed Block Format

For each candidate issue, update or add exactly one managed issue-body block bounded by `<!-- experiment-tldr:start -->` and `<!-- experiment-tldr:end -->`.

Write the block as normal Markdown in this shape:

```md
<!-- experiment-tldr:start -->
## Summary

One short newcomer-friendly summary paragraph.

### Helpful links
- <smallest useful set of links>
<!-- experiment-tldr:end -->
```

## Writing Guidance

- Explain the setup, the investigation, and why it mattered.
- State the current conclusion, recommendation, or unresolved blocker in concrete language.
- Improve existing managed summaries whenever they are inaccurate, stale, vague, or miss the real conclusion.
- Improve unmanaged summaries too when the issue still lacks the `tldr` label and the current body is not adequate.
- Treat `250` words as a soft cap for the summary section, not a target.

## Helpful Links Guidance

- Keep the list short.
- Prefer decisive comments, W&B reports, follow-up PRs, linked issues, and similar artifacts that let a reader verify the summary quickly.
- Omit redundant or low-value links.

## Label And Edit Guidance

- The selector script output is the source of truth for candidate order and provided thread context.
- Use `gh issue view --json <fields>`, `gh pr view --json <fields>`, explicit narrow flags such as `--comments`, and `gh api` to inspect related issues, PRs, or comments when the provided context is not enough.
- Skip issues whose body already matches the desired managed block content.
- The `tldr` label means the issue now has an adequate newcomer-friendly summary plus enough supporting links to dig deeper.
- Add the `tldr` label when the issue now meets that bar. Remove it when the issue no longer meets that bar.
- After updating an issue body, add a short `@dlwh` comment describing what changed.

## Output

- Keep the run focused on useful issue-body improvements rather than broad repository changes.
- If there are zero candidates, report that and exit without mutating GitHub.
- If you mutate any issue bodies or labels, report the affected issue numbers and what changed.
- Always end with the required `HARNESS_SCRUB_LOOP` footer supplied by the base scrub contract.
