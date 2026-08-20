---
name: scrub-experiment-issue-tldrs
description: "Scheduled scrub: TL;DR blocks on experiment issues."
---

# Experiment issue TL;DR scrub

On scheduled turns, maintain experiment issue summaries in
marin-community/marin. The selector script is authoritative for candidate
order and thread context; all judgment, writing, and GitHub editing happen here.
Closed issues are eligible.

For each candidate, maintain exactly one managed block:

~~~markdown
<!-- experiment-tldr:start -->
## Summary

One newcomer-friendly paragraph covering setup, investigation, and the current
conclusion, recommendation, or blocker.

### Helpful links
- <smallest useful set of decisive artifacts>
<!-- experiment-tldr:end -->
~~~

Keep the summary below 250 words when possible, and links short and nonredundant.
Improve existing managed or unmanaged summaries when stale, vague, or
incomplete. Use narrow JSON/comments flags when inspecting related issues and
PRs: gh issue view --json, gh pr view --json, --comments, and gh api. Skip a
body already matching the desired block.

The tldr label means the summary and supporting links meet the newcomer bar.
Add it when they do; remove it when they do not. After changing a body, add a
short @dlwh comment describing the change. With zero candidates, report that and
make no GitHub mutations. Report affected issue numbers and changes otherwise.

Finish with exactly:

~~~text
HARNESS_SCRUB_LOOP {"needs_followup_at":null}
~~~

Use a future RFC 3339 timestamp in needs_followup_at when follow-up is required.
