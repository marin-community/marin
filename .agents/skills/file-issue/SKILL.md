---
name: file-issue
description: File a GitHub issue for a bug or improvement found this session.
---

# File GitHub Issue

Create an issue in marin-community/marin for a bug, regression, improvement, or
experiment identified in the current conversation. Read AGENTS.md and the
matching guidance in .agents/skills/writing-style/ (SKILL.md, issues.md, and
ai-writing-donts.md) before publishing.

## Kind and body

Choose one kind and labels:

| Kind | Use for | Labels |
|---|---|---|
| bug | Bug or regression | bug, agent-generated |
| task | Improvement, refactor, or feature | agent-generated and known priority |
| experiment | Experiment tracking | experiment, agent-generated |

Bug:

~~~markdown
<concrete symptom and impact>

Reproduce:
1. <step>
2. <step>

Expected: <correct behavior>
<optional evidence or confirmed cause>
~~~

Task:

~~~markdown
<what is needed and why>

Done when:
<specific, testable criteria>
~~~

Experiment issues use TL;DR, Description, Hypothesis or Goal, Status, Links,
Decision Log, and Conclusion sections. Omit empty sections except an initially
empty TL;DR when the experiment just started.

Extract concrete symptoms, location, reproduction, cause if known, impact, and
severity from the conversation. Ask if the issue kind or actionable problem is
ambiguous.

## Draft and duplicate check

Search open issues before filing:

~~~bash
gh issue list --repo marin-community/marin --state open --search "<keyword>"
~~~

If a match exists, tell the user and offer a comment instead of a duplicate.
Use a factual bug title or imperative task title, at most 80 characters, with
an optional scope tag; do not add bug: or task: prefixes.

Keep the smallest matching body. Include the facts needed to understand and act
on the issue, numbered reproduction steps for bugs, and testable Done when
criteria for tasks. Trim history and implementation narration; include only
relevant stack frames in code blocks. Do not use images or tables, repeat the
title, or inventory files that links already identify. Apply the writing-style
compression pass to the exact title and body before sending.

If the user explicitly requested filing, file directly. Otherwise show the draft
and wait for approval.

## Publish

Write the body to a unique temporary file. Never inline multiline shell text or
reuse a fixed path. Re-open and inspect the completed file before running
`gh issue create` so shell output or prompt text cannot enter the issue:

~~~bash
body_file="$(mktemp /tmp/issue-body.XXXXXX.md)"
trap 'rm -f "$body_file"' EXIT
cat > "$body_file" <<'EOF'
<body>
EOF
issue_url="$(gh issue create --repo marin-community/marin \
  --title "<title>" --label "agent-generated" --body-file "$body_file")"
~~~

Add bug or experiment, and a specified or clearly warranted p1, p2, or p3; skip
labels that do not exist. After creation, fetch the published text and correct
any alteration:

~~~bash
gh issue view "$issue_url" --json title,body
~~~

Never credit the agent. Report the issue URL.
