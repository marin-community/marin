---
name: file-issue
description: File a GitHub issue only when the user explicitly requests one or invokes a workflow whose stated purpose intrinsically requires a new issue; delegation from another skill alone is insufficient.
---

# File a GitHub issue

Creating an issue is an external publishing action. Do not activate this skill
merely because an issue could help, a problem was found, or another skill lists
an issue among its usual artifacts. A workflow may require issue creation only
when its user-requested purpose depends on that issue, such as CI triage whose
deliverable is a tracked bug. Research, benchmarking, implementation, and PR
work do not imply issue creation.

Before drafting, read `AGENTS.md` and:

- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/issues.md`
- `.agents/skills/writing-style/ai-writing-donts.md`

## Issue Kinds and Body Structure

Pick the smallest matching structure.

| Kind | When to use | Labels |
|---|---|---|
| **bug** | A bug or regression was found | `bug`, `agent-generated` |
| **task** | An improvement, refactor, or feature request | `agent-generated` + priority if known |
| **experiment** | An experiment needs tracking | `experiment`, `agent-generated` |

### Bug body

```markdown
<what is broken and its impact -- concrete symptoms or error messages>

Reproduce:
1. <step>
2. <step>

Expected: <what should happen instead>

<optional: concise evidence or confirmed root cause>
```

### Task body

```markdown
<what needs to be done and why -- enough context for anyone on the team>

Done when:
<specific, testable completion criteria>
```

### Experiment body

```markdown
## TL;DR

<One-paragraph current summary. Leave blank only when the work is just being kicked off.>

## Description

<Context someone outside the thread can understand.>

## Hypothesis or Goal

<What are you trying to learn, fix, or achieve?>

## Status

<Current state; update as evidence lands.>

## Links

* Logbook:
* W&B Report:
* Important updates:

## Decision Log

## Conclusion
```

## Workflow

### 1. Gather and classify

Extract the symptom or desired outcome, impact, location, reproduction, known
cause, and severity. Ask when the issue or its kind is ambiguous.

### 2. Check duplicates

Search for existing issues first:

```bash
gh issue list --repo marin-community/marin --state open --search "<keyword>"
```

If a match exists, tell the user and offer to comment on it instead.

### 3. Draft

**Title**: At most 80 characters, optionally prefixed with a scope tag. State a
factual symptom for a bug (e.g. `[levanter] Gradient accumulation drops the last
microbatch`) and an imperative outcome for a task (e.g. `[levanter] Handle
partial accumulation steps`). Do not add `bug:`, `task:`, or another type
prefix.

Use the body structure above. Keep the facts needed to act; link code with
`file:line`, trim errors to relevant frames, and omit filler, repeated titles,
diff inventories, and unnecessary implementation narration. Bugs need numbered
reproduction steps; tasks need testable completion criteria.

### 4. Inspect the payload

Apply the writing-style final compression pass to the exact title and body that
will be sent to GitHub. For a bug or task, verify the title is at most 80
characters. Every remaining sentence must add
a symptom, impact, reproduction step, observation, expected behavior, or
completion criterion.

### 5. Apply the approval boundary

If the user explicitly asked to file an issue, skip the preview, file it, and
share the link. Do the same when the user invoked a workflow whose stated
deliverable intrinsically requires a new issue.

Otherwise, do not draft or file an issue. Continue the underlying work. If an
issue would materially improve later coordination, mention that option at
handoff without blocking the task.

### 6. File the issue

Write the body to a uniquely named temp file, then pass it with `--body-file`.
Do not inline the body with shell substitution (`--body "$(cat <<'EOF' ...)"`)
— multiline text can be corrupted by pasted output or escaping mistakes. Do not
reuse a fixed path like `/tmp/issue-body.md`; concurrent agent runs can
overwrite each other's drafts on shared hosts.

```bash
body_file="$(mktemp "${TMPDIR:-/tmp}/issue-body.XXXXXX.md")"
trap 'rm -f "$body_file"' EXIT

cat > "$body_file" <<'EOF'
<body>
EOF

issue_url="$(gh issue create --repo marin-community/marin \
  --title "<title>" \
  --label "agent-generated" \
  --body-file "$body_file")"
```

Add kind-appropriate labels (`bug`, `experiment`). If a relevant label does not
exist, skip it rather than creating new labels. For task issues, add a priority
label (`p1`, `p2`, `p3`) if the user specifies one or severity is clear.

Before creating the issue, re-open the body file and verify it contains no
unrelated shell output (pre-commit logs, pytest session headers, prompt
transcripts). If it does, clean the draft before posting.

After creating the issue, fetch its published text with
`gh issue view "$issue_url" --json title,body` and correct any text added or
altered by the publishing tool.

### 7. Report

Print the issue URL.
