---
name: file-issue
description: File a GitHub issue from the current conversation. Use when bugs, regressions, or improvements are identified during a session and need to be captured as a tracked issue.
---

# Skill: File GitHub Issue

Create a GitHub issue in `marin-community/marin` from bugs, regressions, or
improvements identified in the current conversation.

## Background

Read first:

@AGENTS.md

## Issue Templates

This repo has issue templates in `.github/ISSUE_TEMPLATE/`. Choose the right
one based on what was identified:

| Template | When to use | Labels |
|---|---|---|
| **bug-report** | A bug or regression was found | `bug`, `agent-generated` |
| **task** | An improvement, refactor, or feature request | `agent-generated` + priority if known |
| **experiment** | An experiment needs tracking | `experiment`, `agent-generated` |

Always match the section structure of the chosen template. The templates are
the source of truth for issue format in this repo.

### Bug Report format (`.github/ISSUE_TEMPLATE/bug-report.md`)

```markdown
**Describe the bug**
<what is broken -- concrete symptoms, error messages>

**To Reproduce**
1. <step>
2. <step>

**Expected behavior**
<what should happen instead>

**Additional context**
<root cause analysis, file:line references, suggested fix if known>
```

### Task format (`.github/ISSUE_TEMPLATE/task.md`)

```markdown
## Description
<what needs to be done and why -- enough context for anyone on the team>

### Definition of Done
<specific, testable completion criteria>
```

### Experiment format (`.github/ISSUE_TEMPLATE/experiment.md`)

```markdown
## Description
<what is being investigated and why>

## Hypothesis or Goal
<what you expect to learn or achieve>

### Links
* WandB Report: <link>
* <other relevant links>

## Results
<leave empty for new issues>
```

## Workflow

### 1. Gather Context from Conversation

Review the current conversation to extract:

- **What is broken or missing** -- concrete symptoms, error messages, failing test output.
- **Where it happens** -- file paths, line numbers, module names.
- **How to reproduce** -- steps, commands, or minimal config that triggers the problem.
- **Root cause** (if known) -- what the investigation revealed.
- **Severity** -- does it block work, cause data loss, or is it cosmetic?

If the conversation is ambiguous about what to file, ask the user to clarify
before proceeding.

### 2. Classify the Issue

Pick the template (bug-report, task, or experiment) that best fits the
identified problem. If unsure, ask the user.

### 3. Duplicate Check

Before creating a new issue, search for existing ones:

```bash
gh issue list --repo marin-community/marin --state open --search "<keyword>"
```

If a matching issue exists, tell the user and offer to comment on it instead.

### 4. Draft the Issue

**Title**: Short imperative sentence, optionally prefixed with a scope tag
(e.g. `[levanter] Fix gradient accumulation off-by-one`). Under 80 characters.

**Body**: Use the section structure from the chosen template (see above).

**Rules for the body:**

- No filler ("I noticed...", "During our conversation...").
- No markdown images or tables.
- Reference code with `file:line` links, not inline dumps.
- Keep it under ~200 words. A reader should absorb it in under a minute.
- Include error messages or stack traces in code blocks, trimmed to the
  relevant frames.
- For task issues: include a concrete Definition of Done.
- For bug issues: include numbered reproduction steps.

### 5. Confirm or File Directly

If the user explicitly asked to file an issue, skip the draft preview — file it
and share the GitHub link. They can review and edit directly on GitHub.

If the issue was surfaced by the agent (not explicitly requested), show the
drafted title and body before filing. Wait for approval or edits.

### 6. File the Issue

Write the body to a uniquely named temporary file first, then pass it with
`--body-file`.
Do not inline the body with shell substitution such as `--body "$(cat <<'EOF' ...)"`
because multiline issue text can be corrupted by pasted command output or shell
escaping mistakes. Do not reuse a fixed path such as `/tmp/issue-body.md`,
because concurrent agent runs can overwrite each other's drafts on shared
hosts.

```bash
body_file="$(mktemp "${TMPDIR:-/tmp}/issue-body.XXXXXX.md")"
trap 'rm -f "$body_file"' EXIT

cat > "$body_file" <<'EOF'
<body>
EOF

gh issue create --repo marin-community/marin \
  --title "<title>" \
  --label "agent-generated" \
  --body-file "$body_file"
```

Add the template-appropriate labels (e.g. `bug` for bug reports, `experiment`
for experiments). If a relevant label does not exist, skip it rather than
creating new labels.

For task issues, add a priority label (`p1`, `p2`, `p3`) if the user specifies
one or severity is clear from context.

Before creating the issue, re-open the body file once and verify it does not
contain unrelated shell output (for example pre-commit logs, pytest session
headers, or prompt transcripts). If it does, stop and clean the draft before
posting.

### 7. Report Back

Print the issue URL so the user can see it.

## Writing Style

Follow the same terse style from `fix-issue`:

- Every sentence must convey new information.
- No preamble, no editorializing.
- No restating what code does when a link suffices.
- Annotate code links, don't narrate them.

## Tasks

- [ ] Extract bug/issue details from conversation
- [ ] Classify as bug-report, task, or experiment
- [ ] Run duplicate check against open issues
- [ ] Draft issue title and body using the matching template format
- [ ] Show draft to user for confirmation
- [ ] File issue with `gh issue create`
- [ ] Report issue URL to user

## Rules

0. Never credit yourself in the issue.
1. Always add the `agent-generated` label.
2. Confirm with the user before filing only when the agent surfaced the issue
   (not when the user explicitly asked to file).
3. If the conversation does not contain a clear bug or actionable improvement,
   say so and ask the user what they want to file.
4. Always use the section structure from the matching `.github/ISSUE_TEMPLATE/`
   template.
