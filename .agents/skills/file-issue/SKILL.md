---
name: file-issue
description: File a GitHub issue for a bug or improvement found this session.
---

# Skill: File GitHub Issue

Create a GitHub issue in `marin-community/marin` from bugs, regressions, or
improvements identified in the current conversation.

## Background

Read first:

@AGENTS.md

## Issue Kinds and Body Structure

Pick the kind, then use the matching body structure below. There are no GitHub
issue templates — these structures live here.

| Kind | When to use | Labels |
|---|---|---|
| **bug** | A bug or regression was found | `bug`, `agent-generated` |
| **task** | An improvement, refactor, or feature request | `agent-generated` + priority if known |
| **experiment** | An experiment needs tracking | `experiment`, `agent-generated` (use `.agents/skills/run-research/SKILL.md` for the body) |

### Bug body

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

### Task body

```markdown
## Description
<what needs to be done and why -- enough context for anyone on the team>

### Definition of Done
<specific, testable completion criteria>
```

### Experiment body

Use the "Experiment Issue Template" in `.agents/skills/run-research/SKILL.md`.

## Workflow

### 1. Gather Context from Conversation

Extract from the conversation:

- **What is broken or missing** -- concrete symptoms, error messages, failing test output.
- **Where it happens** -- file paths, line numbers, module names.
- **How to reproduce** -- steps, commands, or minimal config that triggers it.
- **Root cause** (if known).
- **Severity** -- blocks work, causes data loss, or cosmetic?

If it's ambiguous what to file, ask the user before proceeding.

### 2. Classify the Issue

Pick the kind (bug, task, or experiment). If unsure, ask the user.

### 3. Duplicate Check

Search for existing issues first:

```bash
gh issue list --repo marin-community/marin --state open --search "<keyword>"
```

If a match exists, tell the user and offer to comment on it instead.

### 4. Draft the Issue

**Title**: Short imperative sentence under 80 characters, optionally prefixed
with a scope tag (e.g. `[levanter] Fix gradient accumulation off-by-one`).

**Body**: Use the section structure for the chosen kind (see above).

**Rules for the body:**

- No filler ("I noticed...", "During our conversation...").
- No markdown images or tables.
- Reference code with `file:line` links, not inline dumps.
- Keep it under ~200 words; readable in under a minute.
- Include error messages or stack traces in code blocks, trimmed to the
  relevant frames.
- For task issues: include a concrete Definition of Done.
- For bug issues: include numbered reproduction steps.

### 5. Confirm or File Directly

If the user explicitly asked to file an issue, skip the preview — file it and
share the link. If the agent surfaced the issue (not explicitly requested),
show the drafted title and body and wait for approval or edits.

### 6. File the Issue

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

gh issue create --repo marin-community/marin \
  --title "<title>" \
  --label "agent-generated" \
  --body-file "$body_file"
```

Add kind-appropriate labels (`bug`, `experiment`). If a relevant label does not
exist, skip it rather than creating new labels. For task issues, add a priority
label (`p1`, `p2`, `p3`) if the user specifies one or severity is clear.

Before creating the issue, re-open the body file and verify it contains no
unrelated shell output (pre-commit logs, pytest session headers, prompt
transcripts). If it does, clean the draft before posting.

### 7. Report Back

Print the issue URL.

## Writing Style

Follow the terse style from `fix-issue`: every sentence conveys new
information; no preamble or editorializing; no restating code a link covers;
annotate code links, don't narrate them.

## Tasks

- [ ] Extract bug/issue details from conversation
- [ ] Classify as bug, task, or experiment
- [ ] Run duplicate check against open issues
- [ ] Draft issue title and body using the matching kind structure
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
4. Always use the section structure for the matching kind (see "Issue Kinds
   and Body Structure" above).
