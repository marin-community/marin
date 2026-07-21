# Pull Requests And Commit Messages

Use this file for commit messages and pull request titles and bodies. The PR
body becomes the squash-merge commit message, so the two media use the same
voice: neutral, compact, and useful in `git log`.

## Write The Title

- Use an imperative sentence of at most 72 characters.
- Name the outcome, not the activity or implementation inventory.
- Use an optional `[scope]` prefix. Do not add `feat:`, `fix:`, or another
  conventional-commit prefix.
- Remove adjectives, authoring-tool or agent-provider names, emoji, and trailing
  punctuation.

Prefer `[sft] Add the OpenCode chat template` over
`feat(sft): opencode tools-aware chat template resource + Levanter chat-format wiring`.

## Write The Body

- Keep an agent-authored PR body at or below 200 words. Most bodies are 50–150
  words in one or two paragraphs with no headings.
- Lead with the behavior that changes. Follow with the reason or constraint that
  shaped it.
- Include a result only when it affects the review decision. State it once and
  link the detailed evidence.
- End with `Fixes #NNNN` or `Part of #NNNN` when applicable.
- Put specifications, reproduction instructions, full benchmark results, and
  research history in an issue, design doc, logbook, or artifact and link it.

The body must stand alone, but it does not need to reproduce the diff. Delete:

- file-by-file or symbol-by-symbol inventories visible in the diff;
- lists of tests and assertions;
- `Testing`, `Validation`, `Verification`, `What`, `Changes`, or `Summary`
  scaffolds;
- claims framed as verdicts, such as `why this is correct`, `cleaner`, or
  `provably`, when a measured result or explicit design choice says more;
- boldface, all-caps emphasis, checkboxes, emoji, and attribution or session
  trailers;
- filler openers such as `This PR`, `In this change`, or `Summary of changes`.

Use a list, table, diagram, or heading only when it conveys a relationship that
is hard to express in two short paragraphs. Markdown is not a completeness
signal.

## Compress An Implementation Report

A template resource PR does not need separate `What`, parity-verdict,
reproduction, and companion-work sections. Keep the review-relevant facts:

```text
Title: [sft] Add the OpenCode chat template

Add the OpenCode/Qwen3 tools-aware chat template and dataset-format builder for
assistant-only loss masking. Generation markers leave rendered token IDs
unchanged while defining the tokens included in the loss.

The template matched Axolotl token IDs on 60 sampled rows. Its static assistant
mask intentionally differs at turn boundaries. Part of #7098; preprocessing is
in #7454.
```

The diff shows constant names and test cases. Linked artifacts can hold the
complete parity data.

## Check The Exact Payload

Before committing or calling `gh pr create` or `gh pr edit`:

1. Read the exact title and body that the command will receive, not the notes
   used to draft them.
2. Count the title characters and body words. Shorten the text before publishing
   if it exceeds 72 characters or 200 words.
3. Apply [ai-writing-donts.md](ai-writing-donts.md). Delete every sentence that
   does not add behavior, motivation, evidence, a caveat, or an issue link.
4. Check that the title, opening sentence, and issue link agree with the actual
   branch scope.
5. After creating or editing the PR, fetch `title,body` with `gh pr view --json`
   and correct any text added or altered by the publishing tool.

Use `printf %s '<title>' | wc -m` for the title and
`wc -w < path/to/body.md` for the body. Inspect the body file itself after
counting it.
