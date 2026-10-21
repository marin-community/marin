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

## Write For A Tired Reviewer

The reader is a person who switches between many reviews and has little attention to spare. Make the body easy to scan in under a minute.

- Put the change and its main result in the first two sentences.
- Keep each paragraph to one topic and two or three short sentences.
- Put measurements of more than one configuration, such as before/after pairs across hardware, node counts, or flags, in a table. Give units in the column headers and add a change column. Do not string several numbers together in one sentence.
- Split any sentence that holds a semicolon list or more than one comparison.

Bad:

> One node goes from 2.531 to 2.479 s per step (-2.0%) without PGLE and from 2.392 to 2.362 s (-1.3%) with PGLE; two nodes go from 3.286 to 3.234 s (-1.6%) and from 3.109 to 3.087 s (-0.7%).

Better:

| Nodes | PGLE | Before (s/step) | After (s/step) | Change |
| --- | --- | --- | --- | --- |
| 1 | no | 2.531 | 2.479 | -2.0% |
| 1 | yes | 2.392 | 2.362 | -1.3% |
| 2 | no | 3.286 | 3.234 | -1.6% |
| 2 | yes | 3.109 | 3.087 | -0.7% |

## Write The Body

- Treat the body as a permanent changelog entry. It must make sense to a
  competent reader who was not part of the authoring session and may encounter
  it months later. This follows the Linux kernel's guidance for
  [describing changes](https://docs.kernel.org/process/submitting-patches.html#describe-your-changes).
- Use as many words as the review-relevant information needs and no more. Most
  bodies are a few plain paragraphs with no headings; benchmarks, reproduction
  details, or compatibility constraints may justify more.
- Lead with the behavior that changes. Follow with the reason or constraint that
  shaped it.
- Keep concrete symptoms, measured results, baselines, and caveats when they
  explain the change or affect the review decision. State them once and link
  detailed evidence when the full record belongs elsewhere.
- End with `Fixes #NNNN` or `Part of #NNNN` when applicable.
- Put extended raw benchmark output and research history in an issue or source
  artifact and link it. Keep the reproduction detail and result summary
  needed to evaluate the change.

The body must stand alone, but it does not need to reproduce the diff. Delete:

- file-by-file or symbol-by-symbol inventories visible in the diff;
- lists of tests and assertions;
- `Testing`, `Validation`, `Verification`, `What`, `Changes`, or `Summary`
  scaffolds;
- claims framed as verdicts, such as `why this is correct`, `cleaner`, or
  `provably`, when a measured result or explicit technical choice says more;
- boldface, all-caps emphasis, checkboxes, emoji, and attribution or session
  trailers;
- filler openers such as `This PR`, `In this change`, or `Summary of changes`.

Use a table for measurements and a list for steps or several parallel items. Use a diagram only for a relationship that is hard to express in prose. Do not add section headings to a normal PR body. Markdown is not a completeness signal.

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
2. Count the title characters. Inspect the body length as a signal, not a gate:
   shorten text that narrates the diff, but retain evidence and caveats a future
   reader needs.
3. Apply [ai-writing-donts.md](ai-writing-donts.md). Delete every sentence that
   does not add behavior, motivation, evidence, a caveat, or an issue link.
4. Check that the title, opening sentence, and issue link agree with the actual
   branch scope.
5. After creating or editing the PR, fetch the published text with
   `gh pr view --json title,body` and correct any text added or altered by the
   publishing tool.

Use `printf %s '<title>' | wc -m` for the title. Inspect the body file itself
after drafting it.

The agent prose cleanup workflow applies an agentic editorial pass after
publication. It preserves technical evidence while removing presentation and
diff narration. It does not replace the author's exact-payload review.
