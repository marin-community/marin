# Reviewing Marin Prose

Use this checklist on a completed draft. For the choices to make before
writing, see the [drafting guide](drafting.md).

## Check evidence and terminology

- Check each material claim against its source. Compare the prose with the
  figures, tables, and code. Make sure each result names the relevant
  baseline, conditions, and limits.
- For a claimed improvement, check that a reader can tell what the baseline
  did, what changed, and how much the outcome improved. Where possible, state
  the difference between the approaches in one or two sentences. When the
  explanation depends on how the approaches work, compare them on the same
  input or workload. Use the same units when quantifying the difference.
  Check that the evidence supports any claim about what caused the improvement.
- Check that each technical term refers to the same concept throughout and
  that the reader meets its definition before they need it.
- Prefer to link directly to projects, people, papers, and companies on first
  mention. Use a source that identifies the named entity or work.
- Scan the lead and body for named methods, runs, diagnostics, and other
  evidence the explanation relies on. At the first mention, check that the
  phrase links to the specific source a reader would want to open. For a
  run-level claim, prefer an accessible W&B run page over a derived JSON
  record. Check that derived analysis links support the calculations or
  comparisons they accompany. A tracker, notebook, or later citation may
  give useful context while leaving that first mention without a direct source.
- Split sentences that mix separate claims. Replace vague verbs and stale
  metaphors with the operation they describe.
- Replace unsupported adjectives such as `robust` or `significant` with
  evidence. State specific unknowns and remove claims that are out of date.

## Check Blog Background

Read the title and opening. State the headline result and the question it
answers in your own words. Before the results, check that the draft explains
what prompted this particular comparison, why the team chose those approaches
when the sources give a reason, what it measured, and what would count as an
improvement. If a follow-up takes over the story, check that the draft explains
what prompted it and which question it answers.

Read the background through the first paragraph of the approach. Check that
the reader can explain why the baseline does the work being changed and which
property or constraint the new approach uses. If the background jumps from
definitions or dimensions to the change, add the missing connection. Move
benchmark-specific details to where they help explain the mechanism or result.

## Check Blog Section Headings

Read the section headings on their own as a table of contents. For each
section, use the label that matches its purpose in the
[blog guide](blog-posts.md#structure), such as “Background,” “Approach,”
“Benchmark,” “Results,” or “Limitations.” Add a short qualifier when it
helps readers choose a section, as in “Single-device benchmark.” Rewrite
headings that describe a technique or state a finding as functional labels.
For example, change “Shard the computation” to “Approach.” Keep the
technical detail in the section itself.

## Cut Empty Sentences

Delete a sentence when removing it loses no information. Common waste:

- bridge prose: `The main change is...`, `Stepping back...`, `What this means is...`
- importance claims: `Importantly`, `It is worth noting`, `This is the key point`
- stage directions: `Let's dive in`, `Here is a summary`, `In this section...`
- effort narration: `After extensive investigation...`
- generic conclusions, recaps, future-work sections, and chatbot sign-offs
- prose that restates a heading, list length, table, diff, or visible structure

Start with the fact. End when the useful information ends.

## Review for Unnecessary Detail

After drafting, identify details that are accurate but unnecessary for
understanding the contribution, interpreting the evidence, or following the
explanation. Cut those details from the main narrative when the linked source
provides enough context for readers who want the full investigation. Keep the
experimental limitations that affect how readers interpret the result.

## Ban Stock Contrast

Remove rhetorical templates built around a staged opposition:

- `X, not Y`
- `not X, but Y`
- `not just X, but also Y`
- `more than X`
- `the question is no longer X; it is Y`
- `the real story is X`
- `what matters is X`
- `rather than Y` when no real comparison is being made

State X directly. Split real comparisons into plain claims.

Bad:

> The issue is not query execution, but the metadata path.

Better:

> Metadata loading takes 12 seconds. Query execution takes 400 ms.

Bad:

> The change improves reliability, not just performance.

Better:

> Median latency fell 18%. Failed retries fell from 7 to 0.

Grammatical negation is fine when the negative fact matters: `The ablation has
not run.` A measured comparison is fine when both sides matter.

## Avoid LLM Cadence

- Skip adjective trios and forced rules of three.
- Skip `Additionally`, `Furthermore`, `Moreover`, `Importantly`, and `Notably`.
- Skip `landscape`, `ecosystem`, `interplay`, `tapestry`, `testament`, `delve`,
  `showcase`, `underscore`, and `foster`.
- Avoid em-dash asides. Make the point a sentence or delete it.
- Avoid bold labels, takeaway boxes, and decorative headings.
- Cut hype, product-announcement language, AGI speculation, and decorative emoji.

## Score Every Paragraph

Each paragraph must contain a concrete result, setup, decision, constraint, or
caveat. Rewrite or delete paragraphs that only frame significance or smooth a
transition.

Delete sentences that carry no information. When a sentence could appear
unchanged in a generic software announcement, name the fact, decision, or
unknown it was meant to convey.

## Prefer Plain Rewrites

- `The main change is that retries are bounded.` → `Retries stop after five attempts.`
- `This highlights a scalability problem.` → `Registration takes 1.2 seconds for 100 workers.`
- `The current state is much clearer.` → Name the resolved and open questions.
- `This is procedural progress.` → `The code is ready. The ablation has not run.`
- `The configuration has three parts:` → Start the list.

## Check For Missing Steps

Check which terms, referents, claims, or causal links a first-time reader
cannot recover from the draft. Fix concrete gaps with the smallest local edit.
Narrow a claim when explaining its full scope would interrupt the text. Link
to optional background.

Before publishing, search the revised draft for stock contrast, bridge phrases,
unsupported adjectives, and stale claims.
