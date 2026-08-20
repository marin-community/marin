# AI-Writing Don'ts

Use this as the final prose pass for agent-written Marin text. Delete first.
Rewrite only when the sentence carries a fact, decision, result, constraint, or
caveat.

## Cut Empty Sentences

Delete a sentence when removing it loses no information. Common waste:

- bridge prose: `The main change is...`, `Stepping back...`, `What this means is...`
- importance claims: `Importantly`, `It is worth noting`, `This is the key point`
- stage directions: `Let's dive in`, `Here is a summary`, `In this section...`
- effort narration: `After extensive investigation...`
- generic conclusions, recaps, future-work sections, and chatbot sign-offs
- prose that restates a heading, list length, table, diff, or visible structure

Start with the fact. End when the useful information ends.

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

## Require Evidence

- Name the code path, configuration, issue, artifact, measurement, or unknown.
- Replace adjectives with evidence. Delete `robust`, `powerful`, `seamless`,
  `significant`, `crucial`, and `pivotal` unless a concrete result follows.
- Say `we have not tested X` when evidence is missing.
- Keep prose consistent with tables, code, and current behavior.
- Remove stale explanations after the implementation changes.
- Use one precise term for one technical object.

## Avoid LLM Cadence

- Skip adjective trios and forced rules of three.
- Skip `Additionally`, `Furthermore`, `Moreover`, `Importantly`, and `Notably`.
- Skip `landscape`, `ecosystem`, `interplay`, `tapestry`, `testament`, `delve`,
  `showcase`, `underscore`, and `foster`.
- Avoid em-dash asides. Make the point a sentence or delete it.
- Avoid bold labels, takeaway boxes, and decorative headings.
- Do not rotate synonyms for the same object.

## Score Every Paragraph

Each paragraph must contain a concrete result, setup, decision, constraint, or
caveat. Rewrite or delete paragraphs that only frame significance or smooth a
transition.

Ask of every sentence:

1. What information disappears if I delete it?
2. Can I replace it with a number, named fact, decision, or explicit unknown?
3. Does it still describe the current code and evidence?
4. Could it appear unchanged in a random software announcement?

Delete sentences that fail this check.

## Prefer Plain Rewrites

- `The main change is that retries are bounded.` → `Retries stop after five attempts.`
- `This highlights a scalability problem.` → `Registration takes 1.2 seconds for 100 workers.`
- `The current state is much clearer.` → Name the resolved and open questions.
- `This is procedural progress.` → `The code is ready. The ablation has not run.`
- `The configuration has three parts:` → Start the list.

Finish with a search for stock contrast, bridge phrases, unsupported adjectives,
and stale claims. Prefer a shorter accurate document.
