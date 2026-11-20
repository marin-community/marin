---
name: press-office
description: Evaluate a new or updated Marin logbook index record or a closed Marin experiment issue. Draft a short, evidence-backed Research Note for each distinct finding that qualifies.
---

# Press office

## Find candidate records

- For a logbook, use the added or materially changed task file under `index/`
  on [marin-logbooks main](https://github.com/marin-community/marin-logbooks/tree/main/index).
  Use the record path and merge revision to identify the input. Compare an
  updated record with its previous version and evaluate the new finding. Skip
  generated `INDEX.md` changes and mechanical edits to task files.
- For a Marin issue, look for closed issues labeled `experiment`. Give priority
  to agent-authored records and threads where readers have to reconstruct the
  result from several comments. Use `agent-generated` and marked agent comments
  as discovery signals. Check why the issue closed. Find the final result,
  read any later corrections, and follow links to the measured runs, code, or
  data. Skip issues that closed without a usable result.

For a manual trial, evaluate the supplied record or issue directly. Check for
a linked logbook or issue before counting a finding as a separate candidate.

## Pick the stories

Read the source record, the linked coordinating issue and commit-pinned
notebook where they exist, and any public benchmark, report, or contribution
needed to verify the claim. A record may support zero, one, or several posts.
Look for distinct findings across systems, data, and ML work. Evaluate each
finding on its own evidence.

Before outlining, compare the candidate takeaways side by side. Keep a
separate post when it gives its intended readers a distinct, useful lesson
with enough evidence to carry the story. When several candidates lead to
the same takeaway, keep one post centered on the strongest evidence. Use a
diagnostic or follow-up as supporting evidence when it helps readers
understand that takeaway. If a follow-up changes the interpretation of an
existing Research Note, propose an update to that note in the editorial
handoff. Leave the remaining findings in the source record.

For each candidate, look for an existing report or blog post covering it. If a
human-authored report or post already covers it, defer to that write-up. Skip
drafting and link to it in the editorial handoff.

Draft when the evidence gives a reader outside Marin a concrete observation,
engineering tradeoff, or failed approach they can learn from. A small result
can qualify.

Give priority to engineering changes and the decisions they informed. A
deployment comparison can support a practical decision when several factors
changed together. Name the factors and scope the result to the deployments
tested. Debugging usually belongs in the logbook or supports the main story.
Give it a standalone post when it establishes a failure mode readers are
likely to encounter or offers a diagnostic technique they can reuse.

When the main finding is negative, skip it unless one of these applies:

- The project took more than one day. Check the project timeline.
- The finding provides evidence against a specific result in a publication
  or arXiv preprint. Check the original claim.
- The finding is especially interesting. State in the editorial handoff what
  a reader could learn from it and what evidence supports that lesson.

Skip records that only announce work, link an archive, or repeat an existing
post without new evidence. If no candidate qualifies, return one sentence
explaining why there is no draft.

For an experiment, recover the original question and motivation from the
design record. Identify the comparison and outcome the team chose to answer
that question. A follow-up can lead when it is the main contribution. Explain
what prompted it and how its question or design differs from the original
experiment.

Establish the baseline, scope, and decision the main finding supports. Check
which revision produced each number. Keep historical and current measurements
distinct, and distinguish kernel benchmarks from
whole-model performance. When the measured and current versions differ,
identify the measured version in the first result sentence. Give the workload,
comparison, and scope of any secondary observation you keep.

## Choose a Marin exemplar

Use the [blog exemplar table](../../../docs/dev-guide/writing-style/blog-exemplars.md)
to choose a primary post by what the new article needs to explain. Read the selected
post before drafting. Borrow its approach to the opening, the sequence of
explanations and evidence, and the placement of figures. Scale the structure
to the length and scope of the new story.

When the primary post leaves a gap, borrow a specific element from another
post. A mechanism explainer can use one post’s worked example and another
post’s benchmark figure. Keep the draft organized around one main finding.
Record the chosen post or posts and what you borrowed in the editorial
handoff outside the article. Use the references for form and voice. Ground
the new article’s facts and claims in its own sources.

## Construct the outline

Read Marin's [drafting guide](../../../docs/dev-guide/writing-style/drafting.md)
and [blog guidance](../../../docs/dev-guide/writing-style/blog-posts.md).

**Identify and frame the contribution before drafting.** For each candidate
post, identify the main question, what the experiments actually establish, and
why that finding may be useful to readers. Distinguish the central conclusion
from the measurements that support it. Choose a title and outline that reflect
the strength and significance of the evidence. State null results or small
differences plainly and at the scope measured.

After reviewing the sources, build the
[explanatory outline](../../../docs/dev-guide/writing-style/blog-posts.md#explanatory-outline)
around the central question or result. Order the material by what the reader
needs to understand at each step. Use the outline to decide which findings
belong in the post before writing prose.

Before saving the outline, read the title by itself. Can you tell what the
post is about? Would you infer a stronger result than the evidence supports?

Save a brief `post-outline.md` alongside the draft. Aim for about 150 words.
Give the proposed title, one sentence each for the question, takeaway, and why
it matters, and three to six short bullets in reading order. Include only the
measurements, caveats, and links needed to assess the main takeaway. When a
record supports multiple posts, give each post its own directory with its
outline, draft, and figures.

## Write the draft

Write for a grad student or ML engineer working on LLMs who may be new to
Marin.
Use “we” for the team's documented work, say what the comparison baseline is
on first mention, and introduce terms when they earn their place. Lead with
the finding and the conditions that bound it. Give enough
background to see why the answer matters. If the central trick takes a
conceptual jump, walk through one tiny case using the objects in the
experiment. Explain what it buys before adding implementation terms. Organize
the article around a few short sections that explain the finding.
Aim for 400–700 words when the material warrants it.

If the source issue or logbook has a TL;DR that works on its own as the lead,
reuse its prose verbatim under the article title. It should make the question,
main result, and scope clear to a reader new to the work. Check it against
the final evidence and any corrections before copying. Write a new lead
when the summary leaves out context the reader needs.

Leave unit-test results, CI status, smoke-test pass counts, and routine
verification out of the post. Describe an experimental safeguard only when
it changes how readers interpret the evidence. State what remains unmeasured
directly.

Keep the article self-contained. Immediately after the lead, link directly
to the original Marin issues and logbook issues when they exist. Link the
canonical public evidence for specific claims and use the notebook for deeper
reproduction details. Return the Markdown draft for review. When preparing a
Research Note for the website, use the date the source PR merged or the
coordinating issue closed. Choose the event that records the finding covered
by the note, and link it in the editorial handoff. Set `date` in `YYYY-MM-DD`
format. If the finding has no merge or close event yet, leave the date open
for the editor.

Set `tags` to a list with the note's main topic. Use `engineering` for systems
and implementation, `modeling` for model, training, or evaluation results,
and `data` for datasets and data curation. Omit `authors`. Follow the
[website's native post guide](https://github.com/marin-community/marin-community.github.io/blob/main/docs/blog-authoring.md).
Publication is a separate editorial decision.

## Make the figures

Aim for at least one figure per post. Show major quantitative results in
both prose and a chart. Make the comparison, units, measurement scope, and
any uncertainty clear from the figure and caption. Use the published data,
and link its source. Give each worked example a figure beside the prose.
Use the same labels and values as the text, and show the step the reader
needs to follow. For a qualitative story, a diagram can carry the main
finding. Provide descriptive alt text and return the figure assets with
the draft.

## Review

Apply the [prose review checklist](../../../docs/dev-guide/writing-style/review-prose.md)
and [blog review checks](../../../docs/dev-guide/writing-style/blog-posts.md#review)
to the draft and figures before the cold read. Check edits that change a
technical claim against its source.

Give a fresh subagent only the article, its figures, and the intended audience
for a cold read. Ask the reader to leave links unopened. Have them say why
the work was done, what question the comparison was designed to answer, what
was measured, and what answer they took away. Ask them to flag unfamiliar
terms or claims that seem broader than the evidence. Ask, “Where did you have
to supply a step the author left implicit?”

Send the cold-read feedback back to the writer. The writer updates the article
and figures to address concrete gaps, checks any amended claims against the
sources, and returns the revised draft for handoff. If the reader could not
explain the main question, answer, or limits, give the revision to a fresh
cold reader and send that feedback back to the writer.
