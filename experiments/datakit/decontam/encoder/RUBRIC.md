# Judge rubric: does the document excerpt hold the eval item?

Each task has an eval item (`eval_text`, and `eval_choices` for multiple-choice items) and a
`document_excerpt` of up to 3,000 characters from a training document, cut around the place where the
two texts overlap. Decide what the excerpt holds. Judge only from the texts in the task. Do not use
any knowledge of how the pair was found.

Formatting never matters: whitespace, case, punctuation, quotes, Markdown, LaTeX delimiters, JSON or
CSV escaping, a prompt wrapper such as `<user>` or "Question:". Meaning matters: numbers, operators,
signs, names, conditions, and what is asked.

## Label

- `COPY`: the excerpt has the eval question itself, complete, with the same content. The choices can
  be present, absent, or in another order. Text before or after the question does not matter.
- `EDITED`: the same question (it asks the same thing and has the same answer), but the wording is not
  identical: a word dropped or replaced, a typo, a sentence reordered, harmless text inserted, or the
  question is cut off but most of it is there and nothing contradicts it.
- `VARIANT`: a different problem made from the same template or source: a number, operator, name or
  condition in the question, or the asked quantity, is changed, or a constraint is added, so the answer
  can differ. Changed choices alone do not make a variant (see below).
- `GENERIC`: the shared text is only a generic phrase, an instruction, boilerplate, or a common
  definition or code idiom ("Which of the following statements is true?", a license header), and the
  excerpt does not hold this particular item.
- `UNRELATED`: none of the above; no meaningful overlap is visible in the excerpt.

For a short generic question (for example "Which of the following is correct?"), the item is the
question together with its choices: label `COPY` only when the choices are there too.

Use `UNSURE` only when the excerpt is cut so that you cannot decide.

## Items with a passage

Some items come with a passage (`eval_passage`): the text that the question is about, such as a
story, a game report or a context sentence. For these items the item is the passage and the
question together.

- `COPY` or `EDITED`: the excerpt holds most of the passage and the question (the rules above for
  small changes apply to both parts). When the excerpt starts or ends inside the passage, judge
  from the part shown and say so in the note.
- `GENERIC`: the excerpt holds the question but not this passage (the same question about another
  text), or the passage but not the question (for example the passage's source article alone).
- `VARIANT`: the excerpt holds this passage with another or a changed question, or this question
  with a changed passage. The same passage with a different question of the same test is a
  `VARIANT`, not a copy of this item.
- `answer`: `YES` only when the excerpt also shows this item's answer to this question.

## Answer

- `YES`: the label is `COPY` or `EDITED`, and the excerpt also shows the answer of this item next to
  the question: for a multiple-choice item, the correct choice is marked or stated ("Answer: B", the
  correct option written as the answer, a worked solution that ends in it); for a free-response item,
  the reference answer or a solution that reaches it.
- `NO`: the label is `COPY` or `EDITED`, but the excerpt does not show which answer is correct, or it
  shows a different answer. Listing the choices without saying which one is correct is `NO`.
- `NA`: the label is `VARIANT`, `GENERIC`, `UNRELATED` or `UNSURE`.

The eval answer is given in `eval_answer`.

## Rules for hard cases

- Compare numbers, decimal points, operators, logic symbols and names carefully.
- The choices can be in another order in the document; order alone does not change the label.
- The document can be a data dump (JSON): a question and its choices stored as fields of one row count
  as shown together.
- The eval question can be in any language; judge it in that language.
- Label `VARIANT` when a number, symbol, name or condition of the question is changed, or a requirement
  is added.
- When the document has the eval question unchanged (or with the small edits of `EDITED`) but its
  choices are changed, removed, added or replaced, the question is copied: label `EDITED` and start the
  note with "choices changed". This does not apply to a short generic stem (next rules).
- A question that is the same except for typesetting is `COPY`.
- For a short generic stem, the item is the stem together with its choices: label `COPY` only when the
  document shows this stem with these choices as one question; label `GENERIC` when the stem words and
  the choice words are in the document but not as this question.
- When `eval_choices` is empty, the item is the question alone.
- The excerpt can start or end inside the question: when most of the question is shown and nothing
  contradicts it, label `EDITED` and say so in the note.

## Output

One JSON object per line, in the input order, nothing else:
`{"uid": "...", "label": "COPY|EDITED|VARIANT|GENERIC|UNRELATED|UNSURE", "answer": "YES|NO|NA", "note": "at most 12 words"}`

The note gives the specific reason. Judges read each task themselves; a script may write the labels a
judge chose but must not choose them.
