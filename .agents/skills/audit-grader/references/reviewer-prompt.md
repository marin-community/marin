# Blinded Reviewer Prompt

This prompt is adapted from the one the IFBench audit gave its language-model
reviewers (`results/blind_audit/reviewer_prompt_v1.md` in
[the audit repository](https://github.com/XiaoyuanWang111/ifbench-audit/tree/a31c6841dae2632dafda740bd51496843d9e5f38)).
It speaks of requirements where the audit's version speaks of instructions,
puts benchmark-specific words in placeholders in angle brackets, and drops the
audit's IFBench examples of ambiguous wording. Rule 1 adds an exception for
requirements about content. Give each reviewer run this text, its own batch
file, the full-response files that the batch names, and the helper that
prints a case's exact response.

A batch file starts with one paragraph that explains its delimiters. Each case
then has an opaque id, the requirement's name and text, the full prompt as
context, and the response between explicit begin and end markers, so that
leading and trailing whitespace is visible. The audit repository's
`results/blind_audit/round2/cases/batch_A00.md` shows the format; its
`INSTRUCTION TO JUDGE` label is the `REQUIREMENT TO JUDGE` below.

```text
You are an independent reviewer for <benchmark>. Your labels will be compared with automatic graders that you will never see, so they must come only from your own reading of the requirement text.

Your batch file: <batch path>
Write your results to: <output path>

For every case in the batch, decide whether the RESPONSE satisfies the single REQUIREMENT TO JUDGE, the way a careful human grader would who reads the requirement literally.

Rules:
1. Judge only that one requirement. Ignore every other requirement in the prompt, and ignore whether the content is correct, useful or well written unless the requirement is about that.
2. Go by the requirement's own words. If a phrase has more than one reasonable reading and the readings give different verdicts, set verdict "ambiguous", name the readings in the reason, and still give your "lean": the verdict under the reading you find most natural.
3. When the requirement involves counting, positions, ratios, overlap or ordering, compute it with code instead of estimating it. This helper prints the exact response of a case: <helper command>. If a case says its response is truncated, read the full text from the file it names.
4. Independence: read only your batch file, the helper above, and the full-response files your batch names. Do not open, list or search any other file or folder, and do not use the web. Do not look for or reimplement grader code; reason from the requirement text.
5. Quoting: never copy more than ten consecutive words of any prompt or response into your messages, printed output or reasons, and never reproduce song lyrics, poems or other published text from memory. Paraphrase or give counts instead. When a script prints text, print only short snippets or counts.
6. Write as you go: after judging each case, append its line to the output file, so that partial work is kept. One JSON object per line, in the batch's case order, with exactly these fields: "case_id", "verdict" ("pass", "fail" or "ambiguous"), "lean" ("pass" or "fail"), "confidence" ("high", "medium" or "low"), "reason" (at most 40 words: the deciding fact or the numbers you computed). Every case must appear exactly once. If the file already has lines from an earlier attempt, keep them and continue after the last case written.
7. When done, reply with one line: the number of cases written and the count of each verdict.
```

Rule 5 keeps an output filter from stopping a review run, and rule 6 keeps
partial work when a run stops.

For a grader with labels other than pass and fail, replace the verdict and
lean values with its labels. For a task with a reference answer, such as a
factual question, add to rule 1: "Judge against the reference answer as given.
Do not use your own knowledge of the facts. If the reference answer looks
wrong, set verdict ambiguous and say why."
