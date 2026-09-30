# Controls for intentional partial credit

A criterion-level mutant can deserve credit for independently correct work.
For example, the protocol-trace task can score six correct numeric criteria while
rejecting its seventh justification criterion. Calling that a whole-answer negative
and demanding reward at most 0.2 would silently change the admitted rubric.

Use `class: "partial"`, `category: "criterion_mutation"` and a concrete
`partial_credit_reason` tied to the admitted rubric. Require a graded outcome,
both reward bounds (upper bound strictly below 1), and at least one assertion on
the actual grading detail. Example:

```json
{
  "id": "wrong-justification",
  "class": "partial",
  "category": "criterion_mutation",
  "source_author": "builder",
  "response": "<the full authored candidate>",
  "partial_credit_reason": "C1-C6 are correct; C7 must fail under the admitted rubric.",
  "expect": {
    "status": "graded",
    "reward_min": 0.857142856,
    "reward_max": 0.857142858,
    "assertions": [
      {"path": ["detail", "stdout", "criteria", "C7", "passed"], "equals": false}
    ]
  }
}
```

Paths traverse objects and arrays in the retained grading result. Embedded JSON
strings, such as a script's JSON stdout, are decoded when traversal continues.
Missing fields, malformed JSON, or a mismatched scalar fail the assertion; Boolean
false is not interchangeable with numeric zero. Use the actual verifier's detail
shape and preserve its raw artifact, rather than inventing a summary field.

Partial controls are replayed authored candidates. They do not replace the required
known-correct, malformed, plausible-wrong and shortcut controls. Whole-answer
negative controls still require reward at most 0.2, while any stricter declared
bound also applies. Independent review must verify that the partial-credit label
matches the admitted rubric; relabeling a critical failure is not a valid repair.

An independent attack earning substantial reward is a separate adjudication
question. A legitimate partial or correct solution is not automatically an exploit;
the attack gate retains such cases as pending adjudication. The independent
review implementation in `capability_pipeline/attack_adjudication.py` can resolve
that specific alarm using a fresh reviewer and a frozen copy of the task, admitted
proposal, candidate transcript and raw grading result. It does not change the
original attack report or reward.

Each rewarded step needs its own disposition: legitimate correct, legitimate
partial, exploit, or uncertain. Legitimate outcomes require rubric-derived reward
bounds and evidence explaining the candidate, rubric and every critical gate.
Partial credit cannot excuse full reward. Exploits and uncertainty remain pending;
an execution failure cannot be waived by this receipt. Any later change to the
task or runtime evidence invalidates the review.

The controller does not resample a reviewer for unchanged evidence. It reuses
the unresolved sidecar and routes the item into bounded repair; a fresh review
is allowed only after the task or runtime snapshot changes. Prior receipts stay
visible to subsequent repair and quality review.

The controller revalidates the sidecar with `validate_resolution`, clears only the
two rewarded-attack alarms, and retains all original trial, extraction, isolation
and private-verifier checks. It passes the immutable manifest, receipt and result
into the later quality-review packet without editing runtime evidence. Resolution
is not runtime or semantic certification. Controller and receipt regressions pass;
a real generated-task adjudication still needs recorded live evidence.
