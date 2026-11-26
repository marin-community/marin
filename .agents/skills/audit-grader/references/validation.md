# Validation

## Five small graders with planted defects

This skill was checked on five small graders written for the purpose, which
cover the four kinds the skill names: two programmatic checkers (one for math
word problems, one that runs code against tests), an LLM-judge pipeline with
recorded judge replies, an RL reward function for JSON answers, and an agent
task whose verifier grades an answer file. Each grader has three planted
defects, one place where its task text allows two verdicts (an ambiguity), and
one behavior that looks wrong but follows the task text (a trap), with 36 to 48
stored responses or trials that stand in for a model panel. The cases, their
answer keys and the runs are in
[`XiaoyuanWang111/audit-grader-suite` at `65da6ba`](https://github.com/XiaoyuanWang111/audit-grader-suite/tree/65da6bac7dfcb57eb57f38d26148258826bee2cd).

Each run was a fresh agent session. Its working directory held SKILL.md,
`references/ifbench.md`, `references/reviewer-prompt.md` and one case
directory, with no answer key. The session was told to work only in that
directory and offline; nothing enforced this. Three runs reported temporary
files or path checks outside the directory, none of them in the suite. Each
report was scored against the case's key: a defect counts as found when the
report names the behavior and gives a counterexample that reproduces; an
ambiguity counts when the report files it as a decision entry; a trap counts as
respected unless the report files it as a defect.

| Case | Kind of grader | Defects found | Ambiguity as a decision | Trap respected | False findings | Real defects found that were not planted |
|---|---|---|---|---|---|---|
| `toymath-extract` | programmatic checker | 3 of 3 | yes | yes | 0 | 3 |
| `toycode-exec` | programmatic checker (code execution) | 3 of 3 | yes | yes | 0 | 4 |
| `toyqa-judge` | LLM-judge pipeline | 3 of 3 | yes | yes (filed as a decision) | 0 | 0 |
| `toyjson-reward` | RL reward function | 3 of 3 | yes | yes | 0 | 3 |
| `toyagent-verify` | agent-task verifier | 3 of 3 | no | yes | 0 | 1 |

Over the five runs: 15 of 15 planted defects found, 4 of 5 ambiguities filed
as decision entries, 5 of 5 traps respected, no false findings, and 11 real
defects found that were not planted. Two more findings read an undefined term
strictly ("spaces" in `toyjson-reward`, "whitespace" in `toyagent-verify`); the
score card counts them in neither column. In each case, one planted defect
changes no stored verdict or score, so only a fixture, a property test, a
fuzzer or a content-free response shows it; every run found its case's. Seven
of the 11 unplanted defects are shared by the case's reference implementation,
and the other four are in the two cases that have none. Each run kept its
fixtures as runnable tests, with strict expected failures where the grader is
wrong; rerunning each run's tests gives the counts it reported.

The `toyagent-verify` run wrote one reading of an undefined term into its
contract (whether other words may surround the answer) and filed no decision
entry for it. Step 2 now says that such a term gets a decision entry. A second
run of that case with the new rule filed the decision entry and kept the other
results. The rule's two examples come from this case, so this rerun shows only
that a run follows the rule.

Limits:

- Each case had one run with the skill as first written, and `toyagent-verify`
  a second run with the new rule, all on Claude Opus 5.5.
- No run ran step 6's two reviewers, because the exercise did not allow
  subagents. The runs drew and froze their review samples but measured no error
  rate.
- The cases are small. Claude Opus 5.5 also wrote them, and each key names the
  skill step where each planted item should surface.

## Blind rerun on IFBench

The audit in [ifbench.md](ifbench.md) was done before this skill was written. In
the rerun, a fresh agent session audited the same grader, Marin's vendored
IFBench checkers, with SKILL.md and `reviewer-prompt.md` as merged in #10422. It
had no `ifbench.md`, no copy of this file and no result of the earlier audit.
Its working directory held the grader, the reference at allenai/IFBench
`1c40f0c`, the dataset, and the earlier audit's stored panel of 68,400
responses. For step 6, two more sessions each labeled the frozen sample of 458
verdicts in a folder that held only its own batches. The run's files and the
score card are in
[`XiaoyuanWang111/audit-grader-suite` at `f4738d6`](https://github.com/XiaoyuanWang111/audit-grader-suite/tree/f4738d6ca7a3cb82a922232358962f371608e2e3),
under `runs/ifbench-r3/` and `runs/SCORES-ifbench.md`.

The rerun's report was scored against the earlier audit's findings. The rule for
a found defect is the one above, and it was written down before the first
match. A finding counts under "decision entry" when the rerun files the
behavior as a decision for the maintainers.

| Earlier audit's findings | Reported as a finding | As a decision entry | Not reported |
|---|---|---|---|
| 34 checker findings that hold under every reading | 28 | 3 | 3 |
| 9 checker findings that rest on a decision | 3 | 5 | 1 |
| 6 protocol findings | 3 | 2 | 1 |

The seventh protocol finding, loose grading, is left out because SKILL.md names
it. The rerun lists 56 findings, and every counterexample reproduces on the
grader. Three concern the working directory and the stored panel. Of the other
53, 40 match the earlier audit, 5 are defects that the earlier audit did not
report, and 8 are behaviors that the earlier audit or the rerun's own reviewers
treat as decisions. None is false. The rerun puts the grader's error rate at
7.2% [5.9, 9.1], against 7.6% [4.7, 11.3] and 6.1% [3.9, 10.1] in the earlier
audit's two rounds, and it also recommends fixing the grader.

The four checker findings that the rerun did not report are each an input form
that it had used to show a defect in another check: a list label, a curly
quote, or punctuation after a repeated word. A fifth was missing before the
review: the flag emoji that the grader rejects. After the review, the auditor
went through the two review cases that every grader got wrong, and the panel
verdicts on which no reading of the decision entries makes the replacement
agree with the grader. The skill does not ask for this step. It added four
findings, the flag emoji among them. #11187 tracks both gaps.

Limits:

- This is one run. The auditor and both reviewers were Claude Opus 5.5, and the
  earlier audit's reviewers were Claude models too, so the two audits can share
  blind spots.
- The auditor's work took two sessions. The first stopped once it had frozen
  the review sample. The second started from the files when both reviews were
  in, and was told to write the report before any further analysis.
- SKILL.md says how many checkers the earlier audit covered, and the reference
  already fixes six of the earlier findings.
- The sessions were told to work offline inside their folders; nothing enforced
  this. A script read each session's tool calls afterwards and found no access
  to the earlier audit or to another session's files.
- The two reports were matched by hand, and no second reader checked the
  matches.
