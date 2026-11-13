# Validation on five small graders

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
