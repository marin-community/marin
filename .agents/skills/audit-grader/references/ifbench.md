# Worked example: IFBench in Evalchemy

This is the audit of the IFBench checkers that Marin's evaluation policy runs
through Evalchemy, for
[#10060](https://github.com/marin-community/marin/issues/10060). Each of the
300 IFBench test prompts carries one or more output constraints, 344 in all.
Each constraint has one of 58 instruction types, such as `format:sub-bullets`,
and one checker per type grades it. The sources are
[the audit repository at `a31c684`](https://github.com/XiaoyuanWang111/ifbench-audit/tree/a31c6841dae2632dafda740bd51496843d9e5f38)
and the Hugging Face dataset
[`xiaoyuanwang/ifbench-audit` at `ceaba90`](https://huggingface.co/datasets/xiaoyuanwang/ifbench-audit/tree/ceaba90ba6a2acac8fd8b51368a37e27c0ba688f),
which has the fixtures, findings, decisions, blinded-review records, and panel
verdicts. This file needs neither: each section gives one step's totals and a
worked example on `format:sub-bullets`. That instruction's one open question
has no proposed reading yet, so section 2's decision example uses
`format:list`.

Terms: Marin's checkers are the IFBench checker files in Marin's Evalchemy fork,
and upstream's checkers, in `allenai/IFBench`, are the reference implementation.
Marin strict is Marin's checkers under strict grading, which checks the response
as written. The official reading of an instruction is what Marin's and
upstream's checker code does with it. The validated checkers are the audit's
replacement, written from the instruction text. Fixes only runs them with the
official reading of every open question, so each difference from Marin strict is
a defect fix. Proposed runs them with the audit's proposed readings. F1 to F43
name checker findings, G1 to G7 protocol findings about the data and the grading
harness, and `D-<topic>` a decision entry. A key is a prompt's row index.

## 1. Inputs

- Grader: Marin's checkers in Evalchemy's `eval/chat_benchmarks/IFBench`. The
  checker files, `instructions.py`, `instructions_registry.py` and
  `instructions_util.py`, are the same in Evalchemy `ab7ae27` and `09d27c9` and
  in Marin's vendored copy, `vendor/evalchemy/eval/chat_benchmarks/IFBench` at
  Marin `2d9bb82` (vendored in #10288).
- Reference: upstream's checkers at `1c40f0c`, the main branch when audited.
- Data: `allenai/IFBench_test` at `2e8a48de`, the head of its main branch on
  2026-10-16, with the sha256 of the parquet file and one sha256 per row over
  its key, prompt, instruction ids and arguments. Key 131, a
  `format:sub-bullets` prompt that section 6 uses, has row hash `1aa5e22d78c2…`.
- Generation: each model's chat template at a pinned model revision, vLLM
  0.30.0, and at most 1,024 new tokens (8,192 with thinking). For a reasoning
  model, the graded text is everything after the last `</think>`.
- Grader dependencies: nltk 3.10.0, emoji 2.15.0 and syllapy 0.8.0, as
  Evalchemy's lockfile pins them, and the NLTK data files by sha256.

Two inputs were unpinned in Evalchemy when audited, and the audit reports both
as finding G5. Evalchemy loaded the data with
`load_dataset("allenai/IFBench_test", split="train")`, so each run reads
whatever the dataset's main branch holds that day. The checkers download missing
NLTK data at import time, and NLTK data has no version.

## 2. Contracts and decisions

Each validated checker's docstring states its contract, written from the
instruction text. The `format:sub-bullets` instruction asks for `*` bullet
points with at least one `-` sub-bullet under each. Its contract, over the whole
visible answer:

- A bullet is a line that starts, after any indentation, with one `*` and the
  item: `* item` or `*item`. A line that starts with `**` (bold) or with an
  italic span such as `*like this*` is not a bullet.
- A sub-bullet is an optionally indented line that starts with `-` and text.
- Pass: there is a bullet, and each bullet has a sub-bullet before the next one.

The audit's decision table has 15 entries where the instruction text allows two
verdicts. For example, `D-list-layout` asks whether each separator must start a
line in `format:list`, whose instruction asks for a list with a separator, here
`!?!?`, in place of bullets. Two fixtures (section 3) show the readings:

| Reading | `Two markers mid-line only: a !?!? b !?!? c` | Three lines: `Here are the items:`, `!?!? apples`, `!?!? pears` |
|---|---|---|
| Official: any two separators count | pass | pass |
| Proposed: each separator starts a line | fail | pass |

The audit proposes the second reading, because the checker's own docstring says
it counts separators "that start a new line", and #10060 expects it for keys 83
and 84. Thirteen entries have a switch in the validated checkers, so the effect
of each reading can be measured. The other two entries need no switch, because
one official reading is itself a defect and the other agrees with the proposal.
Three more open questions have no proposal yet, such as whether a sub-bullet
must be indented under its bullet, and both modes follow the official reading.

## 3. Fixtures

The audit has 143 fixtures, short responses labeled from the instruction text
alone: 81 reproduce a finding or a decision, and 62 are controls. A fixture's id
is its instruction family plus the first 16 hex digits of the sha256 of its
instruction id, arguments and response. One `format:sub-bullets` fixture:

- Id: `layout-and-markup:7afc9e156a0d7d61`. Response:

  ```text
  * **Rivers**
    - Nile
  * **Mountains**
    - Alps
  ```

- Verdicts: the contract and the validated checker pass it, because each `*`
  bullet has a `-` sub-bullet. Marin's and upstream's checkers fail it.

Marin's checker splits the response at every `*` and fails if any piece after
the first lacks a `-`. Pieces such as `Rivers`, between the `**` marks, have
none. The contract is unambiguous here, so the wrong verdict is a defect,
finding F3. F3 has two more symptoms: a response without `*` passes, and a
hyphen anywhere in a bullet's text counts as its sub-bullet. The test suite runs
every fixture on the validated checkers, the finding fixtures and controls on
Marin's and upstream's checkers, the loose-grading test of section 4, and the 10
property tests. 438 tests pass, and the 2 expected failures are
`words:start_verb` fixtures where the NLTK part-of-speech tagger misses an
imperative verb.

## 4. Reference comparison and stress tests

- Upstream's unit tests pass on the validated checkers apart from five
  assertions, each explained by a finding or a decision.
- Harness fidelity: the audit's reimplementation of Evalchemy's strict and loose
  grading matches Evalchemy at `97fffeb`, the commit #10060 audited, and at
  `28631c7`, Marin's pin when audited, on 5,848 verdicts: the greedy run of all
  16 panel configurations and upstream's 300 sample responses.
- Marin's checkers fail 8 of the audit's 10 property tests, which generate
  responses whose correct verdict is known by construction, among them the
  `format:sub-bullets` test.
- Fuzzing found that `custom:csv_quotes` raises `IndexError` on an empty CSV
  field, in 251 of 3,000 random tab-separated outputs with the required header
  (F15). One checker exception fails the whole IFBench task (G6).
- Loose grading accepts a constraint if it passes on any of eight variants of
  the response: the response, a copy with `*` removed, and each of those with
  the first line, the last line, or both dropped. On upstream's sample
  responses, 15 constraints fail strict grading and pass loose grading.

Worked example, a test in the audit's suite. The two-line response `* Rivers` /
`* Lakes` has no sub-bullet, so it fails the contract, and Marin strict fails
it. Loose grading passes it, because the copy with `*` removed has no bullets,
and Marin's checker passes a response without `*` (F3). For the same reason,
loose grading passes every `format:sub-bullets` response with text besides `*`
(finding G1). On the panel, it passes 825 `format:sub-bullets` verdicts that
Marin strict and the validated checker both fail.

## 5. Findings and model panel

The audit has 43 checker findings, each with a minimal counterexample or the
decision it rests on, and seven protocol findings. Nine of the checker findings
depend on a decision. In Marin's checkers, 37 of the 58 instruction types have
at least one unambiguous defect, and those types carry 200 of the 344
constraints. 31 of the 37 types, `format:sub-bullets` among them, are still
defective upstream. Reading the checker code found F1 to F38, the panel's
disagreements found F39 to F42, and the blinded review found F43.

The panel has 16 configurations of 14 open models (7B to 32B), two of them with
thinking on. Each configuration has repeated greedy runs, and the 14 without
thinking also have 8 or 12 sampled runs at temperature 0.7. Every grader graded all 68,400 responses. With thinking off,
as Marin's eval policy runs IFBench, fixing the defects moves a model's greedy
prompt-level score (the share of prompts whose constraints all pass) by −1.7 to
+3.0 points. Rerunning a configuration's greedy decode in the same prompt order
moves its Marin strict score by up to 1.3 points. As a reward, Marin strict is
wrong on 5.7% of the constraint passes it gives, taking fixes only as correct.

Worked example. The 12 `format:sub-bullets` constraints have 2,736 panel
verdicts. Marin strict fails 912 of them that fixes only passes, and passes 208
that fixes only fails. Fixes only moves Qwen3-14B with thinking on the most,
raising its greedy score from 34.0 to 38.7, +4.7 points [+2.3, +7.3], by gaining
15 prompts and losing 1. Nine of the gains are `format:sub-bullets` answers with
bold bullet titles, as in section 3's fixture. Its same-order greedy rerun
changed 5 prompt verdicts and moved its Marin strict score by +1.0 point.

## 6. Blinded review

| | Round 1 | Round 2, held out |
|---|---|---|
| Population | 5,504 verdicts: the greedy run of 16 configurations | 4,668 verdicts: the first sampled run (temperature 0.7) of 14 configurations, without 148 that repeat the greedy response |
| Cases | All 328 disagreements (Marin strict and proposed differ), 100 agreements, 136 fixtures | 150 of 259 disagreements, 50 of 4,409 agreements |
| Cohen's kappa between the reviewers' leans | 0.94 | 0.96 |
| Error rate, Marin strict | 7.6% [4.7, 11.3] | 6.1% [3.9, 10.1] |
| Error rate, fixes only | 4.9% [2.1, 8.6] | 3.2% [1.0, 7.2] |
| Marin strict minus fixes only, points | +2.7 [+2.2, +3.1] | +2.9 [+2.4, +3.4] |

Each reviewer gives every case a verdict (pass, fail, or ambiguous), a lean (the
verdict under the reading the reviewer finds most natural), a confidence, and a
reason. A case has a consensus when both leans agree, and an error is a checker
verdict that contradicts the consensus. Error rates are estimated per stratum
(disagreements, agreements) from the cases with a consensus and weighted by
stratum size, with 95% bootstrap intervals. Round 1's 136 fixtures, the ones
that existed when it was frozen, are checked against their labels and stay out
of the error rate. Round 2 was frozen after
the fixes that round 1 prompted. Its responses were not used to develop the
checkers, and no reviewer had seen them. Round 2 prompted two more checker
changes, one of them the fix for F43 (a list label such as "1." read as a
sentence), and no later round has measured them. Each reviewer batch was a
separate agent run on Claude Opus 5.5 at max effort, and parts of four round-1
batches ran on earlier Claude models. Both reviewers used the same models, so
their agreement does not rule out shared blind spots. Reviewer B read the cases
in a different order from reviewer A. The sandbox did not enforce independence;
the reviewers were instructed to keep it. The reviewer prompt gave the questions
of three decision entries as examples of ambiguous wording, and it left the
choice of reading to the reviewers.

Over the cases where both verdicts are pass or fail and agree, 279 in round 1
and 134 in round 2, Marin strict contradicts the consensus on 6.3% and 6.8%,
fixes only on 2.4% and 2.9%, and the paired difference is +3.9 points in both
rounds, [+3.3, +4.4] in round 2.

Worked example, round-2 case D0082. The response is from the first sampled run
of OLMo-2-1124-7B-Instruct on key 131, a request to shorten a book blurb. Its
third of four `*` bullets has no `-` sub-bullet, but it contains the hyphenated
word "star-gazers". Marin strict passes the response, because that hyphen counts
as a sub-bullet (F3). Fixes only fails it.

| Reviewer | Verdict | Lean | Confidence | Reason, paraphrased |
|---|---|---|---|---|
| A | fail | fail | high | The third bullet has no `-` sub-bullet before the next `*` bullet; the sub-bullet counts are 2, 1, 0, 1 |
| B | fail | fail | high | Of the four bullets, the third is followed directly by the next `*` bullet |

The consensus is fail, so Marin strict's pass is an error. In round 2, the
consensus sided with the validated checker in 36 of the 37 `format:sub-bullets`
disagreements. The other, D0137, has `-` items that are not indented under their
`*` bullet. Both reviewers leaned fail there, and the validated checker passes
it: evidence on section 2's open sub-bullet question.

## 7. Decision gate and recommendations

The decision gate comes from the plan posted on #10060. The validated checkers
count as more reliable than Marin strict only if (a) every fixture passes, (b)
upstream's tests pass apart from the documented disagreements, and the property
tests and the harness-fidelity check pass, (c) scored coverage (the share of
expected verdicts produced) is complete, and (d) the blinded review finds fewer
errors in the validated checkers. On 2026-10-16, all four held, with two
qualifications: (a) held apart from the two tagger fixtures, and (c) held with
the thinking configurations' two sampled runs, which stopped early, reported
and left out.

With the gate passed, the audit report recommends fixing the grader: bringing
Marin's checkers to upstream `1c40f0c`, pinning the dataset revision and the
NLTK data, reporting the validated checkers next to Marin strict and loose, and
recording a checker exception as a per-constraint grading failure. Upstream
still has F3, so only the validated checker grades `format:sub-bullets`
correctly, and the report asks to send F3 and its fixtures upstream. The decision entries go to
Marin's maintainers with the review's evidence. In 40 of the 52 blinded cases
whose disagreement `D-list-layout` caused, the reviewers took the official side.
