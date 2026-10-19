# Worked Example: IFBench in Evalchemy

This is the audit of the 58 IFBench instruction checkers that Marin's evaluation
policy runs through Evalchemy, for
[#10060](https://github.com/marin-community/marin/issues/10060). Paths below are
relative to
[the audit repository at `a31c684`](https://github.com/XiaoyuanWang111/ifbench-audit/tree/a31c6841dae2632dafda740bd51496843d9e5f38).
The Hugging Face dataset
[`xiaoyuanwang/ifbench-audit` at `ceaba90`](https://huggingface.co/datasets/xiaoyuanwang/ifbench-audit/tree/ceaba90ba6a2acac8fd8b51368a37e27c0ba688f)
has the fixtures, the findings, the decisions, the blinded-review records, and
the panel verdicts.

Marin strict is Marin's checkers under Evalchemy's strict grading. The validated
checkers implement the audit's contracts. The official reading of an
instruction is what Marin's and upstream's checker code does with it. Fixes
only is the validated checkers with each of the 13 interpretation switches set
to the official reading, and proposed is the validated checkers with the
proposed readings.

## 1. Inputs

- Data: `allenai/IFBench_test` at `2e8a48de`, with one sha256 per row in
  `protocol/manifest.json`.
- Graders: Marin's checkers in Evalchemy's `eval/chat_benchmarks/IFBench`,
  whose checker code is unchanged from `ab7ae27` to `09d27c9` and in Marin's
  vendored copy at `2d9bb82` (`vendor/evalchemy`, #10288), and upstream
  IFBench at `1c40f0c`, upstream's main branch when audited.
- Dependencies, as the audit pinned them: nltk 3.10.0, emoji 2.15.0, syllapy
  0.8.0, and the NLTK data files by hash.
- When audited, Evalchemy loaded the dataset without a revision. The checkers
  download NLTK data at import time. The audit reports both as finding G5.

## 2. Contracts and decisions

`src/ifbench_audit/contracts.py` has one validated checker per instruction
type. Each docstring states the contract, written from the instruction text.
The decision table in `reports/AUDIT.md` has 15 entries where the text allows
two verdicts. Three more open questions have no proposal yet, and both modes
keep the current behavior for them. `Choices.official()` selects the official
readings (fixes only), and `Choices()` selects the proposals.

## 3. Fixtures

`fixtures/ifbench_fixtures.jsonl` has 143 fixtures, each labeled from the
instruction text: 81 are linked to a finding or a decision, and 62 are
controls. The test suite adds property and metamorphic tests: 438 tests pass,
and the 2 expected failures are the two documented part-of-speech tagger cases.

## 4. Reference comparison and stress tests

- Upstream's unit tests pass on the validated checkers except at five
  documented assertions (`results/upstream_tests_on_validated.txt`).
- The audit's reimplementation of Evalchemy's strict and loose grading matches
  Evalchemy at commits `97fffeb` and `28631c7` on 5,848 verdicts: the greedy
  run of all 16 panel configurations and upstream's sample responses
  (`results/harness_fidelity.txt`).
- Marin's checkers fail 8 of the 10 property tests
  (`results/properties_on_official.txt`).
- Fuzzing found that `custom:csv_quotes` raises `IndexError` on 251 of 3,000
  random tab-separated outputs with the required header (F15). One checker
  exception fails the whole IFBench task (G6).
- Loose grading accepts a constraint if it passes on any of eight variants of
  the response: the response, a copy with `*` removed, and each of those with
  the first line, the last line, or both dropped. It passes any response to
  `format:sub-bullets`. On upstream's sample responses, 15 constraints fail
  strict grading and pass loose grading (G1, `results/loose_rescues.txt`).

## 5. Findings and model panel

The audit has 43 checker findings (F1 to F43), each with a minimal
counterexample or the decision it rests on, and seven protocol findings (G1
to G7) about the dataset and the grading harness, such as loose grading,
unpinned inputs, and exception handling. Nine of the checker findings depend
on a decision. In Marin's
checkers, 37 of the 58 instruction types have at least one unambiguous defect,
and those types carry 200 of the 344 test constraints. 31 of the 37 types are
still defective in upstream at `1c40f0c`. Reading the checker code found F1 to
F38, the panel's disagreements found F39 to F42, and the blinded review found
F43.

The panel has 16 configurations of 14 open models (7B to 32B), generated on one
NVIDIA H100 80GB GPU with vLLM 0.30.0. Every grader graded all 68,400
responses. With thinking off, as Marin's eval policy runs IFBench, fixing the
defects moves a model's greedy prompt-level score by −1.7 to +3.0 points.
Rerunning a configuration's greedy decode in the same prompt order moves its
Marin strict score by up to 1.3 points. As a reward, Marin strict is wrong on
5.7% of the constraint passes it gives, taking the fixes-only verdict as
correct.

## 6. Blinded review

`results/blind_audit/` holds the frozen samples, keys, batches, reviews, and
scores. `analysis/build_blind_audit.py` draws the sample,
`analysis/blind_audit_batches.py` writes the batches, and
`analysis/blind_audit_compare.py` scores the reviews.

| | Round 1 | Round 2, held out |
|---|---|---|
| Population | 5,504 verdicts: the greedy run of 16 configurations | 4,668 verdicts: the first sampled run (temperature 0.7) of 14 configurations, without 148 that repeat the greedy response |
| Cases | All 328 disagreements, 100 agreements, 136 fixtures | 150 of 259 disagreements, 50 of 4,409 agreements |
| Cohen's kappa between reviewers | 0.94 | 0.96 |
| Error rate, Marin strict | 7.6% [4.7, 11.3] | 6.1% [3.9, 10.1] |
| Error rate, fixes only | 4.9% [2.1, 8.6] | 3.2% [1.0, 7.2] |
| Marin strict minus fixes only, points | +2.7 [+2.2, +3.1] | +2.9 [+2.4, +3.4] |

Disagreements are verdicts where Marin strict and proposed differ. Round 1
included the 136 fixtures that existed when it was frozen. The review checks
them against their labels, and they do not enter the error rate. Round 2 was
frozen after the fixes that round 1 prompted. Its responses were not used to
develop the checkers, and no reviewer had seen them.

Each reviewer batch was a separate agent run on Claude Opus 5.5 at max effort.
Parts of four round-1 batches ran on earlier Claude models
(`results/blind_audit/README.md` lists them). Reviewer B read the cases in a
different order from reviewer A. The sandbox did not enforce independence; the
reviewers were instructed to keep it.

## 7. Decision gate and recommendations

Section 10 of `protocol/PROTOCOL.md` sets the decision gate. The validated
checkers count as more reliable than Marin strict only if (a) every fixture
passes, (b) upstream's tests pass apart from the documented disagreements, and
the property tests and the harness-fidelity check pass, (c) scored coverage is
complete, and (d) the blinded review finds fewer errors in the validated
checkers. On 2026-10-16, (a) held apart from the two tagger cases, and (b),
(c), and (d) held.

`reports/AUDIT.md` recommends fixing the grader: bringing Marin's checkers to
upstream `1c40f0c`, pinning the dataset revision and the NLTK data, reporting
the validated checkers next to Marin strict and loose, and recording a checker
exception as a per-constraint grading failure.
