---
name: audit-grader
description: Audit a code check, LLM judge, or RL reward function against its task text, measure its effect on scores, and estimate its error rate with a blinded review of frozen samples. Use when asked to audit a grader or when a grader's scores look implausible; do not activate for routine grader changes.
---

# Audit a Grader

A grader turns a response into a verdict: pass or fail, or a class such as
correct, incorrect, or not attempted. It can be a code check, an LLM judge, or
an RL reward function. This audit measures how often a grader contradicts the
task text and how much that changes scores.
[references/ifbench.md](references/ifbench.md) is a worked example, the audit
of the 58 IFBench instruction checkers. That audit read the checker code before
it wrote its contracts and kept its frozen samples in its own repository; where
it differs from the steps below, follow the steps.

Treat an LLM judge as a pipeline, and audit every stage: graded-text
extraction, judge-prompt rendering, the judge call, reply parsing, label
mapping, and aggregation.

Record the audit in the benchmark's issue: the pinned inputs, the decision
entries, the findings, and the hash of each frozen sample. Keep fixtures as
regression tests in the repository's grader test directory, and mark the ones
that the current grader fails with `pytest.mark.xfail(strict=True)`, so the
change that fixes the grader must remove the mark. Keep the frozen samples, the
keys (each case's source model and grader verdicts), and the reviews in an
immutable artifact, such as a pinned Hugging Face dataset revision, and link it
from the issue.

## 1. Pin the inputs

A benchmark can have several implementations, such as an Evalchemy task and a
Harbor task. If the audit starts from scores that look implausible, find the
implementation that produced them in the run configuration or the stored run
records, and audit that one first.

Record the grader commit, the reference implementation (the benchmark authors'
original grader, if one exists) and its commit, the dataset revision with one
hash per row, how each prompt is rendered for the model, the response
extraction rule, and every grader dependency, including data files such as
NLTK data. Pin the generation settings that change the graded text: output
budget, stop sequences, chat template, the server's reasoning parser, and any
policy that merges a reasoning field into the answer. For an LLM judge, also
pin the judge model, prompt template, decoding settings including the token
limit, and reply parser. For an agent task, where an agent program prompts a
model and runs the model's commands in a sandbox, and a verifier script grades
the result, also pin the agent and its prompt template, the task's image, the
sandbox's network access, and the verifier script from the run's task files.

Take each value from the stored run records when they include it; a run can
override the defaults in code and configuration. Report an input that the
benchmark leaves unpinned as a finding, and list an input you cannot fetch as
unverified, with its location (a URL or a record field) and the command that
would verify it. Before and after each run, check that every file the run
loads matches its pinned commit or hash.

## 2. Write a contract for each check

A check is one separately scored requirement, such as an instruction type or a
rubric item. Write its pass and fail conditions, or the class each kind of
response must get, from the text that the model or the judge sees: the
instruction, rubric, or reference answer. Write the contract before you read
the grader code, so that it does not copy the grader's behavior. For an LLM
judge, read only the rubric in the prompt template first. Record the contract
file's hash and the time before you read any code, and mark each later
contract change or decision entry with its reason.

- Define the graded text: the whole visible answer, an exact answer, or a
  payload such as a CSV block. For a reasoning model, grade the text after the
  model's last end-of-reasoning marker (`</think>` for Qwen3), and never a
  separate reasoning field such as `reasoning_content`. A response whose
  reasoning never closes is an extraction failure: it fails every check, and
  the report counts these responses separately. For a grader with more than
  two labels, give an extraction failure its own label: it counts in the
  denominator, is never correct, and is not an attempt. Never give it the
  not-attempted label, which would hide it among the model's abstentions.
- For an agent task, a trial is one agent run on one task, and the verifier
  often grades an answer file that the agent writes. State how a missing or
  empty answer file counts, which trials enter the denominator (for example,
  trials in which the agent or the verifier raises an error), how each label
  maps to the reward, and whether the agent may use the network.
- If the text allows two verdicts, add a decision entry named `D-<topic>` to
  the report's decision table, with both readings, an example of each, and a
  proposed reading. The maintainers of the evaluation that runs the grader
  decide. Count decisions separately from defects.
- If you implement the contracts as a replacement grader, give it one switch
  per decision, so that the effect of each reading can be measured. For an LLM
  judge, a replacement changes the deterministic stages and keeps the judge
  model and template.

## 3. Build labeled fixtures

Write pass, fail, and near-miss responses for every check, and label them from
the contract alone. Run them against the grader as regression tests. Run an
agent task's verifier as the evaluation runs it: in the task's image, or with
the same interpreter, text encoding, and shell. A fixture that the grader gets
wrong is a finding when the contract is unambiguous, and a decision entry
otherwise. A wrong verdict that matches non-proposed readings of several
decision entries together is also a decision, unless the contract rules it
out under every reading; name every entry involved. Keep the smallest
response that reproduces each wrong verdict, and give each fixture a stable
id, such as the check name plus a content hash.

For an LLM judge, write fixtures for each deterministic stage: judge replies
for the parser and the label mapping, run with a fake judge, and candidate
responses for extraction and judge-prompt rendering. Grade candidate responses
with the real judge only in step 5.

## 4. Compare with the reference and stress the grader

- Run the grader and the reference implementation on the same responses, and
  classify every disagreement as a grader defect, a reference defect, or a
  decision. Run the whole reference pipeline, including its sampling, retry,
  and aggregation code. Skip this comparison when no reference exists.
- If you built a replacement grader, run the reference's unit tests against
  it, and list each assertion that the contract contradicts. If the reference
  has no tests, list the grader's own tests that encode a decision.
- Add property and metamorphic tests. A verdict must stay the same under edits
  that the contract ignores, and must change under edits that it forbids.
- Fuzz the grader, or for an LLM judge, its reply parser and judge-prompt
  renderer. Count an exception as a grading failure, separate from model
  accuracy, and check whether one exception stops the run. Fuzz counts depend
  on the fuzzer's input distribution, so report them as counts; take rates
  from steps 5 and 6.
- Test score-level transforms against the contract, such as IFBench's loose
  grading ([references/ifbench.md](references/ifbench.md), section 4).
- Grade content-free responses: empty text, repeated text, and a restatement of
  the instructions. List each check that they pass. For an LLM judge, check
  offline that they reach the judge unchanged, and grade them in step 5.

## 5. Run a model panel

Start from the stored records of existing runs, and count what they already
show, such as truncated responses or merged reasoning, before you generate
anything. Generate responses from several models and decoding settings with
the pinned prompt rendering. Grade every response with the grader under audit,
the reference, and any replacement. Report pass rates per check, verdict
disagreements, scored coverage (the share of expected items that receive a
verdict), and the change in each model's score.

- Rerun at least one configuration with identical settings, and compare each
  score change with the rerun difference.
- For an LLM judge, repeat each judgment at the deployed settings and report
  how often the verdict flips. Grade each graded-text variant (the grader's and
  the reference's) with each judge setup (model and settings), so that each
  score change belongs to one stage. Give each grader the text that its own
  extraction produces; feeding both the same text hides differences in
  extraction.
- For an RL reward, report the share of passes that contradict the contract, as
  judged by a replacement grader or the blinded review. These are false
  rewards.

## 6. Estimate the error rate with a blinded review

Before the review, record in the issue the result that would change the
recommendation in step 7. For example: the replacement passes every fixture,
scored coverage is complete, and its error rate is lower than the current
grader's, with a paired interval that excludes zero.

1. Draw the sample with a script and a recorded seed. Stratify by whether the
   grader under audit and the replacement (or the reference) disagree, and
   record the population size of each stratum. Sample all or most of the
   disagreements. The agreement stratum is usually most of the population, so
   its sample size sets the width of each grader's interval. Hold out
   responses that nobody used to develop a replacement grader, and draw the
   final review round from them.
2. Freeze the sample: write the sha256 of the case list in the issue before
   any review starts.
3. Render the cases into batch files. Each case shows an opaque case id, the
   check's name and text, the prompt, and the response between explicit begin
   and end markers. For an agent task, the response is what the verifier
   reads, frozen from the trial: the answer file or other graded state, and
   any execution result the verifier uses. Do not show the model, a grader
   verdict, or another reviewer's labels.
4. Have two reviewers label every case, in separate runs and in different
   orders. Each label has a verdict (one of the grader's labels, or
   `ambiguous`), a lean (the label under the reading that the reviewer finds
   most natural), a confidence, and a reason that states the deciding fact or
   count. For language-model reviewers, use
   [references/reviewer-prompt.md](references/reviewer-prompt.md), and give
   them a helper that prints a case's exact response. Isolate each reviewer run
   if the reviewers' environment allows it. Otherwise, report that
   independence was instructed. Report whether the reviewers were people or
   language models.
5. A case has a consensus when both leans agree, and an error is a grader
   verdict that differs from the consensus. A grader's error rate is the share
   of population verdicts with an error. Estimate it per stratum from the cases
   with a consensus, weight it by stratum size, and give 95% bootstrap
   intervals that resample within strata. A lean on an `ambiguous` verdict
   only prefers one of the readings that the text allows, so also report each
   rate over the cases where both verdicts are definite and agree. Report the
   paired difference between graders, Cohen's kappa between the reviewers'
   leans, and the disputed cases. For a grader with more than two labels, also
   report the confusion matrix against the consensus. Do not resolve disputed
   cases yourself.

## 7. Report

Lead with the error rate and its effect on scores. For each finding, give an
id, the check or pipeline stage, the fixture or case that shows it, the
contract that it breaks, the share of verdicts that it affects, and its status
in the reference grader. When the share needs a panel that has not run, name
the stored field that would measure it. List the decision entries for the
evaluation's maintainers. Recommend one action: admit the grader, fix it, or
replace the benchmark. Post the report in the issue, following the issues
guide in the `writing-style` skill.
