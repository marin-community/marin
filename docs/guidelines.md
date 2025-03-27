Today, this document is for the developer who wishes to contribute to Marin.
Tomorrow, this document will be for an AI agent who will help us enforce the
guidelines.

# GitHub issues

- Every issue should have a meaningful title and description that provides
  sufficient context (see more details below).
- Every issue should be assigned to one primary person who is responsible for
  submitting and merging a PR against it.
- Every issue should be tagged with a priority label:
  * p1: Should be done this week
  * p2: Should be done this quarter
  * p3: do it later
- Every issue should be tagged with a type label:
  * bug: represents an unexpected, undesirable behavior that needs to be fixed
  * documentation: a request to add documentation
  * infrastructure: address a problem with the performance or ergonomics of
    working on the cluster
  * experiment: represents a machine learning experiment to test a hypothesis (see below)
- Comments on issues should also have sufficient context and be substantive.
  Minor edits should be made directly to the issue description.

## Issue description

- The description of each issue should provide enough **context** for someone
  who has experience building language models (not necessarily in Marin)
  and has a rough idea of how Marin works.
- **Link** to any relevant papers, GitHub issues, pointers to code, etc.
- State clearly (i) what the **problem to be solved** is and (ii) what the
  **definition of done** is (when the issue can be closed).  Be as concrete as
  possible with specific metrics, stack traces, commands that were run, etc.

# GitHub pull requests

A pull request (PR) should address (or partially address) an issue.  Please
abide by the following guidelines:

- The PR description should link to the issue.
- The PR should have a high-level bulleted list of the changes made.
- The PR should have example output and metrics obtained from running the code
  when appropriate (e.g., the change is substantial).
- The PR author should go through the code and do a **self-review**, adding
  high-level comments that are not obvious from reading the code (e.g., this
  block of code that looks like it was deleted was actually moved to another
  file).  This will make the reviewer's job a lot easier.

## General code style

- Every file should have a top-level docstring that describes the high-level
  purpose and a rough sketch of the design.  For test files, the testing strategy
  should be described if needed.
- Every non-trivial function should have a meaningful docstring that documents
  the arguments and return value.  Include high-level context that's not
  obvious from the function and variable names (don't just paraphrase the
  code).  Give a concrete example of an input and output if appropriate.
- Use type hints for all variables, function arguments, and return values.
- Use `logger.info` for logging instead of `print`.
- Use `dataclass(frozen=True)` instead of untyped dictionaries.
- Write simple unit tests for any tricky piece of code.
- Break code into small pieces as opposed to having giant monolithic functions
  when the pieces could be reusable.

# Experiments

Experiments are the heart of the open development process.  An experiment represents
a set of changes to the model building process (e.g., architectures, learning
schedule, data), and the sequence of experiments over time represents a complete
record of the development process.

Each experiment is captured by:
- a GitHub issue, which describes the experiment and hypothesis or goal, and
- one or more PRs, which implement the experiment in code.

The lifecycle of an experiment is as follows:
- An GitHub issue is created.
- A PR is created against this issue.  The PR should run a **sanity check**
  experiment that can be run at very small scale on a local GPU.  The PR should
  also do a **dry run of the full experiment** so that one can review the full
  experiment and review the estimated cost before running it.
- This PR is reviewed based on both the code and the result of the sanity check
  experimental output.
- If the PR is approved, then the **full experiment** is run.
- Results and analyses are added to the GitHub issue.

## Experiment issues

In addition to the above guidelines for general issues, the description of an
GitHub experiment issue should include the following:
- A description of the set of change(s) to the system (e.g., trying 3 different
  optimizers), as well as the regime that the experiments are going to be run
  (e.g., 1.4B parameter models for 28B tokens), and the metrics that will be
  used to evaluate the changes.
- A **hypothesis or goal**, which is an a priori prediction of what the outcomes
  will be (e.g., optimizer A will be faster).  This serves as a
  [preregistration](https://en.wikipedia.org/wiki/Preregistration).
- Links to the following, which allow anyone to monitor the progress of an
  experiment, which will be filled in over time:
  * wandb report: a link to a wandb report that has all the training runs
    corresponding to this experiment as well as more detailed analysis of the
    results.
  * data browser: a link to the experiment page(s) on the marin data browser,
    which shows the entire dependency structure of all the assets produced
    (datasets, models, predictions) with associated links.

As the experiment progresses, bugs are fixed, and analyses are conducted:
- Make sure issue comments are added to capture the updated thinking.
- The issue description should be updated with a short summary at the end to
  capture a snapshot of the current status.

Keep the scope of an experiment as small as possible, so that experiments do
not stay open for long periods of time.  Follow up experiemnts can be handled
in new issues that link to the original experiment issue.

## Experiment PRs

An experiment PR should contain a file
`experiments/exp${GITHUB_ISSUE_NUMBER}_${DESCRIPTOR}.py` that contains the code
for the experiment.  This file should take no arguments (aside from those the
executor accepts), and running this experiment should launch all the relevant
jobs for this experiment from start to finish.

Experiments are defined using the [executor framework](executor.md),
which represents a DAG over steps.  Each step makes a call to a (Ray)
function with a custom `config`.

Notes:

- In the top file-level docstring, include a brief summary of the experiment and
  link to corresponding GitHub issue.  The summary should be similar to the
  GitHub issue description, but should reference the code.
- Name variables of type `ExecutorStep` based on what the `ExecutorStep`
  produces (e.g., `llama3_8b_model` as opposed to `llama_8b_model_step`).
- Try to avoid using full paths (e.g., gs://marin-us-central2/...).
  Instead, import the corresponding utility or experiment file that generated
  the path and reference the `ExecutorStep`.  This way, the dependency structure
  is made explicit.
- After the full experiment runs, add `override_output_path` to any heavy steps
  that we don't want to accidentally run again (e.g., a large dataset or training
  a new model).
- When possible, use the default functions (e.g., `default_train`,
  `default_eval`) if the configuration details are orthogonal to the aspect
  being varied in the experiment.
