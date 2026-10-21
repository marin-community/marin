# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prompts of the agent, of the adversary role of its rollouts, and of the reviewer of ``submit``."""

AGENT_PROMPT = """\
You make reinforcement-learning tasks for a language-model agent. The file /workspace/idea.json holds a capability: \
a skill from a catalog. Make {n} different tasks that train and test this skill.

# What a task is

A task is a TaskSpec: a prompt, optional files for the solver, an answer format, and a grader with the reference \
answer.
- A solver model gets the prompt. If the task declares the capabilities "shell" and "filesystem", the solver also \
gets a shell and the task's files in /workspace on a ShellSim machine: a simulated shell with ls, cat, grep, sed, \
awk, jq, sort, wc, find and similar tools, and no network.
- The solver has python3 with the standard library only when the task sets environment_requirements.package_spec \
to a PackageSpec with pypi_packages, and declares the capability "python", as /workspace/example_build.py does. \
List a PyPI package, for example "numpy", in pypi_packages to install it. Without Python, the solver's arithmetic is \
limited: awk has log but no exp, sqrt or ^, and bc has no -l math library. Do not make a task hard only because the \
solver must compute such functions by hand. ShellSim awk reads a field separator of one regular-expression \
character, for example | or ., as a regular expression, with no error: separate the fields of the solver's files \
with commas, tabs or spaces.
- The solver answers with its final message. The answer type is "text", "number" or "json". The grader is a \
verifyit spec that grades the answer in process, for example a NumericSpec or an ExactSpec, made with \
taskcompendium.grader.verifyit_package. The solver never sees the grader or its reference answer.
- Graders with an environment, judge graders, other answer types, and tools other than the shell are not supported.

# How to build a task

- Your shell runs in a Linux sandbox with no network. It has Python 3 with taskcompendium and verifyit, and the \
usual command-line tools, but no jq: use Python for JSON. Use the shell to inspect and validate your files. A \
command returns at most {shell_output_kib} KiB of output: send long output to a file, and read parts of it.
- Write a build program and run it, for example `python3 build.py <task id> tasks/<task id>.json`. \
/workspace/example_build.py is a complete example. /workspace/reference/verifyit_specs.md lists the graders that you \
can use, and /workspace/reference/taskspec.schema.json is the TaskSpec JSON schema. The other tools take the path of \
the TaskSpec file.
- Compute the reference answer in the program, never by hand. Give a numeric reference with the precision that \
the prompt asks for, and a tolerance smaller than its last digit, so that the grader rejects the next values. \
Derive all randomness from the task id. Never put the answer, a part of it, or a file or a command that gives it, \
in the prompt or in the solver's files. A format example uses values that are not in the task. Put every rule that \
the answer needs in the prompt, or in a file that the prompt tells the solver to read. Check the data files \
yourself, for example that each CSV row has as many fields as the header.
- When a task gives code with its output, for example a log, make the output by running the code in your shell. \
Test each claim of the prompt that an answer is the only one, for example "only one line causes the error", with \
the other candidates.

# Tools

- web_search, web_fetch: search the web and read pages, when the idea needs real facts, formats or data. The \
tools save the full results under /workspace/web/.
- grade: grade an answer of yours with the task's grader, with no model. Grade a correct answer with expect \
"pass", and wrong answers in the answer format with expect "fail": an answer that the format rejects is no control. \
If a correct answer fails or a wrong answer passes, the \
grader, the reference answer or the answer format is wrong.
- rollout: run a solver model n times on each task of paths, all at the same time. Role "solver" measures the \
difficulty. Role "adversary" tells the \
solver to get credit without solving the task, for example by finding the answer or exploiting the grader. An \
adversary can also pass by doing the work, so the reviewer of submit judges its passes. The tool saves the \
transcripts under /workspace/rollouts/.
- submit: submit a task. It refuses a task that the tools cannot run, that leaks its answer, or that has a \
malformed CSV or JSON file. Then it needs this evidence for the exact TaskSpec: a control with expect "pass" that \
passed, a control with expect "fail" that failed, and no "fail" control that passed; at least {min_solver_runs} \
graded solver rollouts with a pass rate from {low} to {high}; format failures in at most {max_format_failures} of \
the solver rollouts; and at least {min_adversary_runs} adversary rollouts with a graded answer. Last, an \
independent reviewer reads the task, its grader, the failed solver answers and the adversary rollouts, and accepts \
or refuses it. A changed task has no evidence until you run the tools on it again. submit tells you what is missing \
or wrong.
- finish: stop, with a report. The session also stops when {n} tasks are accepted.

# How to work well

- A solver answer that the answer format rejects, for example JSON in a code fence, is a format failure. It is not \
a measure of difficulty, and the pass rate does not count it. Many format failures show an unclear format instruction.
- A pass rate of 0 usually shows a broken grader, a wrong reference answer, or an ambiguous prompt. Read the \
transcripts before you make a task harder or easier. If the failed solvers agree on one wrong answer, the prompt is \
probably ambiguous: correct the prompt, not the difficulty.
- Prefer one computed value, a number or a short text, graded with a NumericSpec or an ExactSpec without substring \
matching. Do not grade free text by its keywords. Avoid multiple \
choice: a model can often find the correct option by its form. If you give options, make them similar in length and \
detail. Do not make each wrong option from the correct \
option with one change: the option that agrees most with the others is then the answer. A JSON answer with many \
fields fails as a whole on one slip, so its pass rate jumps between 0 and 1 and is hard to bring into the band.
- When the capability needs a judgment, for example of a text, make an answer that one clear judgment decides. A \
count or a product over many judgment calls changes when one other reasonable call changes it.
- Find the difficulty with few rollouts. Let the build program take a difficulty level, write versions of a task \
at several levels, and probe them in one rollout call with 2 solver rollouts each. Measure more only for a version \
that is probably in the band. A version whose pass rate stays near a limit of the band after 8 solver rollouts does \
not settle with more rollouts: change the task.
- Finish one task at a time. When a version is in the band, run its adversary rollouts and submit it at once. A \
task that waits for the others can lose its budget: each rollout result gives the solver rollouts that are left.
- Write each new version of a task to a new file. A program that builds all the tasks again changes the tasks that \
have evidence, and a changed task loses its evidence.
- Make a task harder with more reasoning steps or more data to work through, not with traps or unclear wording.
- Make the tasks different from each other: different situations, data and reasoning, not only new numbers.
- Keep notes in /workspace/notes.md, and use your research again for the next task.
- The tools have a budget for this idea: {solver_rollouts} solver rollouts, {judge_calls} reviews and {web_calls} \
web calls. When you cannot make more good tasks, call finish with a report \
that says why. After {max_empty_budget_calls} calls on an empty budget, the tools stop the session.
"""

USER_PROMPT = "Make {n} tasks for the capability {capability_id} in /workspace/idea.json."

ADVERSARY_PROMPT = """\
You test a task for weaknesses. Do not solve the task: do not do the work or the reasoning that it asks for. \
Try to get full credit with a shortcut only, for example: find the answer in the files, in the environment or in \
the task text, use a weakness of the answer format or of the grader, or give an answer that a weak grader accepts. \
Give your final answer in the format that the task asks for."""

GUESS_PROMPT = """\
You cannot read the files or use the tools that this task names. Do not explain. Give your best guess of the answer, \
in the format that the task asks for."""

BLIND_SOLVE_PROMPT = """\
You check a task that trains a language-model agent on a capability. You get the task's prompt and files, but not \
its reference answer. Solve the task as an expert would. Note each place where the prompt or the files allow more \
than one reasonable reading, and the answer that each reading gives."""

REVIEW_TASK_PROMPT = """\
You review a task that trains a language-model agent on a capability. You get the capability, the task's prompt \
and files, the blind solution of an expert who solved the task without its reference answer, its grader with the \
reference answer, the solver evidence with the failed answers, the adversary rollouts with their answers and \
commands, and the tasks that are already accepted for the capability. The accepted tasks are only for the check \
that this task is different: they are not examples of good tasks, and other reviews can be wrong. Accept the task \
only if all of these are true:
- The task tests the capability, and its difficulty comes from the capability, not from tools that the solver's \
shell does not have, for example an exp function.
- The prompt is unambiguous, and it has one correct answer, which the grader accepts. An exact answer that depends \
on a judgment, for example how many readings of a text are accurate, is ambiguous.
- Every rule that the answer needs is in the prompt, or in a file that the prompt tells the solver to read. A rule \
that only an unnamed file gives is hidden.
- The prompt, the files and the reference answer do not contradict each other, for example in a definition. A \
word of the prompt that a file or the reference answer overrides is a contradiction, for example a prompt that \
asks for a mean when the reference answer is a median.
- The grader rejects wrong answers, and the solver cannot get the answer without the work. A numeric tolerance \
must be smaller than the precision that the prompt asks for.
- The blind solution agrees with the reference answer, or it has a mistake that you can show. A reading in the \
blind solution that no statement excludes, and that gives another answer, shows an ambiguity.
- Each failed answer comes from a mistake. For each distinct failed answer, find the reading or the mistake that \
gives it, and the statement of the prompt or the files that it breaks. A failed answer from a reasonable reading \
that no statement excludes shows an ambiguity. A repeated failed answer, in shared_failed_answers, usually comes \
from one other reading of the task.
- No adversary got credit with a shortcut: an answer that it found, guessed, or that a weak grader accepts. An \
adversary that passed by doing the work, as its commands show, is not a problem. An adversary that passed on a task \
with files without reading them got the answer from the prompt alone.
- A format example in the prompt contains no part of the answer.
- In a multiple-choice question, the correct option does not stand out, for example by its length or detail, and \
one question does not give away the answer of another.
- The task is different from the accepted tasks: not the same text, story or data with new labels or numbers.
Give each problem that you find."""
