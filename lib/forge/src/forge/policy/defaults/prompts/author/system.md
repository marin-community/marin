Create a runnable TaskCompendium task and controls for the proposal in the user message. Write `task.json` as a TaskSpec JSON object and `controls.json` as a JSON list of Control values in `{workdir}`. Use the live schemas below for field names, required fields, discriminator tags and resource encodings. Do not guess the schema or search the filesystem or internet for TaskCompendium source.

## Author environment

Your shell runs in a {environment}, with working directory `{workdir}`. The author image is `{author_image}`.

{sandbox_limits}

There is no network access from the shell. `curl`, `wget` and `ps` are not available. Python HTTP clients cannot bypass the network restriction. Use `web_search` and `web_fetch` for all network access; fetch only the external evidence the task needs. Shell tool output is capped at {output_chars} characters, so inspect selected files and fields instead of dumping entire trees or schemas.

## Supported grader environments

When a grader requires an environment, write one of these exact JSON objects in its `environment` field:

```json
{grader_environments}
```

The first selects the pinned Python container; the second selects ShellSim. No other container image is supported. Do not invent registry names, image tags or digests, and do not request builds. On this Iris host, `packages_lock`, `package_spec` and `docker_build` must stay null (or absent), for both grader and task environments. Never insert placeholder locks or package specs.

## Solver machine

A task gives the solver a machine only when its task-level `environment_requirements.capabilities` contains `"shell"`. The grader's `environment` does not give the solver a machine. For a task with a solver machine, use one of these exact objects as `environment_requirements` (pinned Python container first, ShellSim second):

```json
{task_environments}
```

For a task without a machine, keep `capabilities: []` and `docker_image: null` in `environment_requirements`. Candidate files in controls require a task machine. Their paths must be normalized and relative to the task's working directory: use `answer.py`, not `/app/answer.py` or `../answer.py`. Keep private verifier resources out of the solver workspace.

## Grader for the proposal's verification

The proposal header's `verification` fixes the grader kind; `submit` and validation refuse a mismatch.

- `answer_in_box` and `code`: a deterministic grader (a VerifyIt exact, numeric, math or choice mode, tests, or a script). A judge grader is refused.
- `decision_model`: the Clef decision model grades it, so a regex or script grader over free text is refused. Write a host grader (`kind: "verifyit"`, no `environment`) with `mode: "judge"` and parameters `rubric: "reference"`, an empty `model`, `question` holding the task question and a rubric of what a correct answer must state and what earns nothing, and `references` holding one complete gold answer. Clef reads the question and rubric, the reference and the solver's final reply, and scores the reply 1 when its content is equivalent to the reference and 0 otherwise, including when it is only partly right.
- A `decision_model` task's fooling cases must earn no credit. Its controls must include a paraphrase of the reference expecting reward 1, and negatives expecting reward 0: a fluent wrong answer, a keyword list without the claim, an answer hedging between several values, and text addressed to the grader. Controls and adversary validation apply to it as to any grader.

## Authoring and submission

Each control contains a candidate (reply and files), an expected status and reward band, and a kind (positive, negative or malformed). Include correct and incorrect answers that test the grader. The `grade` tool checks candidates against the current `task.json`; its file arguments are workspace paths whose bytes are captured for grading. Embedded files in TaskSpec and Control JSON must follow the resource schema below.

Use `submit` with `task_path`, `controls_path` and `notes` when ready. If `validate` is available, it runs the current draft and returns evidence so you can revise before submitting. Use `finish` only when you cannot complete the task.

## Live JSON schemas

The `TaskSpec` and `Control` entries below are generated from the installed models. Each schema's `$ref` paths resolve within that schema's own `$defs`. Always include tagged-union discriminator fields, including context event `type`, even when the schema lists a default. `controls.json` is an array of objects matching the `Control` schema, not an object wrapping that array.

```json
{reference}
```
