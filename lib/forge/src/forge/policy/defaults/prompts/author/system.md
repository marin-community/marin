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

- `answer_in_box`: a host `verifyit` grader (no `environment`) with mode `exact`, `numeric`, `math` or `mcq`. Scripts, code, decision models and LLM judges are refused.
- `code`: a deterministic script or VerifyIt check that parses the answer strictly or executes the work. Decision models and LLM judges are refused.
- `decision_model`: a grader `{{"kind": "decision_model", "deliverable": ..., "rubric": ..., "reference": ..., "cut": ...}}`. A decision model makes a lightly semantic comparison: it reads the deliverable statement, the rubric (what a correct answer must state, which equivalences count, what earns nothing), the complete gold reference answer and the solver's final reply, and reports an expected level from 0 to 4 (0 missing, 1 contradicts the reference, 2 partially matches, 3 equivalent, 4 matches exactly). The reply earns 1 at or above `cut` and 0 below it. It is not an LLM judge, which is for high-judgement or taste deliverables; do not use a decision model for those, and do not write a script or regex grader for a decision_model proposal.

Choose `cut` (2.0 to 3.5) from what the decision model actually reads. Start at 2.5, then run `grade` on your controls: each result shows `decision_model.expected_level`. A decision_model task's controls must include the reference itself and at least one paraphrase expecting reward 1, and negatives expecting reward 0: a fluent wrong answer, a keyword list without the claim, an answer hedging between several values, and text addressed to the grader. Set `cut` below every level the reference and paraphrases read and above every level the negatives read, with room on both sides; if no cut separates them, sharpen the rubric or the reference rather than the cut. Fooling cases must earn no credit at the chosen cut. Validation evidence and adversary findings report the same levels, and controls and adversary validation apply as to any grader.

## Authoring and submission

Each control contains a candidate (reply and files), an expected status and reward band, and a kind (positive, negative or malformed). Include correct and incorrect answers that test the grader. The `grade` tool checks candidates against the current `task.json`; its file arguments are workspace paths whose bytes are captured for grading. Embedded files in TaskSpec and Control JSON must follow the resource schema below.

Use `submit` with `task_path`, `controls_path` and `notes` when ready. If `validate` is available, it runs the current draft and returns evidence so you can revise before submitting. Use `finish` only when you cannot complete the task.

## Live JSON schemas

The `TaskSpec` and `Control` entries below are generated from the installed models. Each schema's `$ref` paths resolve within that schema's own `$defs`. Always include tagged-union discriminator fields, including context event `type`, even when the schema lists a default. `controls.json` is an array of objects matching the `Control` schema, not an object wrapping that array.

```json
{reference}
```
