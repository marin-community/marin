Create a runnable TaskCompendium task and controls for the proposal in the user message. Write `task.json` as a TaskSpec JSON object and `controls.json` as a JSON list of Control values in `{workdir}`. Use the live schemas below for field names, required fields, discriminator tags and resource encodings. Do not guess the schema or search the filesystem or internet for TaskCompendium source.

## Author environment

Your shell runs in a {environment}, with working directory `{workdir}`. The author image is `{author_image}`. The pinned Python 3.12 Debian image is `{grader_base_image}`.

There is no network access from the shell. `curl`, `wget` and `ps` are not available. Python HTTP clients cannot bypass the network restriction. Use `web_search` and `web_fetch` for all network access; fetch only the external evidence the task needs. Shell tool output is capped at {output_chars} characters, so inspect selected files and fields instead of dumping entire trees or schemas.

## Supported grader environments

When a grader requires an environment, write one of these exact JSON objects in its `environment` field:

```json
{grader_environments}
```

The first selects the pinned Python container; the second selects ShellSim. No other container image is supported. Do not invent registry names, image tags or digests, and do not request builds. Package specs and package locks are unsupported on this Iris host because it has no local factory. Leave those fields absent or null. These restrictions also apply to the task's environment when it needs a machine. A task that needs no machine can keep its environment requirements empty. Keep private verifier resources out of the solver workspace.

## Authoring and submission

Each control contains a candidate (reply and files), an expected status and reward band, and a kind (positive, negative or malformed). Include correct and incorrect answers that test the grader. The `grade` tool checks candidates against the current `task.json`; its file arguments are workspace paths whose bytes are captured for grading. Embedded files in TaskSpec and Control JSON must follow the resource schema below.

Use `submit` with `task_path`, `controls_path` and `notes` when ready. If `validate` is available, it runs the current draft and returns evidence so you can revise before submitting. Use `finish` only when you cannot complete the task.

## Live JSON schemas

The `TaskSpec` and `Control` entries below are generated from the installed models. Each schema's `$ref` paths resolve within that schema's own `$defs`. Always include tagged-union discriminator fields, including context event `type`, even when the schema lists a default. `controls.json` is an array of objects matching the `Control` schema, not an object wrapping that array.

```json
{reference}
```
