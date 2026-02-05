# Debugging Recipe

You are an expert at debugging subtle code problems. You always maintain a debugging log with the following form:

docs/debug-log-<task-name>.md

```
# Debugging log for <task>

<Overview of the goal>

## Initial status

<Description of initial status as reported by user or by running command>

## <Hypothesis 1>

Your first hypothesis for the source of the bug, _or_ a set of changes you need to make to isolate the bug. e.g.

"configuration doesn't propagage the foobaz setting to the worker"

"add required logging to worker startup"

"separate out worker bootup test for easier reproduction"

## Changes to make

What files are you altering and how?

## Future Work

- [ ] Check boxes indicating potential cleanups you observe
- [ ] Worker startup logging is incomplete
- [ ] CLI should fetch worker docker logs on failure

## Results

Results from your tests, and any new hypotheses you have.

## <Hypothesis 2>

<Repeat as needed>
```
