# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: StepRunner shouldn't launch tasks with Fray by default (#3035)

## Context

Every step in a Marin pipeline is currently submitted as a separate Fray job. In Iris, each Fray job creates a new container with RAM/CPU/Disk budgets, adding significant startup overhead. Most steps (downloads, orchestration, evals) don't need isolated containers — only steps with explicit resource requirements (tokenization with specific CPU/RAM, training on TPU, etc.) should go...

### Prompt 2

ok, I'm looking through the code, in the remote decorator, could we attach the fray resource to the dynamically created function instead of the original one?

### Prompt 3

add ParamSpec

### Prompt 4

in the StepSpec remove the `resources`

### Prompt 5

In the remote decorator, would it be cleaner to instead create a "RemoteCallable" instance, that is callable, and has the resources field. Then we could simply check if the fn is `RemoteCallable` instead of hasattr//getattr etc. Wdyt?

### Prompt 6

should we move the `env_vars` and `pip_dependency_groups` to the RemoteCallable as well? is that a bad idea?

### Prompt 7

yes

### Prompt 8

could RemoteCallable be a dataclass?

### Prompt 9

in the StepRunner, should we create the local pool instead of `run` and take into account the `max_concurrenct`?

### Prompt 10

do we need to address the comment https://github.com/marin-community/marin/issues/3035#issuecomment-3963669943 ?

### Prompt 11

what is a good git commnet for your last change?

### Prompt 12

why is that important?

### Prompt 13

ok, can we change the design such that the remote creates a wrapper that submits via fray and then use it in the executor if necessary to create RemoteCallable. Step runner should just execute functions via async executor so we don't need to worry about thread pool

### Prompt 14

is this better?

### Prompt 15

Ok, can we make it even simpler, the StepRunner should just execute function and get future them wrapped in the local handle, no check if something is a RemoteCallable

### Prompt 16

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Analysis:
Let me chronologically trace through the conversation:

1. **Initial Plan**: User provided a detailed plan for making Fray submission opt-in in StepRunner (#3035). The plan had two phases: Phase 1 (local execution by default) and Phase 2 (@remote decorator).

2. **Phase 1 Implementation**:
   - Changed `StepSpec.resources` from `Resour...

### Prompt 17

would it be simpler if we have the RemoteCallable return a handle?

### Prompt 18

let's go with the alternative

### Prompt 19

why do we need the `result_holder` weirdness?

### Prompt 20

yes, the disk_cache runs inside the container (inside RemoteCallable), so we don't need to return the result? Am I missing something?

### Prompt 21

no - no, the RemoteCallable wraps the disk_cache and disributed, so we don't need to return anything from RemoteCallabe

### Prompt 22

ok, here's the thing, I don't want StepRunner to do any kind of re-wrapping or instance checks. I want it to just invoke functions. How about this: can we have the StepSpec always wrap the function in `disk_cache` and `distributed_lock` + optionally `remote` (if resources are non None). And have the StepRunner just execute functions?

### Prompt 23

run tests

### Prompt 24

In Step Runner do we need the `max_concurrent` checks, or would we just queue up work in the thread?

### Prompt 25

yep

### Prompt 26

in executor, is:

### Prompt 27

```
    elif isinstance(step.fn, RemoteCallable):
        final_fn = RemoteCallable(
            fn=resolved_fn,
            resources=step.fn.resources,
            env_vars=step.fn.env_vars,
            pip_dependency_groups=step.fn.pip_dependency_groups,
        )
```

needed?

### Prompt 28

ok, check the diff vs main, is there a way to simplify this change?

### Prompt 29

sure

