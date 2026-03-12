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

