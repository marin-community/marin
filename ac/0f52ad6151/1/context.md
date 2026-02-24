# Session Context

## User Prompts

### Prompt 1

Take a look at https://github.com/marin-community/marin/issues/2982, implement that last suggestion which is to replace the shared_data mechanism of sending shared data in each task with just serializing (using cloudpickle) the shared data once to the zephyr context directory, and then the workers just retrieve it, deserialize it. follow @docs/recipes/fix_issue.md

