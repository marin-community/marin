# Session Context

## User Prompts

### Prompt 1

In the latest changes, let's split the @lib/marin/src/marin/execution/disk_cache.py into 3 invidual util modules, one for disk_cache, one for distriubed lock and another one for fray_exe

### Prompt 2

should the `_write_executor_info(step)` be moved into the disk_cache?

### Prompt 3

I think @lib/marin/src/marin/execution/disk_cache.py shouldn't have the artifact serde embeded, instead that should be injected in @lib/marin/src/marin/execution/step_runner.py and otherwise delegated to the user function, where to user function should first check if the output_path is successful and if yes, do it's own deserialization. Would that approach make sense?

### Prompt 4

ok, let's try that

### Prompt 5

this actually isn't what I expected, I wanted the disk_cached to stay the same, just add 2 extra parameters save and load, but otherwise keep it the same as before

### Prompt 6

ok, but now the disk_cached fn shouldn't take output_path anymore right?

### Prompt 7

ugh, ok, can we change the logic such that when save and load are None, the user provided function is expected to handle the writing/reading from the output_path? Otherwise in @lib/marin/src/marin/execution/step_runner.py we can explicitely set the save/load to Artifact.save/load?

### Prompt 8

can we update the @tests/test_disk_cache.py tests to make em simpler to levarage the new semantic?

