# Grader execution

`command.py` runs grader commands in a process group and kills the group on timeout. `worker.py` runs trusted, importable Python functions in a child process with a bounded deadline. The worker imports the function by its module path, so the module must be available to that child.

This directory owns invocation and process isolation. See [`../modes/README.md`](../modes/README.md) for shared scoring modes.
