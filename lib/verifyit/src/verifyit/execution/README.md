# Grader execution

`command.py` runs grader commands in a process group and kills the group on timeout. `worker.py` runs trusted, importable Python functions in a child process with a bounded deadline. The worker imports the function by its module path, so the module must be available to that child.

`source_callable.py` is an ordinary script staged into a task's private resources. Its descriptor names an original source-package function and maps the task's answer or private contract fields to positional and keyword arguments. The script imports the function from the selected image, calls it once, and writes `{"reward": ..., "detail": ...}`. A descriptor may pin an absolute source file by SHA-256 and select a reward key when the original function returns a mapping. The source image must supply the scorer and its dependencies; staging this transport does not copy the scorer into VerifyIT.

The dataset declaration owns the source pin, image, input projection, and control cases. This directory owns invocation and process isolation. See [`../modes/README.md`](../modes/README.md) for shared scoring modes.
