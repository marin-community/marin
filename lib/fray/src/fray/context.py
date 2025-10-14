# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context variable management for Fray.

Provides thread-safe context management using Python's contextvars, allowing
code to access the appropriate JobContext without explicit passing.
"""

from contextvars import ContextVar

from fray.job import JobContext

# Use ContextVar to handle job context across threads or async tasks
_job_context: ContextVar[JobContext | None] = ContextVar("job_context", default=None)


def get_job_context() -> JobContext:
    """
    Get the current job context.

    Retrieves the JobContext for the current execution context. If no context
    has been explicitly set, automatically initializes a local in-memory context
    for testing/development purposes.

    Returns:
        JobContext: The current job context
    """
    ctx = _job_context.get()
    if not ctx:
        raise RuntimeError(
            "No JobContext is set for the current execution context. "
            "Please ensure that you are running within a job context."
        )

    return ctx


def set_job_context(ctx: JobContext) -> None:
    """
    Set the job context for the current execution context.

    This should typically be called by the job framework (not user code) to
    establish the context for a job's execution.

    Args:
        ctx: The JobContext to set as current

    Example:
        def job_entry_point(ctx: JobContext):
            set_job_context(ctx)
            # Now all code in this context can use get_job_context()
            ...
    """
    _job_context.set(ctx)


def clear_job_context() -> None:
    """
    Clear the current job context.

    Resets the context variable to None. Useful for testing or cleanup.

    Example:
        # In a test fixture
        def test_something():
            ctx = LocalJobContext()
            set_job_context(ctx)
            # ... test code ...
            clear_job_context()  # Clean up
    """
    _job_context.set(None)
