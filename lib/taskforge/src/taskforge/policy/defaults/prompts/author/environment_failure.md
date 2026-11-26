The task's own environment did not start: its setup commands failed with:

{error}

Your shell runs in `{image}` instead, with the task's files hydrated, so the task's environment is not available here. Repair the task's `environment_requirements` (its setup commands) so the environment starts, without changing its image, and keep the rest of the task as it is unless other findings require changes. `grade` uses the task's environment, so it reports the setup failure until the setup is fixed.
