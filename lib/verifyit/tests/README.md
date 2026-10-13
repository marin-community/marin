# VerifyIT tests

Run the package tests from the repository root with `uv run --group test pytest lib/verifyit/tests`, or from `lib/verifyit` with `uv run pytest tests`. Keep the repository's default pytest marker selection when running mixed suites.

`verifyit.test_support` holds trusted scorer functions used by subprocess tests. `verifyit.execution.worker` reimports them by module name in its child process, including when tests run from the repository root.
