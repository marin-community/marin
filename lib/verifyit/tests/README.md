# VerifyIT tests

Run the package tests from the repository root with `uv run --group test pytest lib/verifyit/tests`, or from `lib/verifyit` with `uv run pytest tests`. Keep the repository's default pytest marker selection when running mixed suites.

`verifyit_test_support` holds importable trusted scorer functions used by subprocess tests. The tests import that package under the same name from the repository root and from VerifyIT's package root. The `pythonpath` setting in `pyproject.toml` puts this tests directory on the import path during collection, and `verifyit.execution.worker` passes that path to its child. This checks the real worker import boundary without installing test helpers in the production package.
