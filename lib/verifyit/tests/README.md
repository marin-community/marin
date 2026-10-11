# VerifyIT tests

Run the package tests from the repository root with `uv run --group test pytest lib/verifyit/tests`, or from `lib/verifyit` with `uv run pytest tests`. Keep the repository's default pytest marker selection when running mixed suites.

`verifyit_test_support` holds importable trusted scorer functions used by subprocess tests. The root and VerifyIT `pyproject.toml` files put this tests directory on the import path during collection, and `verifyit.execution.worker` passes that path to its child. This checks the real worker import boundary without installing test helpers in the production package.
