# Marin Agent Notes

Vendored Marin pipeline framework. In this file, leading `/` refers to repository root. Start with `/AGENTS.md`; only Marin-specific conventions are below.

## Key Docs

- `/.agents/skills/add-dataset/SKILL.md` — dataset addition workflow
- `/lib/marin/pyproject.toml` — packaging metadata, extras, dependency groups

## Development

```bash
# Full safe Marin suite
uv run --package marin-core --group test --extra cpu --extra dedup pytest tests

# Lint
./infra/pre-commit.py --all-files --fix
```

## Code Conventions

- Put reusable concrete work here. New pipeline composition, `ArtifactStep` or
  `StepSpec` construction, and references to specific artifacts belong in
  `experiments/**`. The execution framework defines these types; existing
  library step builders are exceptions, not patterns to extend. See
  `/.agents/skills/write-pipeline/SKILL.md`.
- Use `fsspec.open` for filesystem access — do not special-case GCS unless absolutely necessary.
- Do not copy data artifacts to local filesystem; stream through fsspec instead.
- Avoid hard-coding GCS paths like `gs://marin-us-central2/foo/bar`. Bind
  pipeline dependencies in `experiments/**` and resolve their paths at run time.
- NEVER load GCS files from across region if they are more than a few MB.
