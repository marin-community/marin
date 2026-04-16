# Marin Agent Notes

Vendored Marin pipeline framework. In this file, leading `/` refers to repository root. Start with `/AGENTS.md`; only Marin-specific conventions are below.

## Key Docs

- `/.agents/skills/architecture/SKILL.md` — repository structure and core concepts
- `/.agents/skills/add-dataset/SKILL.md` — dataset addition workflow
- `/lib/marin/pyproject.toml` — packaging metadata, extras, dependency groups

## Development

```bash
# Run tests
uv run --package marin pytest

# Lint
./infra/pre-commit.py --all-files --fix
```

## Code Conventions

- Use `fsspec.open` for filesystem access — do not special-case GCS unless absolutely necessary.
- Do not copy data artifacts to local filesystem; stream through fsspec instead.
- Avoid hard-coding GCS paths like `gs://marin-us-central2/foo/bar`. Prefer referencing pipeline steps; if you must use a literal path, wrap with `InputName.hard_coded` and call out the risk.
- NEVER load GCS files from across region if they are more than a few MB.
