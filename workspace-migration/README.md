# Workspace Migration - Step 1

This directory contains the migration script and notes for **Step 1** of the [uv workspace migration plan](../CLAUDE.md#repo-reorg): initializing the workspace and folding `marin` and `data_browser` members into `lib/`.

## Migration Script

Run `./workspace-migration/step-1.sh [REFERENCE_BRANCH]` from the repo root to replay the workspace restructuring on a clean branch.

The script accepts an optional "reference branch" argument, which it can copy from (for files whose content is essentially new in "step 1", not a modification of existing files). If not provided, it will fall back to [`marin-community/marin@rw/ws`].

## Files Updated by Script

The migration script automatically handles all path updates:

1. **`.github/workflows/build-docker-images.yaml`** - Update `src/marin/cluster/config.py` → `lib/marin/src/marin/cluster/config.py`
2. **`.github/workflows/update-leaderboard.yml`** - Update `src/marin/speedrun/` → `lib/marin/src/marin/speedrun/`
3. **`Makefile`** - Update both macOS and Linux sed commands for config.py path
4. **`mkdocs.yml`** - Update `paths: [".", "src"]` → `paths: [".", "lib/marin/src"]`
5. **`.gitignore`** - Remove `lib/` and `CLAUDE.md` exclusions
6. **`pyproject.toml`** - Create workspace root and `lib/marin/pyproject.toml` from reference branch
7. **`CLAUDE.md`** - Create from reference branch
8. **Documentation files** - Update GitHub URLs from `/blob/main/src/marin/` → `/blob/<doc-branch>/lib/marin/src/marin/` (uses `MARIN_DOC_BRANCH` env var, defaults to `main`)

## Testing the Migration

Run [`test-step-1.sh`] to verify the migration is reproducible. This creates an ephemeral test branch from the parent commit, replays the migration, and compares the resulting tree. On success, it cleans up and returns to the original branch.

### Not Impacted

The following do NOT need updates because they work with the new structure:

- **Import statements** (`from marin.X import Y`) - These work unchanged because the package is still named `marin` and installed as an editable package
- **`PYTHONPATH=tests:.`** - The `.` still refers to the repo root, and with the workspace the `marin` package is properly installed
- **Tests** - Import `marin` as a package, which works with the workspace setup
- **Experiments** - Import `from marin.X`, which works because the workspace root depends on the `marin` member

## Testing Checklist

After migration:

- [ ] `uv sync` completes successfully
- [ ] `uv run python -c "import marin; print('Success')"` works
- [ ] `uv run pytest tests/` passes
- [ ] `make check` passes
- [ ] `uv run mkdocs build` generates docs correctly
- [ ] Docker builds work (test locally if possible)
- [ ] Experiments can still import from `marin` package

## Result

After this step, the structure is:

```
marin/
  pyproject.toml        # Workspace root (marin-root)
  experiments/          # Stays at root
  lib/
    marin/              # Workspace member (marin package)
      pyproject.toml
      src/marin/
    data_browser/       # Workspace member
      pyproject.toml
      ...
```

## Next Steps

See [CLAUDE.md](../CLAUDE.md#repo-reorg) for the full workspace migration plan:
- **Step 2**: Add Levanter as a workspace member
- **Step 3**: Add Haliax as a workspace member
- **Step Omega**: Further split into `marin-core`, `marin-crawl`, `ray_tpu`, `rl`, `thalas` packages

[`marin-community/marin@rw/ws`]: https://github.com/marin-community/marin/tree/rw%2Fws
[`test-step-1.sh`]: ./test-step-1.sh
