# Co-develop Marin and Levanter (as a Submodule)

If you need to co-develop [Levanter](https://github.com/stanford-crfm/levanter) and Marin, you can check out Levanter as a working tree inside Marin and install it in editable mode.

## Set up the submodule working tree
```
# Run from ROOT_DIR
git clone https://github.com/stanford-crfm/levanter
git clone git@github.com:stanford-crfm/marin.git
mkdir -p marin/submodules
cd levanter
# Create a new working tree for the LEVANTER_BRANCH branch of the levanter repository,
# located at ROOT_DIR/marin/submodules/levanter. This means ROOT_DIR/marin/submodules/levanter
# is now a working directory specifically for the LEVANTER_BRANCH branch of levanter.
git worktree add ROOT_DIR/marin/submodules/levanter LEVANTER_BRANCH
cd ..
```

Note that this is not the same as `git submodule add` because it does not modify the Marin repository. We find this workflow more convenient for local development.

## Install Levanter (editable)

- Marin itself is installed by `uv sync` during normal installation; you do NOT need `pip install -e .` for Marin.
- To develop Levanter alongside Marin, install the submodule in editable mode:

```
cd marin
uv pip install -e submodules/levanter
```

This ensures changes in `submodules/levanter` are immediately importable in your environment.

## Commit changes to your Levanter branch

Change `LEVANTER_BRANCH` to the branch name in `levanter` you are developing. Now, you can make changes to the `LEVANTER_BRANCH` of levanter by editing files in `ROOT_DIR/marin/submodules/levanter`. After making changes, you can run:
```
cd marin/submodules/levanter
git add .
git commit -m "Update LEVANTER_BRANCH branch with new changes"
```

This commits the changes to the `LEVANTER_BRANCH` branch of Levanter.

Tip: When submitting Ray jobs via `marin/run/ray_run.py`, the runtime automatically includes `submodules/*` and their `src/` in `PYTHONPATH`, which also supports co-development on cluster jobs.
