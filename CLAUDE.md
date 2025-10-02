
## Repo reorg
A long-term goal is to use [`uv` workspaces] to consolidate logical components from across the [Marin] and [Levanter] repos into a single monorepo. This would make it easier to coordinate changes that span multiple components, and simplify dependency management.

Below is a rough sequence of steps to get there, with the goal of minimizing disruption along the way. At time of writing, we're at "step 0", and the `../levanter` sibling directory is a [Levanter] clone we can reference.

### Step 1: initialize workspace, fold `marin` and `data_browser` members into `lib/`

```
marin/
  pyproject.toml
  experiments/
  lib/
    marin/
      pyproject.toml
      src/
    data_browser/
      pyproject.toml
      src/
```

### Step 2: ingest [Levanter]

```
marin/
  pyproject.toml
  experiments/
  lib/
    marin/
      pyproject.toml
      src/
    data_browser/
      pyproject.toml
      src/
    levanter/
      pyproject.toml
      src/
```

At this point, we should also restrict CI to only run on relevant changes.

### Step 3: ingest [Haliax]

```
marin/
  pyproject.toml
  experiments/
  lib/
    marin/
      pyproject.toml
      src/
    data_browser/
      pyproject.toml
      src/
    levanter/
      pyproject.toml
      src/
    haliax/
      pyproject.toml
      src/
```

### Step Omega: `ray_tpu`, `rl`, `thalas`, `marin-core`, `marin-crawl`, `experiments` packages

```
marin/
  pyproject.toml
  experiments/
    hero_runs/
      pyproject.toml
      expXXX_tootsie8b.py
    compel/
      pyproject.toml
      expXXX_compel_v0.py
  lib/
    marin-core/
      pyproject.toml
      src/
    data_browser/
      pyproject.toml
      src/
    haliax/
      pyproject.toml
      src/
    levanter/
      pyproject.toml
      src/
    marin-crawl/
      pyproject.toml
      src/
    ray_tpu/
      pyproject.toml
      src/
    rl/
      pyproject.toml
      src/
    thalas/
      pyproject.toml
      src/
```

[Marin]: https://github.com/marin-community/marin
[Levanter]: https://github.com/marin-community/levanter
[Haliax]: https://github.com/marin-community/haliax

[`uv` workspaces]: https://docs.astral.sh/uv/concepts/projects/workspaces/
