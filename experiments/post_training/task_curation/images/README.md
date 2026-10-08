# Task-curation environments

A declaration states what its machines need as an `Environment`
([`../environment.py`](../environment.py)): `pypi` pins or a uv `lock`, `apt`
packages, `data` such as `nltk:punkt_tab`, or a digest-pinned `image` used
as-is. [`build.py`](build.py) builds every environment that does not name an
image, and the pipeline places each one:

- an `image` runs in a sandbox of that image;
- `apt` packages outside `WORKER_IMAGE_APT` (the `task` stage of
  `lib/iris/Dockerfile`) run in a sandbox of an image built for the
  environment;
- every other environment runs in the Zephyr worker, in a uv environment built
  from its lock ([`../environment_runtime.py`](../environment_runtime.py)).

Agent environments name their image. Iris workers and training pull images
anonymously, so every agent image is a tag of the public
`ghcr.io/marin-community/iris-task` package; a digest copied there with
`crane copy` keeps its reference's `sha256`.

## The grader packages

`GRADER_PACKAGES` in [`../datasets/environments.py`](../datasets/environments.py)
is the environment every grade script in the catalog imports:

- every package in [`../datasets/grader.lock`](../datasets/grader.lock), with
  hashes required: the math, chemistry, schema, instruction-following and
  Reasoning Gym scorer dependencies, `pytest` with `pytest-json-report` and
  `hypothesis`, the packages the Stack Overflow Python-test tasks import,
  `harbor-config`, and `verifiable-instructions` at a pinned commit;
- the NLTK `punkt_tab` and `wordnet` data.

`COMPILER_GRADER_PACKAGES` adds `build-essential` for the TaskTrove
competitive-programming graders, which compile C++ submissions. The worker image
installs `build-essential`, so both run in the worker.

[`../datasets/grader.in`](../datasets/grader.in) lists the direct requirements
grouped by the scorers that need them. Regenerate the lock after changing it:

```bash
uv --directory experiments/post_training/task_curation/datasets pip compile \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes \
  grader.in -o grader.lock
```

Every built environment also puts `verifyit` from this repository on the
grader's import path, because verifyit graders import it. Scorer code is not
installed. A family vendors upstream scorers under `datasets/<family>/scorers/`,
lists that directory in its declaration's `ships`, and its converter packages
the files into each task.

## Building

```bash
uv run python -m experiments.post_training.task_curation.images \
  (--all | --identity IDENTITY_PREFIX ...) [--repository ghcr.io/marin-community/iris-task]
```

`--all` builds every environment the catalog declares; `--identity` selects
declared environments by a prefix of their identity, as printed by
`MissingEnvironmentArtifact`. Run the build with the `MARIN_PREFIX` the
campaign uses; it writes the artifacts there. A CoreWeave prefix needs the
`CW_KEY_ID` and `CW_KEY_SECRET` pair in the environment.

An environment's identity is the SHA-256 of the `pypi` pins or the lock's bytes,
`apt`, `data`, Python 3.12, the platform `linux/amd64`, the digest-pinned base
image, the path, mode and content digest of every `verifyit` file (skipping
`__pycache__`), and its placement. The repository is not part of the identity.
The build:

1. refuses a `lock` file or `verifyit` file that git does not track, and an
   executable `verifyit` file, because a workspace bundle has neither and would
   compute another identity;
2. compiles `pypi` with `uv pip compile --generate-hashes --python-version 3.12
   --python-platform x86_64-unknown-linux-gnu`, or copies `lock`;
3. for an environment placed in a built image, generates a Dockerfile from the
   pinned `iris-task` base that runs `apt-get install` of `apt`, `uv pip sync
   --system --require-hashes` of the lock and the NLTK downloads, and copies
   `verifyit`; it pushes `<repository>:task-curation-env-<identity[:16]>` for
   linux/amd64 without provenance attestations and resolves the pushed digest
   with `docker buildx imagetools inspect`, which needs `docker buildx` and a
   `docker login` for the repository's registry;
4. writes the artifact `images/env-<identity[:16]>`, which stores the lock as
   `requirements.lock` and records the identity, the lock's SHA-256, `apt`,
   `data`, the Python version, the base image, and any built image and tag.

A second run with an unchanged declaration finds the artifact and starts no
build. Planning a source whose environment has no artifact for its current
identity raises `MissingEnvironmentArtifact` with the build command. The
environment's identity and built image digest enter every such source's
artifact identity.

The default repository is the public `iris-task` package: GHCR creates new
packages org-internal, and Iris workers pull anonymously, so an image in a new
package never starts on the cluster until the package is made public. Built
images are separate from the Zephyr worker image, which carries the grading code
that starts each grader machine.
