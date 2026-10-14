# Task-curation environments

A declaration states what its machines need as an `Environment`
([`../environment.py`](../environment.py)): pinned Python `requirements`, `apt`
packages, `files` placed at absolute paths from pinned archives, `env`
variables, a checked-in `dockerfile` fragment that installs a toolchain, or a
digest-pinned `image` used as-is. [`build.py`](build.py) resolves every environment that does not name an
image, and the pipeline places each one:

- an `image` runs in a sandbox of that image;
- a `dockerfile` fragment, or `apt` packages outside `WORKER_IMAGE_APT` (the
  `task` stage of `lib/iris/Dockerfile`), runs in a sandbox of an image built
  for the environment;
- every other environment runs in the Zephyr worker, in a uv environment built
  from its lock ([`../environment_runtime.py`](../environment_runtime.py)).

Agent environments name their image. Iris workers and training pull images
anonymously, so every agent image is a tag of the public
`ghcr.io/marin-community/iris-task` package; a digest copied there with
`crane copy` keeps its reference's `sha256`.

## The grader packages

`GRADER_PACKAGES` in [`../datasets/environments.py`](../datasets/environments.py)
is the environment every grade script in the catalog imports:

- its `requirements`, grouped by the scorers that need them: the scientific
  stack, the math, schema, instruction-following and Reasoning Gym scorer
  dependencies, `pytest` with `pytest-json-report` and `hypothesis`, packages
  used by Stack Overflow Python-test tasks, and `verifiable-instructions` at a
  pinned commit;
- the NLTK `punkt_tab` English tokenizer, placed at
  `/srv/nltk_data/tokenizers/punkt_tab/english` from a pinned `nltk_data`
  archive, with `NLTK_DATA=/srv/nltk_data` in `env`. NLTK's default directory,
  `/usr/local/share/nltk_data`, is read-only inside the worker sandbox.

`COMPILER_GRADER_PACKAGES` adds `build-essential` for the TaskTrove
competitive-programming graders, which compile C++ submissions. The worker image
installs `build-essential`, so both run in the worker.

`LEAN_GRADER_PACKAGES` adds the fragment
[`../datasets/nemotron_ultra/lean.Dockerfile`](../datasets/nemotron_ultra/lean.Dockerfile),
which installs elan under `/opt/elan`, the pinned Lean 4 toolchain and a Mathlib
project at `/lean4/my_project` with its prebuilt oleans, laid out like the NeMo
Skills sandbox. It runs in a built image; the Nemotron Ultra Lean grader
compiles proofs there through
[`../datasets/nemotron_ultra/scorers/local_lean_sandbox.py`](../datasets/nemotron_ultra/scorers/local_lean_sandbox.py).

A `dockerfile` fragment is self-contained: it downloads what it installs inside
its own `RUN` steps, so the build context carries only the lock, the placed
files and the generated Dockerfile. Its lines follow the apt and files steps and
precede the lock sync, so toolchain layers stay cached while the lock changes. A fragment may use only `ARG`, `ENV`
and `RUN` (`FRAGMENT_INSTRUCTIONS` in [`../environment.py`](../environment.py));
the declaration refuses `FROM`, `USER`, `WORKDIR`, `SHELL`, `COPY`, `ADD`,
`ENTRYPOINT`, `CMD` and the rest, because they would replace the base, break the
generated package steps, or change the grader container. `ENV` persists into the
image on purpose: it is how a fragment puts its toolchain on `PATH`.

Each requirement is an exact `name==version` pin or a `name @ https://...`
URL that names a commit; the build compiles them to a hash lock, so there is no
checked-in lock to regenerate. Each `files` entry names an https archive (zip or
tar) by URL and SHA-256 and the member, a file or directory inside it, to place
at the absolute path. The path is the same in a built image and in the worker,
where bubblewrap binds the file read-only, so it cannot lie under the host system
directories (`/usr`, `/etc`, `/opt` and the rest). `env` sets variables for the
grader in both placements.

Every built environment also puts `verifyit` from this repository on the
grader's import path, because verifyit graders import it. Scorer code is not
installed. A family vendors upstream scorers under `datasets/<family>/scorers/`,
and its converter packages the files into each task. Bump the recipe version
when converter or bundled scorer code changes.

## Building

```bash
uv run python -m experiments.post_training.task_curation.images \
  (--all | --identity IDENTITY_PREFIX ...) [--repository ghcr.io/marin-community/iris-task]
```

Register a new buildable grader environment in `BUILDABLE_ENVIRONMENTS` in
[`datasets/environments.py`](../datasets/environments.py), and use that same
declaration in the dataset's config. Prebuilt images need no registration.

`--all` builds every registered environment; `--identity` selects
registered environments by a prefix of their identity, as printed by
`MissingEnvironmentArtifact`. Run the build with the `MARIN_PREFIX` the
campaign uses; it writes the artifacts there. A CoreWeave prefix needs the
`CW_KEY_ID` and `CW_KEY_SECRET` pair in the environment.

An environment's identity is the SHA-256 of its `requirements`, `apt`, `files`
(each path with its archive's URL, SHA-256 and member), `env`, the `dockerfile`
fragment's bytes, the `uv` upload-date cutoff `EXCLUDE_NEWER`, Python 3.12, the
platform `linux/amd64`, the digest-pinned base image, the path, mode and
content digest of every `verifyit` file (skipping `__pycache__`), and its
placement. The repository is not part of the
identity. The build:

1. refuses a `dockerfile` or `verifyit` file that git does not track, and an
   executable `verifyit` file, because a workspace bundle has neither and would
   compute another identity;
2. compiles `requirements` with `uv pip compile --generate-hashes
   --exclude-newer <EXCLUDE_NEWER> --python-version 3.12 --python-platform
   x86_64-unknown-linux-gnu`, so transitive packages resolve to the newest
   release before the cutoff and two builds of one identity produce one lock,
   and downloads each `files` archive, verifies its SHA-256 and extracts the member
   under `files/<path>` in the build context;
3. for an environment placed in a built image, generates a Dockerfile from the
   pinned `iris-task` base that runs `apt-get install` of `apt`, `COPY files/ /`,
   one `ENV` line per `env` entry, the `dockerfile` fragment's lines and `uv pip
   sync --system --require-hashes`
   of the lock, and copies `verifyit`; it pushes
   `<repository>:task-curation-env-<identity[:16]>` for linux/amd64 without
   provenance attestations and resolves the pushed digest with `docker buildx
   imagetools inspect`, which needs `docker buildx` and a `docker login` for the
   repository's registry;
4. writes the artifact `images/env-<identity[:16]>`, which stores the lock as
   `requirements.lock`, `files.tar` (the placed tree, paths relative to `/`)
   when the environment places files, and `environment.json` (the sorted placed
   paths, the SHA-256 of `files.tar` and the `env`) when it declares either, and
   records the identity and any built image digest. A worker builds its environment from the lock URL
   alone, reading the manifest and files beside it.

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
