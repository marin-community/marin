# Task-curation images

Every sandboxed grader in the catalog runs in one image, built from the `grader`
recipe in [`recipes.py`](recipes.py). A declaration names it with
`grader_image=GRADER`, and its converter reads the built image's environment from
`context.grader_environment`. Agent images are separate: a workspace declaration
names the image its agent works in with an `AgentImage` literal.

## The grader recipe

[`grader/Dockerfile`](grader/Dockerfile) starts from `iris-task` pinned by digest
(Python 3.12, linux/amd64) and installs:

- `build-essential` and the shell tools nl2bash checks use (`bzip2`, `gawk`, `jq`,
  `less`, `tmux`, `vim`, `wget`);
- every package in [`grader/requirements.lock`](grader/requirements.lock), with
  hashes required: the math, chemistry, schema, instruction-following and
  Reasoning Gym scorer dependencies, `pytest` with `pytest-json-report` and
  `hypothesis`, `harbor-config`, and `verifiable-instructions` at a pinned commit;
- the NLTK `punkt_tab` and `wordnet` data;
- `verifyit` from this repository, which sandboxed `verifyit` graders import.

[`grader/requirements.in`](grader/requirements.in) lists the direct requirements
grouped by the scorers that need them. Regenerate the lock after changing it:

```bash
uv --directory experiments/post_training/task_curation/images/grader pip compile \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes \
  requirements.in -o requirements.lock
```

Scorer code is not installed in the image. A family vendors upstream scorers under
`datasets/<family>/scorers/`, lists that directory in its declaration's `ships`,
and its converter packages the files into each task.

## Building

```bash
uv run python -m experiments.post_training.task_curation.images.build \
  [--recipe grader] [--registry ghcr.io/marin-community]
```

The build needs `docker buildx` and a `docker login` for the registry. Run it with
the `MARIN_PREFIX` the campaign uses. It writes the artifact there.

An image's identity is the SHA-256 of its recipe: the path, mode and content
digest of every file in the context directory and in each package directory
(skipping `__pycache__`), the digest-pinned base image named by the Dockerfile's
single `FROM` line, and the platform `linux/amd64`. The registry is not part of
the identity. The build:

1. refuses context or package files that git does not track, and executable
   files, because a workspace bundle has neither and would compute another
   identity;
2. pushes `<registry>/task-curation-<name>:<identity[:16]>` for linux/amd64,
   without provenance attestations;
3. resolves the pushed digest with `docker buildx imagetools inspect`;
4. writes the artifact `images/<name>-<identity[:16]>` whose record holds the
   name, identity, tag, digest-pinned image, base image, lock SHA-256, context and
   package file lists, and platform.

A second run with an unchanged recipe finds the artifact and starts no Docker
command. Building a source artifact for a declaration that names a recipe with no
artifact for its current identity raises `MissingImageArtifact` with the build
command to run. The resolved image digest enters every such source's artifact
identity, so a rebuilt image renames those artifacts.

For a local check without pushing:

```bash
docker build --platform linux/amd64 \
  --build-context verifyit=lib/verifyit/src/verifyit \
  -t local/task-curation-grader:test \
  experiments/post_training/task_curation/images/grader
```

Grader images are separate from the Zephyr worker image, which carries the grading
code that starts each grader machine.
