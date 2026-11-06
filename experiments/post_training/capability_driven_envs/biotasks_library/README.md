# BioTasks verifiable-task source library

This campaign starts from the source-project inventory in
[BioTasks issue #5](https://github.com/Open-Athena/biotasks/issues/5) and the
document inventory in [issue #16](https://github.com/Open-Athena/biotasks/issues/16).
The initial cohort is deliberately limited to the 57 counted, content-confirmed
documents with individual source review at BioTasks revision
`a342f41d9d0674cbe280a33a2f537285547552d3`.

`import_reviewed.py` downloads those exact source bytes, verifies every recorded
SHA-256, and stores each unique document once under the private content-addressed
prefix:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/objects/sha256/...
```

It then writes an immutable campaign catalog and publishes `manifest.json` last:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/reviewed57-a342f41d-v1/
    source-catalog.jsonl
    manifest.json
```

Run from the Marin repository root after loading the CoreWeave object-storage
credentials into `CW_KEY_ID` and `CW_KEY_SECRET`:

```bash
uv run python \
  experiments/post_training/capability_driven_envs/biotasks_library/import_reviewed.py
```

The importer executes no notebook code. Its output remains
`internal_unreviewed_do_not_publish`: individual source review in BioTasks is
useful selection evidence, not a redistribution-rights audit, environment audit,
or finding that a document can support a verifiable task. The catalog is an
input to later checklist, curator-packet, proposal, and Taskforge campaigns.
Source project identity must remain the split boundary when tasks are derived.

## Notebook objective-task campaign

`run_notebooks_v1_zephyr.py` processes the complete pinned cohort with one outer
Zephyr shard per source document. It strips Jupyter outputs, asks the author model
for zero to four concrete objective ideas without imposing a quota, runs each
hydrated idea through Taskforge, and obtains a separate blinded usefulness review.
`finalize_notebooks_v1.py` materializes compact campaign views and screening bands.

The complete recovered snapshot is stored at:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/notebooks/v1/snapshots/complete-1/
```

It contains 57 source dispositions, 87 non-null ideas from 49 productive sources,
and eight explicit null-source records. Taskforge structurally accepted all 87
non-null TaskSpecs; this only establishes buildability and valid reference/negative
controls. The blinded GLM screen recommended 40 accept, 17 revise, and 27 reject,
with three review calls retained as errors. These screening bands are not scientific
review, rights approval, solver calibration, or release approval. All artifacts
remain `internal_unreviewed_do_not_publish`.

## Repository objective-task campaign

`import_repositories.py` materializes bounded, hash-verified evidence for the 405
issue-#5 repositories that had detected notebook or literate-document formats. It
executes no repository code and caps each source at its README plus three documents.
`run_repositories_v1_zephyr.py` authors zero to several concrete ideas per repository
and obtains author/self and blinded GLM reviews before Taskforge lowering.

The recovered immutable snapshot is stored at:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/repositories/v1/snapshots/complete-2/
```

It contains 405 repository dispositions: 255 productive repositories, 144 explicit
null repositories, and six repositories whose proposed ideas all failed deterministic
hydration. The snapshot retains 398 hydrated ideas. The blinded GLM screen placed 265
in accept, 89 in revise, 15 in reject, and 29 in review-error. These are pre-Taskforge
screening bands, not structural acceptance or domain validation.

## Combined Taskforge campaign

`run_combined_v1_taskforge.py` combines only frozen blinded-GLM accepts from the
notebook and repository snapshots, preserving source-campaign and screening-record
identity. It excludes revise, reject, null, and review-error records rather than
silently treating them as failures. `finalize_combined_v1.py` freezes the complete
TaskSpecs and manifest at:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/combined/v1/snapshots/complete-1/
```

The combined campaign contains 305 unique ideas (40 notebook and 265 repository)
and 305 unique, structurally accepted TaskSpecs. Structural acceptance establishes
buildability and deterministic control validity only; it is not scientific review,
rights approval, solver calibration, training approval, or release approval. The
snapshot remains `internal_unreviewed_do_not_publish`.

## GLM OpenAI-compatible solver calibration

`calibrate_glm_v1.py` runs frozen TaskSpecs through RolloutEngine's generic
OpenAI-compatible chat transport and the ShellSim shell/filesystem harness. The
runner is deliberately provider-neutral: a later OpenRouter panel changes only
the endpoint, API key, and model identifier. The frozen v2 calibration is at:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/combined/v1/calibration/
    glm-5.3-openai-shellsim-v2/snapshots/complete-1/
```

GLM-5.3 solved 182 of 305 tasks (59.7%). Of 291 trials that reached the grader,
182 were correct, 49 returned valid but incorrect JSON, and 60 violated the
required bare-JSON submission format. Fourteen trials ended before a grade: eight
hit the per-turn generation limit, five exhausted the eight-turn interaction
budget, and one exhausted ShellSim CPU after choosing an exponential algorithm.
These are ordinary single-trial model results, not evidence that the fourteen
TaskSpecs are structurally invalid. ShellSim results must remain separate from
future gVisor or other execution-backend populations.
