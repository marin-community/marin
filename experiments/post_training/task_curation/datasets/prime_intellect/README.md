# Verified SWE intake

`swe.py` declares the pinned Scale-SWE-Verified and SWE-Lego-Real-Data-Verified
populations. These sources remain in `PENDING_MEASUREMENT_SOURCES` until their
current release has all Atlas admission receipts. Select them by name for intake:

```bash
uv run python -m experiments.post_training.task_curation.driver \
  --source scale-swe-verified --source swe-lego-real-data-verified \
  --mode quick --output-root /private/new-swe-version --run
```

The checked-in license receipts record the dataset-level user decision and
named teacher licenses. QUICK conversion supplies no admission. Follow
[task curation](../../../../../docs/references/task-curation.md) for lexical
decontamination, full-population oracle and no-op trials, SAMPLE and final-pin
qualification.

Each row becomes a native Harbor package. Version `2026.11.16.1` binds the
complete immutable image map through a private pinned auxiliary dataset and
uses the landed `shared-source` protection. Its Dockerfile uses the original
image by digest with the source work directory; public setup runs before the actor.
The golden patch stays under `solution/`, and test material is supplied only
at verification. Source-declared 4 CPU, 4 GiB memory and 10 GiB disk limits
are retained. Absent source phase deadlines remain absent in the package. The
[full-population oracle filter](../../../../../docs/references/task-curation.md#full-population-oracle-filter)
records execution calibration and final per-task deadlines separately; retain
the source's explicit CPU, memory and disk limits.

Scoring preserves the source implementation pinned at
`PrimeIntellect-ai/research-environments@b41d8759070715dc102000e06fba0d73ebce9955`:

- `scale_score.py` retains the Scale-SWE JUnit scorer, with context-managed
  UTF-8 file reads.
- `native_scripts.py` contains the source setup, restoration and evaluation
  functions. Annotation strings keep the packaged guest code importable by
  source images with Python 3.6. The grade entry points use equivalent
  `subprocess.run` options available in those interpreters.
- `parse_pytest_outcomes` is from
  `PrimeIntellect-ai/verifiers@672d020c4dcabf5127a76f51aeb4adaf5b89d3fc`.
- The grade entry points adapt source runtime calls to the native verifier
  process and preserve its patch fallback order and test-result interpretation.

Both upstream projects license their code under Apache-2.0. Their license
notices are retained beside the adapted code.

## Multi-SWE private image inputs

Multi-SWE stages the original image's top-level `/home/*.sh` and `test.patch`
on the trusted host before removing them from the actor. Its converter needs a
frozen private archive from each original image. Resolve every pinned source
row's image tag to a digest before capture. The rows are
[`PrimeIntellect/Multi-SWE-RL-Verified@64b3ebf62750b1373e26106cfc658a1861809d0f`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL-Verified/tree/64b3ebf62750b1373e26106cfc658a1861809d0f),
in `data/train-00000-of-00003.parquet` through `data/train-00002-of-00003.parquet`.
Retain those files' SHA-256 hashes under `raw_files` in the
[`CaptureInput` schema](capture_artifacts.py). Each `ImageBinding` contains the
original references, their pinned digest reference and source `instance_id`
values. Retain registry descriptors beside the index. The stage requires all
2,232 source tasks and one record per distinct image digest.

First generate an exact source-ID/pin receipt with the documented
[stage-0 gate](../../../../../docs/references/task-curation.md#stage-0-binding-license-gate).
Supply its JSON registry with `--license-receipts RECEIPTS.json`, or use the
canonical checked-in registry. `PRIVATE_REGIONAL_ROOT` is a private `s3://` prefix
in the acquisition region; for CW RNO2A, use a new prefix under
`s3://marin-rno2a/`. Run full capture inside a trusted Iris CPU job on that
regional controller. The stage queries its current controller task record and
matching backend advertisement, checks the advertised region against the S3 bucket
and inherits its parent's controller for child sandboxes. Keep large source
inputs and image data in-region; the Mac may hold small indexes and receipts.
Preserve the artifact version when resuming identical inputs.

```bash
uv run python -m experiments.post_training.task_curation.datasets.prime_intellect.capture_artifacts \
  --images FROZEN_IMAGE_INDEX.json --images-sha256 INDEX_SHA256 \
  --cluster cw-rno2a --concurrency 4 --artifact-root PRIVATE_REGIONAL_ROOT \
  --artifact-version YYYY.MM.DD.N --output capture-result.json --run
```

The stage enforces the exact-pin license gate before creating its graph and
again before capture. It uses secret-free SANDBOX machines with network denied,
one CPU, 2 GiB memory and a requested 10 GiB job disk budget. This auxiliary
metadata operation supplies no source execution or storage-enforcement
qualification; original task allocations remain in the task contract.
Each archive transfer must match the source-machine SHA-256, and its receipt
retains member bytes, modes and mtimes. Completed captures resume only when
their image/task binding and stored hashes still match. Failed attempts and
partial archives stay preserved; a failure prevents a complete-source result.
`capture-result.json` references the complete `captures.json` manifest and its
hash. Preserve both with the intake evidence. The private auxiliary input is frozen at
`open-athena/atlas-multiswe-grading-artifacts-20261115-v1@acc1f291ad6ad293c095bd8cde868ab2715717a2`.
The TaskSpec converter binds this input under version2026.11.16.2; current execution qualification remains pending.

Docker Hub resolution must be single-threaded and honor `Retry-After`, with
exponential backoff capped at ten minutes. Anonymous registry `HEAD` version
checks avoid image downloads; retain response digests and rate-limit evidence.
Do not replace unavailable images with another environment.
The documented endpoint is
`https://registry-1.docker.io/v2/REPOSITORY/manifests/TAG`; use Docker's normal
anonymous registry negotiation or an already authorized Docker CLI sign-in.
Keep safe response headers and retry timestamps beside the index, excluding
authorization tokens. See [Docker's version-check procedure](https://docs.docker.com/docker-hub/usage/pulls/#view-pull-rate-and-limit).


### Multi-SWE report parser

`multiswe/data.py` and `multiswe/score.py` retain the source taskset's row restoration,
instance construction, report generation and expected-test checks at
`research-environments@b41d8759070715dc102000e06fba0d73ebce9955`. The source's
Qiskit import shim is retained under `multiswe/_qiskit/`. The fresh private grader uses `multiswe/parser/Dockerfile`, with a digest-pinned
Python3.12 base and `requirements.lock` compiled by the standard environment builder
from the original `multi-swe-bench`1.1.2 source distribution and its SHA-256.
`parser/provenance.json` records the cutoff, platform and lock hash. Export and
execute the unchanged context through the explicit EVAL declarative route; no baked
image publication is required. Actor and parser sandboxes share the validation cap.

`multiswe/host_grade.py` runs on the fresh grading machine after original
`/home` harness execution on the actor's machine. It consumes the raw source row,
cat output and exit code, and the harness stdout/stderr as separate artifacts.
A successful nonempty cat output is passed unchanged to the original report
parser; otherwise it uses the source's stdout/stderr fallback. It preserves
UTF-8 and line endings, and reports missing required transport artifacts as
errors. Harness exit status alone does not replace the source score.

The host parser and its dependency image do not change the actor image or
install Python in it. Parser probes supply no execution qualification; complete
oracle, NoP, final-pin SAMPLE and admission receipts are still required.


The Multi-SWE converter is `multiswe/convert.py`, version2026.11.16.2. It binds
all2,232 rows to the2,231 captured original image digests, preserves source
4CPU/4GiB/10GiB actor limits, and keeps the original setup, prompt and golden
patch. Source-private /home files are removed during public preparation and
restored from the frozen archive only at grading time. The original fix-extraction
script and pipeline run on that actor machine; the original parser runs in a
fresh machine. Original member bytes, modes and mtimes remain in the archive.

The source's largest raw row is28.1MB and largest golden patch14.6MB; its
recipe declares a64MiB decoded-resource budget. Archive delivery uses bounded
commands rather than excluding large patches. Undeclared source deadlines remain
absent in the grader and are recorded separately from calibration attempt budgets.
Conversion, census and static export do not establish execution qualification.


The executable TaskSpec declares the original shell capability explicitly, so the
candidate model receives the maintained harness's shell tool. Native Harbor wrappers
supply that tool through Harbor; the converted ScriptGrader path must declare it.
Oracle and NoP alone do not test the candidate's tool interface.
