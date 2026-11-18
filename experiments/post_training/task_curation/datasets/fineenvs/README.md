# FineEnvs public-input images

The native FineEnvs tasks stage packaged data from `/setup_files/inputs` into
`/home/user/input` with their original `stage_inputs.py` checksum checks. Some
input sets exceed the converter’s inline-file byte budget. The image builder stores
those files in a derived image based on the digest-pinned native image.

Each input set lives at `/opt/fineenvs-inputs/<input-signature>`. The signature
covers every input file's path, size, content hash, mode and modification time.
The builder verifies the native input archive and adds only public input files.
It compares the filesystem before OCI export with the filesystem recovered from
the OCI layout. Base file contents and metadata are preserved, except that
adding inputs can update their parent directories' modification times. Original
image layers and runtime configuration are preserved.

The approved FineEnvs plan uses at most ten grouped images to fit the available
Daytona snapshot budget. Grouping
exposes other tasks' public input sets outside the workdir to the root user.
The image provenance records this deviation. Conversion must bind only the
task's own set at `/setup_files/inputs` before running the unchanged staging
script. Original prompts, graders and resource allocations remain unchanged,
including any allocation that is too small for its inputs.

Run `python -m experiments.post_training.task_curation.datasets.fineenvs.images
--help` for the artifact graph's pinned input arguments. A dry run prints the
artifact paths and fingerprints. Building requires a regional Iris CPU job,
verified tool hashes and passing Daytona CPU, memory, disk, network ALLOW, file
metadata and setup-command probes for both native profiles: 1 CPU / 1024 MiB memory / 5120 MiB disk / network ALLOW, and
2 CPU / 4096 MiB memory / 10240 MiB disk / network ALLOW.

Each artifact contains the OCI layout, complete filesystem inventory and
checksum-bound provenance, including the source's typed restriction tier and
checked license-receipt binding. Publish images to an internal or private registry
package and verify their digests before conversion. After publication, convert the tasks, run original-oracle and no-op trials,
complete SAMPLE review, and obtain Atlas admission.
