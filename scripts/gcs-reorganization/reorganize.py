"""
reorganize.py

Script for reorganizing the `marin-data` GCS bucket (transferring files to `marin-us-central2`) via the Google Cloud
Storage Transfer Service API.

Mappings (before / after) are defined as a top-level `*MAPPINGS` dictionary, which auto-generates the Storage Transfer
configuration, then programmatically creates & launches each job.

Reference :: https://cloud.google.com/storage-transfer/docs/manifest
"""

import json

from google.cloud import storage_transfer

# Defines each Transfer Job as a mapping :: `job_name` -> (`src_bucket_root`, `destination_bucket_root`)
# fmt: off
RAW_TRANSFER_JOB_MAPPINGS = {
    # === External Semi-Formatted (dolma, datacomp, fineweb-edu) ===
    # TODO (@percyliang) :: Do we want to move the "external" data somewhere special vs. just dumping under "raw"?
    "raw-dolma (v1.7)": (
        {"bucket_name": "marin-data", "path": "raw/dolma/dolma-v1.7/"},

        # TODO (@percyliang) :: Dropping `dolma` from version?
        {"bucket_name": "marin-us-central2", "path": "raw/dolma/v1.7/"},
    ),

    "raw-dclm (v2024-07-09-baseline-dedup)": (
        {"bucket_name": "marin-data", "path": "datacomp/dclm-baseline-dedup-07-09/"},
        {"bucket_name": "marin-us-central2", "path": "raw/dclm/v2024-07-09-baseline-dedup/"},
    ),

    # TODO (@percyliang) :: Should this be treated as "external" or is this a raw source we're doing stuff with?
    "raw-fineweb-edu (v1 :: #5b89d1e on HF)": (
        # Note :: This was created via a Transfer Job from HF Datasets --> GCS (resulting in path artifacts). Clean?
        {
            "bucket_name": "marin-data",
            "path": (
                "raw/fineweb-edu/huggingface.co/datasets/HuggingFaceFW/"
                "resolve/5b89d1ea9319fe101b3cbdacd89a903aca1d6052/data/"
            ),
        },

        # TODO (@percyliang) :: Added HF Datasets short commit hash to canonical version?
        {"bucket_name": "marin-us-central2", "path": "raw/fineweb-edu/v1-5b89d1e/"},
    ),
    # === Raw Data ===
    "raw-algebraic-stack (v2023-10-13)": (
        # Note :: Version `v2023-10-13` is from GDrive (Marin / Curation / Marin - Math Datasets)
        #   =>> Subdirectories are split as `< train / validation / test > / <lang>{XXXX}.jsonl.zst`
        {"bucket_name": "marin-data", "path": "raw/algebraic-stack/"},
        {"bucket_name": "marin-us-central2", "path": "raw/algebraic-stack/v2023-10-13/"},
    ),

    # ar5iv is from: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/
    #   - Note version is just `04.2024`
    "raw-ar5iv (v04.2024)": (
        {"bucket_name": "marin-data", "path": "raw/arxiv/data.fau.de/"},
        {"bucket_name": "marin-us-central2", "path": "raw/ar5iv/v04.2024/"}
    ),

    "raw-fineweb (v1.0 - #???)": (
        # TODO (@percyliang) :: Think this was also a HF Datasets transfer... should add commit hash?
        {"bucket_name": "marin-data", "path": "raw/fineweb/fw-v1.0/"},
        # {"bucket_name": "marin-us-central2", "path": "raw/fineweb/v1.0-<COMMIT-HASH>/"}   # TODO
    ),

    "raw-falcon-refinedweb (v1.0 - #c735840)": (
        {
            "bucket_name": "marin-data",
            "path": (
                "raw/huggingface.co/datasets/tiiuae/falcon-refinedweb/"
                "resolve/c735840575b629292b41da8dde11dcd523d4f91c/data/"
            ),
        },
        {"bucket_name": "marin-us-central2", "path": "raw/falcon-refinedweb/v1.0-c735840/"},
    ),

    # TODO (@percyliang) :: Adding `legal` prefix because not familiar with individual corpora tags... ok?
    "raw-legal-edgar (v1.0 - #f7d3ba7)": (
        {
            "bucket_name": "marin-data",
            "path": (
                "raw/huggingface.co/datasets/eloukas/edgar-corpus/resolve/f7d3ba73d65ff10194a95b84c75eb484d60b0ede/"
            ),
        },
        {"bucket_name": "marin-us-central2", "path": "raw/legal-edgar/v1.0-f7d3ba7/"},
    ),

    "raw-legal-hupd (v1.0 - f570a84)": (
        {"bucket_name": "marin-data", "path": "raw/huggingface.co/datasets/HUPD/hupd/resolve/main/data"},
        {"bucket_name": "marin-us-central2", "path": "raw/legal-hupd/v1.0-f570a84/"},
    ),

    "raw-legal-multi-legal-wikipedia-filtered (v1.0 - #483f6c8)": {
        {
            "bucket_name": "marin-data",
            "path": "raw/huggingface.co/datasets/joelniklaus/MultiLegalPileWikipediaFiltered/resolve/main/data/",
        },
        {"bucket_name": "marin-us-central2", "path": "legal-multi-legal-wikipedia-filtered/v1.0-483f6c8/"},
    },

    "raw-legal-open-australian-legal-corpus": (
        {
            "bucket_name": "marin-data",
            "path": (
                "raw/huggingface.co/datasets/umarbutler/open-australian-legal-corpus/"
                "resolve/66e7085ff50b8d71d3089efbf60e02ef5b53cf46/"
            ),
        },
        {"bucket_name": "marin-us-central2", "path": "raw/legal-open-australian-legal-corpus/v1.0-66e7085/"},
    ),

    # TODO (@percyliang) :: Similarly, adding `instruct` prefix?
    "raw-instruct-tulu-sft (v2.0 - #6248b17)": {
        {
            "bucket_name": "marin-data",
            "path": (
                "raw/instruct/huggingface.co/datasets/allenai/tulu-v2-sft-mixture/"
                "resolve/6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752/data/"
            ),
        },
        {"bucket_name": "marin-us-central2", "path": "raw/instruct-tulu-sft/v2.0-6248b17/"},
    },

    # TODO (@percyliang) :: `pubmed` has subdirectories (`europe_pmc`, `pubmed_abstracts`, `pubmed_central`). Should
    #                       these be handled as separate top-level raw datasets?
    #                       Separately --> Missing version / provenance?
    # "raw-pubmed (???)": (
    #     {"bucket_name": "marin-data", "path": "raw/pubmed/"},
    #     {"bucket_name": "marin-us-central2", "path": "???"}
    # ),

    "raw-slim-pajama (v1.0 - #2d0accd)": (
        {"bucket_name": "marin-data", "path": "raw/slim-pajama/2d0accdd/SlimPajama-627B/"},
        {"bucket_name": "marin-us-central2", "path": "raw/slim-pajama/v1.0-2d0accd/"},
    ),

    "raw-stackexchange (v2024-04-02)": (
        {"bucket_name": "marin-data", "path": "raw/stackexchange/archive.org/download/stackexchange/"},
        {"bucket_name": "marin-us-central2", "path": "raw/stackexchange/v2024-04-02/"},
    ),

    # TODO (@percyliang) :: Missing Wikipedia version / provenance
    # "raw-wikipedia (???)": (
    #     {"bucket_name": "marin-data", "path": "raw/wikipedia/"},
    #     {"bucket_name": "marin-us-central2", "path": "???"}
    # )
}
# fmt: on

# === Set `TRANSFER_JOB_MAPPINGS` ===
TRANSFER_JOB_MAPPINGS = RAW_TRANSFER_JOB_MAPPINGS
TRANSFER_JOB_PREFIX = "raw"


def reorganize() -> None:
    print("[*] Reorganizing GCS Bucket `gs://marin-data` -> `gs://marin-us-central2`")
    client = storage_transfer.StorageTransferServiceClient()

    # Iterate though Transfer Job Mappings =>> fire off transfers!
    transfer_job_operations = {}
    for transfer_job_name, (gs_src, gs_sink) in TRANSFER_JOB_MAPPINGS.items():
        assert gs_src["bucket_name"] == "marin-data", f"Unexpected `src` bucket name: {gs_src['bucket_name']}"
        assert gs_sink["bucket_name"] == "marin-us-central2", f"Unexpected `sink` bucket name: {gs_sink['bucket_name']}"

        request = storage_transfer.CreateTransferJobRequest(
            {
                "transfer_job": {
                    "project_id": "hai-gcp-models",
                    "description": f"Reorganization Transfer Job for {transfer_job_name}",
                    "status": storage_transfer.TransferJob.Status.ENABLED,
                    "transfer_spec": {"gcs_data_source": gs_src, "gcs_data_sink": gs_sink},
                }
            }
        )

        # Create Job and Run
        #   => `creation_request` =>> Specifies Job, creates new "entry" in `console.cloud.google.com/jobs/transferJobs`
        #   => `run_response` =>> Actually invokes the transfer operation and tracks status
        creation_request = client.create_transfer_job(request)
        run_response = client.run_transfer_job({"job_name": creation_request.name, "project_id": "hai-gcp-models"})

        # Add Job Metadata to Trackers
        #   => Check https://console.cloud.google.com/transfer/jobs/<transfer_job_id>
        transfer_job_operations[transfer_job_name] = {
            "transfer_job_id": creation_request.name,
            "run_operation": run_response,
        }

    # Wait until all `run_operations` are done!
    try:
        still_running = True
        while still_running:
            running_jobs = [job for job in transfer_job_operations if job["run_operation"].running()]
            still_running = len(running_jobs) > 0

            print(f"[*] {len(running_jobs)} / {len(transfer_job_operations)} Transfer Jobs are still in progress!")

    finally:
        # If Ctrl-C or Interrupt --> serialize `transfer_job_operations` to Disk
        print("[*] Interrupt --> Serializing `transfer_job_operations` to `scripts/gcs-reorganization/transfer.json`")
        with open(f"scripts/gcs-reorganization/{TRANSFER_JOB_PREFIX}-transfer.json", "w") as f:
            json.dump(
                {
                    k: {"transfer_job_id": v["transfer_job_id"], "run_operation_id": v["run_operation"].operation.name}
                    for k, v in transfer_job_operations.items()
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    reorganize()
