# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry job for the docling backfill of the router's OCR route.

The production fleet run converted the text-extractable documents; this converts the ones the
router sent to OCR, so the union of the two outputs is a docling conversion of the full classified
sample. Same fleet, same senders, same record shape -- only the routing key set and the output
prefix differ. A thin shim over :func:`~experiments.datakit.build_pdf_source.extract_fleet.run_fleet` for
the same cloudpickle-by-reference reason as
:mod:`~experiments.datakit.build_pdf_source.extract_fleet_run`.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-extract-backfill \\
        -- python -m experiments.datakit.build_pdf_source.extract_fleet_backfill_run
"""

from experiments.datakit.build_pdf_source.extract_fleet import fleet_backfill_step, run_fleet

if __name__ == "__main__":
    run_fleet(fleet_backfill_step, "Backfill")
