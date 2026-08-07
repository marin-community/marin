# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry job for the production fleet extraction of the text route.

A thin shim over :func:`~experiments.datakit.build_pdf_source.extract_fleet.run_fleet`, kept as its own
module on purpose: running a module as ``__main__`` makes cloudpickle serialize its functions by
value, and the ``@cache``-wrapped client/pool helpers in ``extract_fleet`` can only pickle as
references -- so everything a task touches must live in the importable module, not here. Same
split as ``extract_ocr_all`` over ``extract_ocr``.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-extract-fleet \\
        -- python -m experiments.datakit.build_pdf_source.extract_fleet_run
"""

from experiments.datakit.build_pdf_source.extract_fleet import fleet_extract_step, run_fleet

if __name__ == "__main__":
    run_fleet(fleet_extract_step, "Text-route")
