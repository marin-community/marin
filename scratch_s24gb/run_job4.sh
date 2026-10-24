#!/bin/bash
set -x
cd scratch_s24gb
uv venv /tmp/tv --python 3.12 -q
uv pip install -q --python /tmp/tv/bin/python torch triton numpy 'nvmath-python[cu13]==1.0.0' nvidia-cusparselt-cu13 \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
/tmp/tv/bin/python -u bench_v3.py 2>&1 | sed 's/^/[torch] /'
