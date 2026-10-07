# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

RUN python -m pip install --no-cache-dir pyext==0.7 numpy==1.23.5
RUN mkdir -p /opt/apps/eval && python - <<'PY'
import hashlib
import urllib.request
from pathlib import Path

url = 'https://raw.githubusercontent.com/hendrycks/apps/b45c0ed78517a3a6492eb77b21cffbb79b1096f1/eval/testing_util.py'
body = urllib.request.urlopen(url, timeout=30).read()
assert hashlib.sha256(body).hexdigest() == '9a4e58ff2634ef606c42597457c0733910862e0588ea750d901265bbfe65d36f'
Path('/opt/apps/eval/testing_util.py').write_bytes(body)
PY
