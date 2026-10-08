# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# The image every task-curation ScriptGrader and sandboxed verifyit grader runs in. Build it with
# `python -m experiments.post_training.task_curation.images`, which also supplies the
# `verifyit` build context (lib/verifyit/src/verifyit); see ../README.md.
# The base is iris-task:latest as of 2026-10-07 (Python 3.12.13, linux/amd64).
FROM ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab

# Compilers for code graders, and the shell tools archived TaskTrove graders and nl2bash tasks call.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential bzip2 gawk jq less tmux vim wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /opt/task-curation/requirements.lock
RUN pip install --no-cache-dir --root-user-action=ignore --require-hashes -r /opt/task-curation/requirements.lock

# NLTK looks under /usr/local/share/nltk_data without an environment variable.
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab wordnet

# Sandboxed verifyit graders run `python3 -c "from verifyit.grade import main"` in this image.
COPY --from=verifyit . /opt/task-curation/runtime/verifyit
RUN find /opt/task-curation/runtime -name __pycache__ -prune -exec rm -rf {} + \
    && echo /opt/task-curation/runtime > "$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/task-curation-runtime.pth"

RUN python3 -c "import nltk, pytest_jsonreport, rdkit, reasoning_gym, verifiable_instructions; \
from sympy.parsing.latex import parse_latex; from verifyit.grade import main; \
nltk.data.find('tokenizers/punkt_tab/english/'); nltk.data.find('corpora/wordnet.zip'); \
assert str(parse_latex(r'\frac{1}{2}')) == '1/2'"
