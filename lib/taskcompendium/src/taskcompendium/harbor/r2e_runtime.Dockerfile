# R2E-Gym verifier image: preserve the source image's repository and Python 3.7
# environment while adding the trusted TaskCompendium supervisor runtime.
ARG R2E_SOURCE_IMAGE

FROM python:3.12-slim-bullseye@sha256:411fa4dcfdce7e7a3057c45662beba9dcd4fa36b2e50a2bfcd6c9333e59bf0db AS supervisor
RUN pip install --no-cache-dir \
    msgspec==0.19.0 \
    tomlkit==0.13.3 \
    fsspec==2025.3.0 \
    pyarrow==23.0.1 \
    openai==2.24.0 \
    pytest==8.4.2 \
    pytest-json-report==1.5.0

FROM ${R2E_SOURCE_IMAGE}
COPY --from=supervisor /usr/local /usr/local
COPY --from=supervisor /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1
COPY --from=supervisor /usr/lib/x86_64-linux-gnu/libssl.so.1.1 /usr/lib/x86_64-linux-gnu/libssl.so.1.1
RUN source_python=$(readlink -f /testbed/.venv/bin/python) \
    && source_python_root=$(dirname "$(dirname "$source_python")") \
    && source_python_name=$(basename "$source_python") \
    && cp -a "$source_python_root" /opt/r2e-python \
    && ln -sf "/opt/r2e-python/bin/$source_python_name" /testbed/.venv/bin/python \
    && ln -sf "/opt/r2e-python/bin/$source_python_name" /testbed/.venv/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python
