FROM ghcr.io/marin-community/iris-task@sha256:ba67e05c032fd79776fe7eb48819868a6485df377af2201bb3cc23d00c6aa972
USER root
COPY requirements.lock /opt/tool-normalization/requirements.lock
RUN python3 -m pip install --no-cache-dir --require-hashes -r /opt/tool-normalization/requirements.lock
RUN python3 -c 'import importlib.metadata,json,platform; print(json.dumps({"python":platform.python_version(),"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}},sort_keys=True))' > /opt/python-packages.json
USER actor
WORKDIR /workspace
