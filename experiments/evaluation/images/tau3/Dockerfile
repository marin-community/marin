ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NODE_VERSION
ARG NODE_SHA256
RUN python3 -c 'import hashlib, os, tarfile, urllib.request; from io import BytesIO; version = os.environ["NODE_VERSION"]; data = urllib.request.urlopen(f"https://nodejs.org/dist/v{version}/node-v{version}-linux-x64.tar.gz").read(); assert hashlib.sha256(data).hexdigest() == os.environ["NODE_SHA256"]; archive = tarfile.open(fileobj=BytesIO(data)); archive.extractall("/opt/node", filter="data")' \
    && ln -s /opt/node/node-v${NODE_VERSION}-linux-x64/bin/node /usr/local/bin/node \
    && ln -s /opt/node/node-v${NODE_VERSION}-linux-x64/bin/npm /usr/local/bin/npm \
    && ln -s /opt/node/node-v${NODE_VERSION}-linux-x64/bin/npx /usr/local/bin/npx

COPY package.json package-lock.json /opt/harbor-pi/
RUN cd /opt/harbor-pi && npm ci --omit=dev --no-audit --no-fund \
    && ln -s /opt/harbor-pi/node_modules/.bin/pi /usr/local/bin/pi \
    && ln -s /opt/harbor-pi/node_modules/.bin/pi-acp /usr/local/bin/pi-acp \
    && ln -s /opt/harbor-pi/node_modules/.bin/acpx /usr/local/bin/acpx \
    && npm cache clean --force

COPY mcp.lock acp.lock /opt/harbor-deps/
RUN python3 -m pip install --no-cache-dir --require-hashes -r /opt/harbor-deps/mcp.lock \
    && python3 -m venv /opt/harbor-acp-venv \
    && /opt/harbor-acp-venv/bin/pip install --no-cache-dir --require-hashes -r /opt/harbor-deps/acp.lock \
    && python3 -c 'import mcp' \
    && /opt/harbor-acp-venv/bin/python -c 'import acp' \
    && node --version && pi --version && acpx --version

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
