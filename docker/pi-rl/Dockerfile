# Development image only. Keep task data, model credentials, and endpoint URLs out
# of the image; Harbor supplies them per trial after sandbox creation.
FROM docker.io/library/node:22.19.0-trixie-slim@sha256:a174cf99ada4707fb5a81a9233af04db83911c68b6bf58199f956fbdf8389926 AS node

FROM docker.io/library/python@sha256:e529028263dbe6910a2d96f7d2b8f5266385e917fd45d286ef166977c094a51e

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /opt/pi-rl
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund \
    && mkdir -p /usr/local/lib/node_modules/@earendil-works \
        /opt/pi-rl/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works \
    && ln -s /opt/pi-rl/node_modules/@earendil-works/pi-ai \
        /opt/pi-rl/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai \
    && ln -s /opt/pi-rl/node_modules/@earendil-works/pi-coding-agent \
        /usr/local/lib/node_modules/@earendil-works/pi-coding-agent \
    && ln -s /opt/pi-rl/node_modules/.bin/pi /usr/local/bin/pi \
    && node --version && pi --version

WORKDIR /tmp
