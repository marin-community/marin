FROM oven/bun:1.4.0-slim@sha256:e0ee68d16ccb9927bf02aa7dd8fd4bf3369ee6d46da04faa72b05ce8bfd135f6 AS bun
FROM node:24.21.0-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20 AS source
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
WORKDIR /source
COPY settings.json /package/settings.json
RUN git init && git remote add origin https://github.com/SemiAnalysisAI/InferenceX-app.git && \
    git fetch --depth=1 origin "$(node -p 'require("/package/settings.json").app_source')" && \
    git checkout --detach FETCH_HEAD
COPY app.patch /package/app.patch
RUN git apply /package/app.patch
ENV CI=1 CYPRESS_INSTALL_BINARY=0 NEXT_TELEMETRY_DISABLED=1
RUN bun install --frozen-lockfile

FROM source AS ingest
COPY ingest.mjs ingest.test.mjs /package/
RUN node --test /package/ingest.test.mjs
ENTRYPOINT ["bun", "/package/ingest.mjs"]

FROM source AS build
COPY build.mjs /package/build.mjs
RUN node /package/build.mjs

FROM node:24.21.0-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20 AS runtime
WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=8080 NEXT_TELEMETRY_DISABLED=1 DATABASE_DRIVER=postgres DATABASE_SSL=false
COPY --from=build --chown=node:node /source/packages/app/.next/standalone ./
COPY --from=build --chown=node:node /source/packages/app/.next/static ./packages/app/.next/static
COPY --from=build --chown=node:node /source/packages/app/public ./packages/app/public
COPY --from=build /source/LICENSE ./LICENSE
COPY entrypoint.mjs ./entrypoint.mjs
USER node
EXPOSE 8080
CMD ["node", "entrypoint.mjs"]
