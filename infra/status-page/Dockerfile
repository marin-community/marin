# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# --- build stage -------------------------------------------------------------
FROM node:20-slim AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY tsconfig.base.json tsconfig.server.json tsconfig.web.json vite.config.ts postcss.config.cjs ./
COPY server ./server
COPY web ./web

RUN npm run build

# Second npm install, dev deps pruned, for the runtime stage.
RUN npm prune --omit=dev

# --- runtime stage -----------------------------------------------------------
FROM node:20-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=8080

COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/server/dist ./server/dist
COPY --from=build /app/web/dist ./web/dist
COPY --from=build /app/package.json ./package.json

EXPOSE 8080
CMD ["node", "server/dist/main.js"]
