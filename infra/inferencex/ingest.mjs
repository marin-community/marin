// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Validate the collector's original source identity, then use the upstream importer.
import { readFileSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { isDeepStrictEqual } from 'node:util';

const artifacts = resolve(process.argv[2] ?? '/artifacts');
const read = (path) => JSON.parse(readFileSync(path, 'utf8'));
const bundle = read(join(artifacts, 'agentx-source.json'));
const source = bundle.source;
if (
  source.kind !== 'github-actions' ||
  source.repository !== 'marin-community/marin' ||
  !Number.isSafeInteger(source.run_id) ||
  source.run_id <= 0 ||
  !Number.isSafeInteger(source.run_attempt) ||
  source.run_attempt <= 0
) {
  throw new Error('Import requires an actual Marin GitHub source run and attempt');
}
const response = await fetch(
  `https://api.github.com/repos/${source.repository}/actions/runs/${source.run_id}/attempts/${source.run_attempt}`,
  {
    headers: {
      Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
      Accept: 'application/vnd.github+json',
    },
    signal: AbortSignal.timeout(60_000),
  },
);
if (!response.ok) throw new Error(`Cannot verify GitHub source: HTTP ${response.status}`);
const actual = await response.json();
if (
  actual.head_sha !== source.head_sha ||
  actual.run_attempt !== source.run_attempt ||
  actual.status !== 'completed' ||
  !['success', 'failure', 'timed_out'].includes(actual.conclusion)
) {
  throw new Error('Source attempt is not the completed, uncancelled run saved by submission');
}
const expected = [];
const allowed = ['agentx-source.json'];
for (const point of bundle.points) {
  if (point.receipt?.status !== 'valid') continue;
  const { profile, concurrency } = point.manifest;
  if (!['h100', 'gb200'].includes(profile) || ![8, 32].includes(concurrency)) {
    throw new Error('Point is outside the fixed comparison grid');
  }
  const suffix = `agentic_snowball011_${profile}_vllm_bf16_agentx_c${concurrency}`;
  const aggregateName = `bmk_${suffix}`;
  const aggregate = read(join(artifacts, aggregateName, 'agg.json'));
  const attempt = read(join(artifacts, suffix, 'attempt.json'));
  if (
    aggregate.num_gpus !== (profile === 'h100' ? 8 : 4) ||
    attempt.status !== 'succeeded' ||
    attempt.iris_attempt_uid !== point.receipt.attempt_uid ||
    !isDeepStrictEqual(attempt.source, source) ||
    attempt.recipe_fingerprint !== point.manifest.recipe_fingerprint ||
    aggregate.recipe_fingerprint !== point.manifest.recipe_fingerprint
  ) {
    throw new Error(
      'Aggregate, physical GPU count or Iris attempt provenance differs from its receipt',
    );
  }
  expected.push(aggregateName);
  allowed.push(aggregateName, suffix);
}
if (readdirSync(artifacts).some((name) => !allowed.includes(name))) {
  throw new Error('Bundle has an unexpected entry; source-override metadata is not accepted');
}
const present = readdirSync(artifacts)
  .filter((name) => name.startsWith('bmk_'))
  .sort();
if (JSON.stringify(present) !== JSON.stringify(expected.sort())) {
  throw new Error('Bundle contains an unexpected or missing benchmark aggregate');
}
if (!expected.length) {
  console.log('No validated points to import');
  process.exit(0);
}
const environment = {
  ...process.env,
  INGEST_RUN_ID: String(source.run_id),
  INGEST_RUN_ATTEMPT: String(source.run_attempt),
  INGEST_REPO: source.repository,
  INGEST_ARTIFACTS_PATH: artifacts,
  INFERENCEX_SOURCE_REPOS: source.repository,
};
if (!environment.DATABASE_WRITE_URL) {
  for (const key of [
    'DATABASE_USER',
    'DATABASE_PASSWORD',
    'DATABASE_NAME',
    'DATABASE_SOCKET_DIR',
  ]) {
    if (!environment[key]) throw new Error(`${key} is required`);
  }
  const url = new URL('postgresql://localhost/');
  url.username = environment.DATABASE_USER;
  url.password = environment.DATABASE_PASSWORD;
  url.pathname = `/${environment.DATABASE_NAME}`;
  environment.DATABASE_WRITE_URL = url.href;
}
delete environment.DATABASE_PASSWORD;
const child = spawnSync('bun', ['packages/db/src/ingest-ci-run.ts'], {
  cwd: '/source',
  env: environment,
  stdio: 'inherit',
  timeout: 900_000,
});
if (child.error) throw child.error;
if (child.status !== 0) process.exit(child.status ?? 1);
if (process.env.DASHBOARD_URL) {
  if (!process.env.INVALIDATE_SECRET)
    throw new Error('INVALIDATE_SECRET is required for cache refresh');
  const refreshed = await fetch(new URL('/api/v1/invalidate', process.env.DASHBOARD_URL), {
    method: 'POST',
    signal: AbortSignal.timeout(120_000),
    headers: {
      Authorization: `Bearer ${process.env.INVALIDATE_SECRET}`,
      ...(process.env.IAP_ID_TOKEN
        ? { 'Proxy-Authorization': `Bearer ${process.env.IAP_ID_TOKEN}` }
        : {}),
    },
  });
  if (!refreshed.ok)
    throw new Error(`Cache refresh failed: HTTP ${refreshed.status}; safely retry import`);
  console.log(`Imported source run ${source.run_id}/${source.run_attempt}; cache refreshed`);
}
