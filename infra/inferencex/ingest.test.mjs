// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

let fixtureNumber = 0;

async function recoveredPoint(t, conclusion, { interrupted = false, changedHead = false } = {}) {
  const directory = mkdtempSync(join(tmpdir(), 'inferencex-admission-'));
  const artifacts = join(directory, 'artifacts');
  const admitted = join(directory, 'admitted.json');
  const source = {
    kind: 'github-actions', repository: 'marin-community/marin',
    run_id: 1234, run_attempt: 2, head_sha: '4'.repeat(40),
  };
  const suffix = 'agentic_snowball011_h100_vllm_bf16_agentx_c8';
  mkdirSync(join(artifacts, suffix), { recursive: true });
  mkdirSync(join(artifacts, `bmk_${suffix}`));
  const write = (path, value) => writeFileSync(path, JSON.stringify(value));
  write(join(artifacts, 'agentx-source.json'), {
    source, points: [{
      manifest: { profile: 'h100', concurrency: 8, recipe_fingerprint: 'recipe' },
      receipt: { status: 'valid', attempt_uid: 'iris-attempt' },
    }],
  });
  write(join(artifacts, suffix, 'attempt.json'), {
    source, status: interrupted ? 'failed' : 'succeeded',
    iris_attempt_uid: 'iris-attempt', recipe_fingerprint: 'recipe',
  });
  write(join(artifacts, `bmk_${suffix}`, 'agg.json'), { num_gpus: 8, recipe_fingerprint: 'recipe' });
  const oldArguments = process.argv;
  const oldEnvironment = { ...process.env };
  t.after(() => {
    t.mock.restoreAll();
    syncBuiltinESMExports();
    process.argv = oldArguments;
    process.env = oldEnvironment;
    rmSync(directory, { recursive: true, force: true });
  });
  process.argv = [process.execPath, 'ingest.mjs', artifacts];
  process.env.DATABASE_WRITE_URL = 'postgresql://test@localhost/unused';
  delete process.env.DASHBOARD_URL;
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.equal(String(url), 'https://api.github.com/repos/marin-community/marin/actions/runs/1234/attempts/2');
    return Response.json({
      status: 'completed', conclusion, run_attempt: 2,
      head_sha: changedHead ? '5'.repeat(40) : source.head_sha,
    });
  });
  // The upstream CLI is the I/O boundary. Record the identity it would ingest;
  // never write fixture measurements into an actual database.
  t.mock.method(childProcess, 'spawnSync', (_command, _arguments, options) => {
    write(admitted, {
      run_id: options.env.INGEST_RUN_ID,
      attempt: options.env.INGEST_RUN_ATTEMPT,
      repository: options.env.INGEST_REPO,
    });
    return { status: 0 };
  });
  syncBuiltinESMExports();
  const execute = () => import(`./ingest.mjs?fixture=${fixtureNumber++}`);
  if (interrupted || changedHead || conclusion === 'cancelled') {
    await assert.rejects(execute(), Error);
    assert.equal(existsSync(admitted), false);
  } else {
    await execute();
    assert.deepEqual(JSON.parse(readFileSync(admitted, 'utf8')), {
      run_id: '1234', attempt: '2', repository: 'marin-community/marin',
    });
  }
}

for (const conclusion of ['success', 'failure', 'timed_out']) {
  test(`a valid recovered window retains its source identity after Action ${conclusion}`,
    (t) => recoveredPoint(t, conclusion));
}
test('an interrupted native window cannot enter the importer',
  (t) => recoveredPoint(t, 'failure', { interrupted: true }));
test('a cancelled source cannot enter the importer',
  (t) => recoveredPoint(t, 'cancelled'));
test('a different source commit cannot enter the importer',
  (t) => recoveredPoint(t, 'failure', { changedHead: true }));
