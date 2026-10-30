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

async function recoveredPoint(t, conclusion, {
  interrupted = false, changedHead = false, profile = 'h100', concurrency = 8,
  wrongGpuCount = false, replacedHost = false, shortWindow = false,
} = {}) {
  const directory = mkdtempSync(join(tmpdir(), 'inferencex-admission-'));
  const artifacts = join(directory, 'artifacts');
  const admitted = join(directory, 'admitted.json');
  const source = {
    kind: 'github-actions', repository: 'marin-community/marin',
    run_id: 1234, run_attempt: 2, head_sha: '4'.repeat(40),
  };
  const hero = profile === 'gb200';
  const prefix = hero ? 'hero144k' : 'snowball011';
  const suffix = `agentic_${prefix}_${profile}_vllm_bf16_agentx_c${concurrency}`;
  mkdirSync(join(artifacts, suffix), { recursive: true });
  mkdirSync(join(artifacts, `bmk_${suffix}`));
  const write = (path, value) => writeFileSync(path, JSON.stringify(value));
  write(join(artifacts, 'agentx-source.json'), {
    source, points: [{
      manifest: { profile, concurrency, recipe_fingerprint: 'recipe', placement: { nodes: ['n0','n1','n2','n3'] } },
      receipt: { status: 'valid', attempt_uid: 'iris-attempt', validation: {
        coverage: { expected_duration_seconds: shortWindow ? 60 : 3600 },
        profiling_requests: 1000, profiling_error_ratio: 0,
      } },
    }],
  });
  write(join(artifacts, suffix, 'attempt.json'), {
    source, status: interrupted ? 'failed' : 'succeeded',
    iris_attempt_uid: 'iris-attempt', recipe_fingerprint: 'recipe',
  });
  write(join(artifacts, `bmk_${suffix}`, 'agg.json'), {
    num_gpus: wrongGpuCount ? 4 : hero ? 16 : 8, recipe_fingerprint: 'recipe',
    conc: concurrency, hw: profile, infmax_model_prefix: prefix, disagg: false, is_multinode: hero,
    model: hero ? 'marin-community/Hero-535B-step144000' : 'open-athena/Marin-Snowball-67B-A2B-v0.1.1',
    dataset: { num_dataset_entries: 393 },
  });
  if (hero) {
    const tasks = Array.from({ length:4 }, (_,i) => ({
      source, status:'succeeded', iris_task_id:`job/${i}:0`, iris_attempt_uid:`attempt${i}`,
      iris_node_name:`n${i}`, recipe_fingerprint:'recipe',
      gpu_uuids:Array.from({ length:4 },(_,gpu)=>`GPU-${i}-${gpu}`),
    }));
    const attempts = Object.fromEntries(tasks.map((task,i)=>[`job/${i}`,task.iris_attempt_uid]));
    if (replacedHost) tasks[3].iris_attempt_uid = 'replacement';
    write(join(artifacts,suffix,'gang.json'),{tasks,attempts});
  }
  mkdirSync(join(artifacts,'changelog-metadata'));
  write(join(artifacts,'changelog-metadata','agentx.json'),{
    base_ref:'agentx', head_ref:source.head_sha,
    entries:[{description:'AgentX measured load','append-only':true}],
  });
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
  if (interrupted || changedHead || conclusion === 'cancelled' || wrongGpuCount || replacedHost || shortWindow) {
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
test('Snowball calibration can import a load outside the weekly defaults',
  (t) => recoveredPoint(t, 'success', { concurrency: 16 }));
test('Hero imports all sixteen physical GPUs at an arbitrary positive load',
  (t) => recoveredPoint(t, 'success', { profile: 'gb200', concurrency: 64 }));
test('the old four-GB200 profile cannot masquerade as Hero',
  (t) => recoveredPoint(t, 'success', { profile: 'gb200', wrongGpuCount: true }));
test('a replaced Hero follower cannot enter the measured gang',
  (t) => recoveredPoint(t, 'success', { profile: 'gb200', replacedHost: true }));
test('startup checks cannot enter the full-hour comparison',
  (t) => recoveredPoint(t, 'success', { shortWindow: true }));
