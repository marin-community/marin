import { spawn } from 'node:child_process';

for (const key of ['PGHOST', 'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'CACHE_NAMESPACE']) {
  if (!process.env[key]) throw new Error(`${key} is required`);
}
if (process.env.DATABASE_WRITE_URL)
  throw new Error('The web service must not receive a database write URL');
const environment = { ...process.env, DATABASE_READONLY_URL: 'postgresql:///' };
const child = spawn(process.execPath, ['packages/app/server.js'], {
  stdio: 'inherit',
  env: environment,
});
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
child.on('error', (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
child.on('exit', (code, signal) => {
  process.exitCode = code ?? (signal ? 128 : 1);
});
