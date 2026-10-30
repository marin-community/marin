import { spawn } from 'node:child_process';

const required = [
  'DATABASE_USER',
  'DATABASE_PASSWORD',
  'DATABASE_NAME',
  'DATABASE_SOCKET_DIR',
  'CACHE_NAMESPACE',
];
for (const key of required) {
  if (!process.env[key]) throw new Error(`${key} is required`);
}
if (process.env.DATABASE_WRITE_URL)
  throw new Error('The web service must not receive a database write URL');
const connection = new URL('postgresql://localhost/');
connection.username = process.env.DATABASE_USER;
connection.password = process.env.DATABASE_PASSWORD;
connection.pathname = `/${process.env.DATABASE_NAME}`;
const environment = { ...process.env, DATABASE_READONLY_URL: connection.href };
delete environment.DATABASE_PASSWORD;
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
