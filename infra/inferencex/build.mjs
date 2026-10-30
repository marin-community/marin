import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const settings = JSON.parse(readFileSync('/package/settings.json', 'utf8'));
const source = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
if (source !== settings.app_source) throw new Error('InferenceX-app source differs from its pin');
const patch = createHash('sha256').update(readFileSync('/package/app.patch')).digest('hex');
const profile = createHash('sha256')
  .update(readFileSync('/package/internal.mjs'))
  .update(readFileSync('/package/internal-layout.tsx'))
  .digest('hex');
writeFileSync(
  'packages/app/public/marin-build.json',
  JSON.stringify(
    {
      upstream: `https://github.com/SemiAnalysisAI/InferenceX-app/tree/${source}`,
      source,
      patch_sha256: patch,
      profile: 'internal',
      profile_sha256: profile,
      node: process.version,
      bun: settings.bun,
    },
    null,
    2,
  ),
);
execFileSync('bun', ['run', 'build'], {
  stdio: 'inherit',
  env: {
    ...process.env,
    NEXT_PUBLIC_SITE_URL: settings.site_url,
    NEXT_PUBLIC_MARIN_PERFORMANCE: '1',
    CIRCLE_NODE_TOTAL: '1',
  },
});
