import { createHash } from 'node:crypto';
import { copyFileSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

// Keep complete upstream benchmark routes, APIs and their modules. Only the
// deployment entrypoints change; charts, providers and the data pipeline stay upstream.
const routes = new Set([
  '(dashboard)',
  '[...notFound]',
  'agentx',
  'api',
  'compare',
  'compare-per-dollar',
  'compare-precision',
  'compare-spec-decode',
  'embed',
  'model',
  'run',
  'zh',
]);
const root = 'packages/app/src/app';
for (const directory of [root, join(root, 'zh')]) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && !routes.has(entry.name) && entry.name !== 'fonts') {
      rmSync(join(directory, entry.name), { recursive: true });
    }
  }
}
// Private pages do not need database-backed social images or static SEO discovery.
function removeSocialImages(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) removeSocialImages(path);
    else if (entry.name === 'opengraph-image.tsx' || entry.name === 'twitter-image.tsx')
      rmSync(path);
  }
}
removeSocialImages(root);
copyFileSync('/package/layout.tsx', join(root, 'layout.tsx'));
writeFileSync(
  join(root, 'robots.ts'),
  "export default function robots() { return { rules: [{ userAgent: '*', disallow: '/' }] }; }\n",
);
writeFileSync(
  join(root, 'sitemap.ts'),
  "import type { MetadataRoute } from 'next';\nexport default function sitemap(): MetadataRoute.Sitemap { return []; }\n",
);

const settings = JSON.parse(readFileSync('/package/settings.json', 'utf8'));
const source = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
if (source !== settings.app_source) throw new Error('InferenceX-app source differs from its pin');
const patch = createHash('sha256').update(readFileSync('/package/app.patch')).digest('hex');
const profile = createHash('sha256')
  .update(readFileSync('/package/build.mjs'))
  .update(readFileSync('/package/layout.tsx'))
  .digest('hex');
writeFileSync(
  'packages/app/public/knotmeter-build.json',
  JSON.stringify(
    {
      upstream: `https://github.com/SemiAnalysisAI/InferenceX-app/tree/${source}`,
      source,
      patch_sha256: patch,
      profile: 'knotmeter',
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
    CIRCLE_NODE_TOTAL: '1',
  },
});
