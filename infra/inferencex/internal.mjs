import { copyFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Keep complete upstream benchmark routes, APIs and their modules. Only the
// deployment entrypoints change; charts, providers and the data pipeline stay upstream.
const routes = new Set([
  '(dashboard)',
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
copyFileSync('/package/internal-layout.tsx', join(root, 'layout.tsx'));
writeFileSync(
  join(root, 'robots.ts'),
  "export default function robots() { return { rules: [{ userAgent: '*', disallow: '/' }] }; }\n",
);
writeFileSync(
  join(root, 'sitemap.ts'),
  "import type { MetadataRoute } from 'next';\nexport default function sitemap(): MetadataRoute.Sitemap { return []; }\n",
);
