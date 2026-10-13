import { copyFile, mkdir } from 'node:fs/promises';
import { build } from 'esbuild';

await mkdir('dist', { recursive: true });
await build({entryPoints: ['main.js'], outfile: 'dist/main.js', bundle: true, format: 'iife', target: 'es2022'});
await Promise.all(['index.html', 'style.css', 'markdown.css'].map(file => copyFile(file, `dist/${file}`)));
