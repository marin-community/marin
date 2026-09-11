import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

/** Where `npm run dev` forwards RPC. Override with FINELOG_DEV_SERVER. */
const FINELOG_DEV_SERVER = process.env.FINELOG_DEV_SERVER ?? 'http://127.0.0.1:10001'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: {
      index: './src/main.ts',
    },
  },
  output: {
    distPath: { root: 'dist' },
    // 'auto' makes chunk URLs resolve against the script tag's origin, so the
    // bundle works under either / or a reverse-proxy prefix like
    // /proxy/log-server/. The base is set via <base href> at serve time.
    assetPrefix: 'auto',
  },
  html: {
    template: './src/template.html',
    templateParameters: { title: 'Finelog Dashboard' },
  },
  server: {
    // `npm run dev` serves the SPA with HMR and forwards RPC to a finelog
    // running on its default port, so frontend work does not need a rebuild
    // and reinstall of the dist the Rust server reads from disk.
    proxy: {
      '/finelog.stats.StatsService': FINELOG_DEV_SERVER,
      '/finelog.logging.LogService': FINELOG_DEV_SERVER,
    },
  },
})
