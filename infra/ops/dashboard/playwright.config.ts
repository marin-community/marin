import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: process.env.OPS_BASE_URL ?? 'http://127.0.0.1:8088',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
})
