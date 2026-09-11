import { defineConfig, devices } from '@playwright/test'

/**
 * Drives the built dashboard against an already-running finelog server; it does
 * not start one. `scripts/demo.py --keep` serves a seeded store on port 10001,
 * which is the default target.
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: process.env.FINELOG_BASE_URL ?? 'http://127.0.0.1:10001',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
