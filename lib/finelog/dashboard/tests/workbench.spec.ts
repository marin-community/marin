/**
 * Query-workbench smoke test against a real finelog server.
 *
 * Start one with data and point the suite at it:
 *
 *     uv run python lib/finelog/dashboard/scripts/demo.py --keep     # port 10001
 *     cd lib/finelog/dashboard && npm run build && npx playwright test
 *
 * `FINELOG_BASE_URL` overrides the target, which is how this runs against a
 * store holding real segments rather than the demo's seeded rows.
 */
import { expect, test } from '@playwright/test'

const NAMESPACE = process.env.FINELOG_TEST_NAMESPACE ?? 'metrics.cpu'

function queryUrl(sql: string): string {
  return `/query?sql=${encodeURIComponent(sql)}`
}

test('runs a query and renders its rows', async ({ page }) => {
  await page.goto(queryUrl(`SELECT * FROM "${NAMESPACE}" LIMIT 5`))
  await expect(page.locator('tbody tr').first()).toBeVisible()
  await expect(page.getByText(/\d+ rows? · \d+ ms/)).toBeVisible()
})

test('reports a SQL error instead of rendering an empty table', async ({ page }) => {
  await page.goto(queryUrl('SELECT * FROM no_such_namespace'))
  await expect(page.getByText(/not found|Error/i)).toBeVisible()
  await expect(page.locator('tbody tr')).toHaveCount(0)
})

test('opens the full value of a truncated cell', async ({ page }) => {
  await page.goto(queryUrl(`SELECT * FROM "${NAMESPACE}" LIMIT 5`))
  await page.locator('tbody tr').first().locator('td button').first().click()

  const drawer = page.getByRole('dialog')
  await expect(drawer).toBeVisible()
  await expect(drawer.getByText('Rest of row')).toBeVisible()

  await page.keyboard.press('Escape')
  await expect(drawer).toBeHidden()
})

test('charts a query result and honours the axis pickers', async ({ page }) => {
  await page.goto(queryUrl(`SELECT * FROM "${NAMESPACE}" LIMIT 200`))
  await page.getByRole('button', { name: 'chart', exact: true }).click()

  const chart = page.locator('svg[role=img]')
  await expect(chart).toBeVisible()
  // A mark is drawn, not just axes.
  await expect(chart.locator('path, rect, circle').first()).toBeVisible()

  await page.getByRole('button', { name: 'bar', exact: true }).click()
  await expect(chart.locator('rect').first()).toBeVisible()
})

test('switches timestamps between UTC, local, and epoch', async ({ page }) => {
  await page.goto(queryUrl('SELECT 1758000000000 AS captured_ms'))
  const cell = page.locator('tbody tr').first().locator('td').first()

  await page.getByRole('button', { name: 'utc', exact: true }).click()
  await expect(cell).toHaveText('2025-09-16 05:20:00.000')

  await page.getByRole('button', { name: 'epoch', exact: true }).click()
  await expect(cell).toHaveText('1758000000000')
})
