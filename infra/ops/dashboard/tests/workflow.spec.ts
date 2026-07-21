import { createHmac } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { expect, test } from '@playwright/test'

const WEBHOOK_SECRET = 'local-grafana-secret'
const fixture = readFileSync(resolve(import.meta.dirname, '../../fixtures/dns-warning-firing.json'))

test('a signed Grafana group becomes one visible agent investigation', async ({ page, request }) => {
  const timestamp = String(Math.floor(Date.now() / 1000))
  const signature = createHmac('sha256', WEBHOOK_SECRET).update(`${timestamp}:`).update(fixture).digest('hex')
  const ingestion = await request.post('/api/ingest/grafana', {
    data: fixture,
    headers: {
      'content-type': 'application/json',
      'x-grafana-alerting-signature': signature,
      'x-grafana-alerting-signature-timestamp': timestamp,
    },
  })
  expect([200, 202]).toContain(ingestion.status())
  const body = (await ingestion.json()) as { case_ids: string[] }
  expect(body.case_ids).toHaveLength(1)

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent investigations' })).toBeVisible()
  await page.getByText('DNSConfigForming cw-us-east-08a', { exact: false }).last().click()

  await expect(page.getByRole('heading', { name: /DNSConfigForming cw-us-east-08a/ })).toBeVisible()
  await expect(page.getByText('fp 2b05ef3b1641c79a', { exact: true })).toBeVisible()
  await expect(page.getByText('fp ef356383208c86c5', { exact: true })).toBeVisible()
  const chat = page.locator('aside')
  await expect(chat.getByText('Validated the alert against the target cluster', { exact: false })).toBeVisible()

  await page.getByPlaceholder('Ask a follow-up or add context…').fill('Check whether both namespaces are affected by the same node resolver configuration.')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByText('follow up queued')).toBeVisible()
  await expect(chat.getByText('Validated the alert against the target cluster', { exact: false })).toHaveCount(2)
})

test('an operator can queue a one-off question through the same surface', async ({ page }) => {
  await page.goto('/ask')
  await page.getByLabel('What should the agent investigate?').fill('Validate the current free image filesystem space without modifying the node.')
  await page.getByRole('button', { name: 'Start investigation' }).click()
  await expect(page).toHaveURL(/\/cases\//)
  await expect(page.getByText('Validate the current free image filesystem space', { exact: false }).first()).toBeVisible()
})
