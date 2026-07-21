import { expect, test } from '@playwright/test'

test('a polled Grafana group becomes one visible agent investigation', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent investigations' })).toBeVisible()
  await page.getByText(/DNSConfigForming.*cw-us-east-08a/).last().click()

  await expect(page.getByRole('heading', { name: /DNSConfigForming.*cw-us-east-08a/ })).toBeVisible()
  await expect(page.getByText('fp 1:dns-config-forming:2b05ef3b1641c79a', { exact: true })).toBeVisible()
  await expect(page.getByText('fp 1:dns-config-forming:ef356383208c86c5', { exact: true })).toBeVisible()
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

test('diagnostics exposes durable polls and the live process buffer', async ({ page }) => {
  await page.goto('/diagnostics')
  await expect(page.getByRole('heading', { name: 'Polling and process logs' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Recent successful Grafana polls' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Live process buffer' })).toBeVisible()
  await expect(page.getByText('reconciled Grafana snapshot', { exact: false }).last()).toBeVisible()
})
