import { expect, test } from '@playwright/test'

test('a polled Grafana group becomes one visible agent investigation', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Alert cases' })).toBeVisible()
  await page.getByText(/DNSConfigForming.*cw-us-east-08a/).last().click()

  await expect(page.getByRole('heading', { name: /DNSConfigForming.*cw-us-east-08a/ })).toBeVisible()
  await expect(page.getByText('fp 1:dns-config-forming:2b05ef3b1641c79a', { exact: true })).toBeVisible()
  await expect(page.getByText('fp 1:dns-config-forming:ef356383208c86c5', { exact: true })).toBeVisible()
  const chat = page.locator('aside')
  await expect(chat.getByText('Agent session')).toBeVisible()

  const followUp = 'Check whether both namespaces are affected by the same node resolver configuration.'
  await page.getByPlaceholder('Ask a follow-up or add context…').fill(followUp)
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByText('follow up queued').last()).toBeVisible()
})

test('an operator can queue a one-off question through the same surface', async ({ page }) => {
  await page.goto('/ask')
  await page.getByLabel('Investigation request').fill('Validate the current free image filesystem space without modifying the node.')
  await page.getByRole('button', { name: 'Queue investigation' }).click()
  await expect(page).toHaveURL(/\/cases\//)
  await expect(page.getByText('Validate the current free image filesystem space', { exact: false }).first()).toBeVisible()
})

test('diagnostics exposes durable polls and the live process buffer', async ({ page }) => {
  await page.goto('/diagnostics')
  await expect(page.getByRole('heading', { name: 'Service diagnostics' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Grafana polling' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Slack deliveries' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Process logs' })).toBeVisible()
  await expect(page.getByRole('table').first().getByRole('cell', { name: '2', exact: true }).first()).toBeVisible()
  await expect(page.getByText('No agent escalations.')).toBeVisible()
  await expect(page.locator('pre').first()).toBeVisible()
})
