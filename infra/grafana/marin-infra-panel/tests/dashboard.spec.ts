import { expect, test } from '@playwright/test';

test('compact infra dashboard renders the overview and deep-dive sections', async ({ page }) => {
  await page.goto('/d/infra/infra?orgId=1&kiosk');
  await expect(page.getByLabel('Nightly regression status')).toBeVisible();
  await expect(page.getByLabel('Main branch build history')).toBeVisible();
  await expect(page.getByText('Healthy workers', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-first-viewport.png' });

  await page.getByText('Fleet and provisioning', { exact: true }).scrollIntoViewIfNeeded();
  await expect(page.getByText('Workers by region', { exact: true })).toBeVisible();
  await expect(page.getByText('us-east5', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-operations.png' });

  await page.getByText('Hero training', { exact: true }).scrollIntoViewIfNeeded();
  await expect(page.getByRole('link', { name: 'W&B report ↗' }).first()).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-hero-training.png' });
});
