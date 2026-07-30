import { expect, test } from '@playwright/test';

test('infra status page renders regressions, CI, capacity, provisioning, and the hero run', async ({ page }) => {
  await page.goto('/d/infra/infra?orgId=1&kiosk');
  await expect(page.getByRole('main', { name: 'Marin infrastructure status' })).toBeVisible();
  await expect(page.getByLabel('Nightly regression status')).toBeVisible();
  await expect(page.getByLabel('Main branch build history')).toBeVisible();
  await expect(page.getByText('healthy workers', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-first-viewport.png' });

  await page.getByRole('region', { name: 'Provisioning status' }).scrollIntoViewIfNeeded();
  await expect(page.getByText('us-east5', { exact: true })).toBeVisible();
  await expect(page.getByText('pools without ready outcome', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-operations.png' });

  await page.getByRole('region', { name: 'Hero run charts' }).scrollIntoViewIfNeeded();
  await expect(page.getByRole('link', { name: 'W&B report ↗' }).first()).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-hero-training.png' });
});
