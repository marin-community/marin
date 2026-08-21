import { expect, test } from '@playwright/test';

test('infra status page renders regressions, CI, capacity, and the hero run', async ({ page }) => {
  await page.goto('/d/infra/infra?orgId=1&kiosk');
  await expect(page.getByRole('main', { name: 'Marin infrastructure status' })).toBeVisible();
  await expect(page.getByLabel('Nightly regression status')).toBeVisible();
  await expect(page.getByLabel('Main branch build history')).toBeVisible();
  await expect(page.getByText('healthy workers', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-first-viewport.png' });

  await page.getByRole('region', { name: 'Hero run charts' }).scrollIntoViewIfNeeded();
  await expect(page.getByRole('link', { name: 'W&B report ↗' }).first()).toBeVisible();
  await page.screenshot({ path: 'artifacts/infra-hero-training.png' });
});

test('cluster capacity page shows job rollups and GPU packing for one cluster', async ({ page }) => {
  await page.goto('/d/marin-cluster-capacity/cluster-capacity?orgId=1&kiosk');
  await expect(page.getByRole('main', { name: 'Cluster capacity' })).toBeVisible();
  await expect(page.getByText('Cluster capacity · cw-us-east-02a')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Active jobs' }).getByText('/alice/llama')).toBeVisible();
  await expect(page.getByRole('article', { name: 'Node gpu-a' })).toBeVisible();
  await expect(page.getByRole('list', { name: 'GPU slots on gpu-b' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Unbound tasks' }).getByText('/dave/train/0')).toBeVisible();
  await page.screenshot({ path: 'artifacts/cluster-capacity.png', fullPage: true });
});
