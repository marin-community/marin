import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { API_FIXTURES } from "../fixtures/dashboard";

const FIXED_NOW = new Date("2026-07-17T12:30:00.000Z");
const VIEWPORTS = [
  { width: 1920, height: 1080 },
  { width: 1440, height: 900 },
  { width: 1366, height: 768 },
] as const;

async function openDashboard(page: Page): Promise<void> {
  await page.clock.install({ time: FIXED_NOW });
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const fixture = API_FIXTURES[path];
    if (fixture === undefined) {
      await route.abort("blockedbyclient");
      return;
    }
    await route.fulfill({ json: fixture });
  });
  await page.route("https://**", (route) => route.abort("blockedbyclient"));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Nightly regressions" })).toBeVisible();
  await expect(page.getByText("Today: 6/10 healthy")).toBeVisible();
}

test("matrix communicates status, confidence, duration, and recovery", async ({ page }) => {
  await page.setViewportSize(VIEWPORTS[0]);
  await openDashboard(page);

  const table = page.getByRole("table", {
    name: "Seven UTC days of scheduled regression status and duration by lane",
  });
  await expect(table).toBeVisible();
  await expect(table.locator("tbody tr")).toHaveCount(7);
  await expect(page.getByRole("columnheader", { name: "Data T3 · Mon" })).toBeVisible();

  const falseGreen = page.getByLabel(
    /vLLM GPU, 2026-07-17: GitHub success; 1 minute; suspiciously short; expected 6 minutes to 15 minutes/,
  );
  await expect(falseGreen).toBeVisible();
  await falseGreen.click();
  const falseGreenDetails = falseGreen.locator("xpath=ancestor::details");
  await expect(falseGreenDetails.getByText(/expected 6m–15m/)).toBeVisible();

  const recovered = page.getByLabel(
    /Harbor, 2026-07-17: GitHub success; 8 minutes; within expected range; failed then passed/,
  );
  await expect(recovered).toBeVisible();
  await recovered.focus();
  await expect(recovered).toBeFocused();
  const fridayT3 = table.locator('tbody tr:first-child td[headers~="lane-datakit-t3"]');
  await expect(fridayT3).toContainText("not scheduled");

  const vllmCell = falseGreen.locator("xpath=ancestor::td");
  await expect(vllmCell).toHaveAttribute(
    "headers",
    "date-2026-07-17 group-forks subgroup-inference lane-vllm-gpu",
  );

  const accessibility = await new AxeBuilder({ page })
    .include("section[aria-labelledby='nightly-heading']")
    .analyze();
  expect(accessibility.violations).toEqual([]);
});

for (const viewport of VIEWPORTS) {
  test(`full dashboard remains glanceable at ${viewport.width}x${viewport.height}`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await openDashboard(page);

    const layout = await page.evaluate(() => {
      const table = document.querySelector(".nightly-table-shell")?.getBoundingClientRect();
      const build = [...document.querySelectorAll("h2")]
        .find((heading) => heading.textContent === "GitHub Build")
        ?.getBoundingClientRect();
      return {
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
        tableRight: table?.right ?? Number.POSITIVE_INFINITY,
        tableBottom: table?.bottom ?? Number.POSITIVE_INFINITY,
        buildTop: build?.top ?? Number.POSITIVE_INFINITY,
        viewportHeight: window.innerHeight,
        durationFontSize: Number.parseFloat(
          getComputedStyle(document.querySelector(".nightly-duration") as Element).fontSize,
        ),
        iconWidth: Number.parseFloat(
          getComputedStyle(document.querySelector(".nightly-status-icon") as Element).width,
        ),
      };
    });
    expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
    expect(layout.tableRight).toBeLessThanOrEqual(layout.viewportWidth);
    expect(layout.tableBottom).toBeLessThan(layout.viewportHeight);
    expect(layout.buildTop).toBeLessThan(layout.viewportHeight);
    expect(layout.durationFontSize).toBeGreaterThanOrEqual(10.5);
    expect(layout.iconWidth).toBeGreaterThanOrEqual(16);

    await expect(page).toHaveScreenshot(`nightly-dashboard-${viewport.width}x${viewport.height}.png`, {
      fullPage: false,
    });
  });
}
