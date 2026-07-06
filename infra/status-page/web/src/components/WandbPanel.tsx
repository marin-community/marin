// W&B training charts, embedded as an iframe of a dedicated two-panel
// "status page" report (defined by wandb_report.py next to package.json).
//
// A purpose-built mini report is the only reliable way to embed
// individual W&B panels: public reports serve frame-friendly headers and
// embed chrome, but individual panels are not URL-addressable, and
// pixel-cropping the full hero report proved flaky (its layout shifts
// with progressive rendering). The mini report's runset is a run-name
// search ("sw2k_v4_2048_muon"), so new hero resumes appear on the page
// without touching this file.
const REPORT_EMBED_URL =
  "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T:-status-page-hero-charts--VmlldzoxNzQzMTI4MQ==";
const PARENT_REPORT_URL =
  "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ";
// Fits the fluid-width report's title block + single panel row on a wide
// viewport, with W&B's cookie banner landing below the charts. On narrow
// viewports the panels stack and the iframe scrolls instead.
const EMBED_HEIGHT_PX = 840;

export function WandbPanel() {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Training
        </h3>
        <a
          href={PARENT_REPORT_URL}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-500 hover:text-emerald-300"
        >
          full wandb report ↗
        </a>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <iframe
          src={REPORT_EMBED_URL}
          title="W&B: 67B-A2B MoE on 10T tokens — status-page hero charts"
          className="w-full rounded-md bg-slate-950"
          style={{ height: EMBED_HEIGHT_PX, border: "none" }}
        />
      </div>
    </div>
  );
}
