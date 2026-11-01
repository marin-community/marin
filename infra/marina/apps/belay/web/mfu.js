// Hero Scale MFU page: how hero-scale MFU moved, from mfu.json (one-rack ladder from the hero to the Belay
// architecture, the S7 push, the production hero's cutovers, and other one-rack measurements). Loaded after common.js.

const SVGNS = 'http://www.w3.org/2000/svg'

function svg(tag, attrs, parent) {
  const node = document.createElementNS(SVGNS, tag)
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v)
  if (parent) parent.append(node)
  return node
}

function label(parent, x, y, text, attrs = {}) {
  const t = svg('text', { x, y, 'font-size': 11, fill: 'var(--slate)', 'font-family': 'var(--mono)', ...attrs }, parent)
  t.textContent = text
  return t
}

function pct(v, d = 1) {
  return v === null || v === undefined ? '—' : `${v.toFixed(d)}%`
}

// Small mono links to the branch, commit or PR behind a result.
function codeLinks(links) {
  if (!links || !links.length) return null
  const short = (label) =>
    label.replace('marin_belay_v1_mfuladder_', '').replace('marin_belay_v1_mfuladder', 'mfuladder').replace('marin_belay_v1_', '')
  return el(
    'span',
    { class: 'code-links' },
    links.map((l) =>
      el('a', { href: l.url, target: '_blank', rel: 'noopener', title: l.label }, el('span', { class: 'lfull' }, l.label), el('span', { class: 'lshort' }, short(l.label))),
    ),
  )
}

function runCell(run, links) {
  return el('td', { class: 'mono muted small run' }, el('span', { class: 'rname' }, run), codeLinks(links))
}

function num(v, d) {
  return v === null || v === undefined ? el('span', { class: 'muted' }, '—') : v.toFixed(d)
}

// Step chart over the ladder stages: latest tuned MFU.
function ladderChart(ladder) {
  const box = document.getElementById('ladder-chart')
  // viewBox width follows the box, so labels stay readable at phone width.
  const W = Math.max(340, Math.min(760, box.clientWidth))
  const H = 300
  const M = { l: 44, r: 16, t: 18, b: 34 }
  const lo = 28
  const hi = 42
  const stages = ladder.stages
  const X = (i) => M.l + ((i + 0.5) / stages.length) * (W - M.l - M.r)
  const half = (W - M.l - M.r) / stages.length / 2
  const Y = (v) => M.t + ((hi - v) / (hi - lo)) * (H - M.t - M.b)
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Analytic MFU by stage, one GB200 rack' })
  for (const v of [30, 34, 38, 42]) {
    svg('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }, root)
    label(root, M.l - 6, Y(v) + 4, `${v}%`, { 'text-anchor': 'end' })
  }
  const d = stages.map((s, i) => `${i ? 'L' : 'M'}${X(i) - half},${Y(s.mfu)}H${X(i) + half}`).join('')
  svg('path', { d, fill: 'none', stroke: 'var(--rope)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, root)
  stages.forEach((s, i) => {
    svg('circle', { cx: X(i), cy: Y(s.mfu), r: 4.5, fill: 'var(--rope)' }, root)
    label(root, X(i), Y(s.mfu) - 9, pct(s.mfu), { 'text-anchor': 'middle', fill: 'var(--ink)', 'font-weight': 600 })
    label(root, X(i), H - M.b + 18, s.stage, { 'text-anchor': 'middle', fill: 'var(--ink)', 'font-weight': 600 })
  })
  box.replaceChildren(root)
}

// MFU against training step for the production hero, one segment per run.
function productionChart(production) {
  const box = document.getElementById('prod-chart')
  const runs = production.runs
  const W = Math.max(340, Math.min(760, box.clientWidth))
  const H = 260
  const M = { l: 44, r: 16, t: 18, b: 34 }
  const x0 = 50000
  const x1 = 380000
  const lo = 20
  const hi = 32
  const X = (s) => M.l + ((s - x0) / (x1 - x0)) * (W - M.l - M.r)
  const Y = (v) => M.t + ((hi - v) / (hi - lo)) * (H - M.t - M.b)
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Production hero MFU by training step' })
  for (const v of [20, 24, 28, 32]) {
    svg('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }, root)
    label(root, M.l - 6, Y(v) + 4, `${v}%`, { 'text-anchor': 'end' })
  }
  for (const s of [100000, 200000, 300000]) label(root, X(s), H - M.b + 18, `${s / 1000}k`, { 'text-anchor': 'middle' })
  label(root, W - M.r, H - M.b + 18, 'step', { 'text-anchor': 'end', fill: 'var(--faint)' })
  const d = runs
    .map((r, i) => {
      const end = i + 1 < runs.length ? runs[i + 1].from_step : x1
      return `${i ? 'V' : 'M' + X(r.from_step) + ','}${Y(r.mfu)}H${X(end)}`
    })
    .join('')
  svg('path', { d, fill: 'none', stroke: 'var(--teal)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, root)
  runs.forEach((r, i) => {
    svg('circle', { cx: X(r.from_step), cy: Y(r.mfu), r: 4, fill: 'var(--teal)' }, root)
    // Segments under 5k steps (the data-mix swap) are too short to label; the table has them.
    const end = i + 1 < runs.length ? runs[i + 1].from_step : x1
    if (end - r.from_step < 5000) return
    label(root, X(r.from_step) - 4, Y(r.mfu) - 8, pct(r.mfu, 2), { fill: 'var(--ink)', 'font-weight': 600, 'text-anchor': 'end' })
  })
  box.replaceChildren(root)
}

function ladderTable(ladder) {
  setChildren(
    'ladder-rows',
    ladder.stages.map((s) => {
      return el(
        'tr',
        {},
        el('td', { class: 'mono strong' }, s.stage),
        el('td', {}, s.change),
        runCell(s.run, s.links),
        el('td', { class: 'r mono' }, num(s.step, 2)),
        el('td', { class: 'r mono hide-sm' }, num(s.tps_k, 1), 'k'),
        el('td', { class: 'r mono strong' }, pct(s.mfu)),
      )
    }),
  )
}

function pushList(push) {
  setChildren(
    'push-rows',
    push.steps.map((p) =>
      el(
        'tr',
        { dataset: { wall: String(Boolean(p.wall)) } },
        el('td', { class: 'mono muted small' }, p.date),
        el('td', {}, p.what),
        runCell(p.run, p.links),
        el('td', { class: 'r mono' }, p.step.toFixed(2), p.wall ? el('small', { class: 'muted' }, ' wall') : null),
        el('td', { class: 'r mono strong' }, pct(p.mfu)),
      ),
    ),
  )
  setChildren(
    'push-wins',
    push.wins_ms.map((w) => el('span', { class: 'win' }, el('b', { class: 'mono' }, `${w.ms} ms`), ` ${w.what}`, codeLinks(w.links))),
  )
}

function productionTable(production) {
  setChildren(
    'prod-rows',
    production.runs.map((r) => {
      const tps = r.tps_m ?? (r.step ? production.tokens_per_step / r.step / 1e6 : null)
      return el(
        'tr',
        { title: r.note ?? '' },
        el('td', { class: 'mono muted small hide-sm' }, r.date),
        el('td', {}, r.change),
        runCell(r.run.replace('hero-', ''), r.links),
        el('td', { class: 'r mono' }, num(r.step, 2)),
        el(
          'td',
          { class: 'r mono hide-sm' },
          tps === null ? el('span', { class: 'muted' }, '—') : `${tps.toFixed(2)}M`,
          tps !== null && !r.tps_m ? el('sup', { class: 'muted' }, '*') : null,
        ),
        el('td', { class: 'r mono strong' }, pct(r.mfu, 2)),
      )
    }),
  )
}

function otherTable(other) {
  setChildren(
    'other-rows',
    other.rows.map((r) =>
      el(
        'tr',
        {},
        el('td', {}, r.what),
        el('td', { class: 'r mono muted' }, r.before ?? '—'),
        el('td', { class: 'r mono strong' }, r.after),
        el('td', { class: 'muted small' }, r.note, codeLinks(r.links)),
      ),
    ),
  )
}

async function mfuPage() {
  showView('mfu')
  const response = await fetch(STATIC ? 'mfu.json' : `${BELAY_BASE}mfu.json`, { headers: { accept: 'application/json' } })
  const data = await response.json()
  const h = data.headline
  const latestHero = data.production.runs[data.production.runs.length - 1]
  document.getElementById('kpi-mfu').textContent = pct(h.mfu)
  document.getElementById('kpi-sub').textContent =
    `${h.stage} · ${h.step.toFixed(3)} s/step · ${h.tps_k.toFixed(1)}k tok/s · control ${h.control_step.toFixed(3)} s`
  document.getElementById('kpi-hero').textContent = pct(latestHero.mfu, 2)
  document.getElementById('kpi-hero-sub').textContent = `${latestHero.run} · ${latestHero.step.toFixed(2)} s/step`
  document.getElementById('flop-note').textContent = data.flop_note
  document.getElementById('ladder-setup').textContent = data.ladder.setup
  document.getElementById('prod-setup').textContent = data.production.setup
  document.getElementById('push-note').textContent = data.s7_push.note
  ladderChart(data.ladder)
  ladderTable(data.ladder)
  pushList(data.s7_push)
  productionChart(data.production)
  productionTable(data.production)
  otherTable(data.other)
  setChildren('unsourced', data.unsourced.map((u) => el('li', {}, u)))
}

mfuPage()
