// Marin Belay status page: the idea checklist (one row per registry idea, its gates, recipe status and potential;
// a row expands in place to the agent's thread for it), what is running, the agent log and the activity feed, from
// the registry results plus the public state (read-only, the same data any anonymous visitor gets). status#X####
// opens that idea's row. Agent text is rendered by markdown(), which builds DOM text nodes only (no HTML) and links
// only http(s) URLs. Loaded after common.js.

let registry = null // the registry's results JSON (ideas with trials, events, potential)
let state = null // the public state last read from the API
// tracks.json, generated from experiments/belay/config.py: { tracks, recipe_tag, default_author }. Read once per
// refresh and passed explicitly to the gate and recipe helpers.
let config = { tracks: [], recipe_tag: '', default_author: '' }

const GATE = {
  win: { icon: '✓', cls: 'won' },
  noise: { icon: '~', cls: 'noise' },
  loss: { icon: '✗', cls: 'lost' },
  running: { icon: '◷', cls: 'run' },
  none: { icon: '–', cls: 'tested' },
  na: { icon: '·', cls: 'tested' },
}
const OUTSIDE = /outside the verdict|not part of the verdict/

function trackInfo(cfg, id) {
  return cfg.tracks.find((t) => t.track === id) ?? {}
}

function verdictOf(d, noise) {
  if (d === undefined || d === null) return 'none'
  return d <= -noise ? 'win' : d >= noise ? 'loss' : 'noise'
}

// The registry build shows macro loss as placeholder BPB; gates judge the recorded macro loss (``loss_delta``).
function counted(trials) {
  return trials
    .filter((t) => !OUTSIDE.test(t.note ?? '') && (t.delta ?? []).length)
    .map((t) => ({ ...t, delta: t.loss_delta ?? t.delta }))
}

// Gate 1, beat a vanilla transformer: the latest vanilla-track trial (vanilla-d512 to vanilla-d1280).
function denseGate(idea, cfg) {
  const t = counted(idea.trials.filter((x) => x.rung?.startsWith('vanilla-'))).at(-1)
  if (!t) return { kind: 'none', text: 'not run', title: '' }
  const d = t.delta[0]
  const kind = verdictOf(d, trackInfo(cfg, t.rung).noise)
  const where = t.protocol === 'fixed_time' ? 'equal time' : 'equal data'
  return { kind, short: signed(d), text: `${kind} at ${where}, Δ ${signed(d)} · ${t.rung.replace('vanilla-', '')}`, title: `${t.variant} (${t.rung}, ${t.date})` }
}

// A gate with a run in flight: its progress and, from the W&B step rate or the expected finish, the time left.
function runningGate(run) {
  const pct = run.total_steps ? Math.floor((100 * Math.min(run.step ?? 0, run.total_steps)) / run.total_steps) : null
  const left = eta(run) ?? (run.expected_at ? `due ${relative(run.expected_at)}` : null)
  const text = ['running', pct !== null ? `${pct}%` : null, left ? `${left}${left.startsWith('due') ? '' : ' left'}` : null]
  const step = run.total_steps ? `step ${(run.step ?? 0).toLocaleString()} of ${run.total_steps.toLocaleString()}` : 'starting'
  return { kind: 'running', short: pct !== null ? `${pct}%` : '', text: text.filter(Boolean).join(' · '), title: `${run.name}: ${step}` }
}

// Gate 2, the 8-minute speedrun: the speedrun track's stages, or a legacy d512 trial from before Tracks existed.
function speedrunGate(idea, liveRun, cfg) {
  const sr = trackInfo(cfg, 'speedrun')
  const ours = counted(idea.trials.filter((t) => t.rung === 'speedrun'))
  const legacy = counted(idea.trials.filter((t) => t.rung === 'd512'))
  const data = ours.filter((t) => t.protocol === 'equal_steps').at(-1)
  const time = ours.filter((t) => t.protocol === 'fixed_time').at(-1)
  if (liveRun) return runningGate(liveRun)
  if (idea.track && idea.track !== 'speedrun' && !ours.length) return { kind: 'na', text: `n/a (track ${idea.track})`, title: '' }
  const final = time ?? data ?? legacy.at(-1)
  if (!final) return { kind: 'none', text: 'not run', title: '' }
  const d = final.delta[0]
  const kind = verdictOf(d, sr.noise)
  const where = time ? 'equal time' : data ? 'equal data' : `legacy ${final.protocol === 'fixed_time' ? 'fixed time' : 'equal steps'}`
  const steps = final.steps ? ` · ${final.steps.toLocaleString()}${sr.baseline_steps && (time || data) ? ` vs ${sr.baseline_steps.toLocaleString()}` : ''} steps` : ''
  const stages = [data, time].filter(Boolean).map(
    (t) => `${t.protocol === 'fixed_time' ? 'Equal time' : 'Equal data'}: Δ ${signed(t.delta[0])}${t.steps ? ` at ${t.steps.toLocaleString()} steps` : ''}${t.step_time_change ? `, step time ${t.step_time_change}` : ''}`,
  )
  return { kind, short: signed(d), text: `${kind} at ${where}, Δ ${signed(d)}${steps}`, title: stages.join('\n') || `${final.variant} (${final.rung}, ${final.date})` }
}

// The idea's standing in the live recipe: its latest adopted/removed event for the recipe's checkpoint tag.
function recipeEvent(idea, recipeTag) {
  return (idea.events ?? []).filter((e) => e.checkpoint === recipeTag && ['adopted', 'removed'].includes(e.type)).at(-1)
}

function inLiveRecipe(idea, recipeTag) {
  return recipeEvent(idea, recipeTag)?.type === 'adopted'
}

// A gate cell: a compact chip (icon, and Δ for a win or loss) whose details show on hover or keyboard focus, and
// expand inline on a tap at phone width.
function gateChip(g, more = null) {
  const look = GATE[g.kind] ?? GATE.none
  const shown = ['win', 'loss', 'running'].includes(g.kind) && g.short ? ` ${g.short}` : ''
  const chip = el('span', { class: `gate-chip ${look.cls}` }, el('b', {}, look.icon), shown)
  const lines = [g.text, ...(g.title ? g.title.split('\n') : [])].filter((x, i, a) => x && a.indexOf(x) === i)
  const tip = el('span', { class: 'gate-tip', role: 'tooltip' }, ...lines.map((x) => el('span', {}, x)), more)
  const cell = el('span', { class: 'gcell', tabindex: '0', 'aria-expanded': 'false', 'aria-label': g.text }, chip, tip)
  cell.addEventListener('click', (e) => {
    if (e.target.closest('a')) return
    cell.setAttribute('aria-expanded', String(cell.getAttribute('aria-expanded') !== 'true'))
  })
  return cell
}
const checklistView = { sort: 'potential', query: '' }
const openRows = new Set() // checklist rows expanded to their thread, by registry id
const logView = { query: '' }
const openPosts = new Set() // agent log entries expanded, by registry id
const REGISTRY_BLOB = 'https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay'
const XID = /^X\d{4,}$/

const expandedIdeas = new Set()
const KIND_CLASS = { better: 'won', worse: 'lost', neutral: 'noise' }
const KIND_WORD = { better: 'better', worse: 'worse', neutral: 'noise' }
const STATE_CLASS = { recorded: 'in', running: 'won', planned: 'tested', approved: 'tested', pending: 'tested', declined: 'lost' }
const STATE_WORD = {
  pending: 'waiting',
  approved: 'approved, queued',
  planned: 'planned',
  running: 'running',
  recorded: 'recorded',
  declined: 'declined',
}
const FEED_ICON = { intake: '＋', approved: '✓', plan: '✎', started: '▶', finished: '■', result: '◆', declined: '✕' }

function verdictChip(result) {
  const kind = result.kind ?? 'neutral'
  return el('span', { class: `st ${KIND_CLASS[kind] ?? 'noise'}` }, KIND_WORD[kind] ?? kind)
}

function variantDelta(idea) {
  const row = (idea.result?.rows ?? []).find((r) => !r.control)
  return row ? delta(row.delta) : el('span', { class: 'muted' }, '—')
}

function ideaRuns(idea) {
  return state.runs.filter((r) => idea.registry_id && r.registry_id === idea.registry_id)
}

function detail(idea, cfg) {
  const plan = idea.plan
  const runs = ideaRuns(idea)
  const facts = [
    idea.url ? ['Source', link(idea.url, idea.author ? `#9514 · ${idea.author}` : 'GitHub')] : ['Source', link(`https://github.com/${cfg.default_author}`, `proposal by @${cfg.default_author}`)],
    plan?.cost ? ['Cost', plan.cost] : null,
    plan?.commit ? ['Code', link(`https://github.com/marin-community/marin/commit/${plan.commit}`, el('code', {}, plan.commit.slice(0, 10)))] : null,
    plan?.spec_url ? ['Spec', link(plan.spec_url, `runs/${idea.registry_id}.yaml`)] : null,
    idea.registry_url ? ['Registry', link(idea.registry_url, idea.registry_id)] : null,
  ].filter(Boolean)
  return el(
    'div',
    { class: 'detail-c' },
    el('div', { class: 'kv' }, facts.map(([k, v]) => el('span', { class: 'kvp' }, el('span', { class: 'k' }, k), v))),
    runs.length
      ? el(
          'div',
          { class: 'mini-runs' },
          runs.map((r) =>
            el(
              'div',
              { class: 'mini-run' },
              el('code', {}, r.name),
              runPill(r),
              progress(r),
              el('span', { class: 'mono' }, loss(r.loss)),
              link(wandbUrl(r), 'W&B ↗'),
            ),
          ),
        )
      : null,
    idea.result ? el('p', { class: 'verdict-line' }, verdictChip(idea.result), ' ', idea.result.verdict) : null,
    idea.note ? el('p', { class: 'muted small' }, idea.note) : null,
    !idea.result && !idea.note ? el('p', { class: 'muted small clamp' }, plan?.summary ?? idea.body) : null,
  )
}

function ideaLine(idea, cells) {
  const key = idea.id
  const expanded = expandedIdeas.has(key)
  const button = el(
    'button',
    {
      type: 'button',
      class: 'line',
      'aria-expanded': String(expanded),
      onclick: () => {
        expanded ? expandedIdeas.delete(key) : expandedIdeas.add(key)
        render()
      },
    },
    ...cells,
  )
  return el('div', { class: 'line-wrap' }, button, expanded ? detail(idea, config) : null)
}

function idCell(idea) {
  return el('span', { class: 'xid mono' }, idea.registry_id ?? `#${idea.id}`)
}

function renderRunning() {
  const live = state.runs.filter((r) => !TERMINAL.has(r.state))
  if (!live.length) {
    setChildren('running', el('div', { class: 'none' }, 'Nothing running'))
    return
  }
  setChildren(
    'running',
    live.map((r) => {
      const idea = state.ideas.find((i) => i.registry_id && i.registry_id === r.registry_id)
      const time = eta(r)
      const cells = [
        el('span', { class: 'xid mono' }, r.registry_id ?? ''),
        el('span', { class: 'nm' }, el('b', {}, idea?.title ?? r.name), el('small', {}, r.name)),
        progress(r),
        el('span', { class: 'eta mono' }, r.state === 'pending' ? 'queued' : time ? `${time} left` : '—'),
      ]
      return idea ? ideaLine(idea, cells) : el('div', { class: 'line static' }, ...cells)
    }),
  )
}

function renderRecent() {
  const done = state.ideas.filter((i) => i.result || i.state === 'declined').slice(0, 8)
  if (!done.length) {
    setChildren('recent', el('div', { class: 'none' }, 'No results yet'))
    return
  }
  setChildren(
    'recent',
    done.map((idea) =>
      ideaLine(idea, [
        idCell(idea),
        el('span', { class: 'nm' }, el('b', {}, idea.title)),
        el('span', { class: 'd mono' }, idea.result ? variantDelta(idea) : ''),
        idea.result ? verdictChip(idea.result) : el('span', { class: 'st lost' }, 'declined'),
      ]),
    ),
  )
}

// Gate 3, hero-scale MFU: the hero-mfu track's relative step-time Δ, else the speedrun step-time cost as a hint.
function heroGate(idea, cfg) {
  const hero = counted(idea.trials.filter((t) => t.rung === 'hero-mfu')).at(-1)
  if (hero) {
    const d = hero.delta[0]
    return { kind: verdictOf(d, trackInfo(cfg, 'hero-mfu').noise), short: `${signed(d * 100, 1)}%`, text: `s/step ${signed(d * 100, 1)}%`, title: hero.variant }
  }
  const stc = idea.trials.find((t) => t.rung === 'speedrun' && t.protocol === 'equal_steps' && t.step_time_change)
  return { kind: 'none', text: stc ? `not run · speedrun ${stc.step_time_change} step time` : 'not run', title: '' }
}

// Gate 4, the hero-scale projection: the competitive ladder's Δ per rung (legacy wider rungs count too).
function scalingGate(idea, cfg) {
  const rungs = counted(idea.trials.filter((t) => t.rung?.startsWith('competitive-') || ['d768', 'd1024', 'd1280'].includes(t.rung)))
  if (!rungs.length) return { kind: 'none', text: 'not run', title: '' }
  const latest = new Map(rungs.map((t) => [t.rung.replace('competitive-', ''), t]))
  const parts = [...latest].map(([r, t]) => `${r} ${signed(t.delta[0])}`)
  const worst = [...latest.values()].map((t) => verdictOf(t.delta[0], trackInfo(cfg, 'competitive-d512').noise))
  const kind = worst.includes('loss') ? 'loss' : worst.every((v) => v === 'win') ? 'win' : 'noise'
  const last = [...latest.values()].at(-1)
  return { kind, short: signed(last.delta[0]), text: `${parts.join(' · ')} · projection pending`, title: parts.join('\n') }
}

function recipeCell(idea, recipeTag) {
  const event = recipeEvent(idea, recipeTag)
  const day = event ? new Date(`${event.date}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''
  if (event?.type === 'adopted') return el('span', { class: 'gate-chip won', title: event.note ?? '' }, el('b', {}, '✓'), ` ${recipeTag}`)
  if (event?.type === 'removed') {
    return el(
      'span',
      { class: 'recipe-cell' },
      el('span', { class: 'gate-chip noise', title: event.note ?? '' }, el('b', {}, '↺'), ` removed at ${day}`),
      el('span', { class: 'gate-chip run', title: 'In Belay Ckpt 1, not in the live recipe: retested on the current recipe' }, retestStage(idea, event)),
    )
  }
  return el('span', { class: 'gate-chip tested' }, el('b', {}, '–'))
}

// A Ckpt 1 idea the live recipe dropped is retested on the current recipe: "retest" until a speedrun trial lands.
function retestStage(idea, event) {
  return idea.trials.some((t) => t.rung === 'speedrun' && t.date >= event.date) ? 'retested' : 'retest'
}

function approvalDate(idea, pageIdea) {
  const dates = [pageIdea?.approved_at?.slice(0, 10), ...(idea.trials ?? []).map((t) => t.date), ...(idea.events ?? []).map((e) => e.date)]
  return dates.filter(Boolean).sort()[0] ?? ''
}

function checklistRows(cfg) {
  const live = new Map(state.runs.filter((r) => !TERMINAL.has(r.state)).map((r) => [r.registry_id, r]))
  return registry.ideas.map((idea) => {
    const pageIdea = state.ideas.find((i) => i.registry_id === idea.xid)
    // A link from before the agent log: an idea's post on the old running issue, kept in place there.
    const comment = (idea.trials ?? []).map((t) => t.comment_url).find((u) => (u ?? '').includes('issuecomment-'))
    return {
      idea,
      comment,
      thread: (state.threads ?? []).find((t) => t.registry_id === idea.xid),
      date: approvalDate(idea, pageIdea),
      author: idea.author,
      accepted: inLiveRecipe(idea, cfg.recipe_tag),
      score: idea.potential?.score ?? null,
      dense: denseGate(idea, cfg),
      speedrun: speedrunGate(idea, live.get(idea.xid), cfg),
      hero: heroGate(idea, cfg),
      scaling: scalingGate(idea, cfg),
    }
  })
}

const SORTS = {
  potential: (a, b) => a.accepted - b.accepted || (b.score ?? -1) - (a.score ?? -1) || b.date.localeCompare(a.date),
  date: (a, b) => b.date.localeCompare(a.date),
  accepted: (a, b) => b.accepted - a.accepted || b.date.localeCompare(a.date),
}

function renderChecklist() {
  if (!registry) return
  const q = checklistView.query.toLowerCase()
  const rows = checklistRows(config)
    .filter((r) => !q || `${r.idea.xid} ${r.idea.name} ${r.author ?? ''} ${r.idea.category} ${recipeEvent(r.idea, config.recipe_tag)?.type === 'removed' ? 'retest' : ''}`.toLowerCase().includes(q))
    .sort(SORTS[checklistView.sort])
  document.getElementById('checklist-n').textContent = String(rows.length)
  const registryUrl = (x) => `https://github.com/marin-community/marin/blob/marin_belay_v1_registry/experiments/belay/registry/${x.xid}-${x.id}.yaml`
  const head = el(
    'div',
    { class: 'ck-row ck-head', role: 'row' },
    ['Idea', 'Author', 'Date', 'Dense', 'Speedrun', 'Hero MFU', 'Projection', 'Recipe', 'Potential'].map((h) => el('span', { role: 'columnheader' }, h)),
  )
  setChildren('checklist', head, rows.map((r) => checklistRow(r, registryUrl)))
}

// The analysis reports of the idea's Track trials (one per trial, on the registry branch).
function analysisLinks(idea, cfg) {
  const tracks = new Set(cfg.tracks.map((t) => t.track))
  return (idea.trials ?? [])
    .filter((t) => tracks.has(t.rung) && (t.wandb ?? []).length)
    .map((t) => link(`${REGISTRY_BLOB}/analysis/${idea.xid}/${t.id}.md`, `${idea.xid} ${t.id}`))
}

function threadDetail(r, cfg) {
  const analyses = analysisLinks(r.idea, cfg)
  return el(
    'div',
    { class: 'ck-detail', role: 'cell' },
    r.thread
      ? [el('div', { class: 'muted small' }, 'Agent thread · updated ', when(r.thread.updated_at)), markdown(r.thread.body)]
      : el('p', { class: 'muted small' }, 'No agent thread for this idea.'),
    el(
      'div',
      { class: 'kv' },
      analyses.length ? el('span', { class: 'kvp' }, el('span', { class: 'k' }, 'Analysis'), ...analyses.flatMap((a, i) => (i ? [' · ', a] : [a]))) : null,
      r.comment ? el('span', { class: 'kvp' }, el('span', { class: 'k' }, 'Earlier post'), link(r.comment, 'GitHub ↗')) : null,
    ),
  )
}

function toggleRow(xid) {
  openRows.has(xid) ? openRows.delete(xid) : openRows.add(xid)
  renderChecklist()
  document.getElementById(xid)?.focus({ preventScroll: true })
}

function checklistRow(r, registryUrl) {
  const xid = r.idea.xid
  const open = openRows.has(xid)
  const row = el(
    'div',
    {
      class: `ck-row${r.accepted ? ' live' : ''}${open ? ' open' : ''}`,
      id: xid,
      role: 'row',
      tabindex: '0',
      'aria-expanded': String(open),
      title: r.accepted ? `In the live recipe, ${config.recipe_tag}` : '',
    },
    el(
      'span',
      { class: 'ck-idea', 'data-label': 'Idea' },
      el('span', { class: 'xid mono' }, el('a', { href: `#${xid}` }, xid)),
      ' ',
      link(registryUrl(r.idea), r.idea.name),
      el('span', { class: 'ck-caret', 'aria-hidden': 'true' }, open ? '▾' : '▸'),
    ),
    el('span', { 'data-label': 'Author' }, link(`https://github.com/${r.author}`, `@${r.author}`)),
    el('span', { class: 'mono', 'data-label': 'Date' }, r.date),
    el('span', { 'data-label': 'Dense' }, gateChip(r.dense)),
    el('span', { 'data-label': 'Speedrun' }, gateChip(r.speedrun)),
    el('span', { 'data-label': 'Hero MFU' }, gateChip(r.hero)),
    el('span', { 'data-label': 'Projection' }, gateChip(r.scaling)),
    el('span', { 'data-label': 'Recipe' }, recipeCell(r.idea, config.recipe_tag)),
    el(
      'span',
      { 'data-label': 'Potential' },
      r.score === null
        ? el('span', { class: 'muted', title: r.idea.potential?.rationale ?? '' }, r.idea.potential ? 'n/a' : '–')
        : el('span', { class: 'score', title: `${r.idea.potential.rationale} (${r.idea.potential.model}, ${r.idea.potential.date})` }, `${r.score}/10`),
    ),
    open ? threadDetail(r, config) : null,
  )
  // A click on the row (not on a link, a gate chip or the open thread) or Enter toggles it.
  row.addEventListener('click', (e) => {
    if (e.target.closest('a, .gcell, .ck-detail')) return
    toggleRow(xid)
  })
  row.addEventListener('keydown', (e) => {
    if (e.target !== row || (e.key !== 'Enter' && e.key !== ' ')) return
    e.preventDefault()
    toggleRow(xid)
  })
  return row
}

function wireChecklist() {
  document.getElementById('ck-sort').addEventListener('change', (e) => {
    checklistView.sort = e.target.value
    renderChecklist()
  })
  document.getElementById('ck-filter').addEventListener('input', (e) => {
    checklistView.query = e.target.value
    renderChecklist()
  })
  document.getElementById('log-filter').addEventListener('input', (e) => {
    logView.query = e.target.value
    renderAgentLog()
  })
  window.addEventListener('hashchange', () => openLinkedRow())
}

// Every agent post, one per idea, each updated in place as the idea moves on; the latest update first.
function renderAgentLog() {
  const q = logView.query.toLowerCase()
  const posts = [...(state.threads ?? [])]
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .filter((t) => !q || `${t.registry_id} ${t.title} ${t.body}`.toLowerCase().includes(q))
  document.getElementById('log-n').textContent = String(posts.length)
  if (!posts.length) {
    setChildren('agent-log', el('li', { class: 'none' }, q ? 'No posts match' : 'No posts yet'))
    return
  }
  setChildren(
    'agent-log',
    posts.map((t) => {
      const open = openPosts.has(t.registry_id)
      return el(
        'li',
        { class: open ? 'open' : '' },
        el(
          'button',
          {
            type: 'button',
            class: 'log-head linkish',
            'aria-expanded': String(open),
            onclick: () => {
              open ? openPosts.delete(t.registry_id) : openPosts.add(t.registry_id)
              renderAgentLog()
            },
          },
          el('span', { class: 'ft' }, t.title),
          when(t.updated_at),
        ),
        open ? el('div', { class: 'log-body' }, markdown(t.body), el('a', { href: `#${t.registry_id}` }, `${t.registry_id} in the checklist`)) : null,
      )
    }),
  )
}

function renderFeed() {
  const items = state.activity.slice(0, 15)
  if (!items.length) {
    setChildren('feed', el('li', { class: 'none' }, 'Nothing yet'))
    return
  }
  setChildren(
    'feed',
    items.map((a) =>
      el(
        'li',
        { dataset: { kind: a.kind } },
        el('span', { class: 'ic', 'aria-hidden': 'true' }, FEED_ICON[a.kind] ?? '•'),
        el('span', { class: 'ft' }, a.registry_id ? [el('code', {}, a.registry_id), ' '] : null, a.text),
        when(a.at),
      ),
    ),
  )
}

function render() {
  renderNext(state.next)
  renderPipeline(state.pipeline)
  renderRunning()
  renderRecent()
  renderChecklist()
  renderAgentLog()
  renderFeed()
  setChildren('updated', 'updated ', when(state.now))
}

// status#X#### (from a #9514 reply, the agent log or the admin page) opens that idea's row and scrolls to it.
let linkedRowShown = false
function openLinkedRow() {
  const xid = decodeURIComponent(location.hash.slice(1))
  if (xid === 'agent-log') document.getElementById('agent-log-block')?.scrollIntoView({ block: 'start' })
  if (!XID.test(xid) || !registry) return
  openRows.add(xid)
  renderChecklist()
  document.getElementById(xid)?.scrollIntoView({ block: 'start' })
}

async function refresh() {
  try {
    ;[state, registry, config] = await Promise.all([
      loadJson('state'),
      loadJson('results'),
      fetch(asset('tracks.json')).then((r) => r.json()),
    ])
    render()
    if (!linkedRowShown) {
      linkedRowShown = true
      openLinkedRow()
    }
  } catch (error) {
    renderNext({ tone: 'blocked', text: `Status unavailable (${error.message})` })
  }
}

showView('status')
wireChecklist()
refresh()
setInterval(refresh, REFRESH_MS)
