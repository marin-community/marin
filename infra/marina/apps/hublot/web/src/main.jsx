import { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Flow } from './Flow.jsx';
import { InspectorCard } from './Inspector.jsx';
import { NodeInfo } from './NodeInfo.jsx';
import { ChartLegend, FlowLegend, TimeLegend } from './Legends.jsx';
import { flowGraph, MAX_WINDOW_MINUTES, MIN_WINDOW_MS } from './contracts.js';
import { cardsWithSamples, elapsed, shortName, utc } from './display.js';
import { createRunModel } from './run.js';
import './style.css';

const minRange = MIN_WINDOW_MS;
const appBase = import.meta.env.BASE_URL;

function urlParams(run, window) {
  const params = new URLSearchParams({ run });
  if (window) {
    params.set('from', window.from);
    params.set('to', window.to);
  }
  return '?' + params.toString();
}

function windowFields(params) {
  const toISO = (key) => {
    const value = params.get(key);
    const date = new Date(Number(value));
    return value && Number.isFinite(date.getTime()) ? date.toISOString() : '';
  };
  return { from: toISO('from'), to: toISO('to') };
}

function TimeBar({ loaded, range, setRange, cursor, waits, origin, hasRunStart, reset }) {
  const duration = loaded.to - loaded.from;
  const left = (range.from - loaded.from) / duration * 100;
  const right = (range.to - loaded.from) / duration * 100;
  const zoomed = range.from !== loaded.from || range.to !== loaded.to;
  const sliderEnd = Math.ceil(duration / 1000);
  const sliderValue = (time) => time === loaded.to ? sliderEnd : Math.round((time - loaded.from) / 1000);
  const timestamp = (event) => Math.min(loaded.to, loaded.from + Number(event.target.value) * 1000);
  const elapsedText = (time) => elapsed(time, origin) + ' elapsed. ' + utc(time);
  return (
    <section className="time-bar" aria-label="Shared time range">
      <div className="time-display">
        <strong>{hasRunStart ? 'Run time' : 'Window time'}</strong>
        {zoomed && <span data-testid="selected-range" title={utc(range.from) + ' to ' + utc(range.to)}>
          {elapsed(range.from, origin)}–{elapsed(range.to, origin)}
        </span>}
      </div>
      <div className="time-ruler">
        <div className="ruler-track" aria-hidden="true">
          <i className="ruler-range" style={{ left: left + '%', right: (100 - right) + '%' }} />
          {waits.map((span) => {
            const start = (Math.max(span.start, loaded.from) - loaded.from) / duration * 100;
            const end = (Math.min(span.end, loaded.to) - loaded.from) / duration * 100;
            return <i key={span.id} className="ruler-wait" style={{ left: start + '%', width: (end - start) + '%' }} />;
          })}
          {cursor !== null && <i className="ruler-cursor" style={{ left: ((cursor - loaded.from) / duration * 100) + '%' }} />}
        </div>
        <input type="range" aria-label="Range start" aria-valuetext={elapsedText(range.from)}
          min={0} max={sliderEnd} value={sliderValue(range.from)}
          onChange={(event) => setRange({ ...range, from: Math.min(timestamp(event), range.to - minRange) })} />
        <input type="range" aria-label="Range end" aria-valuetext={elapsedText(range.to)}
          min={0} max={sliderEnd} value={sliderValue(range.to)}
          onChange={(event) => setRange({ ...range, to: Math.max(timestamp(event), range.from + minRange) })} />
        <div className="ruler-endpoints" aria-hidden="true">
          <span>{elapsed(loaded.from, origin)}</span><span>{elapsed(loaded.to, origin)}</span>
        </div>
      </div>
      <div className="time-actions">
        <TimeLegend />
        <button type="button" onClick={reset} disabled={range.from === loaded.from && range.to === loaded.to}>Full window</button>
      </div>
    </section>
  );
}

function RunHeading({ model, onLoadFull }) {
  const { start_ms, end_ms, inferenceGpus, policyGpus, fullRun, batchPolicy, trainingRows } = model.run;
  const canLoadFull = fullRun && (fullRun.from !== model.window.from || fullRun.to !== model.window.to);
  return (
    <section className="run-heading" aria-label="Resolved execution"
      data-execution={model.run.execution_uid} data-cluster={model.run.cluster} data-job={model.run.job_id}>
      <div className="run-title">
        <time>{new Date(model.window.from).toISOString().slice(0, 10)}</time>
        <h2 title={model.run.run_id ?? model.run.job_id}>{shortName(model.run.run_id || model.run.job_id)}</h2>
      </div>
      <div className="run-origin">
        <span>{model.run.cluster}</span>
        <details>
          <summary>Run details</summary>
          <dl>
            <dt>Run ID</dt><dd>{model.run.run_id || 'Unavailable'}</dd>
            <dt>Execution UID</dt><dd>{model.run.execution_uid}</dd>
            <dt>Iris job</dt><dd>{model.run.job_id || 'Unavailable'}</dd>
            <dt>Source</dt><dd>Finelog · {model.deployment} · {model.namespace}</dd>
            {start_ms != null && <><dt>Run started</dt><dd>{utc(start_ms)}</dd></>}
            {end_ms != null && <><dt>Run ended</dt><dd>{utc(end_ms)} · {elapsed(end_ms, start_ms)} elapsed</dd></>}
            {(inferenceGpus || policyGpus) && <>
              <dt>Configured GPUs</dt>
              <dd>{[inferenceGpus && inferenceGpus + ' inference', policyGpus && policyGpus + ' trainer']
                .filter(Boolean).join(' · ')}</dd>
            </>}
            {(batchPolicy || trainingRows != null || model.batchTarget != null) && <>
              <dt>Batching</dt><dd>{[
                batchPolicy === 'full_batch' ? 'Full batch' : batchPolicy === 'rolling' ? 'Rolling' : null,
                trainingRows != null && trainingRows + ' training rows',
                model.batchTarget != null && model.batchTarget + ' groups per batch',
              ].filter(Boolean).join(' · ')}</dd>
            </>}
            {model.maxStaleness != null && <><dt>Staleness limit</dt><dd>{model.maxStaleness} policy steps</dd></>}
            <dt>Loaded window</dt><dd>{utc(model.window.from)} to {utc(model.window.to)}</dd>
            {model.run.driverReadFailed && <><dt>Driver logs</dt><dd>Finelog could not load driver timings for this window.</dd></>}
            {canLoadFull && <dd><button type="button" onClick={() => onLoadFull(model.run.id, fullRun)}>Open full run</button></dd>}
          </dl>
        </details>
      </div>
    </section>
  );
}

function App() {
  const [navigation, setNavigation] = useState(() => ({ search: window.location.search }));
  const initial = new URLSearchParams(navigation.search);
  const [runInput, setRunInput] = useState(initial.get('run') ?? '');
  const [fields, setFields] = useState(() => windowFields(initial));
  const [inputError, setInputError] = useState('');
  const [model, setModel] = useState(null);
  const [load, setLoad] = useState({ state: 'idle' });
  const [expanded, setExpanded] = useState(false);
  const [range, setRange] = useState(null);
  const [hover, setHover] = useState(null);
  const [point, setPoint] = useState(null);
  const activeCursor = hover ?? point;
  const cursor = activeCursor?.time ?? null;
  const [selected, setSelected] = useState(null);
  const [infoNode, setInfoNode] = useState(null);
  const [pins, setPins] = useState([]);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [sideWidth, setSideWidth] = useState(438);
  const [resizing, setResizing] = useState(false);
  const workspace = useRef(null);
  const cardRefs = useRef(new Map());
  const availableCards = useMemo(() => model ? cardsWithSamples(model) : new Set(), [model]);
  const edgeMap = useMemo(() => model
    ? new Map(flowGraph(expanded, model.run.separateGrading).edges.map((edge) => [edge.id, edge]))
    : new Map(), [expanded, model]);

  useEffect(() => {
    const pop = () => setNavigation({ search: window.location.search });
    window.addEventListener('popstate', pop);
    return () => window.removeEventListener('popstate', pop);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(navigation.search);
    const id = params.get('run');
    setRunInput(id ?? '');
    setFields(windowFields(params));
    setInputError('');
    setModel(null);
    setHover(null);
    setPoint(null);
    setExpanded(false);
    setPins([]);
    setSelected(null);
    setInfoNode(null);
    setInspectorOpen(false);
    if (!id) {
      setLoad({ state: 'idle' });
      return;
    }
    setLoad({ state: 'loading' });
    const pending = new AbortController();
    (async () => {
      try {
        const response = await fetch(appBase + 'api/run' + navigation.search, { signal: pending.signal });
        const result = await response.json();
        if (pending.signal.aborted) return;
        if (!response.ok) {
          setLoad({ state: 'error', message: result.error ?? result.detail, choices: result.choices });
          return;
        }
        setModel(createRunModel(result));
        setRange(result.window);
        setFields(windowFields(new URLSearchParams({ from: result.window.from, to: result.window.to })));
        setLoad({ state: 'ready' });
      } catch (error) {
        if (!pending.signal.aborted) {
          setLoad({ state: 'error', message: 'Hublot could not load the run from Marina.' });
        }
      }
    })();
    return () => pending.abort();
  }, [navigation]);

  useEffect(() => {
    if (!inspectorOpen || !selected) return;
    const frame = requestAnimationFrame(() => cardRefs.current.get(selected)?.scrollIntoView({ block: 'nearest' }));
    return () => cancelAnimationFrame(frame);
  }, [inspectorOpen, selected]);

  function openRun(run, bounds) {
    if (!run.trim()) {
      setInputError('Enter a run ID.');
      return;
    }
    const search = urlParams(run.trim(), bounds);
    if (window.location.search !== search) history.pushState(null, '', search);
    setNavigation({ search });
  }

  function openWindow(event) {
    event.preventDefault();
    const endsInZone = (text) => /(?:Z|[+-]\d\d:\d\d)$/.test(text);
    const from = Date.parse(fields.from);
    const to = Date.parse(fields.to);
    if (!endsInZone(fields.from) || !endsInZone(fields.to) || !Number.isFinite(from) ||
        !Number.isFinite(to) || to - from < minRange || to - from > MAX_WINDOW_MINUTES * 60_000) {
      setInputError('Enter ISO times with a UTC offset, ' + MIN_WINDOW_MS / 1000 + ' seconds to ' + MAX_WINDOW_MINUTES + ' minutes apart.');
      return;
    }
    openRun(runInput, { from, to });
  }

  function selectRange(next) {
    setHover(null);
    setPoint((current) => current && current.time >= next.from && current.time < next.to ? current : null);
    setRange(next);
  }

  function focusSpan(span) {
    const from = Math.max(model.window.from, Math.min(span.start, model.window.to - minRange));
    const to = Math.min(model.window.to, Math.max(span.end + 1, from + minRange));
    selectRange({ from, to });
  }

  function inspect(id) {
    if (!availableCards.has(id)) return;
    if (id === 'dispatch' || id === 'workers') setExpanded(true);
    setInspectorOpen(true);
    setSelected(id);
    if (point && point.id !== id && !pins.includes(point.id)) setPoint(null);
    if (inspectorOpen && selected === id) cardRefs.current.get(id)?.scrollIntoView({ block: 'nearest' });
  }

  function closeCard(id) {
    setPins((current) => current.filter((pin) => pin !== id));
    if (selected === id) setSelected(null);
    if (point?.id === id) setPoint(null);
  }

  function moveCard(id, direction) {
    setPins((current) => {
      const next = [...current];
      const index = next.indexOf(id);
      const destination = index + direction;
      if (destination < 0 || destination >= next.length) return current;
      [next[index], next[destination]] = [next[destination], next[index]];
      return next;
    });
  }

  function resizeSidebar(width) {
    const size = workspace.current.clientWidth;
    setSideWidth(Math.max(322, Math.min(width, size - 445)));
  }

  const waits = model?.phases.filter(({ kind, start, end }) =>
    kind === 'rollout_wait' && start < model.window.to && end >= model.window.from) ?? [];
  const chartOrder = selected && !pins.includes(selected) ? [selected, ...pins] : pins;
  const origin = model?.run.start_ms ?? model?.window.from;
  return (
    <div className="app">
      <header className="top-bar">
        <div className="brand">
          <img className="brand-mark" src={appBase + 'hublot.svg'} width="44" height="44" alt="" />
          <div><h1>Hublot</h1><p>RL performance</p></div>
        </div>
        <div className="top-controls">
          <form className="run-form" onSubmit={(event) => { event.preventDefault(); openRun(runInput); }}>
            <label className="sr-only" htmlFor="run-id">Run ID</label>
            <input id="run-id" placeholder="Paste a Finelog run ID or Iris execution UID"
              spellCheck={false} autoComplete="off" value={runInput}
              onChange={(event) => setRunInput(event.target.value)} />
            <button type="submit" className="primary-button">Open run</button>
          </form>
          <details className="load-window">
            <summary>Load window</summary>
            <form onSubmit={openWindow}>
              <label>From<input placeholder="2026-11-21T00:27:00Z"
                value={fields.from} onChange={(event) => setFields({ ...fields, from: event.target.value })} /></label>
              <label>To<input placeholder="2026-11-21T00:54:00Z"
                value={fields.to} onChange={(event) => setFields({ ...fields, to: event.target.value })} /></label>
              <button className="primary-button" type="submit">Load UTC window</button>
            </form>
          </details>
        </div>
      </header>
      {inputError && <div className="form-error" role="alert">{inputError}</div>}
      {!model && <main className="empty">
        {load.state === 'idle' && <div className="empty-box">
          <h2>Open a run</h2>
          <p>Paste an exact run ID, Iris job path, or execution UID to inspect activity across the loop.</p>
        </div>}
        {load.state === 'loading' && <div className="empty-box" role="status">
          <h2>Reading Finelog</h2><p>Resolving the execution and loading the window.</p>
        </div>}
        {load.state === 'error' && <div className="empty-box" role="alert">
          <h2>Run unavailable</h2><p>{load.message}</p>
          {load.choices?.length > 0 && <ul className="run-choices">
            {load.choices.map((choice) => <li key={choice.id}>
              <button type="button" onClick={() => openRun(choice.id, initial.has('from')
                ? { from: initial.get('from'), to: initial.get('to') } : undefined)}>
                <strong>{choice.cluster} · {choice.execution_uid}</strong>
                <small>{choice.job_id} · {utc(choice.last_ms)}</small>
              </button>
            </li>)}
          </ul>}
        </div>}
      </main>}
      {model && range && <main className="dashboard">
        <RunHeading model={model} onLoadFull={openRun} />
        <TimeBar loaded={model.window} range={range} setRange={selectRange} cursor={cursor}
          waits={waits} origin={origin} hasRunStart={model.run.start_ms != null}
          reset={() => selectRange(model.window)} />
        <div className={'workspace' + (resizing ? ' resizing' : '') + (inspectorOpen ? '' : ' inspector-folded')} ref={workspace}
          style={{ '--sidebar': sideWidth + 'px' }}>
          <section className="flow-pane" aria-label="Rollout flow">
            <div className="pane-heading">
              <h2>Flow</h2>
              <FlowLegend />
            </div>
            <Flow model={model} availableCards={availableCards} expanded={expanded} setExpanded={setExpanded} selected={selected}
              onInspect={inspect} onInfo={setInfoNode} range={range} cursor={cursor} resizing={resizing} />
            <div className="flow-foot">
              <span>{cursor === null ? 'Latest sample in view' : 'At ' + elapsed(cursor, origin) + ' elapsed'}</span>
              <span>Click a node or stream to inspect it</span>
            </div>
          </section>
          {inspectorOpen && <div className="sidebar-handle" role="separator" aria-label="Resize inspector"
            aria-orientation="vertical" aria-valuemin={322} aria-valuenow={sideWidth}
            tabIndex={0} onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              setResizing(true);
            }} onPointerMove={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                resizeSidebar(workspace.current.getBoundingClientRect().right - event.clientX);
              }
            }} onLostPointerCapture={() => setResizing(false)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                event.preventDefault();
                resizeSidebar(sideWidth + (event.key === 'ArrowLeft' ? 24 : -24));
              }
            }} />}
          <aside className="inspector" aria-label="Inspector">
            <div className="pane-heading inspect-heading">
              <h2>Inspector</h2>
              {inspectorOpen && <ChartLegend phases={model.phases} />}
              <button type="button" className="fold-button"
                title={inspectorOpen ? 'Fold inspector' : 'Open inspector'}
                aria-label={inspectorOpen ? 'Fold inspector' : 'Open inspector'}
                aria-expanded={inspectorOpen} aria-controls="inspector-charts"
                onClick={() => { setHover(null); setInspectorOpen(!inspectorOpen); }}>
                <i className="panel-icon" aria-hidden="true" />
              </button>
            </div>
            <div id="inspector-charts" className="inspect-stack" data-testid="chart-stack" hidden={!inspectorOpen}>
              {!chartOrder.length && <p className="inspector-empty">Select a node or stream on the graph.</p>}
              {chartOrder.map((id) => <InspectorCard key={id} id={id} edge={edgeMap.get(id)}
                model={model} availableCards={availableCards} range={range} origin={origin} hover={hover} setHover={setHover}
                cursor={activeCursor} point={point?.id === id ? point : null} onPoint={setPoint} onInspect={inspect}
                onPin={(pin) => setPins((current) => current.includes(pin) ? current : [...current, pin])}
                onClose={closeCard} onMove={moveCard} onFocusSpan={focusSpan}
                pinned={pins.includes(id)} pinIndex={pins.indexOf(id)} pinCount={pins.length}
                active={selected === id} register={(node) => {
                  if (node) cardRefs.current.set(id, node);
                  else cardRefs.current.delete(id);
                }} />)}
            </div>
          </aside>
        </div>
        {infoNode && <NodeInfo id={infoNode} model={model} onClose={() => setInfoNode(null)} />}
      </main>}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
