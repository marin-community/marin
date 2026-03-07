/**
 * Unified log viewer component for Iris dashboards.
 *
 * All log panes (controller process logs, worker process logs, autoscaler logs,
 * task logs) share this single component backed by the FetchLogs RPC.
 */

import { h } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
const html = htm.bind(h);

const LOG_LEVELS = ['', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
const LEVEL_LABELS = { '': 'All Levels', DEBUG: 'Debug', INFO: 'Info', WARNING: 'Warning', ERROR: 'Error', CRITICAL: 'Critical' };
const LEVEL_CLASSES = { 1: 'debug', 2: 'info', 3: 'warning', 4: 'error', 5: 'critical' };

/**
 * Unified log viewer.
 *
 * @param {function} rpc - RPC function (controllerRpc or workerRpc)
 * @param {string} source - Log source key for FetchLogs (e.g. "/process")
 * @param {string} [title] - Section title
 * @param {boolean} [showControls=true] - Show regex/level/limit controls
 * @param {string} [defaultRegex=''] - Pre-filled regex filter
 * @param {string} [defaultLevel=''] - Pre-filled min level
 * @param {number} [defaultMaxLines=500] - Default max lines
 * @param {number} [pollInterval=3000] - Poll interval in ms
 */
export function LogViewer({
  rpc,
  source,
  title = 'Logs',
  showControls = true,
  defaultRegex = '',
  defaultLevel = '',
  defaultMaxLines = 500,
  pollInterval = 3000,
}) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [regex, setRegex] = useState(defaultRegex);
  const [minLevel, setMinLevel] = useState(defaultLevel);
  const [maxLines, setMaxLines] = useState(defaultMaxLines);
  const [regexError, setRegexError] = useState('');
  const containerRef = useRef(null);
  const wasAtBottom = useRef(true);

  const doFetch = useCallback(async () => {
    try {
      const params = {
        source,
        maxLines: maxLines || 500,
        tail: true,
      };
      if (regex) params.substring = regex;
      if (minLevel) params.minLevel = minLevel;

      const resp = await rpc('FetchLogs', params);
      setEntries(resp.entries || []);
      setRegexError('');
      setLoading(false);
    } catch (error) {
      const msg = error.message || '';
      if (msg.includes('regex') || msg.includes('Regex')) {
        setRegexError(msg);
      } else {
        console.error('Failed to fetch logs:', error);
      }
      setLoading(false);
    }
  }, [rpc, source, regex, minLevel, maxLines]);

  // Track scroll position to auto-scroll only when at bottom
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      wasAtBottom.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    };
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    const el = containerRef.current;
    if (el && wasAtBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [entries]);

  useEffect(() => {
    doFetch();
    const interval = setInterval(doFetch, pollInterval);
    return () => clearInterval(interval);
  }, [doFetch, pollInterval]);

  const formatTime = (entry) => {
    const ms = entry.timestamp ? (entry.timestamp.epochMs || 0) : 0;
    if (!ms) return '';
    return new Date(Number(ms)).toLocaleTimeString();
  };

  const levelClass = (entry) => LEVEL_CLASSES[entry.level] || '';

  return html`
    <div class="log-viewer">
      ${title && html`<h2 style="margin:0 0 8px 0;font-size:16px">${title}</h2>`}
      ${showControls && html`
        <div class="log-controls">
          <label class="log-control">
            <span>Filter:</span>
            <input
              type="text"
              placeholder="substring filter"
              value=${regex}
              onInput=${(e) => setRegex(e.target.value)}
              class=${regexError ? 'input-error' : ''}
              style="width:200px"
            />
          </label>
          <label class="log-control">
            <span>Level:</span>
            <select value=${minLevel} onChange=${(e) => setMinLevel(e.target.value)}>
              ${LOG_LEVELS.map(l => html`<option value=${l}>${LEVEL_LABELS[l]}</option>`)}
            </select>
          </label>
          <label class="log-control">
            <span>Lines:</span>
            <select value=${String(maxLines)} onChange=${(e) => setMaxLines(parseInt(e.target.value))}>
              <option value="100">100</option>
              <option value="200">200</option>
              <option value="500">500</option>
              <option value="1000">1000</option>
              <option value="5000">5000</option>
            </select>
          </label>
        </div>
        ${regexError && html`<div class="log-regex-error">${regexError}</div>`}
      `}
      <div class="log-container" ref=${containerRef}>
        ${loading && html`<div class="log-empty">Loading logs...</div>`}
        ${!loading && entries.length === 0 && html`<div class="log-empty">No logs found</div>`}
        ${entries.map((entry, i) => html`
          <div class="log-line ${levelClass(entry)}" key=${i}>
            <span class="log-time">${formatTime(entry)}</span>
            <span class="log-data">${entry.data || ''}</span>
          </div>
        `)}
      </div>
    </div>
  `;
}
