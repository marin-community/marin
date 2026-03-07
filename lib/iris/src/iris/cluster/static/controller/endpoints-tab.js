import { h } from 'preact';
import { useState, useRef } from 'preact/hooks';
import htm from 'htm';

const html = htm.bind(h);

const DEFAULT_PAGE_SIZE = 100;

export function EndpointsTab({ endpoints, prefix, onPrefixChange }) {
  const [showAll, setShowAll] = useState(false);
  const [localPrefix, setLocalPrefix] = useState(prefix || '');
  const filterInputRef = useRef(null);

  if (!endpoints || endpoints.length === 0) {
    return html`
      <div style="margin-bottom:15px;display:flex;align-items:center;gap:10px">
        <form onSubmit=${handleFilterSubmit} style="display:flex;gap:8px">
          <input
            ref=${filterInputRef}
            type="text"
            placeholder="Filter by prefix..."
            value=${localPrefix}
            onInput=${e => setLocalPrefix(e.target.value)}
            style="padding:6px 10px;border:1px solid #d1d9e0;border-radius:4px;font-size:13px;width:250px"
          />
          <button type="submit" style="padding:6px 12px;font-size:13px">Filter</button>
          ${prefix && html`<button type="button" onClick=${handleFilterClear} style="padding:6px 12px;font-size:13px">Clear</button>`}
        </form>
      </div>
      <div class="no-jobs">No endpoints${prefix ? ' matching prefix' : ' registered'}</div>`;
  }

  function handleFilterSubmit(e) {
    e.preventDefault();
    setShowAll(false);
    onPrefixChange(localPrefix);
  }

  function handleFilterClear() {
    setLocalPrefix('');
    setShowAll(false);
    onPrefixChange('');
  }

  const totalCount = endpoints.length;
  const displayEndpoints = showAll ? endpoints : endpoints.slice(0, DEFAULT_PAGE_SIZE);
  const hasMore = totalCount > DEFAULT_PAGE_SIZE && !showAll;

  return html`
    <div style="margin-bottom:15px;display:flex;align-items:center;gap:10px">
      <form onSubmit=${handleFilterSubmit} style="display:flex;gap:8px">
        <input
          ref=${filterInputRef}
          type="text"
          placeholder="Filter by prefix..."
          value=${localPrefix}
          onInput=${e => setLocalPrefix(e.target.value)}
          style="padding:6px 10px;border:1px solid #d1d9e0;border-radius:4px;font-size:13px;width:250px"
        />
        <button type="submit" style="padding:6px 12px;font-size:13px">Filter</button>
        ${prefix && html`<button type="button" onClick=${handleFilterClear} style="padding:6px 12px;font-size:13px">Clear</button>`}
      </form>
      <span style="color:#57606a;font-size:13px">${totalCount} endpoint${totalCount !== 1 ? 's' : ''}${!showAll && hasMore ? ', showing ' + DEFAULT_PAGE_SIZE : ''}</span>
    </div>
    <table>
      <thead><tr><th>Name</th><th>Address</th><th>Job</th><th>Metadata</th></tr></thead>
      <tbody>
        ${displayEndpoints.map(e => {
          const metaStr = e.metadata
            ? Object.entries(e.metadata).map(([k, v]) => k + '=' + v).join(', ')
            : '-';
          return html`<tr>
            <td>${e.name}</td>
            <td>${e.address}</td>
            <td><a href=${'/job/' + e.job_id} class="job-link">${(e.job_id || '').slice(0, 8)}...</a></td>
            <td>${metaStr}</td>
          </tr>`;
        })}
      </tbody>
    </table>
    ${hasMore && html`
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px">
        <button onClick=${() => setShowAll(true)} style="padding:6px 12px;font-size:13px">
          Show all ${totalCount} endpoints
        </button>
      </div>
    `}`;
}
