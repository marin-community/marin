import { h } from 'preact';
import htm from 'htm';

const html = htm.bind(h);

export function EndpointsTab({ endpoints }) {
  if (!endpoints || endpoints.length === 0) {
    return html`<div class="no-jobs">No endpoints registered</div>`;
  }

  return html`
    <table>
      <thead><tr><th>Name</th><th>Address</th><th>Job</th><th>Metadata</th></tr></thead>
      <tbody>
        ${endpoints.map(e => {
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
    </table>`;
}
