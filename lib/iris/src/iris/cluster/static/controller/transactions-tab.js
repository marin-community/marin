import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { timestampFromProto } from '/static/shared/utils.js';

const html = htm.bind(h);

function formatTime(timestampMs) {
  if (!timestampMs) return '-';
  const date = new Date(timestampMs);
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  const milliseconds = String(date.getMilliseconds()).padStart(3, '0');
  return `${hours}:${minutes}:${seconds}.${milliseconds}`;
}

function formatDetails(detailsJson) {
  if (!detailsJson) return '-';
  try {
    const details = JSON.parse(detailsJson);
    const entries = Object.entries(details)
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    return entries || '-';
  } catch (e) {
    return detailsJson;
  }
}

export function TransactionsTab() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTransactions = async () => {
    try {
      const resp = await controllerRpc('GetTransactions', { limit: 200 });
      const actions = (resp.actions || []).map(action => ({
        timestamp: timestampFromProto(action.timestamp),
        action: action.action,
        entity_id: action.entityId,
        details: action.details
      }));
      setTransactions(actions);
      setError(null);
    } catch (e) {
      setError('Failed to load transactions: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTransactions();
    const interval = setInterval(fetchTransactions, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return html`<div>Loading transactions...</div>`;
  }

  if (error) {
    return html`<div class="error-message">${error}</div>`;
  }

  if (!transactions || transactions.length === 0) {
    return html`<div class="no-jobs">No transactions recorded</div>`;
  }

  return html`
    <table>
      <thead><tr>
        <th>Time</th>
        <th>Action</th>
        <th>Entity</th>
        <th>Details</th>
      </tr></thead>
      <tbody>
        ${transactions.map(tx => html`<tr>
          <td>${formatTime(tx.timestamp)}</td>
          <td>${tx.action}</td>
          <td>${tx.entity_id}</td>
          <td style="font-size:12px">${formatDetails(tx.details)}</td>
        </tr>`)}
      </tbody>
    </table>`;
}
