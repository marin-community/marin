import { h } from 'preact';
import htm from 'htm';

const html = htm.bind(h);

export function UsersTab({ users }) {
  if (!users || users.length === 0) {
    return html`<div class="no-jobs">No users found</div>`;
  }

  return html`
    <table>
      <thead>
        <tr>
          <th>User</th>
          <th>Active Jobs</th>
          <th>Running Jobs</th>
          <th>Pending Jobs</th>
          <th>Total Tasks</th>
          <th>Running Tasks</th>
          <th>Completed Tasks</th>
        </tr>
      </thead>
      <tbody>
        ${users.map(user => html`
          <tr>
            <td>${user.user || '-'}</td>
            <td>${user.activeJobs || 0}</td>
            <td>${user.runningJobs || 0}</td>
            <td>${user.pendingJobs || 0}</td>
            <td>${user.totalTasks || 0}</td>
            <td>${user.runningTasks || 0}</td>
            <td>${user.completedTasks || 0}</td>
          </tr>
        `)}
      </tbody>
    </table>
  `;
}
